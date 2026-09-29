"""Create, inspect or remove the AWS resources of GeoInv3D's *Batch* backend.

NOT NEEDED for the default EC2 backend (geoinv3d/cloud/ec2.py), which only
uses EC2.  This script needs S3, ECR, Batch and IAM permissions: in an account
where you may only use EC2/EBS, ask the account administrator before running
``apply``.  Every resource is tagged Owner=<your IAM user name>.

    python deploy/setup_aws.py plan         # read-only: what exists, what would be created
    python deploy/setup_aws.py apply        # create what is missing (safe to re-run)
    python deploy/setup_aws.py push-image   # build deploy/Dockerfile, push it to ECR
    python deploy/setup_aws.py teardown     # delete the GeoInv3D resources again

Everything is named ``geoinv3d-*`` and tagged ``Project=geoinv3d`` and ``Owner``:

* S3 bucket ``geoinv3d-<account>-<region>`` (no public access) for job data,
  progress reports and results;
* ECR repository ``geoinv3d-worker`` for the worker image (last 10 kept);
* CloudWatch log group ``/aws/batch/geoinv3d`` (90-day retention);
* IAM: an instance role/profile for the Batch hosts, and a job role that may
  only read and write ``s3://<bucket>/geoinv3d/jobs/*``;
* a security group without inbound rules in the default VPC;
* a managed, on-demand Batch compute environment (c5.xlarge-c5.9xlarge,
  0-MAX_VCPUS vCPUs: nothing runs, and nothing is billed, while idle), a job
  queue and a job definition (12 h timeout per job).

``apply`` writes ``~/.geoinv3d/aws.json``; the API server
(``python -m geoinv3d.api``) reads it to find the bucket, queue and job
definition.  The worker code lives in the image: run ``push-image`` again
after changing anything under ``geoinv3d/``.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path.home() / ".geoinv3d" / "aws.json"
TAGS = {"Project": "geoinv3d"}
ECS_INSTANCE_POLICY = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"


@dataclass
class Settings:
    region: str = "ap-south-1"
    max_vcpus: int = 36
    instance_types: tuple = ("c5.xlarge", "c5.2xlarge", "c5.4xlarge", "c5.9xlarge")
    job_timeout_s: int = 12 * 3600
    log_retention_days: int = 90
    default_vcpus: int = 4          # job definition default; jobs override per instance
    default_memory_mib: int = 7000
    name: str = "geoinv3d"


@dataclass
class Names:
    """Resource names derived from the settings and the account id."""
    account: str
    s: Settings
    bucket: str = field(init=False)

    def __post_init__(self):
        n, r = self.s.name, self.s.region
        self.bucket = f"{n}-{self.account}-{r}"
        self.ecr_repo = f"{n}-worker"
        self.log_group = f"/aws/batch/{n}"
        self.instance_role = f"{n}-batch-instance-role"
        self.instance_profile = f"{n}-batch-instance-profile"
        self.job_role = f"{n}-worker-job-role"
        self.security_group = f"{n}-batch"
        self.compute_env = f"{n}-ondemand"
        self.job_queue = f"{n}-queue"
        self.job_definition = f"{n}-worker"
        self.image = f"{self.account}.dkr.ecr.{r}.amazonaws.com/{self.ecr_repo}:latest"


def _code(err) -> str:
    return getattr(err, "response", {}).get("Error", {}).get("Code", "")


def job_role_policy(bucket: str) -> dict:
    """The worker may only read and write job files under geoinv3d/jobs/."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
             "Resource": f"arn:aws:s3:::{bucket}/geoinv3d/jobs/*"},
            {"Effect": "Allow", "Action": "s3:ListBucket",
             "Resource": f"arn:aws:s3:::{bucket}",
             "Condition": {"StringLike": {"s3:prefix": "geoinv3d/jobs/*"}}},
        ],
    }


def trust_policy(service: str) -> dict:
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]}


class Deployer:
    """Checks and creates the resources in order; ``clients`` maps service -> client."""

    def __init__(self, settings: Settings, clients: dict, log=print):
        self.s, self.c, self.log = settings, clients, log
        identity = clients["sts"].get_caller_identity()
        self.n = Names(identity["Account"], settings)
        TAGS["Owner"] = identity["Arn"].split("/")[-1]   # required on every created resource

    # ── existence checks (read-only) ──
    def _exists(self, call, missing_codes=(), **kw) -> bool:
        try:
            call(**kw)
            return True
        except Exception as e:
            if _code(e) in missing_codes:
                return False
            raise

    def has_bucket(self):
        return self._exists(self.c["s3"].head_bucket, ("404", "NoSuchBucket", "NotFound"),
                            Bucket=self.n.bucket)

    def has_ecr_repo(self):
        return self._exists(self.c["ecr"].describe_repositories, ("RepositoryNotFoundException",),
                            repositoryNames=[self.n.ecr_repo])

    def has_log_group(self):
        groups = self.c["logs"].describe_log_groups(logGroupNamePrefix=self.n.log_group)
        return any(g["logGroupName"] == self.n.log_group for g in groups["logGroups"])

    def _has_role(self, name):
        return self._exists(self.c["iam"].get_role, ("NoSuchEntity",), RoleName=name)

    def has_instance_role(self):
        return self._has_role(self.n.instance_role)

    def has_instance_profile(self):
        return self._exists(self.c["iam"].get_instance_profile, ("NoSuchEntity",),
                            InstanceProfileName=self.n.instance_profile)

    def has_job_role(self):
        return self._has_role(self.n.job_role)

    def default_vpc(self):
        vpcs = self.c["ec2"].describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
        if not vpcs["Vpcs"]:
            raise RuntimeError(f"No default VPC in {self.s.region}")
        vpc = vpcs["Vpcs"][0]["VpcId"]
        subnets = self.c["ec2"].describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc]}])["Subnets"]
        return vpc, [sn["SubnetId"] for sn in subnets]

    def security_group_id(self):
        vpc, _ = self.default_vpc()
        groups = self.c["ec2"].describe_security_groups(Filters=[
            {"Name": "group-name", "Values": [self.n.security_group]},
            {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
        return groups[0]["GroupId"] if groups else None

    def has_security_group(self):
        return self.security_group_id() is not None

    def _batch_state(self, kind):
        if kind == "ce":
            items = self.c["batch"].describe_compute_environments(
                computeEnvironments=[self.n.compute_env])["computeEnvironments"]
        else:
            items = self.c["batch"].describe_job_queues(jobQueues=[self.n.job_queue])["jobQueues"]
        items = [i for i in items if i.get("status") != "DELETED"]
        return items[0] if items else None

    def has_compute_env(self):
        return self._batch_state("ce") is not None

    def has_job_queue(self):
        return self._batch_state("queue") is not None

    def _job_definition_properties(self):
        role_arn = f"arn:aws:iam::{self.n.account}:role/{self.n.job_role}"
        return {
            "image": self.n.image,
            "resourceRequirements": [
                {"type": "VCPU", "value": str(self.s.default_vcpus)},
                {"type": "MEMORY", "value": str(self.s.default_memory_mib)}],
            "jobRoleArn": role_arn,
            "logConfiguration": {"logDriver": "awslogs", "options": {
                "awslogs-group": self.n.log_group, "awslogs-region": self.s.region,
                "awslogs-stream-prefix": "worker"}},
        }

    def has_job_definition(self):
        want = self._job_definition_properties()
        defs = self.c["batch"].describe_job_definitions(
            jobDefinitionName=self.n.job_definition, status="ACTIVE")["jobDefinitions"]
        for d in defs:
            cp = d.get("containerProperties", {})
            if (cp.get("image") == want["image"] and cp.get("jobRoleArn") == want["jobRoleArn"]
                    and d.get("timeout", {}).get("attemptDurationSeconds") == self.s.job_timeout_s):
                return True
        return False

    # ── creation ──
    def create_bucket(self):
        s3 = self.c["s3"]
        kw = {} if self.s.region == "us-east-1" else {
            "CreateBucketConfiguration": {"LocationConstraint": self.s.region}}
        s3.create_bucket(Bucket=self.n.bucket, **kw)
        s3.put_public_access_block(Bucket=self.n.bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        s3.put_bucket_tagging(Bucket=self.n.bucket, Tagging={
            "TagSet": [{"Key": k, "Value": v} for k, v in TAGS.items()]})
        s3.put_bucket_lifecycle_configuration(Bucket=self.n.bucket, LifecycleConfiguration={
            "Rules": [{"ID": "abort-incomplete-uploads", "Status": "Enabled",
                       "Filter": {"Prefix": ""},
                       "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}}]})

    def create_ecr_repo(self):
        ecr = self.c["ecr"]
        ecr.create_repository(repositoryName=self.n.ecr_repo,
                              imageScanningConfiguration={"scanOnPush": True},
                              tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])
        ecr.put_lifecycle_policy(repositoryName=self.n.ecr_repo, lifecyclePolicyText=json.dumps({
            "rules": [{"rulePriority": 1, "description": "keep the last 10 images",
                       "selection": {"tagStatus": "any", "countType": "imageCountMoreThan",
                                     "countNumber": 10},
                       "action": {"type": "expire"}}]}))

    def create_log_group(self):
        logs = self.c["logs"]
        logs.create_log_group(logGroupName=self.n.log_group, tags=TAGS)
        logs.put_retention_policy(logGroupName=self.n.log_group,
                                  retentionInDays=self.s.log_retention_days)

    def create_instance_role(self):
        iam = self.c["iam"]
        iam.create_role(RoleName=self.n.instance_role,
                        AssumeRolePolicyDocument=json.dumps(trust_policy("ec2.amazonaws.com")),
                        Description="GeoInv3D AWS Batch container instances",
                        Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])
        iam.attach_role_policy(RoleName=self.n.instance_role, PolicyArn=ECS_INSTANCE_POLICY)

    def create_instance_profile(self):
        iam = self.c["iam"]
        iam.create_instance_profile(InstanceProfileName=self.n.instance_profile,
                                    Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])
        iam.add_role_to_instance_profile(InstanceProfileName=self.n.instance_profile,
                                         RoleName=self.n.instance_role)

    def create_job_role(self):
        iam = self.c["iam"]
        iam.create_role(RoleName=self.n.job_role,
                        AssumeRolePolicyDocument=json.dumps(trust_policy("ecs-tasks.amazonaws.com")),
                        Description="GeoInv3D worker: read/write its job files in S3",
                        Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])
        iam.put_role_policy(RoleName=self.n.job_role, PolicyName="geoinv3d-job-files",
                            PolicyDocument=json.dumps(job_role_policy(self.n.bucket)))

    def create_security_group(self):
        vpc, _ = self.default_vpc()
        self.c["ec2"].create_security_group(
            GroupName=self.n.security_group, VpcId=vpc,
            Description="GeoInv3D Batch hosts: no inbound, all outbound",
            TagSpecifications=[{"ResourceType": "security-group",
                                "Tags": [{"Key": k, "Value": v} for k, v in TAGS.items()]}])

    def create_compute_env(self):
        _, subnets = self.default_vpc()
        profile_arn = self.c["iam"].get_instance_profile(
            InstanceProfileName=self.n.instance_profile)["InstanceProfile"]["Arn"]
        params = dict(
            computeEnvironmentName=self.n.compute_env, type="MANAGED", state="ENABLED",
            computeResources={
                "type": "EC2", "allocationStrategy": "BEST_FIT_PROGRESSIVE",
                "minvCpus": 0, "desiredvCpus": 0, "maxvCpus": self.s.max_vcpus,
                "instanceTypes": list(self.s.instance_types), "subnets": subnets,
                "securityGroupIds": [self.security_group_id()],
                "instanceRole": profile_arn,
                "ec2Configuration": [{"imageType": "ECS_AL2023"}],
                "tags": {**TAGS, "Name": f"{self.s.name}-batch"},
            },
            tags=TAGS,
        )
        # A new instance profile takes a few seconds to become usable
        for attempt in range(12):
            try:
                self.c["batch"].create_compute_environment(**params)
                break
            except Exception as e:
                if _code(e) == "ClientException" and "profile" in str(e).lower() and attempt < 11:
                    time.sleep(5)
                    continue
                raise
        self._wait_valid("ce")

    def create_job_queue(self):
        self.c["batch"].create_job_queue(
            jobQueueName=self.n.job_queue, state="ENABLED", priority=1,
            computeEnvironmentOrder=[{"order": 1, "computeEnvironment": self.n.compute_env}],
            tags=TAGS)
        self._wait_valid("queue")

    def create_job_definition(self):
        self.c["batch"].register_job_definition(
            jobDefinitionName=self.n.job_definition, type="container",
            containerProperties=self._job_definition_properties(),
            retryStrategy={"attempts": 1},
            timeout={"attemptDurationSeconds": self.s.job_timeout_s},
            propagateTags=True, tags=TAGS)

    def _wait_valid(self, kind, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            item = self._batch_state(kind)
            status = item and item.get("status")
            if status == "VALID":
                return
            if status == "INVALID":
                raise RuntimeError(f"{kind} is INVALID: {item.get('statusReason')}")
            time.sleep(5)
        raise TimeoutError(f"{kind} did not become VALID in {timeout} s")

    def steps(self):
        n = self.n
        return [
            (f"S3 bucket {n.bucket}", self.has_bucket, self.create_bucket),
            (f"ECR repository {n.ecr_repo}", self.has_ecr_repo, self.create_ecr_repo),
            (f"Log group {n.log_group}", self.has_log_group, self.create_log_group),
            (f"IAM role {n.instance_role}", self.has_instance_role, self.create_instance_role),
            (f"Instance profile {n.instance_profile}", self.has_instance_profile,
             self.create_instance_profile),
            (f"IAM role {n.job_role}", self.has_job_role, self.create_job_role),
            (f"Security group {n.security_group}", self.has_security_group,
             self.create_security_group),
            (f"Batch compute environment {n.compute_env} (on-demand, 0-{self.s.max_vcpus} vCPU)",
             self.has_compute_env, self.create_compute_env),
            (f"Batch job queue {n.job_queue}", self.has_job_queue, self.create_job_queue),
            (f"Batch job definition {n.job_definition} -> {n.image}", self.has_job_definition,
             self.create_job_definition),
        ]

    def plan(self) -> list[tuple[str, bool]]:
        rows = [(label, exists()) for label, exists, _ in self.steps()]
        for label, ok in rows:
            self.log(f"  {'exists ' if ok else 'CREATE '} {label}")
        return rows

    def apply(self) -> None:
        for label, exists, create in self.steps():
            if exists():
                self.log(f"  exists   {label}")
            else:
                self.log(f"  creating {label} ...")
                create()
        self.write_config()

    def config(self) -> dict:
        n = self.n
        return {"region": self.s.region, "account": n.account, "bucket": n.bucket,
                "job_queue": n.job_queue, "job_definition": n.job_definition,
                "log_group": n.log_group, "image": n.image}

    def write_config(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.config(), indent=2), encoding="utf-8")
        self.log(f"  wrote {path}")

    # ── removal ──
    def teardown(self, delete_data: bool = False) -> None:
        batch, iam, n = self.c["batch"], self.c["iam"], self.n
        queue = self._batch_state("queue")
        if queue:
            if queue["state"] == "ENABLED":
                batch.update_job_queue(jobQueue=n.job_queue, state="DISABLED")
                self._wait_settled("queue")
            batch.delete_job_queue(jobQueue=n.job_queue)
            self._wait_gone("queue")
            self.log(f"  deleted job queue {n.job_queue}")
        ce = self._batch_state("ce")
        if ce:
            if ce["state"] == "ENABLED":
                batch.update_compute_environment(computeEnvironment=n.compute_env,
                                                 state="DISABLED")
                self._wait_settled("ce")
            batch.delete_compute_environment(computeEnvironment=n.compute_env)
            self._wait_gone("ce")
            self.log(f"  deleted compute environment {n.compute_env}")
        for d in batch.describe_job_definitions(jobDefinitionName=n.job_definition,
                                                status="ACTIVE")["jobDefinitions"]:
            batch.deregister_job_definition(jobDefinition=d["jobDefinitionArn"])
            self.log(f"  deregistered {d['jobDefinitionArn'].split('/')[-1]}")
        sg = self.security_group_id()
        if sg:
            self.c["ec2"].delete_security_group(GroupId=sg)
            self.log(f"  deleted security group {n.security_group}")
        if self.has_instance_profile():
            iam.remove_role_from_instance_profile(InstanceProfileName=n.instance_profile,
                                                  RoleName=n.instance_role)
            iam.delete_instance_profile(InstanceProfileName=n.instance_profile)
            self.log(f"  deleted instance profile {n.instance_profile}")
        if self.has_instance_role():
            iam.detach_role_policy(RoleName=n.instance_role, PolicyArn=ECS_INSTANCE_POLICY)
            iam.delete_role(RoleName=n.instance_role)
            self.log(f"  deleted role {n.instance_role}")
        if self.has_job_role():
            iam.delete_role_policy(RoleName=n.job_role, PolicyName="geoinv3d-job-files")
            iam.delete_role(RoleName=n.job_role)
            self.log(f"  deleted role {n.job_role}")
        if self.has_log_group():
            self.c["logs"].delete_log_group(logGroupName=n.log_group)
            self.log(f"  deleted log group {n.log_group}")
        if self.has_ecr_repo():
            self.c["ecr"].delete_repository(repositoryName=n.ecr_repo, force=True)
            self.log(f"  deleted ECR repository {n.ecr_repo} and its images")
        if self.has_bucket():
            objects = self.c["s3"].list_objects_v2(Bucket=n.bucket, MaxKeys=1).get("KeyCount", 0)
            if objects and not delete_data:
                self.log(f"  kept S3 bucket {n.bucket}: it holds job data "
                         f"(re-run with --delete-data to remove it)")
            else:
                if objects:
                    paginator = self.c["s3"].get_paginator("list_objects_v2")
                    for page in paginator.paginate(Bucket=n.bucket):
                        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                        if keys:
                            self.c["s3"].delete_objects(Bucket=n.bucket,
                                                        Delete={"Objects": keys})
                self.c["s3"].delete_bucket(Bucket=n.bucket)
                self.log(f"  deleted S3 bucket {n.bucket}")

    def _wait_settled(self, kind, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            item = self._batch_state(kind)
            if not item or item.get("status") not in ("CREATING", "UPDATING"):
                return
            time.sleep(5)

    def _wait_gone(self, kind, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._batch_state(kind) is None:
                return
            time.sleep(5)
        raise TimeoutError(f"{kind} was not deleted in {timeout} s")

    # ── image ──
    def push_image(self, run=subprocess.run) -> None:
        if run(["docker", "info"], capture_output=True).returncode != 0:
            raise RuntimeError("Docker is not running: start Docker Desktop and try again")
        n = self.n
        local = f"{n.ecr_repo}:latest"
        sha = run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                  cwd=REPO_ROOT).stdout.strip() or "dev"
        self.log(f"  building {local} from deploy/Dockerfile ...")
        run(["docker", "build", "-f", str(REPO_ROOT / "deploy" / "Dockerfile"), "-t", local,
             str(REPO_ROOT)], check=True)
        auth = self.c["ecr"].get_authorization_token()["authorizationData"][0]
        user, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
        registry = auth["proxyEndpoint"].replace("https://", "")
        run(["docker", "login", "--username", user, "--password-stdin", registry],
            input=password, text=True, check=True, capture_output=True)
        for tag in ("latest", f"git-{sha}"):
            remote = f"{registry}/{n.ecr_repo}:{tag}"
            run(["docker", "tag", local, remote], check=True)
            self.log(f"  pushing {remote} ...")
            run(["docker", "push", remote], check=True)


def make_clients(region: str) -> dict:
    import boto3
    session = boto3.session.Session(region_name=region)
    return {svc: session.client(svc) for svc in
            ("sts", "s3", "ecr", "logs", "iam", "ec2", "batch")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", default="plan",
                        choices=["plan", "apply", "push-image", "teardown"])
    parser.add_argument("--region", default=Settings.region)
    parser.add_argument("--max-vcpus", type=int, default=Settings.max_vcpus)
    parser.add_argument("--delete-data", action="store_true",
                        help="teardown: also empty and delete the S3 bucket")
    parser.add_argument("--yes", action="store_true", help="teardown without asking")
    args = parser.parse_args(argv)

    settings = Settings(region=args.region, max_vcpus=args.max_vcpus)
    d = Deployer(settings, make_clients(args.region))
    print(f"GeoInv3D on AWS account {d.n.account}, region {settings.region}")
    if args.command == "plan":
        d.plan()
    elif args.command == "apply":
        d.apply()
    elif args.command == "push-image":
        d.push_image()
    elif args.command == "teardown":
        if not args.yes and input("Type 'delete geoinv3d' to remove the resources: ") \
                != "delete geoinv3d":
            print("Nothing deleted.")
            return
        d.teardown(delete_data=args.delete_data)


if __name__ == "__main__":
    sys.exit(main())

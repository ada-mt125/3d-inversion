#!/bin/bash
# GeoInv3D on a Slurm cluster: one job, MT's frequencies on MPI ranks (geoinv3d/methods/parallel.py).
# Rank 0 runs the job (data, mesh, inversion, the result); ranks 1..N each solve a group of the
# frequencies (a model goes to them once; only vectors travel). The ranks beyond the frequencies
# with data stay idle, so ask for at most 1 + the number of frequencies.
#
#   sbatch deploy/hpc/slurm_mt_mpi.sh params.json DATA_DIR OUT_DIR
#
# Needs, in the job's Python: geoinv3d's requirements, mpi4py built against the cluster's MPI,
# and a fast sparse solver (pydiso / Intel MKL PARDISO on x86, or python-mumps).
#SBATCH --job-name=geoinv3d-mt
#SBATCH --ntasks=10               # 1 + 9 frequency groups
#SBATCH --cpus-per-task=8         # each rank's solver threads (PARDISO)
#SBATCH --mem-per-cpu=4G          # a frequency's factorization: ~3 GB at 30k cells, ~23 GB at 220k
#SBATCH --time=04:00:00
set -euo pipefail
PARAMS=${1:?params.json}; DATA=${2:?data dir}; OUT=${3:?out dir}
# module load ...                 # the cluster's MPI and Python (site specific)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GEOINV3D_PARALLEL=mpi      # (also chosen by itself when started on several ranks)
srun python -m geoinv3d.cloud.worker --local "$PARAMS" "$DATA" "$OUT"

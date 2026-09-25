FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    simpeg \
    discretize \
    boto3

COPY geoinv3d/ /app/geoinv3d/

ENTRYPOINT ["python", "-m", "geoinv3d.cloud.worker"]

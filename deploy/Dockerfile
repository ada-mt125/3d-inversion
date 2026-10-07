FROM python:3.13-slim

WORKDIR /app

# rasterio reads GeoTIFF data and DEMs uploaded from the web page; pydiso (with Intel
# MKL) is PARDISO, the sparse solver of MT and DC
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    simpeg \
    discretize \
    numba \
    rasterio \
    scikit-learn \
    pydiso \
    boto3

# The worker code is baked into the image: rebuild and push it to ECR after
# every change under geoinv3d/, or AWS Batch keeps running the old version.
COPY geoinv3d/ /app/geoinv3d/

ENTRYPOINT ["python", "-m", "geoinv3d.cloud.worker"]

#!/bin/bash
# GeoInv3D on a PBS Pro cluster (e.g. Imperial's RCS): one job, MT's frequencies on MPI ranks
# (see slurm_mt_mpi.sh for what the ranks do).
#
#   qsub -v PARAMS=params.json,DATA=DATA_DIR,OUT=OUT_DIR deploy/hpc/pbs_mt_mpi.sh
#
#PBS -N geoinv3d-mt
#PBS -l select=1:ncpus=64:mem=256gb:mpiprocs=8:ompthreads=8   # 8 ranks: rank 0 + 7 frequency groups
#PBS -l walltime=04:00:00
set -euo pipefail
cd "$PBS_O_WORKDIR"
# module load ...                 # the cluster's MPI and Python (site specific)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export GEOINV3D_PARALLEL=mpi
mpiexec python -m geoinv3d.cloud.worker --local "$PARAMS" "$DATA" "$OUT"

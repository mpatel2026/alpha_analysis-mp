#!/bin/bash
#SBATCH --job-name=a5cpu_ref
#SBATCH --output=logs/cpu_ref_%j.out
#SBATCH --error=logs/cpu_ref_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --account=m4477
#
# CPU reference for the GPU guiding-center path: the same source built for CPU
# (tools/build_variant.sh cpu CPU=1), on the same input.
#
#   sbatch cpu_ref.sh <case> [tag]        (tag defaults to cpu)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./common.sh
cpu_env

CASE=${1:?usage: sbatch cpu_ref.sh <case> [tag]}
TAG=${2:-cpu}
check_binary "$TAG"
OUT=$(fresh_copy "$CASE" "$TAG")

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=threads

srun -n 8 -c 32 --cpu-bind=cores \
    "$VARDIR/ascot5_main.$TAG" --in="$OUT" --d="GPUVAL $TAG" \
    > "logs/cpu_ref_${SLURM_JOB_ID}.log" 2>&1
grep "Simulation finished\|Simulation time per process" \
    "logs/cpu_ref_${SLURM_JOB_ID}.log"
echo
echo "Compare with:"
echo "  python compare_runs.py $OUT <GPU output of the same case>"

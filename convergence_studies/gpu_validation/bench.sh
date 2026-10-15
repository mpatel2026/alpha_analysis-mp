#!/bin/bash
#SBATCH --job-name=a5gpu_bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=02:30:00
#SBATCH --account=m4477
#
# Production-size benchmark: one input on all GPUs of the allocation, the same
# way run-sims-GPU.sh runs it (one rank per GPU).
#
#   sbatch bench.sh <case> [tag]          (tag defaults to opt)
#   sbatch --nodes=2 bench.sh <case> opt  (8 GPUs)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./common.sh
gpu_env

CASE=${1:?usage: sbatch bench.sh <case> [tag]}
TAG=${2:-opt}
check_binary "$TAG"
OUT=$(fresh_copy "$CASE" "bench_${TAG}")

t0=$(date +%s)
srun -n "$SLURM_NTASKS" -c 32 --cpu-bind=cores --gpu-bind=single:1 \
    "$VARDIR/ascot5_main.$TAG" --in="$OUT" --d="GPUVAL bench $TAG" \
    > "logs/bench_${SLURM_JOB_ID}_${TAG}.log" 2>&1
t1=$(date +%s)

grep "PCI\|Simulation time per process\|Simulation finished\|markers had\|aborted" \
    "logs/bench_${SLURM_JOB_ID}_${TAG}.log"
echo "Total ascot5_main wall time: $((t1 - t0)) s"
echo "Output: $OUT"

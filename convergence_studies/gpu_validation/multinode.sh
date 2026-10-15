#!/bin/bash
#SBATCH --job-name=a5gpu_mn
#SBATCH --output=logs/multinode_%j.out
#SBATCH --error=logs/multinode_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --account=m4477
#
# Multi-node check: the same input on 2 nodes (8 GPUs) and on 1 node (4 GPUs).
# Check in the log that the 8 ranks report 8 distinct (host, GPU) pairs, look
# at the per-process time spread, then compare the two outputs statistically.
#
#   sbatch multinode.sh <case> [tag]      (tag defaults to opt)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./common.sh
gpu_env

CASE=${1:?usage: sbatch multinode.sh <case> [tag]}
TAG=${2:-opt}
check_binary "$TAG"

OUT8=$(fresh_copy "$CASE" "${TAG}_8gpu")
OUT4=$(fresh_copy "$CASE" "${TAG}_4gpu")

echo "=== 2 nodes, 8 ranks"
srun -N 2 -n 8 -c 32 --cpu-bind=cores --gpu-bind=single:1 \
    "$VARDIR/ascot5_main.$TAG" --in="$OUT8" --d="GPUVAL $TAG 8gpu" \
    > "logs/multinode_${SLURM_JOB_ID}_8gpu.log" 2>&1
grep "node-local rank\|Simulation time per process\|Simulation finished" \
    "logs/multinode_${SLURM_JOB_ID}_8gpu.log"
echo "distinct physical GPUs (host, PCI bus id), expect 8:"
grep -o "on [a-z0-9]*: .*PCI [0-9A-Fa-f:.]*" \
    "logs/multinode_${SLURM_JOB_ID}_8gpu.log" \
    | awk '{print $2, $NF}' | sort -u | wc -l

echo "=== 1 node, 4 ranks"
srun -N 1 -n 4 -c 32 --cpu-bind=cores --gpu-bind=single:1 \
    "$VARDIR/ascot5_main.$TAG" --in="$OUT4" --d="GPUVAL $TAG 4gpu" \
    > "logs/multinode_${SLURM_JOB_ID}_4gpu.log" 2>&1
grep "Simulation time per process\|Simulation finished" \
    "logs/multinode_${SLURM_JOB_ID}_4gpu.log"

echo
echo "Compare with:"
echo "  python compare_runs.py $OUT4 $OUT8"

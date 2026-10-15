#!/bin/bash
#SBATCH --job-name=a5opt
#SBATCH --output=logs/run_opt_%j.out
#SBATCH --error=logs/run_opt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --account=m4477
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mpatel26@thea.energy
#
# Production run with the optimized GPU build. Results are written into the
# input file itself, as with run-sims-GPU.sh.
#
#   sbatch --nodes=N [--time=HH:MM:SS] run_opt.sh <input.h5> [tag]
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./common.sh
gpu_env

IN=${1:?usage: sbatch run_opt.sh <input.h5> [tag]}
TAG=${2:-opt}
check_binary "$TAG"
[ -f "$IN" ] || { echo "Missing $IN" >&2; exit 1; }

echo "Input: $IN"
echo "Binary: $VARDIR/ascot5_main.$TAG, $SLURM_NTASKS ranks on $SLURM_NNODES nodes"
t0=$(date +%s)
srun -n "$SLURM_NTASKS" -c 32 --cpu-bind=cores --gpu-bind=single:1 \
    "$VARDIR/ascot5_main.$TAG" --in="$IN" --d="opt $SLURM_JOB_ID" \
    > "logs/run_opt_${SLURM_JOB_ID}_${TAG}.log" 2>&1
t1=$(date +%s)

grep "Simulation time per process\|Simulation finished\|markers had\|aborted" \
    "logs/run_opt_${SLURM_JOB_ID}_${TAG}.log"
echo "Total ascot5_main wall time: $((t1 - t0)) s"

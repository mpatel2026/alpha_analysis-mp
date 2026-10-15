#!/bin/bash
#SBATCH --job-name=a5gpu_nsys
#SBATCH --output=logs/nsys_%j.out
#SBATCH --error=logs/nsys_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --account=m4477
#
# Nsight Systems profile of one or two variants (one GPU each), to see how much
# of a time step is kernel time versus launch/synchronization overhead.
# Use a short case (small --tmax in prep_inputs.py); profiles get large.
#
#   sbatch nsys_profile.sh <case> <tag1> [tag2]
#
# Open the .nsys-rep files in Nsight Systems, or read the *_stats.txt
# summaries (cuda_gpu_kern_sum, cuda_api_sum, openacc_sum).
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./common.sh
gpu_env

CASE=${1:?usage: sbatch nsys_profile.sh <case> <tag1> [tag2]}
shift
mkdir -p "$VALDIR/nsys"

# Run the steps side by side, one GPU each. By default a step takes all of the
# job memory, and it inherits --gpus-per-node=4 from SLURM_GPUS_PER_NODE, so
# the other steps would wait for it to finish.
unset SLURM_GPUS_PER_NODE
for tag in "$@"; do
    check_binary "$tag"
    out=$(fresh_copy "$CASE" "nsys_$tag")
    rep="$VALDIR/nsys/${CASE}_${tag}_${SLURM_JOB_ID}"
    srun --exact -n 1 -c 32 --gpus-per-task=1 --mem=50G --cpu-bind=cores \
        nsys profile -t cuda,openacc -s none --force-overwrite=true \
        -o "$rep" --stats=true \
        "$VARDIR/ascot5_main.$tag" --in="$out" --d="GPUVAL nsys $tag" \
        > "${rep}_stats.txt" 2>&1 &
done
wait
ls -la "$VALDIR/nsys" | grep "$SLURM_JOB_ID"

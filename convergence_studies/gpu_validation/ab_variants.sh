#!/bin/bash
#SBATCH --job-name=a5gpu_ab
#SBATCH --output=logs/ab_%j.out
#SBATCH --error=logs/ab_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --account=m4477
#
# Run up to four ascot5_main variants side by side, one GPU each, on copies of
# the same input. Used for bit-exact checks and for timing sweeps.
#
#   sbatch ab_variants.sh <case> <tag1> [tag2 tag3 tag4]
#   e.g. sbatch ab_variants.sh gc20k base-minfo legacy s12 opt
#
# <case> is an input made by prep_inputs.py ($VALDIR/<case>.h5), <tag> picks
# $VARDIR/ascot5_main.<tag>.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./common.sh
gpu_env

CASE=${1:?usage: sbatch ab_variants.sh <case> <tag>...}
shift
TAGS=("$@")
[ ${#TAGS[@]} -ge 1 ] && [ ${#TAGS[@]} -le 4 ] || {
    echo "Give 1 to 4 variant tags" >&2; exit 1; }

declare -A OUT
for tag in "${TAGS[@]}"; do
    check_binary "$tag"
    OUT[$tag]=$(fresh_copy "$CASE" "$tag")
done

# Run the steps side by side, one GPU each. By default a step takes all of the
# job memory, and it inherits --gpus-per-node=4 from SLURM_GPUS_PER_NODE, so
# the other steps would wait for it to finish.
unset SLURM_GPUS_PER_NODE
for tag in "${TAGS[@]}"; do
    srun --exact -n 1 -c 32 --gpus-per-task=1 --mem=50G --cpu-bind=cores \
        "$VARDIR/ascot5_main.$tag" --in="${OUT[$tag]}" --d="GPUVAL $tag" \
        > "logs/ab_${SLURM_JOB_ID}_${tag}.log" 2>&1 &
done
wait

echo "Variant       sim time [s]   output"
for tag in "${TAGS[@]}"; do
    t=$(grep -o "Simulation finished in [0-9.]*" \
        "logs/ab_${SLURM_JOB_ID}_${tag}.log" | awk '{print $4}')
    printf "%-12s %12s   %s\n" "$tag" "${t:-FAILED}" "${OUT[$tag]}"
done
echo
echo "Compare with e.g.:"
echo "  python compare_runs.py ${OUT[${TAGS[0]}]} <other output> [--bitwise]"

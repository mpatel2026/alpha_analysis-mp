#!/bin/bash
#SBATCH --job-name=ascot_array
#SBATCH --array=2-3
#SBATCH --output=logs/ascot_%a.out
#SBATCH --error=logs/ascot_%a.err

# Resource Allocations per Array Task
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96       # Stellar Intel Ice Lake nodes have 96 physical cores
#SBATCH --time=08:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mpatel26@thea.energy

# 1. Identify the input file for this specific array index
INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sim-list-Eos.txt)

# 2. Safety check: If the line is empty, don't run anything
if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file found for Task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Starting ASCOT5 Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using Input File: $INPUT_FILE"
echo "Running on $SLURM_JOB_NUM_NODES nodes with $SLURM_CPUS_PER_TASK CPUs per task"

#RUN_MESSAGE="Using Input File: G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p02cellarea__FLR-mode_FPP-power_03242026.h5"
HOME_DIR="/home/mp3096/"
SCRATCH_DIR="/scratch/s/gpfs/mp3096/ascot_h5s/Eos10-std/"

# 3. Execute ASCOT5
# Note: Since you requested 10 nodes, you likely need srun to distribute the load
srun -n $SLURM_NNODES -c 96 --cpu_bind=cores "${HOME_DIR}ascot5/ascot5_develop_mp_hpc/build/ascot5_main" --in="${SCRATCH_DIR}$INPUT_FILE" --d="$RUN_MESSAGE"

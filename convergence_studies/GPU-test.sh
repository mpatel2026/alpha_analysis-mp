#!/bin/bash
#SBATCH --job-name=ascot_gpu_debug
#SBATCH --output=logs/debug_ascot.out
#SBATCH --error=logs/debug_ascot.err

# --- GPU Debug Resource Allocations ---
#SBATCH --nodes=1                     # Use 1 node for debugging (4 GPUs total)
#SBATCH --ntasks-per-node=4           # 1 MPI task per GPU
#SBATCH --gpus-per-node=4             # Request all 4 GPUs on the node
#SBATCH --cpus-per-task=1            # 128 cores / 4 tasks = 32
#SBATCH --constraint=gpu              
#SBATCH --qos=debug                   # CRITICAL: Change from regular to debug
#SBATCH --time=00:30:00               # CRITICAL: Max time for debug is 30 mins
#SBATCH --account=m4477

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load PrgEnv-nvidia
module load craype-x86-milan
module load cudatoolkit
module load cray-hdf5-parallel

# Crucial for GPU-to-GPU communication and OpenACC offloading
export MPICH_GPU_SUPPORT_ENABLED=1
export ACC_DEVICE_TYPE=nvidia

# 3. Paths
HOME_DIR="/global/homes/m/mpatel26/"
SCRATCH_DIR="/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"

# 1. Define internal variables from Slurm environment
# NTASKS is the total number of MPI ranks (Nodes * ntasks-per-node)
TOTAL_TASKS=${SLURM_NTASKS}

# CPUS_PER_TASK is the -c value from your header
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Total GPUs requested for the whole job
TOTAL_GPUS=${SLURM_GPUS}

echo "------------------------------------------------------"
echo "Running with Variables:"
echo "Total MPI Tasks (-n): $TOTAL_TASKS"
echo "CPUs per Task (-c):   $CPUS_PER_TASK"
echo "------------------------------------------------------"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 2. Execute with variables
srun -n "$TOTAL_TASKS" \
     -c "$CPUS_PER_TASK" \
     --cpu-bind=cores \
     --gpu-bind=closest \
     "${HOME_DIR}ascot5/ascot5_develop_mp_hpc_GPU/build/ascot5_main" --in="${SCRATCH_DIR}G1600-free-reopt_I_10cm-wall_200k-mrk_biot-bfield_p01cellarea_FO-mode_alpha-power_04012026.h5"
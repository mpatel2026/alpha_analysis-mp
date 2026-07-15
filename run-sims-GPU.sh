#!/bin/bash
#SBATCH --job-name=ascot_gpu_array
#SBATCH --array=20
#SBATCH --output=logs/ascot_%a.out
#SBATCH --error=logs/ascot_%a.err

# --- GPU Resource Allocations ---
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4           # 1 MPI Task per GPU
#SBATCH --gpus-per-node=4             # Request all 4 A100s per node
#SBATCH --cpus-per-task=1            # Full node CPU allocation (128 threads / 4 tasks)
#SBATCH --constraint=gpu              
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH --account=m4477
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mpatel26@thea.energy
 
# 1. Identify the input file
# Note: Ensure sim_list_GPU.txt is in the same directory where you submit the job
LIST_FILE="sim-list-Eos.txt"
INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")

if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file found for Task ID $SLURM_ARRAY_TASK_ID in $LIST_FILE"
    exit 1
fi

# 2. Load the GPU Environment
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load PrgEnv-nvidia
module load craype-accel-nvidia80 
module load cudatoolkit
module load cray-hdf5-parallel

export MPICH_GPU_SUPPORT_ENABLED=1
export ACC_DEVICE_TYPE=nvidia
export HDF5_USE_FILE_LOCKING=FALSE


# 3. Paths
HOME_DIR="/global/homes/m/mpatel26/"
SCRATCH_DIR="/pscratch/sd/m/mpatel26/ascot_h5s/"

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
     --gpu-bind=single:1 \
     "${HOME_DIR}ascot5/ascot5_develop_mp_hpc_GPU/build/ascot5_main" --in="${SCRATCH_DIR}$INPUT_FILE"

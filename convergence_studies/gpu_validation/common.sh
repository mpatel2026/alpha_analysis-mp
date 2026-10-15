# Shared settings for the GPU validation jobs (sourced by the job scripts).

# Binaries built with tools/build_variant.sh in the optimized tree
VARDIR=/global/homes/m/mpatel26/ascot5/ascot5_developed_mp_hpc_GPU_optimized/variants
# Inputs made by prep_inputs.py, and per-run copies that the runs write into
VALDIR=/pscratch/sd/m/mpatel26/gpu_validation
RUNDIR=$VALDIR/runs
mkdir -p "$RUNDIR" logs

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

gpu_env() {
    module purge
    source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
    # cudatoolkit must come before craype-accel-nvidia80
    module load PrgEnv-nvidia cudatoolkit cray-hdf5-parallel craype-x86-milan
    module load craype-accel-nvidia80
    export MPICH_GPU_SUPPORT_ENABLED=1
    export ACC_DEVICE_TYPE=nvidia
}

cpu_env() {
    # Must match tools/build_variant.sh CPU=1
    module purge
    source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh
    module load PrgEnv-gnu craype-x86-milan cray-hdf5-parallel
}

# fresh_copy <case> <tag>: copy the case input to a new file for one run and
# print its path, so that concurrent runs never write to the same file.
fresh_copy() {
    local src="$VALDIR/$1.h5"
    local dst="$RUNDIR/${1}_${2}_${SLURM_JOB_ID}.h5"
    [ -f "$src" ] || { echo "Missing $src (run prep_inputs.py)" >&2; exit 1; }
    cp "$src" "$dst"
    echo "$dst"
}

# check_binary <tag>
check_binary() {
    [ -x "$VARDIR/ascot5_main.$1" ] || {
        echo "Missing $VARDIR/ascot5_main.$1" >&2; exit 1; }
}

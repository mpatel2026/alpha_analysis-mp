"""Automatic CPU/GPU detection and JAX environment setup.

Must be imported (and configure() called) before any module that imports
jax/desc, since JAX reads JAX_PLATFORMS/JAX_ENABLE_X64/XLA_FLAGS at import
time and ignores changes made afterwards.
"""
import os
import shutil
import subprocess


def _nonzero(val):
    try:
        return int(val) > 0
    except (TypeError, ValueError):
        return False


def detect_device() -> str:
    """Return 'gpu' if a GPU is visible to this process, else 'cpu'."""
    for var in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_JOB_GPUS"):
        if _nonzero(os.environ.get(var)):
            return "gpu"

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible and cuda_visible != "-1":
        return "gpu"

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return "gpu"
        except (subprocess.SubprocessError, OSError):
            pass

    return "cpu"


def configure() -> str:
    """Detect the device and set the JAX/XLA/threading environment for it.

    Returns the resolved device string ('gpu' or 'cpu').
    """
    device = detect_device()

    if device == "gpu":
        # Deliberately do NOT set JAX_PLATFORMS here. JAX's own default
        # (unset) auto-probes the CUDA plugin and falls back to CPU with a
        # warning if it fails to initialize. Explicitly listing "cuda,cpu"
        # instead makes JAX treat "cuda" as the required default backend, so
        # if the plugin fails to register, later calls asking for the
        # default backend hard-error instead of silently falling back.
        os.environ.setdefault("JAX_ENABLE_X64", "True")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
        os.environ.setdefault("OMP_NUM_THREADS", "16")
        os.environ.setdefault("MKL_NUM_THREADS", "16")
        os.environ.setdefault("NUMBA_NUM_THREADS", "16")
        reason = "GPU detected"
    else:
        n_cpu = os.environ.get("SLURM_CPUS_PER_TASK", "64")
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_ENABLE_X64", "True")
        os.environ.setdefault(
            "XLA_FLAGS",
            f"--xla_force_host_platform_device_count={n_cpu} "
            "--xla_cpu_multi_thread_eigen=true",
        )
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("OMP_NUM_THREADS", n_cpu)
        os.environ.setdefault("MKL_NUM_THREADS", n_cpu)
        os.environ.setdefault("NUMBA_NUM_THREADS", n_cpu)
        reason = "no GPU detected"

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    print(f"[device_setup] resolved device = {device} ({reason})")
    return device

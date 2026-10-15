# GPU validation for ascot5_developed_mp_hpc_GPU_optimized

These checks cover the optimized GPU tree (`~/ascot5/ascot5_developed_mp_hpc_GPU_optimized`,
branch `gpu-multinode-a100`). They confirm that:
- the new commits leave results unchanged where they should,
- the RNG fix only changes results statistically,
- GPU agrees with CPU on the same source and input,
- 2 nodes agree with 1 node.

Binaries live in `variants/` of that tree and are built with `tools/build_variant.sh`.

| tag | what it is |
|---|---|
| `base-minfo` | merged 5.6.4 code, before any optimization (reference) |
| `s12` | + device selection, async queue, n_running every 16 steps, binary rho search |
| `spline` | + streamed tricubic spline (changes rounding only) |
| `opt` | + rank-dependent RNG seeds (all commits; the one to use) |
| `legacy` | `opt` source built with `LEGACY=1` (old spline + old seeds) |
| `reg168`, `reg128`, `vlen64`, `vlen32` | `opt` with a register cap or vector length |
| `wall8` | `opt` with wall octree depth 8 (should be bit-exact with `opt`) |
| `cpu` | `opt` source built for CPU |

## 0. Inputs

These have already been made in `$PSCRATCH/gpu_validation/`. Each is a random subset of the ISHW 100k
GC case, adaptive GC:
- `gc20k_10ms.h5`: 20k markers, stopped at 10 ms
- `gc40kdist_10ms.h5`: 40k markers with dist5D and rho5D on, stopped at 10 ms
- `gc5k_short.h5`: 5k markers, stopped at 2 ms (profiling, quick checks)

The 50 ms versions (`gc20k`, `gc40k`, `gc40kdist`) are too long for the debug QOS on one GPU with
the old code.

To make others, e.g. a short case for profiling:

    module load conda; conda activate ascot-develop-mp
    export HDF5_USE_FILE_LOCKING=FALSE JAX_PLATFORMS=cpu
    python prep_inputs.py <src.h5> $PSCRATCH/gpu_validation/gc5k_short.h5 --nmrk 5000 --tmax 0.002

If a run hits the 30 min debug limit, make a new case with a smaller `--tmax`.

## 1. Bit-exact check and first timing (1 GPU node, debug)

    sbatch ab_variants.sh gc20k_10ms base-minfo legacy s12 opt

The job prints the simulation time per variant and the output file of each. Then run:

    python compare_runs.py <base-minfo out> <legacy out> <s12 out> --bitwise   # must PASS
    python compare_runs.py <base-minfo out> <opt out>                         # statistical

## 2. Tuning sweep

    sbatch ab_variants.sh gc20k_10ms opt reg168 reg128 vlen64
    sbatch ab_variants.sh gc20k_10ms opt vlen32 spline
    sbatch ab_variants.sh gc5k_short opt wall8      # then compare --bitwise

Keep the fastest settings. If `reg168` or `vlen64` wins, rebuild with that knob, e.g.
`tools/build_variant.sh prod MAXREGCOUNT=168`, or make it the default.

## 3. CPU reference (1 CPU node, debug)

    sbatch cpu_ref.sh gc20k_10ms
    python compare_runs.py <cpu out> <opt out>

## 4. Multi-node (2 GPU nodes, debug)

    sbatch multinode.sh gc40kdist_10ms

Check that the log shows 8 distinct (host, GPU) pairs and a sensible per-process time spread. Then:

    python compare_runs.py <4gpu out> <8gpu out>

The dist5d and distrho5d total weight ratio should be close to 1.

## 5. Profile (optional)

    sbatch nsys_profile.sh gc5k_short base-minfo opt

## Reading compare_runs.py

- **`--bitwise`:** every end-state field except cputime must be identical for every marker id.
- **Statistical mode:**
  - End-condition fractions are compared with a binomial z-score; |z| > 3 is flagged.
  - End energy, end time, end rho and wall-hit time are compared with a two-sample KS test;
    p < 0.001 is flagged.
  - No new aborted markers are allowed.

## Notes on the job scripts

- **Concurrent steps:** in `ab_variants.sh` and `nsys_profile.sh`, each step needs its own `--mem`. The
  scripts also `unset SLURM_GPUS_PER_NODE`; otherwise every `srun` step inherits all 4 GPUs of the
  job and the steps run one after another.
- **Binaries:** never overwrite a `variants/ascot5_main.<tag>` binary while a job may be using it.
  Build under a new tag instead.

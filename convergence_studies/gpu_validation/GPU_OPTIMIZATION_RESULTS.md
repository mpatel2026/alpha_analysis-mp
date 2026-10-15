# ASCOT5 GPU optimization: what changed, how it was tested, results

*Status as of 2026-10-01. Author: Manav Patel.*

## Summary

The OpenACC GPU build of ASCOT5 used for the G1600/Eos alpha-particle studies was optimized for NERSC
Perlmutter's A100 GPUs.

- **Speed:** the optimized code is **2.2–3.0× faster** for the same input on the same hardware. A
  100k-marker guiding-center run on one GPU node went from 2.05 h to 44 min. A full-length 100k full-orbit
  run went from 38.2 h on 16 GPUs to 14.7 h on 4 GPUs, i.e. 10× fewer GPU-hours.
- **Physics:** unchanged.
  - Most changes give bit-for-bit identical results.
  - Two changes (a new random-number seeding and a reordered spline sum) change individual markers, but
    loss fractions, power loss, wall loads and end-state distributions agree within Monte Carlo error.
- **Bugs fixed:**
  - All MPI ranks were drawing identical random numbers.
  - A thread race, an undersized GPU data range, and missing frees.
- **Where:** the optimized executable is now the main GPU executable,
  `~/ascot5/ascot5_develop_mp_hpc_GPU/build/ascot5_main`. It's on git branch `GPU-opt`, pushed to
  https://github.com/mpatel2026/ascot5-mp/tree/GPU-opt.

---

## 1. Background and terms

- **Simulation modes:**
  - **GC (guiding-center)** follows the gyro-averaged motion. The production GC runs use the adaptive
    Cash-Karp integrator with Milstein collisions.
  - **FO (full-orbit)** follows the full gyration with a fixed time step (1/20 of a gyro-period).
- **Markers** are the simulated alpha particles; each has a weight (particles/s). A run ends when every
  marker has hit an *end condition*: wall collision, thermalization, energy limit, or simulation-time
  limit.
- **End state:** the final position, momentum, time, end condition and wall tile of each marker. All
  physics outputs used here (loss fractions, power loss, wall loads) are computed from the end states.
- **Hardware:** a Perlmutter GPU node has 4 A100 GPUs, and runs use one MPI rank per GPU, each simulating
  a contiguous block of markers.
- **"Bit-identical":** every end-state field of every marker is exactly equal between two runs, except
  the wall-clock `cputime` field. Physics outputs are then identical by construction.
- **"Statistically consistent":** two runs use different random numbers, so individual markers differ, but
  their ensembles agree within Monte Carlo error. Criteria:
  - end-condition fractions within |z| < 3 (binomial z-score);
  - end energy, end time, end ρ and wall-hit time agree by a two-sample Kolmogorov–Smirnov (KS) test
    (p > 0.001);
  - power loss and wall loads within their bootstrap or Monte Carlo errors.

## 2. What was changed

The code is in `~/ascot5/ascot5_develop_mp_hpc_GPU`, branch `GPU-opt`. The base is the group's
`conformal-walls` branch, merged with upstream ASCOT5 5.6.4 (commit `5c7ed7bb`).

| Change | Why it matters | Changes results? |
|---|---|---|
| **Fewer CPU–GPU synchronizations.** GPU kernels are queued asynchronously. The CPU checks how many markers are still running every 16 steps instead of every step. Kernels that did nothing unless time runs backwards are skipped | Each check made the CPU wait for the GPU, every step | No (bit-identical) |
| **Plasma profile lookup:** binary search instead of a linear scan of the ρ grid | The inputs have a 2047-point ρ grid, and the scan ran several times per marker per step. Collisions got 72% faster and end-condition checks 60% faster | No (bit-identical) |
| **Magnetic-field spline:** the tricubic spline sums over cell corners in turn instead of loading all 64 coefficients at once, and B_R, B_φ, B_z are evaluated together | The old version overflowed GPU registers into slow memory (about 2.4 KB per evaluation, now 132 B). The orbit step got 29% faster | Rounding only (about 1e-14 relative) |
| **Random-number seeding:** each MPI rank and GPU slot gets an independent seed (SplitMix64 hash of rank and slot) | **Bug:** previously every rank used the same seeds, so marker *j* on every rank got identical collision noise | Yes: new random numbers (statistically equivalent) |
| **Multi-node support:** each rank picks its own GPU and prints host and PCI bus ID. The timing spread across ranks and per-rank progress every 10 minutes are printed | Makes multi-node runs verifiable and imbalance visible | No |
| **Bug fixes:** thread race in `recalculate_acceleration`, undersized random-number range in `mccc_gc_euler`, GPU data freed at the end | Latent bugs, not triggered by the production settings | No |
| **Build options:** `MINFO`, `MAXREGCOUNT`, `GPU_VLEN`, `WALL_DEPTH`, `LEGACY=1` (reproduces the old spline and RNG exactly) | Testing and tuning; the defaults are the production settings | – |

## 3. How before and after were compared

Each test ran the **same input** with two builds of the code. The binaries are in
`~/ascot5/ascot5_developed_mp_hpc_GPU_optimized/variants/`:

| Build | What it is |
|---|---|
| `base-minfo` | **Before:** upstream 5.6.4 merge, with none of the changes above |
| `opt` | **After:** all changes. Same source as the current main executable |
| `compact` | `opt` plus active-marker compaction. Tried and reverted (section F) |
| `legacy` | `opt` source built with `LEGACY=1` (old spline and RNG). Used to prove that all *other* changes are bit-identical to the old code |

For the 100k GC benchmark, "before" is the earlier production run of the same input.

**Tools** (in `~/alpha_analysis-mp/convergence_studies/gpu_validation/`):
- `compare_runs.py`: bit-exact or statistical comparison of end states.
- `wall_compare.py`: power loss and wall loads, with error bars.

**Locations:**
- Job logs: `gpu_validation/logs/`.
- Run outputs: `/pscratch/sd/m/mpatel26/gpu_validation/runs/<case>_<build>_<jobid>.h5`.

## 4. Results

### Speed

| Benchmark | Hardware | Before | After | Speedup |
|---|---|---|---|---|
| A. GC, 100k markers, 300 ms | 1 node (4 GPUs) | 2.05 h job / 6,626 s simulation | **43.8 min** job / 2,609 s | **2.8× job, 2.5× simulation** |
| B. GC, 200k markers, 300 ms | 2 nodes (8 GPUs) | 97.5 min / 5,837 s | **41.8 min** / 2,497 s | **2.3×** |
| C. GC, 20k markers, 10 ms | 1 GPU | 724.5 s | 333.4 s | 2.2× |
| D. FO, 20k markers, 0.5 ms | 1 GPU | 423.9 s | 141.4 s | **3.0×** |

### Resource use: before vs after (all GPU runs compared in this study)

**CPU reference column:** a CPU run of the **same G1600 case type and marker count**, but **not the
  identical input**: different wall and/or marker set, from earlier CPU studies. It's a rough guide to CPU
  cost, not a strict comparison.
  - All CPU runs used 10 CPU nodes, 1 rank × 256 threads each.
  - Node-hours = 10 × simulation time.
  - Core-hours = node-hours × 128 physical cores.

| Run | Mode | Markers | GPUs (nodes) | **Before:** wall clock | **Before:** GPU node-h | **Before:** GPU-h | **After:** wall clock | **After:** GPU node-h | **After:** GPU-h | Speedup | Relevant CPU run: wall clock / CPU node-h / core-h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A. ISHW eng-wall, 300 ms | GC | 100k | 4 (1) | 2:03:00 | 2.05 | 8.2 | 0:43:45 | 0.73 | 2.9 | **2.8×** | 1:47:27 (sim) / 17.9 / 2,290 *(1)* |
| A′. Same, with compaction (reverted) | GC | 100k | 4 (1) | – | – | – | 0:43:12 | 0.72 | 2.9 | 2.8× | – |
| B. 10 cm wall (p05) subset, 300 ms | GC | 200k | 8 (2) | 1:37:33 | 3.25 | 13.0 | 0:41:50 | 1.39 | 5.6 | **2.3×** | 3:30:16 (sim) / 35.0 / 4,490 *(2)* |
| C. ISHW subset, 10 ms | GC | 20k | 1 (¼) | 0:12:05 (sim) | 0.050 | 0.20 | 0:05:33 (sim) | 0.023 | 0.09 | 2.2× | – |
| D. ISHW FO subset, 0.5 ms | FO | 20k (16.8k nonzero weight) | 1 (¼) | 0:07:04 (sim) | 0.029 | 0.12 | 0:02:21 (sim) | 0.010 | 0.04 | **3.0×** | – |
| E. LCFS wall, 300 ms | FO | 100k | before 16 (4); after 4 (1) | 38:13:46 (sim) | 152.9 | 611.7 | 14:43:29 | 14.7 | 58.9 | **2.6× wall, 10.4× GPU-h** | 18:44:37 (sim) / 187 / 23,990 *(3)* |
| F. LCFSmark100k (production; no "before" run) | FO | 115k | 4 (1) | – | – | – | 9:05:23 | 9.09 | 36.4 | – | – |
| G. LCFSmark1M (production; no "before" run) | FO | 1M | 40 (10) | – | – | – | 7:35:05 | 75.8 | 303.4 | – | – |

**Column definitions:**
- **Wall clock** is elapsed time as h:mm:ss. It's the Slurm job time unless marked "(sim)", which is
  ASCOT's reported simulation time and excludes a few minutes of input reading and output writing.
- **GPU node-h** = nodes × wall clock (a single-GPU run counts as ¼ node).
- **GPU-h** = GPUs × wall clock.

**CPU reference runs** (logs in `~/alpha_analysis-mp/convergence_studies/logs-CPU/`):
1. `G1600-free-reopt_I_10cm-newwall_100k-mrk_biot-bfield_p01cellarea_FLR-mode_zero-profile_alpha-power_04222026.h5`
   (GC, 100k; `ascot_14.out`): 6,447 s simulation. It lost 9.85% of markers to the wall.
2. `G1600-engineeringwall-withtargets-200kmrk-gc-biot-decay-04292026.h5` (GC, 200k; `ascot_26.out`):
   12,616 s. A 1M-marker CPU run of the 10 cm-wall case (`ascot_1.out`) took 50,603 s, i.e. 141 CPU
   node-hours.
3. `G1600-free-reopt_I_10cm-wall_100k-mrk_biot-bfield_p01cellarea_GCFO-mode_zero-profile_05052026.h5`
   (GC→FO hybrid, 100k; `ascot_37.out`): 67,477 s. It's the closest CPU full-orbit case available; there
   is no CPU run of the LCFS-wall FO input.

**How to read it:**
- **GC:** the optimized GPU code does 100k GC markers in about 0.7 GPU node-hours, against about 18 CPU
  node-hours for a comparable CPU run, and 200k in 1.4 GPU node-hours against about 35.
- **Allocations:** NERSC charges GPU and CPU node-hours from separate allocations at different rates, so
  compare within one type.
- **Sizing:** 1 GPU node per about 100k markers is the efficient size (section 6).

### Physics consistency

| Benchmark | Before | After | Verdict |
|---|---|---|---|
| A. GC 100k | thermalized 80.31%, wall 19.69%, 0 aborted | thermalized 80.58%, wall 19.41%, 0 aborted | Consistent: z = 1.5; KS p = 0.49–0.61 |
| B. GC 200k | **power loss 4.924 ± 0.038%**, marker loss 10.147 ± 0.066%, 1.927 MW on the wall | **power loss 4.933 ± 0.039%**, marker loss 10.188 ± 0.064%, 1.930 MW | Consistent: power loss differs by 0.16σ; wall loads agree bin by bin (see below); KS p = 0.12–0.95 |
| C. GC 20k | wall 9.21% | wall 9.21% | Consistent: KS p = 0.73–0.97. `legacy` is **bit-identical** to before |
| D. FO 20k | wall 0.95% | wall 0.98% | Consistent: z = −0.3; KS p = 0.82–1.0. `legacy` is **bit-identical** to before |

### A. GC, 100k markers, one node

- **Input:** ISHW G1600 engineering-wall GC case,
  `/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/ISHW/G1600-free-reopt_I_eng-wall_100k-mrk_biot-bfield_p01cellarea_GC-mode_zero-profile_09012026.h5`.
  Adaptive GC, 300 ms limit; all markers end by 0.24 s.
- **Before:** the production run of 2026-09-08 (`results/run_0318234011` in that file; log
  `~/alpha_analysis-mp/logs/ascot_1.out`), same 4-GPU layout.
- **After:** job 59090327 on an exact copy of the inputs (`$PSCRATCH/gpu_validation/ishw100k_300ms.h5`).
  Output `runs/ishw100k_300ms_bench_opt_59090327.h5`; logs `logs/bench_59090327.out` and
  `logs/bench_59090327_opt.log`.
- **Timing spread:** the 4 GPUs finished between 2,042 s and 2,608 s. Markers are split into contiguous
  blocks, and the input's marker order makes some blocks slower.

### B. GC, 200k markers, two nodes

- **Input:** `$PSCRATCH/gpu_validation/g1600_200k_300ms.h5`, a random 200k subset of the 1M-marker G1600
  10 cm-wall input (`...10cm-wall_1M-mrk..._p05cellarea_FLR-mode...09102026.h5`), 300 ms limit.
- **Jobs:**
  - Before: job 59122647 (`base-minfo`), output `runs/g1600_200k_300ms_bench_base-minfo_59122647.h5`.
  - After: job 59122650 (`opt`), output `runs/g1600_200k_300ms_bench_opt_59122650.h5`.
  - Logs: `logs/bench_59122647*` and `logs/bench_59122650*`.
- **Wall loads** (`wall_compare.py`):
  - **Binned loads:** binned by wall-tile centroid (36 φ × 10 z bins, 214 bins loaded), the bin-by-bin
    differences have an rms of 0.95σ, with no bin beyond 3σ. That's what Monte Carlo noise alone gives.
  - **Hot spots:** the same bins (φ 210–230° and 80–90°, z +1.4 to +2.1 m), agreeing within 1.3σ.
  - **Peak single-tile load:** 12% higher after. This is noise: the peak tile collects only 125–146
    markers (±8–9% each), and it falls on a different tile of a cluster of near-equal tiles.
  - **Wetted area:** −2%. This counts tiles touched at least once (median 3 markers per tile), so it isn't
    precise at this marker count.

### C. GC, 20k markers, one GPU, 10 ms

- **Jobs:**
  - Job 59089329: `base-minfo`, `legacy`, `s12` and `opt` side by side, one GPU each.
  - Job 59089330: a clean repeat of `opt` plus tuning variants.
  - Outputs `runs/gc20k_10ms_*`; logs `logs/ab_59089329*` and `logs/ab_59089330*`.
- **Bit-exact checks:**
  - `legacy` and `s12` are bit-identical to `base-minfo` for all markers. So everything except the spline
    and RNG changes is bit-identical to the old code.
  - The tuning variants are bit-identical to `opt`.
- **Tuning** (simulation time; the defaults were kept):

  | Variant | Time |
  |---|---|
  | `opt` | 333.4 s |
  | register cap 168 | 355.2 s |
  | register cap 128 | 389.8 s |
  | `GPU_VLEN=64` | 329.9 s (noise) |

  `WALL_DEPTH=8` (finer wall-collision grid) was 6% faster and bit-identical in a separate 5k-marker test
  (job 59089524).

### D. FO, 20k markers, one GPU, 0.5 ms

- **Input:** a 20k subset of the ISHW FO input. Job 59124428 ran `base-minfo`, `legacy` and `opt` side by
  side. Outputs `runs/fo20k_0p5ms_*`; logs `logs/ab_59124428*`.
- **Times:** 423.9 s before, 161.7 s for `legacy` (2.6×), 141.4 s for `opt` (3.0×).
- **Bit-exact:** `legacy` is bit-identical to `base-minfo` for all markers that ran.
- **Zero-weight markers:** 16% of markers in this input have weight 0. They are rejected at start in every
  build; see Other findings.
- **Limitation:** 0.5 ms only covers prompt losses. The full-length FO test is item G.

### E. Multi-node correctness

- **Job:** 59089914, 40k GC markers with distribution output on, run on 8 GPUs (2 nodes) and on 4 GPUs
  (1 node). Logs `logs/multinode_59089914*`.
- **GPU binding:** the 8 ranks ran on 8 distinct physical GPUs (by PCI bus ID).
- **End states:** wall loss 9.27% vs 9.10% (z = 0.8), KS p > 0.6.
- **Distributions:** dist5D and ρ5D totals agree within 0.05% and their 1D marginals within 1.8%.

### F. Active-marker compaction: tried and reverted

**The idea:** as markers finish, move the running ones to the front of the arrays so the GPU only works
on those. Each marker keeps its original random-number stream, so results are bit-identical. It was
implemented (commit `87101928`), validated, measured, and then **reverted** (commit `e17f6351`), because it
doesn't speed up runs at production sizes.

**Correctness: bit-identical in every test.** `compact_stress` is a build that compacts at nearly every
check, to exercise the code heavily in short runs.

| Case | Job | `opt` | `compact` | `compact_stress` | Result |
|---|---|---|---|---|---|
| GC adaptive 5k, 2 ms | 59159303 | 52.1 s | 56.2 s | 58.2 s | Bit-identical |
| GC adaptive 20k, 10 ms | 59159305 | 333.8 s | – | 344.5 s | Bit-identical |
| FO 20k, 0.5 ms | 59159307 | 141.0 s | 141.2 s | 140.3 s | Bit-identical |
| GC fixed-step 5k, 0.5 ms | 59159312 | 145.4 s | – | 151.5 s | Bit-identical |
| GC 100k, 300 ms (full benchmark A) | 59159313 | 2,609 s | 2,572 s | – | Bit-identical |

**Speed: no useful gain.**
- **GC, 100k markers (25k per GPU):** −1.4%, within noise.
- **FO, 5k markers from the LCFSmark input, 0.3 ms, under the profiler (job 59163899):** compaction fired
  once (confirmed from the kernel counts), and the run got **23% slower** (60.3 vs 49.1 s). The orbit-push
  kernel went from 45 to 57 µs per step.

**Why:**
- **Below one wave:** at production sizes (up to about 30k markers per GPU) all groups of 32 markers
  already run at the same time, and a step lasts as long as the slowest group. Packing live markers into
  full groups makes each group, and so the step, slower.
- **The long FO tails are latency-bound:** in the LCFSmark100k run (section 5), a step cost about 400 µs
  with most markers running, and still about **115 µs with only a few dozen markers left per GPU**. That's
  the cost of one marker's step on a GPU, about 10 kernel launches plus dependent memory loads. Compaction
  can't reduce it.

**Ideas for the tail instead (not implemented):**
- Finish the last few percent of markers on a CPU, where one marker's step costs microseconds rather than
  about 115 µs.
- Fuse the step into fewer kernels.
- A hybrid GC/FO mode on GPU.

### G. Full-length FO comparison

- **Input:** the 100k-marker FO case with the LCFS wall, 300 ms limit:
  `.../convergence_studies/G1600-free-reopt_I_LCFS-wall_100k-mrk_biot-bfield_p01cellarea_FO-mode_zero-profile_06112026.h5`.
- **Before:** the June 12 production run (`results/run_1379985987` in that file; log
  `convergence_studies/logs-GPU/ascot_13.out`). 137,626 s simulation on 16 GPUs (4 nodes) = 611.7 GPU-h.
- **After:** job 59125235 with `opt` on 1 node (4 GPUs), on a copy of the inputs
  (`$PSCRATCH/gpu_validation/lcfs_fo100k_300ms.h5`). Output `runs/lcfs_fo100k_300ms_bench_opt_59125235.h5`;
  logs `logs/bench_59125235*`.
- **Time:** job 14:43:29 (52,992 s simulation; ranks 47,371–52,992 s) = **58.9 GPU-h, 10.4× less GPU
  time**. That's 2.6× less wall time on a quarter of the GPUs.
- **Physics: consistent.**

  | Quantity | Before | After | Difference |
  |---|---|---|---|
  | Thermalized | 81.14% | 81.06% | z = 0.48 |
  | Wall | 12.45% | 12.57% | z = −0.78 |
  | **Power loss** | **7.138 ± 0.066%** | **7.177 ± 0.074%** | +0.39σ |
  | Particle loss | 12.46% | 12.57% | +0.69σ |
  | Wall-deposited power | 13.84 MW | 13.92 MW | +0.6% |

  - **KS tests:** p = 0.21–0.63 on end energy, end time, ρ and wall-hit time.
  - **Binned wall loads:** 203 bins loaded; differences have an rms of 0.96σ, with no bin beyond 3σ.
  - **Hot spots:** the hottest bins (φ ≈ 180–200° and 350–360°, z −0.7–0 m) agree within 2.3σ.
  - **Peak tile load:** −3%.
  - **Aborts:** 6,407 vs 6,375 aborted with the same error types, mostly field-grid errors near the LCFS
    wall. These come from the input, not the code.

## 5. Production runs using the new code

| Job | Input (in `/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/ISHW/`) | Nodes | Status |
|---|---|---|---|
| 59126689 | `G1600-free-reopt_I-engwall-LCFSmark100k_..._09302026.h5` (115k FO markers) | 1 | **Done in 9:05:23** (`opt`). 94,248 wall, 20,756 thermalized, 0 aborted |
| 59126690 | `G1600-free-reopt_I-engwall-LCFSmark1M_..._09302026.h5` (1M FO markers) | 10 | **Done in 7:35:05** (`opt`). 819,958 wall, 180,025 thermalized, 17 energy limit, 0 aborted; ranks 23,753–27,286 s |

**Where the time went in the 100k run:**
- 95% of markers were done at 5.0 h and 99% at 7.3 h.
- **The last 5% of markers took 45% of the run.** A step cost about 400 µs with most markers running and
  fell to about 115 µs at the end (section F).
- The longest-lived marker needed 1.57 × 10⁸ steps (0.18 s simulated, 1.15 ns per step), and that sets
  the run length.

**Backups:** 24 h copies with a 23 h CPU-time cutoff (`..._cpulim.h5`, `ENDCOND_CPUTIMELIM=1`,
`ENDCOND_MAX_CPUTIME=82800` s) were queued in case the 10 h runs timed out. Both were cancelled unused once
the originals finished in time.

- **Outputs and logs:** results are written into the input files; logs are `logs/run_opt_<jobid>*`.

## 6. Using the new code

**Run:** `~/ascot5/ascot5_develop_mp_hpc_GPU/build/ascot5_main`.
- **Existing scripts:** the `run-sims-GPU*.sh` scripts in `~/alpha_analysis-mp` already use it.
- **Confirming the build:** a run's log shows `Tag v1-100-ge17f635`, plus a line per rank with its host and
  GPU PCI ID.
- **Rebuild:** `~/ascot5/install_develop_mp_hpc_GPU.sh`.
- **Previous binary:** the pre-optimization binary (2026-08-31) is kept in
  `~/ascot5/ascot5_develop_mp_hpc_GPU/old_builds/`.

**Sizing:** an A100 is used efficiently with about 25–30k markers per GPU.
- **GC:** fewer markers per GPU barely shortens the run, so use 1 node per 100k markers, or add markers
  when adding nodes.
- **FO:** time per step grows with markers per GPU even below that, so more GPUs help somewhat.
- **Long tails:** set `ENDCOND_CPUTIMELIM`/`ENDCOND_MAX_CPUTIME` about an hour under the Slurm limit, so
  stragglers can't cost the whole run.

## 7. Other findings (not caused by the optimization)

- **FLR losses were never active**, in the CPU or GPU code: in `endcond.c`,
  `enable_flr_losses & active_wall` is `1 & 0x8 = 0`. Runs labelled "FLR-mode" were plain GC as far as
  wall losses go. The code was left unchanged.
- **The ISHW FO marker input has 15,937 of 100,000 markers with weight 0.** They are rejected at start and
  reported as "aborted". There's no physics bias, but 16% of markers are wasted; the fix belongs in the FO
  marker preparation.
- **`create-input.py` distribution bins** (R 2.25–4.25 m, z ±1.5 m, |p| < 3.6e-20 kg·m/s) don't cover the
  G1600/Eos markers (R 5.5–9.9 m, |p| up to 1e-19). With those settings the dist5D and ρ5D outputs are
  empty.
- **Markers are split into contiguous blocks per rank.** If the marker file is ordered, some ranks take up
  to 1.6× longer. Shuffling markers when creating inputs avoids this.
- **The 2026-09-04 GC run in the ISHW 100k file** (`run_0268518078`) has 14,283 aborted markers from an
  older buggy build. Don't use it; the 2026-09-08 run is clean.

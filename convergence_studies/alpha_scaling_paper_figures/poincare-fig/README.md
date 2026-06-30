# Helios G1600 Poincare comparison — pure Biot-Savart vs mixed (eq.compute) field

Two-panel Poincare plot at φ=0 comparing two field constructions for the same
finite-β Helios G1600-12-89 QA2e-1 Bxdl25 equilibrium:

- **Pure Biot-Savart** — coil field + DESC `PlasmaField` (spectral fit of
  the equilibrium plasma current contribution) **everywhere**.
- **Mixed** — `eq.compute(["B_R","B_phi","B_Z"])` inside the LCFS,
  coil + `PlasmaField` outside. This reproduces the same construction Manav
  uses for ASCOT (`use_mixed_field=True` in `ascotBfield.py`).

**Output:** `out/poincare_manav_v5_sym.png`

## Folder layout

```
to-send/
├── README.md                # this file
├── build_field.py           # field construction (CPU)
├── wall_slices.py           # VTK → φ=0 (R, Z) polylines
├── make_figure.py           # spline trace + plot for the comparison figure
├── ascotBfield.py           # Manav's reference field-build (for traceability)
├── fl_tracer.py             # Todd's general-purpose FL tracer CLI
├── FL_TRACER_MANUAL.md      # docs for fl_tracer.py
├── data/
│   ├── equil_*.h5
│   ├── encircling_*.h5
│   ├── shaping_*.h5
│   └── *.vtk                # 4 wall files (LCFS_CX, 10cm, 30cm, engineering)
└── out/
    ├── field_pure_A64.h5    # gridded (BR,Bphi,BZ) — pure path
    ├── field_mixed_A64.h5   # gridded (BR,Bphi,BZ) — mixed path
    ├── trace_pure_A64.h5    # field-line trace + |B| plane
    ├── trace_mixed_A64.h5
    ├── walls_phi0.npz       # cached wall slices at φ=0
    └── poincare_manav_v5_sym.png
```

**Two-tier scripting.** `build_field.py` + `make_figure.py` are the pair that
produce *this comparison* figure (custom mixed-vs-pure field construction).
`fl_tracer.py` is the standalone general-purpose Poincare CLI — useful for
ordinary single-method traces (vacuum, finite-β with stock PlasmaField, mgrid
generation, 3D B·n) and is not invoked by `make_figure.py`. See
`FL_TRACER_MANUAL.md` for its full options.

## Environment

DESC + JAX + h5py + vtk + matplotlib. On Perlmutter the author used:

```bash
source ~/vdesc.sh                 # loads conda desc-env, CUDA/cuDNN
export PYTHONPATH=~/DESC:$PYTHONPATH
```

## Reproducing the figure

All three scripts use script-relative `data/` and `out/` paths, so just run
them from inside this folder.

### 1. Slice the walls (~30 s, CPU)

```bash
python wall_slices.py
```

Reads the 4 VTK files, cuts each by the y=0 plane, keeps the φ=0 hemisphere,
saves polylines (NaN-joined for non-convex walls) to `out/walls_phi0.npz`.

### 2. Build the field arrays (~6 min, CPU)

```bash
python build_field.py --A-res 64 --tag A64
```

Builds the PlasmaField **once** at A_res=64 (shared between pure and mixed),
loops over 50 phi sections × 200 R × 200 Z, and produces both
`field_pure_A64.h5` and `field_mixed_A64.h5`.

Domain: LCFS bbox + 1.2 m cushion (matches Manav's `wall_offset=100cm`).
Grid (200, 50, 200) over (R, φ, Z) at φ ∈ [0, 2π/NFP). NFP=2.

A_res=64 chosen as a fast/safe alternative to Manav's A_res=128 — at A_res=64
the spectral fit is already converged enough that the visible Poincare
structure is identical to A_res=128.

### 3. Trace + plot (~9 min, GPU; or ~10 s in --replot mode)

```bash
# First run: trace and save (uses GPU 0 by default)
python make_figure.py --tag A64 --orbits 1000 \
    --no-modB --no-cube-marker --no-line-marker \
    --dot-color black --symmetrize \
    --out poincare_manav_v5_sym.png

# Re-plot from cached traces (no GPU, ~10 s) — for aesthetic iteration
python make_figure.py --replot --tag A64 \
    --no-modB --no-cube-marker --no-line-marker \
    --dot-color black --symmetrize \
    --out poincare_manav_v5_sym.png
```

### Key flags for `make_figure.py`

| Flag | Effect |
|---|---|
| `--replot` | Skip retrace; load cached `out/trace_*.h5` |
| `--no-modB` | Hide \|B\| contour overlay |
| `--no-cube-marker` | Hide red squares (cube initial points) |
| `--no-line-marker` | Hide blue ×'s (radial line initial points) |
| `--dot-color black` | Use solid color instead of viridis-by-line-index |
| `--symmetrize` | Mirror every Poincare dot about Z=0 (see below) |
| `--gpu cpu` | Force CPU tracing (slow, but no GPU needed) |
| `--xlim`, `--ylim` | Crop the plot region |

### Sampling

- **Two cubes** (mirrored about Z=0): centered at (R, Z) = (8.01, ±3.67),
  side 0.40 m, 14×14 = 196 points each → 392 cube points covering both
  X-points at φ=0.
- **Radial line** at Z=0: 45 points across R ∈ [0.85·R0, 1.15·R0].
- Total: 437 starting points × 1000 orbits (500 fwd + 500 bwd).

## Why `--symmetrize`

The equilibrium and saved (BR, Bphi, BZ) arrays are stell-symmetric to ~1e-10
(verified). However, diffrax's adaptive ODE integrator does **not** respect
stell-sym exactly — it picks slightly different step paths for top-going vs
bottom-going field lines, which leads to ~25% asymmetry in finite-orbit
counts. `--symmetrize` mirrors every Poincare dot about Z=0 at plot time,
which is physically correct (the field IS symmetric) and yields a clean
publication figure.

## Notes / caveats

- The VTK `lcfs_CX.vtk` slice overlays exactly with the DESC LCFS at φ=0; the
  figure plots only the DESC LCFS to avoid redundancy.
- The `engineering` wall has divertor target structure at the **top** (Z>+3.4)
  for this equilibrium revision — the manual's `--foci 8.01 -3.67` X-point
  location is the up-down mirror from a different revision; here both
  X-points sit at Z = ±3.67.
- `--orbits 1000` is sufficient for visible flux-surface structure. Higher
  orbits → more island detail in the pure panel but no qualitative change.

— Todd

# FL Tracer Manual

`fl_tracer.py` — Field line tracer for stellarator Poincare plots, mgrid generation, and 3D B*n visualization. Built on DESC's `field_line_integrate` (diffrax ODE solver).

## Environment Setup

```bash
source ~/vdesc.sh
export PYTHONPATH=~/DESC:$PYTHONPATH
```

## Quick Start

```bash
# Vacuum trace (coils only, no plasma field)
python ~/bin/fl_tracer.py \
  --equil eq.h5 --shaping shaping.h5 --encircling encircling.h5 \
  --vacuum --gpu 0

# Finite-beta trace (coils + plasma current)
python ~/bin/fl_tracer.py \
  --equil eq.h5 --shaping shaping.h5 --encircling encircling.h5 \
  --A-res 16 --build-on-cpu --gpu 3

# Single coilset instead of separate shaping/encircling
python ~/bin/fl_tracer.py \
  --equil eq.h5 --coils coils.h5 --vacuum --gpu 0
```

## How It Works

### Two-Level Field Interpolation (finite-beta)

The finite-beta field has two components: external coils and internal plasma currents.

1. **PlasmaField** — Computes the plasma current contribution via Biot-Savart integral, then fits the vector potential A to a Chebyshev-Fourier spectral basis (resolution controlled by `--A-res`). B is obtained via spectral curl(A).

2. **SplineMagneticField** — The total field (coils + PlasmaField) is evaluated on a uniform 3D grid (`--interp-res` points in each of R, phi, Z) and fit to a cubic spline. This spline is what the ODE integrator actually evaluates during tracing.

For vacuum runs, only the coil field is used (no PlasmaField), and the spline is built directly from the coil field.

### Forward + Backward Tracing

All traces are done in **both directions**: half the requested orbits forward along B, half backward along -B. This doubles the coverage of the divertor leg and separatrix structure for the same compute cost.

The `--orbits` parameter specifies the total. E.g., `--orbits 1000` = 500 forward + 500 backward.

### The `--build-on-cpu` Flag (Critical for Finite-Beta)

PlasmaField construction requires large matrix operations (pseudoinverse of the Chebyshev-Fourier basis). For typical equilibria (L=12, M=12, N=20), this exceeds 80GB GPU RAM at A_res > 18.

`--build-on-cpu` solves this by:
1. Spawning a **separate CPU subprocess** (access to ~256GB system RAM)
2. Building PlasmaField + evaluating the total field + building the SplineMagneticField
3. Saving the spline to a temp file on disk
4. The main GPU process loads the spline and runs the fast ODE integration

**This flag is required for finite-beta runs** unless A_res <= 18 and interp-res is small.

Not needed for vacuum runs (no PlasmaField).

### Source Grid

The Biot-Savart integral discretizes the equilibrium current density using a `QuadratureGrid`. The code currently uses `QuadratureGrid(L=64, M=64, N=64)` (DESC's default). This is well-converged for typical equilibria.

### Memory Limits (A100 80GB GPU, Perlmutter)

| Parameter | GPU max | CPU max | Recommended | Notes |
|-----------|---------|---------|-------------|-------|
| A_res     | 18      | 32      | 32          | For G1600 L=12,M=12,N=20 eq |
| interp_res| 32      | 32      | 32          | 40 OOMs with A_res=32; 64 OOMs always |

## Standard Run Settings

The built-in defaults for `--foci`, `--cube-l`, and the fallback `--contour-levels` are **Helios-tuned**. At other scales you must override them — see `Scaling between machines` below.

### Helios (R0 ≈ 7.95, B ≈ 5 T)

```bash
python ~/bin/fl_tracer.py \
  --equil <eq.h5> \
  --shaping <shaping.h5> --encircling <encircling.h5> \
  --interp-res 32 --A-res 32 --build-on-cpu \
  --orbits 1000 --nphi 4 \
  --n-cube 14 --cube-l 0.40 --foci 8.01 -3.67 \
  --n-lines 45 \
  --rtol 1e-8 --atol 1e-8 --max-steps 200000 \
  --gpu 3 \
  -o <output_dir> --tag <run_tag>
```

**Typical runtime:** ~10 min (3 min CPU build + 7 min GPU tracing).

For vacuum, drop `--A-res`, `--build-on-cpu`, and add `--vacuum`.

### Eos (R0 ≈ 3.34, B = 5 T)

```bash
python ~/bin/fl_tracer.py \
  --equil <eq.h5> \
  --shaping <shaping.h5> --encircling <encircling.h5> \
  --interp-res 32 --A-res 32 --build-on-cpu \
  --orbits 200 --nphi 4 \
  --n-cube 14 --cube-l 0.17 --foci 3.19 -1.56 \
  --n-lines 45 \
  --rtol 1e-8 --atol 1e-8 --max-steps 200000 \
  --gpu <N> \
  --contour-levels 1 2 3 4 5 \
  -o <output_dir> --tag <run_tag>
```

**Typical runtime:** ~9 min finite-β (~3 min CPU build + 6 min GPU tracing), ~7 min vacuum.

Validated with the 6 MW on-axis heating equilibrium (`<β>_vol ≈ 0.5%`) on 2026-04-21, both vacuum and finite-β. Foci are the X-point location at φ=0 — see below.

### Scaling between machines

`--foci`, `--cube-l`, and `--contour-levels` are **not portable** between machines. Override per-equilibrium:

| Parameter      | How it scales                    | Helios          | Eos             |
|----------------|----------------------------------|-----------------|-----------------|
| `--foci R Z`   | X-point of that specific eq      | `8.01 -3.67`    | `3.19 -1.56`    |
| `--cube-l`     | ∝ R0 (same fraction of machine)  | `0.40`          | `0.17`          |
| `--contour-levels` | \|B\| range in the SOL       | `5 6 7` (default) | `1 2 3 4 5`   |

The code's hardcoded `linspace(5, 7, 5)` fallback for `--contour-levels` is Helios-tuned; at smaller machines it produces empty overlays because X-point-region |B| is lower. Always pass `--contour-levels` explicitly (or `--no-contours`) for sub-Helios equilibria.

**Finding the X-point for `--foci`:** datasets may ship an `x-point` marker — a zero-current `FourierRZCoil` named literally `"x-point"` whose curve traces the X-point vs φ. Evaluate it at φ=0:

```python
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
c = load(".../coils_..._xpoint.h5")
d = c.compute(["x"], grid=LinearGrid(zeta=np.linspace(0, 2*np.pi, 400)), basis="rpz")
R, phi, Z = np.asarray(d["x"]).T
i = np.argmin(np.abs(np.mod(phi + np.pi, 2*np.pi) - np.pi))
print(R[i], Z[i])   # → --foci R Z
```

**Bounds caveat.** Auto bounds (`R0*0.6..R0*1.4`, `±R0*0.5`) scale with R0 but can leave little margin below the X-point. If divertor-leg structure is important, set `--bounds-Z` explicitly (e.g., `-2.5 1.7` at Eos scale) to give legs room to trace before they exit the spline domain.

## All CLI Options

### Input Files
| Flag | Description |
|------|-------------|
| `--equil` | Equilibrium .h5 file (required) |
| `--coils` | Single coilset .h5 |
| `--shaping` | Shaping coils .h5 (use with `--encircling`) |
| `--encircling` | Encircling coils .h5 (use with `--shaping`) |

Provide either `--coils` OR both `--shaping` and `--encircling`.

### Physics
| Flag | Default | Description |
|------|---------|-------------|
| `--vacuum` | off | Coils only, no plasma current |
| `--A-res` | 32 | PlasmaField Chebyshev-Fourier resolution |

### Sampling (Starting Points)
| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | both | `line`, `cube`, or `both` |
| `--foci R Z` | 7.95 -3.61 | Cube center in (R, Z) |
| `--cube-l` | 0.20 | Cube side length (meters) |
| `--n-cube` | 12 | Points per side (total = n^2) |
| `--n-lines` | 10 | Number of radial line points |
| `--r-start` | R0*0.85 | Line start R |
| `--r-end` | R0*1.15 | Line end R |
| `--z-start` | 0.0 | Line Z coordinate |

The **cube** samples a square grid in the R-Z plane centered at `--foci`. Good for mapping divertor leg structure near the X-point.

The **line** samples along R at fixed Z=`--z-start`. Good for mapping the core flux surfaces.

### Tracing
| Flag | Default | Description |
|------|---------|-------------|
| `--orbits` | 100 | Total toroidal transits (split fwd/bwd) |
| `--nphi` | 4 | Poincare sections per field period |
| `--interp-res` | 64 | SplineMagneticField grid resolution |
| `--rtol` | 1e-8 | ODE relative tolerance |
| `--atol` | 1e-8 | ODE absolute tolerance |
| `--max-steps` | 100000 | Max ODE steps (per direction) |
| `--bounds-R` | auto | R domain [min, max] |
| `--bounds-Z` | auto | Z domain [min, max] |

### Device
| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | 0 | GPU index, or `cpu` |
| `--build-on-cpu` | off | Build PlasmaField on CPU, trace on GPU |

### Output
| Flag | Default | Description |
|------|---------|-------------|
| `-o` | coil dir | Output directory |
| `--tag` | fl | Filename tag for outputs |

### Tasks
| Flag | Default | Description |
|------|---------|-------------|
| `--fl` | yes | Poincare field line trace |
| `--mgrid` | no | Generate .mgrid file |
| `--plot-3d` | off | 3D B*n + coil HTML visualization |

### |B| Contours
| Flag | Default | Description |
|------|---------|-------------|
| `--no-contours` | off | Disable |B| overlay on Poincare |
| `--contour-levels` | linspace(5,7,5) | Custom |B| levels in Tesla |

## Poincare Plot Details

- **Colored dots**: Traced field line positions (viridis colormap, one color per field line), starting from the 1st orbit (initial positions excluded)
- **Red dots (alpha=0.01)**: Initial cube/line starting positions
- **Red dashed line**: LCFS from the equilibrium, drawn on top of the Poincare dots
- **Contours**: |B| magnitude from the spline field (inferno colormap)
- **Panels**: One per Poincare section within one field period

## 3D B*n Plot

- In **vacuum** mode: evaluates B*n on `eq.surface` (the equilibrium boundary geometry)
- In **finite-beta** mode: evaluates B*n on the full equilibrium object
- Overlays unique coils from the coilset
- Outputs both `.html` (interactive) and `.png` (static)

## Tips

- **Always check GPU status** before launching: `nvidia-smi`
- **Kill zombie GPU processes** after canceling a run: find PIDs with `nvidia-smi`, then `kill -9 <PID>`
- **Two simultaneous CPU builds will OOM** — run `--build-on-cpu` jobs sequentially
- **Vacuum runs are much faster** (~7 min) since they skip PlasmaField construction
- **You can run vacuum and finite-beta in parallel** on different GPUs since only finite-beta uses CPU RAM
- **Compute time scales roughly linearly** with orbits and number of field lines

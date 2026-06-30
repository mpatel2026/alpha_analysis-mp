#!/usr/bin/env python
"""Field line tracer: Poincare plots, mgrid generation, and 3D B*n visualization.

Examples:
  # Finite-beta Poincare trace (separate coils):
  python fl_tracer.py --equil eq.h5 --shaping sc.h5 --encircling ec.h5

  # Vacuum trace with single coilset:
  python fl_tracer.py --equil eq.h5 --coils coils.h5 --vacuum

  # With mgrid generation:
  python fl_tracer.py --equil eq.h5 --coils coils.h5 --mgrid yes

  # With 3D B*n + coil visualization:
  python fl_tracer.py --equil eq.h5 --shaping sc.h5 --encircling ec.h5 --plot-3d

  # Circle-only sampling with custom center:
  python fl_tracer.py --equil eq.h5 --coils coils.h5 --mode circle --foci 8.0 0.0

  # Custom resolution and tolerances:
  python fl_tracer.py --equil eq.h5 --coils coils.h5 --interp-res 128 --orbits 200

  # CPU mode:
  python fl_tracer.py --equil eq.h5 --coils coils.h5 --gpu cpu
"""

import argparse
import datetime
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/DESC"))


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Field line tracer with Poincare plots, mgrid, and 3D B*n.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Task toggles ──
    task = p.add_argument_group("tasks")
    task.add_argument(
        "--fl", default="yes", choices=["yes", "no"],
        help="Field line tracing + Poincare plot (default: yes)",
    )
    task.add_argument(
        "--mgrid", default="no", choices=["yes", "no"],
        help="Generate .mgrid file (default: no)",
    )
    task.add_argument(
        "--plot-3d", action="store_true",
        help="3D B*n surface + coil visualization",
    )

    # ── Input files ──
    inp = p.add_argument_group("input files")
    inp.add_argument("--equil", type=str, default=None,
                     help="Equilibrium .h5 (required for --fl and --plot-3d)")
    inp.add_argument("--coils", type=str, default=None,
                     help="Single coilset .h5")
    inp.add_argument("--shaping", type=str, default=None,
                     help="Shaping coils .h5")
    inp.add_argument("--encircling", type=str, default=None,
                     help="Encircling coils .h5")

    # ── Physics ──
    phys = p.add_argument_group("physics")
    phys.add_argument(
        "--vacuum", action="store_true",
        help="Vacuum tracing (coils only, no PlasmaField)",
    )
    phys.add_argument(
        "--A-res", type=int, default=32,
        help="PlasmaField vector-potential resolution (default: 32)",
    )

    # ── Field line sampling ──
    samp = p.add_argument_group("sampling")
    samp.add_argument(
        "--mode", default="both", choices=["line", "cube", "both"],
        help="Starting-point pattern (default: both)",
    )
    samp.add_argument(
        "--foci", type=float, nargs=2, metavar=("R", "Z"),
        default=[7.95, -3.61],
        help="Cube center (R, Z) (default: 7.95 -3.61)",
    )
    samp.add_argument(
        "--cube-l", type=float, default=0.20,
        help="Cube side length in meters (default: 0.20)",
    )
    samp.add_argument(
        "--n-cube", type=int, default=12,
        help="Points per side of cube (default: 12, total = n^2)",
    )
    samp.add_argument(
        "--r-start", type=float, default=None,
        help="Line sampling start R (default: R0*0.85)",
    )
    samp.add_argument(
        "--r-end", type=float, default=None,
        help="Line sampling end R (default: R0*1.15)",
    )
    samp.add_argument(
        "--n-lines", type=int, default=10,
        help="Number of radial field lines (default: 10)",
    )
    samp.add_argument(
        "--z-start", type=float, default=0.0,
        help="Z for line sampling (default: 0.0)",
    )

    # ── Tracing ──
    tr = p.add_argument_group("tracing")
    tr.add_argument(
        "--orbits", type=int, default=100,
        help="Toroidal transits (default: 100)",
    )
    tr.add_argument(
        "--nphi", type=int, default=4,
        help="Poincare sections per field period (default: 4)",
    )
    tr.add_argument(
        "--interp-res", type=int, default=64,
        help="SplineMagneticField resolution (default: 64)",
    )
    tr.add_argument(
        "--spline-chunk", type=int, default=None,
        help="Chunk size for spline build (default: auto)",
    )
    tr.add_argument("--rtol", type=float, default=1e-8,
                    help="ODE relative tolerance (default: 1e-8)")
    tr.add_argument("--atol", type=float, default=1e-8,
                    help="ODE absolute tolerance (default: 1e-8)")
    tr.add_argument(
        "--max-steps", type=int, default=100000,
        help="Max ODE solver steps (default: 100000)",
    )
    tr.add_argument(
        "--bounds-R", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="R domain bounds (default: derived from eq axis)",
    )
    tr.add_argument(
        "--bounds-Z", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="Z domain bounds (default: derived from eq axis)",
    )

    # ── mgrid ──
    mg = p.add_argument_group("mgrid parameters")
    mg.add_argument("--mgrid-res", type=int, default=128,
                    help="nR=nZ=nphi for mgrid (default: 128)")
    mg.add_argument("--Rmin", type=float, default=5.0, help="(default: 5.0)")
    mg.add_argument("--Rmax", type=float, default=11.0, help="(default: 11.0)")
    mg.add_argument("--Zmin", type=float, default=-5.0, help="(default: -5.0)")
    mg.add_argument("--Zmax", type=float, default=5.0, help="(default: 5.0)")
    mg.add_argument("--mgrid-output", type=str, default=None,
                    help="Output .mgrid path (default: auto)")

    # ── |B| contours ──
    ctr = p.add_argument_group("|B| contour overlay")
    ctr.add_argument(
        "--no-contours", action="store_true",
        help="Disable |B| contour overlay on Poincare plots",
    )
    ctr.add_argument(
        "--contour-levels", type=float, nargs="+", default=None,
        help="|B| contour levels in Tesla (default: linspace(5, 7, 5))",
    )
    ctr.add_argument(
        "--contour-cmap", type=str, default="inferno",
        help="Matplotlib colormap for |B| contours (default: inferno)",
    )

    # ── 3D plot ──
    p3d = p.add_argument_group("3D plot parameters")
    p3d.add_argument(
        "--plot-res", type=int, nargs=2, metavar=("M", "N"), default=[12, 12],
        help="B*n surface grid resolution (default: 12 12)",
    )

    # ── Output ──
    p.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory (default: directory of coil file)",
    )
    p.add_argument(
        "--tag", type=str, default="fl",
        help="Output filename tag (default: fl)",
    )

    # ── Device ──
    p.add_argument(
        "--gpu", type=str, default="0",
        help="GPU index or 'cpu' (default: 0)",
    )
    p.add_argument(
        "--build-on-cpu", action="store_true",
        help="Build PlasmaField + spline on CPU (more RAM), then trace on GPU",
    )

    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def get_starting_points(args, R0):
    """Generate (r0, z0) arrays based on --mode, using magnetic axis R0."""
    r_parts, z_parts = [], []

    if args.mode in ("cube", "both"):
        foci_r, foci_z = args.foci
        half = args.cube_l / 2.0
        r_1d = np.linspace(foci_r - half, foci_r + half, args.n_cube)
        z_1d = np.linspace(foci_z - half, foci_z + half, args.n_cube)
        rr, zz = np.meshgrid(r_1d, z_1d)
        r_parts.append(rr.ravel())
        z_parts.append(zz.ravel())
        print(
            f"  Cube: center=({foci_r:.3f}, {foci_z:.3f}), "
            f"l={args.cube_l}, n={args.n_cube}, total={args.n_cube**2}"
        )

    if args.mode in ("line", "both"):
        r_start = args.r_start if args.r_start is not None else R0 * 0.85
        r_end = args.r_end if args.r_end is not None else R0 * 1.15
        r_line = np.linspace(r_start, r_end, args.n_lines)
        z_line = np.full_like(r_line, args.z_start)
        r_parts.append(r_line)
        z_parts.append(z_line)
        print(
            f"  Line:   R=[{r_start:.3f}, {r_end:.3f}], "
            f"Z={args.z_start}, n={args.n_lines}"
        )

    return np.concatenate(r_parts), np.concatenate(z_parts)


def compute_modB_on_plane(field, phi_sec, R_bounds, Z_bounds, res=64):
    """Compute |B| on an R-Z plane at a given toroidal angle."""
    R_1d = np.linspace(R_bounds[0], R_bounds[1], res)
    Z_1d = np.linspace(Z_bounds[0], Z_bounds[1], res)
    R_2d, Z_2d = np.meshgrid(R_1d, Z_1d)

    coords = np.zeros((R_2d.size, 3))
    coords[:, 0] = R_2d.flatten()
    coords[:, 1] = phi_sec
    coords[:, 2] = Z_2d.flatten()

    B_vec = field.compute_magnetic_field(coords, basis="rpz")
    B_mag = np.linalg.norm(np.asarray(B_vec), axis=1)
    return R_2d, Z_2d, B_mag.reshape(R_2d.shape)


def load_coils(args):
    """Load coils -> (coil_field, [coils_for_plotting])."""
    from desc.io import load
    from desc.magnetic_fields import SumMagneticField

    if args.coils:
        print(f"Loading coils: {args.coils}")
        coils = load(args.coils)
        if isinstance(coils, (list, tuple)):
            coils = coils[-1]
        return coils, [coils]

    print(f"Loading shaping:    {args.shaping}")
    print(f"Loading encircling: {args.encircling}")
    shaping = load(args.shaping)
    encircling = load(args.encircling)
    if isinstance(shaping, (list, tuple)):
        shaping = shaping[-1]
    if isinstance(encircling, (list, tuple)):
        encircling = encircling[-1]
    return SumMagneticField(encircling, shaping), [shaping, encircling]


# =============================================================================
# Task: mgrid
# =============================================================================


def run_mgrid(coil_field, args, out_dir, timestamp):
    """Generate and save a .mgrid file from the coil field."""
    if args.mgrid_output:
        mgrid_path = args.mgrid_output
    else:
        mgrid_path = os.path.join(out_dir, f"mgrid_{args.tag}_{timestamp}.mgrid")

    print(f"\n{'=' * 60}")
    print(f"  MGRID GENERATION")
    print(f"{'=' * 60}")
    print(f"  Resolution: {args.mgrid_res}^3")
    print(f"  R: [{args.Rmin}, {args.Rmax}] m")
    print(f"  Z: [{args.Zmin}, {args.Zmax}] m")
    print(f"  Output: {mgrid_path}")

    coil_field.save_mgrid(
        mgrid_path,
        Rmin=args.Rmin, Rmax=args.Rmax,
        Zmin=args.Zmin, Zmax=args.Zmax,
        nR=args.mgrid_res, nZ=args.mgrid_res, nphi=args.mgrid_res,
    )
    print(f"  Saved: {os.path.abspath(mgrid_path)}")


# =============================================================================
# Task: field line tracing
# =============================================================================


def run_fl_trace(spline_field, eq, args, out_dir, timestamp):
    """Field line integration via field_line_integrate + Poincare plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from desc.grid import LinearGrid
    from desc.magnetic_fields import field_line_integrate

    NFP = eq.NFP
    R0 = float(eq.get_axis().R_n[0])
    R_min, R_max = args._R_bounds
    Z_min, Z_max = args._Z_bounds
    nphi = args.nphi
    orbits = args.orbits

    # ── Starting points ──
    print(f"\n{'=' * 60}")
    print(f"  FIELD LINE TRACING")
    print(f"{'=' * 60}")
    print(f"Sampling starting points (mode={args.mode})...")
    r0, z0 = get_starting_points(args, R0)
    n_fl = len(r0)
    print(f"  Total field lines: {n_fl}")

    # ── Phi arrays (forward + backward, half orbits each) ──
    half_orbits = orbits // 2
    n_phi_half = nphi * half_orbits * NFP + 1
    phi_fwd = np.linspace(0, half_orbits * 2 * np.pi, num=n_phi_half)
    phi_bwd = np.linspace(0, -half_orbits * 2 * np.pi, num=n_phi_half)
    half_max_steps = args.max_steps // 2

    print(f"\nTracing {n_fl} field lines, {half_orbits} orbits fwd + {half_orbits} bwd, "
          f"{n_phi_half} phi points each...")
    print(f"  Bounds: R[{R_min:.2f}, {R_max:.2f}], Z[{Z_min:.2f}, {Z_max:.2f}]")
    print(f"  rtol={args.rtol}, atol={args.atol}, max_steps={half_max_steps} (per direction)")

    # Forward trace
    R_fwd, Z_fwd = field_line_integrate(
        r0, z0, phi_fwd, spline_field,
        bounds_R=(R_min, R_max),
        bounds_Z=(Z_min, Z_max),
        rtol=args.rtol,
        atol=args.atol,
        max_steps=half_max_steps,
        options={"throw": False},
    )
    print(f"  Forward done: {np.sum(np.isfinite(R_fwd))}/{R_fwd.size} finite")

    # Backward trace
    R_bwd, Z_bwd = field_line_integrate(
        r0, z0, phi_bwd, spline_field,
        bounds_R=(R_min, R_max),
        bounds_Z=(Z_min, Z_max),
        rtol=args.rtol,
        atol=args.atol,
        max_steps=half_max_steps,
        options={"throw": False},
    )
    print(f"  Backward done: {np.sum(np.isfinite(R_bwd))}/{R_bwd.size} finite")

    # ── Clip out-of-bounds ──
    R_fwd = np.where((R_fwd < R_min) | (R_fwd > R_max), np.nan, R_fwd)
    Z_fwd = np.where((Z_fwd < Z_min) | (Z_fwd > Z_max), np.nan, Z_fwd)
    R_bwd = np.where((R_bwd < R_min) | (R_bwd > R_max), np.nan, R_bwd)
    Z_bwd = np.where((Z_bwd < Z_min) | (Z_bwd > Z_max), np.nan, Z_bwd)

    n_total = R_fwd.size + R_bwd.size
    n_finite = int(np.sum(np.isfinite(R_fwd)) + np.sum(np.isfinite(R_bwd)))
    print(f"\nResults: fwd={R_fwd.shape}, bwd={R_bwd.shape}")
    print(f"  Finite: {n_finite}/{n_total} ({100 * n_finite / n_total:.1f}%)")
    if n_finite > 0:
        R_all = np.concatenate([R_fwd, R_bwd])
        Z_all = np.concatenate([Z_fwd, Z_bwd])
        print(f"  R range: [{np.nanmin(R_all):.4f}, {np.nanmax(R_all):.4f}]")
        print(f"  Z range: [{np.nanmin(Z_all):.4f}, {np.nanmax(Z_all):.4f}]")

    # ── Poincare plot: one panel per section within one field period ──
    # Stride between repeated visits to the same toroidal angle:
    stride = nphi * NFP

    ncols = min(nphi, 4)
    nrows = (nphi + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 6 * nrows),
        squeeze=False,
    )

    phi_sections = np.linspace(0, 2 * np.pi / NFP, nphi, endpoint=False)
    cmap = plt.cm.viridis

    # ── Precompute |B| contour data per section ──
    plot_contours = not args.no_contours
    contour_levels = args.contour_levels
    if contour_levels is None:
        contour_levels = np.linspace(5, 7, 5)
    else:
        contour_levels = np.array(contour_levels)

    modB_data = {}
    if plot_contours:
        print("Computing |B| contours...")
        for i, phi_sec in enumerate(phi_sections):
            modB_data[i] = compute_modB_on_plane(
                spline_field, phi_sec,
                (R_min, R_max), (Z_min, Z_max),
                res=args.interp_res,
            )
        print("  |B| contours OK")

    for i, phi_sec in enumerate(phi_sections):
        ax = axes[i // ncols, i % ncols]

        # |B| contours (behind everything)
        if plot_contours and i in modB_data:
            R_g, Z_g, B_mag = modB_data[i]
            cs = ax.contour(
                R_g, Z_g, B_mag,
                levels=contour_levels,
                cmap=args.contour_cmap,
                alpha=0.6,
                linewidths=1.5,
            )
            ax.clabel(cs, inline=True, fontsize=8, fmt="%1.1f T")

        # Poincare dots from forward trace (skip initial point at orbit 0)
        Rf = R_fwd[i::stride, :]
        Zf = Z_fwd[i::stride, :]
        # Backward trace: section i in forward corresponds to section
        # (stride - i) % stride in the backward phi array
        bwd_idx = (stride - i) % stride
        Rb = R_bwd[bwd_idx::stride, :]
        Zb = Z_bwd[bwd_idx::stride, :]

        for j in range(n_fl):
            color = cmap(j / max(n_fl - 1, 1))
            ax.plot(Rf[1:, j], Zf[1:, j],
                    ls="", marker="o", ms=0.5, c=color)
            ax.plot(Rb[1:, j], Zb[1:, j],
                    ls="", marker="o", ms=0.5, c=color)

        # Initial points: cube in red, line in blue (only on phi=0 panel)
        if i == 0:
            n_cube_pts = args.n_cube**2 if args.mode in ("cube", "both") else 0
            if n_cube_pts > 0:
                ax.plot(Rf[0, :n_cube_pts], Zf[0, :n_cube_pts],
                        ls="", marker="s", ms=2, c="red", alpha=0.3, zorder=5)
            if n_cube_pts < n_fl:
                ax.plot(Rf[0, n_cube_pts:], Zf[0, n_cube_pts:],
                        ls="", marker="x", ms=4, mew=1, c="blue", alpha=0.8, zorder=5)

        # LCFS at this toroidal angle (on top of Poincare dots)
        grid = LinearGrid(
            rho=1.0, theta=360, zeta=phi_sec, NFP=NFP, endpoint=True,
        )
        lcfs = eq.compute(["R", "Z"], grid=grid)
        ax.plot(lcfs["R"], lcfs["Z"], "r--", lw=1.5, label="LCFS", zorder=10)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$R$ (m)")
        ax.set_ylabel(r"$Z$ (m)")
        mode_str = "vacuum" if args.vacuum else r"finite-$\beta$"
        ax.set_title(
            rf"$\phi \cdot N_{{FP}} / 2\pi = "
            rf"{phi_sec * NFP / (2 * np.pi):.2f}$ ({mode_str})"
        )

    # Hide unused axes
    for i in range(nphi, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"poincare_{args.tag}_{timestamp}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Poincare plot: {out_path}")


# =============================================================================
# Task: 3D B*n visualization
# =============================================================================


def run_3d_plot(coil_field, eq, coil_list, args, out_dir, timestamp):
    """3D B*n on LCFS + unique coils."""
    from desc.grid import LinearGrid
    from desc.plotting import plot_3d, plot_coils

    print(f"\n{'=' * 60}")
    print(f"  3D B*n VISUALIZATION")
    print(f"{'=' * 60}")

    M, N = args.plot_res
    print(f"  Grid: M={M}, N={N}")
    grid = LinearGrid(M=M, N=N, NFP=1, endpoint=True)

    if args.vacuum:
        fig = plot_3d(eq.surface, "B*n", field=coil_field, grid=grid)
    else:
        fig = plot_3d(eq, "B*n", field=coil_field, grid=grid)

    for coils in coil_list:
        fig_c = plot_coils(coils, unique=True)
        for trace in fig_c.data:
            fig.add_trace(trace)

    fig.update_layout(title=f"B*n ({args.tag})", height=800)

    base = os.path.join(out_dir, f"Bn3d_{args.tag}_{timestamp}")
    fig.write_html(base + ".html")
    print(f"  Saved: {base}.html")
    try:
        fig.write_image(base + ".png", scale=2)
        print(f"  Saved: {base}.png")
    except Exception as e:
        print(f"  PNG export skipped ({e})")


# =============================================================================
# CPU spline builder (subprocess)
# =============================================================================


def _build_spline_on_cpu(args):
    """Spawn a CPU-only subprocess to build PlasmaField + SplineMagneticField.

    Returns the path to the saved spline .h5 file.
    """
    import subprocess
    import tempfile

    spline_path = tempfile.mktemp(suffix="_spline.h5", prefix="fl_tracer_")

    # Build the inline script that runs on CPU
    coil_args = ""
    if args.coils:
        coil_args = f'coils = load("{args.coils}")\n'
        coil_args += '    if isinstance(coils, (list, tuple)): coils = coils[-1]\n'
        coil_args += '    coil_field = coils'
    else:
        coil_args = f'shaping = load("{args.shaping}")\n'
        coil_args += f'    encircling = load("{args.encircling}")\n'
        coil_args += '    if isinstance(shaping, (list, tuple)): shaping = shaping[-1]\n'
        coil_args += '    if isinstance(encircling, (list, tuple)): encircling = encircling[-1]\n'
        coil_args += '    coil_field = SumMagneticField(encircling, shaping)'

    script = f'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.expanduser("~/DESC"))

from desc import set_device
set_device("cpu")

import numpy as np
from desc.io import load
from desc.grid import QuadratureGrid
from desc.magnetic_fields import PlasmaField, SplineMagneticField, SumMagneticField

def build():
    print("  [CPU] Loading data...")
    eq = load("{args.equil}")
    if hasattr(eq, "__getitem__"): eq = eq[-1]
    {coil_args}

    R0 = float(eq.get_axis().R_n[0])
    NFP = eq.NFP
    R_min, R_max = {repr(args.bounds_R)} or (R0 * 0.6, R0 * 1.4)
    Z_min, Z_max = {repr(args.bounds_Z)} or (-R0 * 0.5, R0 * 0.5)

    print(f"  [CPU] NFP={{NFP}}, R0={{R0:.3f}}")
    print(f"  [CPU] Building PlasmaField (A_res={args.A_res})...")
    # PlasmaField bounds wider than spline bounds to avoid Chebyshev edge artifacts
    R_pad = 0.3 * (R_max - R_min)
    Z_pad = 0.3 * (Z_max - Z_min)
    pf_R_bounds = (R_min - R_pad, R_max + R_pad)
    pf_Z_bounds = (Z_min - Z_pad, Z_max + Z_pad)
    print(f"  [CPU] PlasmaField bounds: R=[{{pf_R_bounds[0]:.2f}}, {{pf_R_bounds[1]:.2f}}], "
          f"Z=[{{pf_Z_bounds[0]:.2f}}, {{pf_Z_bounds[1]:.2f}}]")
    print(f"  [CPU] Spline bounds: R=[{{R_min:.2f}}, {{R_max:.2f}}], Z=[{{Z_min:.2f}}, {{Z_max:.2f}}]")
    sg = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=NFP)
    plasma = PlasmaField(eq, source_grid=sg, R_bounds=pf_R_bounds,
                         Z_bounds=pf_Z_bounds, A_res={args.A_res}, chunk_size=200)
    print("  [CPU] PlasmaField OK")
    total = SumMagneticField(coil_field, plasma)

    print(f"  [CPU] Building SplineMagneticField (res={args.interp_res})...")
    R_sp = np.linspace(R_min, R_max, {args.interp_res})
    Z_sp = np.linspace(Z_min, Z_max, {args.interp_res})
    phi_sp = np.linspace(0, 2*np.pi/NFP, {args.interp_res}, endpoint=False)
    spline = SplineMagneticField.from_field(
        total, R=R_sp, phi=phi_sp, Z=Z_sp,
        method="cubic", extrap=False, NFP=NFP,
    )
    print("  [CPU] SplineMagneticField OK")

    spline.save("{spline_path}")
    print(f"  [CPU] Saved spline to {spline_path}")

build()
'''

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=False,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: CPU spline build failed (exit code {result.returncode})")
    if not os.path.exists(spline_path):
        sys.exit(f"ERROR: Spline file not created at {spline_path}")
    return spline_path


# =============================================================================
# Main
# =============================================================================


def main():
    import time as _time
    t_start = _time.time()

    args = parse_args()

    do_fl = args.fl == "yes"
    do_mgrid = args.mgrid == "yes"
    do_3d = args.plot_3d

    # ── Validate ──
    if args.coils and (args.shaping or args.encircling):
        sys.exit("ERROR: Provide --coils OR --shaping + --encircling, not both.")
    if not args.coils and not (args.shaping and args.encircling):
        sys.exit(
            "ERROR: Must provide --coils or both --shaping and --encircling.\n"
            "Run with --help for usage."
        )
    if (do_fl or do_3d) and not args.equil:
        sys.exit("ERROR: --equil required for --fl and --plot-3d.")
    if not do_fl and not do_mgrid and not do_3d:
        sys.exit("ERROR: Nothing to do. Enable --fl, --mgrid, or --plot-3d.")

    # ── Device setup (before any JAX/DESC imports) ──
    use_gpu = args.gpu.lower() != "cpu"
    build_on_cpu = getattr(args, "build_on_cpu", False) and use_gpu

    if build_on_cpu:
        # Phase 1: build spline on CPU in a subprocess, save to disk.
        # Phase 2: load spline on GPU in this process for tracing.
        print("--build-on-cpu: building spline in CPU subprocess...")
        t_cpu = _time.time()
        spline_path = _build_spline_on_cpu(args)
        print(f"  Spline saved to: {spline_path}")
        print(f"  CPU build time: {_time.time() - t_cpu:.1f}s")

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

    from desc import set_device
    set_device("gpu" if use_gpu else "cpu")

    import jax
    os.makedirs("./jax_cache", exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    from desc.io import load
    from desc.grid import QuadratureGrid
    from desc.magnetic_fields import (
        PlasmaField,
        SplineMagneticField,
        SumMagneticField,
    )

    # Workaround: field_line_integrate passes method= to
    # compute_magnetic_field, but SplineMagneticField doesn't accept it.
    _orig = SplineMagneticField.compute_magnetic_field

    @functools.wraps(_orig)
    def _patched(self, *a, **kw):
        kw.pop("method", None)
        return _orig(self, *a, **kw)

    SplineMagneticField.compute_magnetic_field = _patched

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load data ──
    coil_field, coil_list = load_coils(args)

    eq = None
    if do_fl or do_3d:
        print(f"Loading equilibrium: {args.equil}")
        eq = load(args.equil)
        if hasattr(eq, "__getitem__"):
            eq = eq[-1]

    # ── Derived geometry ──
    R0, NFP = None, None
    if eq is not None:
        R0 = float(eq.get_axis().R_n[0])
        NFP = eq.NFP
        print(f"  NFP={NFP}, L={eq.L}, M={eq.M}, N={eq.N}, R0={R0:.3f}")

    if args.bounds_R is not None:
        R_min, R_max = args.bounds_R
    elif R0 is not None:
        R_min, R_max = R0 * 0.6, R0 * 1.4
    else:
        R_min, R_max = args.Rmin, args.Rmax

    if args.bounds_Z is not None:
        Z_min, Z_max = args.bounds_Z
    elif R0 is not None:
        Z_min, Z_max = -R0 * 0.5, R0 * 0.5
    else:
        Z_min, Z_max = args.Zmin, args.Zmax

    # Stash derived values for task functions
    args._R_bounds = (R_min, R_max)
    args._Z_bounds = (Z_min, Z_max)

    # ── Output directory ──
    if args.output_dir:
        out_dir = args.output_dir
    elif args.coils:
        out_dir = os.path.dirname(os.path.abspath(args.coils))
    elif args.shaping:
        out_dir = os.path.dirname(os.path.abspath(args.shaping))
    else:
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    cpu_note = " (spline built on CPU)" if build_on_cpu else ""
    print(f"  FL Tracer  |  {'GPU ' + args.gpu if use_gpu else 'CPU'}{cpu_note}")
    print(f"  {'Vacuum' if args.vacuum else 'Finite-beta'}  |  Output: {out_dir}")
    print(f"{'=' * 60}")

    # ── Build total field (coils + optional plasma) ──
    if not build_on_cpu:
        if args.vacuum:
            print("\nVacuum mode: coils only")
            total_field = coil_field
        else:
            print(f"\nBuilding PlasmaField (A_res={args.A_res})...")
            source_grid = QuadratureGrid(
                L=64, M=64, N=64, NFP=NFP,
            )
            R_pad = 0.3 * (R_max - R_min)
            Z_pad = 0.3 * (Z_max - Z_min)
            plasma = PlasmaField(
                eq,
                source_grid=source_grid,
                R_bounds=(R_min - R_pad, R_max + R_pad),
                Z_bounds=(Z_min - Z_pad, Z_max + Z_pad),
                A_res=args.A_res,
                chunk_size=200,
            )
            print("  PlasmaField OK")
            total_field = SumMagneticField(coil_field, plasma)

    # ── mgrid (uses coil field, not total) ──
    if do_mgrid:
        run_mgrid(coil_field, args, out_dir, timestamp)

    # ── Build or load spline for FL tracing ──
    if do_fl:
        if build_on_cpu:
            print(f"\nLoading CPU-built spline from {spline_path}...")
            spline_field = load(spline_path)
            print("  SplineMagneticField loaded OK")
            # Clean up temp file
            os.remove(spline_path)
        else:
            print(f"\nBuilding SplineMagneticField (res={args.interp_res})...")
            R_sp = np.linspace(R_min, R_max, args.interp_res)
            Z_sp = np.linspace(Z_min, Z_max, args.interp_res)
            phi_sp = np.linspace(
                0, 2 * np.pi / NFP, args.interp_res, endpoint=False,
            )
            spline_field = SplineMagneticField.from_field(
                field=total_field,
                R=R_sp, phi=phi_sp, Z=Z_sp,
                method="cubic", extrap=False, NFP=NFP,
                chunk_size=args.spline_chunk,
            )
            print("  SplineMagneticField OK")
        t_trace = _time.time()
        run_fl_trace(spline_field, eq, args, out_dir, timestamp)
        print(f"  Tracing time: {_time.time() - t_trace:.1f}s")

    # ── 3D B*n (uses coil field for external B) ──
    if do_3d:
        run_3d_plot(coil_field, eq, coil_list, args, out_dir, timestamp)

    elapsed = _time.time() - t_start
    m, s = divmod(elapsed, 60)
    print(f"\n[DONE] All tasks complete. Total elapsed: {int(m)}m {s:.1f}s")


if __name__ == "__main__":
    main()

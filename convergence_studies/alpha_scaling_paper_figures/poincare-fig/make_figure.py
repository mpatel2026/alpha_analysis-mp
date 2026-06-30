#!/usr/bin/env python
"""Trace biotsavart vs nested fields and produce two-panel Poincare figure with walls.

Loads field arrays from `build_field.py`, builds SplineMagneticField, runs
forward+backward field_line_integrate on GPU, plots φ=0 panel for each field
with |B| contours, Poincare dots, DESC LCFS, and 3 wall overlays
(VTK LCFS_CX overlays exactly with DESC LCFS so we plot only the latter).

Usage:
    python make_figure.py                # default: A64 build, GPU 0
    python make_figure.py --gpu cpu      # CPU tracing (slower)
    python make_figure.py --orbits 500   # faster
"""

import argparse
import functools
import os
import sys
import time


# GPU setup must happen before any JAX/DESC imports.
def _peek_gpu():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", default="0")
    a, _ = p.parse_known_args()
    return a.gpu


_gpu_id = _peek_gpu()
if _gpu_id.lower() != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
sys.path.insert(0, os.path.expanduser("~/DESC"))

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from desc import set_device  # noqa: E402

set_device("gpu" if _gpu_id.lower() != "cpu" else "cpu")

import jax  # noqa: E402

from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.magnetic_fields import (  # noqa: E402
    SplineMagneticField,
    field_line_integrate,
)


# ── Config ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "out")
EQ_FILE = f"{DATA_DIR}/equil_Helios_G1600-12-89_QA2e-1_Bxdl25_free_L12_M12_N20.h5"

plt.rcParams.update({
    'font.size': 10,          # General font size
    'axes.labelsize': 10,     # x and y labels
    'axes.titlesize': 10,     # Title size
    'xtick.labelsize': 10,    # x-axis tick labels
    'ytick.labelsize': 10,    # y-axis tick labels
    'legend.fontsize': 10,     # Legend size
    'font.family': 'sans-serif' # A clean, professional serif font
})

# X-point at +Z for this equilibrium (manual's -3.67 is the up-down mirror
# from a different revision; the engineering wall here has the divertor
# target at the top, confirming +Z X-point at φ=0).
FOCI = (8.01, 3.67)
CUBE_L = 0.40
N_CUBE = 14
N_LINES = 45
NPHI_SECTIONS = 4  # per FP; we only plot one (φ=0)
RTOL = 1e-8
ATOL = 1e-8
MAX_STEPS = 100000
PHI_PLOT = 0.0
MODB_RES = 128
MODB_LEVELS = np.linspace(5.0, 7.0, 5)


# ── SplineMagneticField method= workaround (from fl_tracer.py) ─────────────
_orig_compute = SplineMagneticField.compute_magnetic_field


@functools.wraps(_orig_compute)
def _patched_compute(self, *a, **kw):
    kw.pop("method", None)
    return _orig_compute(self, *a, **kw)


SplineMagneticField.compute_magnetic_field = _patched_compute


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_cached_trace(path):
    """Load a saved trace_*.h5 (produced by an earlier run).

    Returns (trace_data, modB, bounds_R, bounds_Z) — same shape as the live
    trace path so plotting code is shared.
    """
    with h5py.File(path, "r") as f:
        Rf = np.asarray(f["R_fwd"])
        Zf = np.asarray(f["Z_fwd"])
        Rb = np.asarray(f["R_bwd"])
        Zb = np.asarray(f["Z_bwd"])
        r0 = np.asarray(f["r0"])
        z0 = np.asarray(f["z0"])
        modB_R = np.asarray(f["modB_R"])
        modB_Z = np.asarray(f["modB_Z"])
        modB_val = np.asarray(f["modB"])
        NFP = int(f.attrs["NFP"])
        n_cube = int(f.attrs["n_cube_pts"])
        bounds_R = tuple(float(x) for x in f.attrs["bounds_R"])
        bounds_Z = tuple(float(x) for x in f.attrs["bounds_Z"])
    trace_data = (Rf, Zf, Rb, Zb, r0, z0, n_cube, NFP)
    modB = (modB_R, modB_Z, modB_val)
    return trace_data, modB, bounds_R, bounds_Z


def load_spline_h5(path):
    with h5py.File(path, "r") as f:
        R = np.asarray(f["R_1d"])
        phi = np.asarray(f["phi_1d"])
        Z = np.asarray(f["Z_1d"])
        BR = np.asarray(f["BR"])
        Bphi = np.asarray(f["Bphi"])
        BZ = np.asarray(f["BZ"])
        NFP = int(f.attrs["NFP"])
        bounds_R = tuple(float(x) for x in f.attrs["bounds_R"])
        bounds_Z = tuple(float(x) for x in f.attrs["bounds_Z"])
    spline = SplineMagneticField(
        R=R, phi=phi, Z=Z,
        BR=BR, Bphi=Bphi, BZ=BZ,
        NFP=NFP, method="cubic", extrap=False,
    )
    return spline, NFP, bounds_R, bounds_Z


def starting_points(R0):
    """Two mirrored cubes (±FOCI[1]) around both X-points + radial line."""
    fr, fz = FOCI
    h = CUBE_L / 2
    r1 = np.linspace(fr - h, fr + h, N_CUBE)
    z_top = np.linspace(fz - h, fz + h, N_CUBE)
    z_bot = np.linspace(-fz - h, -fz + h, N_CUBE)
    rr_t, zz_t = np.meshgrid(r1, z_top)
    rr_b, zz_b = np.meshgrid(r1, z_bot)
    rcube = np.concatenate([rr_t.ravel(), rr_b.ravel()])
    zcube = np.concatenate([zz_t.ravel(), zz_b.ravel()])
    rline = np.linspace(R0 * 0.85, R0 * 1.15, N_LINES)
    zline = np.zeros_like(rline)
    return (
        np.concatenate([rcube, rline]),
        np.concatenate([zcube, zline]),
        len(rcube),
    )


def trace(spline, NFP, bounds_R, bounds_Z, orbits, R0):
    r0, z0, n_cube_pts = starting_points(R0)
    n_fl = len(r0)
    half = orbits // 2
    nphi_pts = NPHI_SECTIONS * half * NFP + 1
    phi_fwd = np.linspace(0.0, half * 2 * np.pi, nphi_pts)
    phi_bwd = np.linspace(0.0, -half * 2 * np.pi, nphi_pts)

    print(f"  {n_fl} field lines × {orbits} orbits "
          f"({half} fwd + {half} bwd), {nphi_pts} phi pts each")

    Rf, Zf = field_line_integrate(
        r0, z0, phi_fwd, spline,
        bounds_R=bounds_R, bounds_Z=bounds_Z,
        rtol=RTOL, atol=ATOL,
        max_steps=MAX_STEPS // 2,
        options={"throw": False},
    )
    print(f"  fwd: {np.sum(np.isfinite(np.asarray(Rf)))}/"
          f"{Rf.size} finite")

    Rb, Zb = field_line_integrate(
        r0, z0, phi_bwd, spline,
        bounds_R=bounds_R, bounds_Z=bounds_Z,
        rtol=RTOL, atol=ATOL,
        max_steps=MAX_STEPS // 2,
        options={"throw": False},
    )
    print(f"  bwd: {np.sum(np.isfinite(np.asarray(Rb)))}/"
          f"{Rb.size} finite")

    Rf, Zf, Rb, Zb = (np.asarray(x) for x in (Rf, Zf, Rb, Zb))
    Rf = np.where((Rf < bounds_R[0]) | (Rf > bounds_R[1]), np.nan, Rf)
    Zf = np.where((Zf < bounds_Z[0]) | (Zf > bounds_Z[1]), np.nan, Zf)
    Rb = np.where((Rb < bounds_R[0]) | (Rb > bounds_R[1]), np.nan, Rb)
    Zb = np.where((Zb < bounds_Z[0]) | (Zb > bounds_Z[1]), np.nan, Zb)
    return Rf, Zf, Rb, Zb, r0, z0, n_cube_pts, NFP


def compute_modB_plane(spline, phi_sec, bR, bZ, res=MODB_RES):
    R1 = np.linspace(bR[0], bR[1], res)
    Z1 = np.linspace(bZ[0], bZ[1], res)
    R2, Z2 = np.meshgrid(R1, Z1)
    coords = np.column_stack(
        [R2.ravel(), np.full(R2.size, phi_sec), Z2.ravel()]
    )
    B = np.asarray(spline.compute_magnetic_field(coords, basis="rpz"))
    return R2, Z2, np.linalg.norm(B, axis=1).reshape(R2.shape)


def plot_panel(ax, trace_data, walls, lcfs, modB, bR, bZ, title,
               show_modB=True, show_cube=True, show_line=True,
               dot_color=None, dot_ms=0.6, symmetrize=False):
    if show_modB:
        R2, Z2, B_mag = modB
        cs = ax.contour(
            R2, Z2, B_mag, levels=MODB_LEVELS,
            cmap="inferno", alpha=0.6, linewidths=1.2,
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%1.1f T")

    Rf, Zf, Rb, Zb, r0, z0, n_cube, NFP = trace_data
    n_fl = Rf.shape[1]
    stride = NPHI_SECTIONS * NFP
    Rf_p = Rf[0::stride, :]
    Zf_p = Zf[0::stride, :]
    Rb_p = Rb[0::stride, :]
    Zb_p = Zb[0::stride, :]
    cmap = plt.cm.viridis
    for j in range(n_fl):
        c = dot_color if dot_color is not None else cmap(
            j / max(n_fl - 1, 1)
        )
        Rfj, Zfj = Rf_p[1:, j], Zf_p[1:, j]
        Rbj, Zbj = Rb_p[1:, j], Zb_p[1:, j]
        ax.plot(Rfj, Zfj, ls="", marker="o", ms=dot_ms, c=c)
        ax.plot(Rbj, Zbj, ls="", marker="o", ms=dot_ms, c=c)
        if symmetrize:
            # Field is stell-sym at φ=0 → mirror every dot about Z=0
            # to fill in coverage the diffrax integrator missed.
            ax.plot(Rfj, -Zfj, ls="", marker="o", ms=dot_ms, c=c)
            ax.plot(Rbj, -Zbj, ls="", marker="o", ms=dot_ms, c=c)

    if show_cube:
        ax.plot(r0[:n_cube], z0[:n_cube],
                ls="", marker="s", ms=2, c="red", alpha=0.3, zorder=5)
    if show_line:
        ax.plot(r0[n_cube:], z0[n_cube:],
                ls="", marker="x", ms=4, mew=1,
                c="blue", alpha=0.8, zorder=5)

    ax.plot(lcfs[0], lcfs[1], "r-", lw=1.0, label="DESC LCFS", zorder=10)

    wall_styles = [
        ("offset_30cm", dict(color="tab:orange", lw=1.0,
                             label="Conformal walls: 10cm, 20, 30cm")),
        ("offset_10cm", dict(color="tab:orange",   lw=1.0
                            #,label="10 cm offset"
                            )),
        ("offset_20cm", dict(color="tab:orange",   lw=1.0
                                #,label="20 cm offset"
                                )),
        ("engineering", dict(color="black",      lw=1.0,
                             label="First wall with divertor")),
    ]
    for k, s in wall_styles:
        rz = walls[k]
        ax.plot(rz[:, 0], rz[:, 1], zorder=11, **s)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("R (m)", fontsize=12)
    ax.set_ylabel("Z (m)", fontsize=12)
    ax.set_xlim(bR)
    ax.set_ylim(bZ)
    ax.set_title(title, fontsize = 12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0",
                        help="GPU index, or 'cpu'")
    parser.add_argument("--tag", default="A64",
                        help="Field-build tag, e.g. 'A64' (matches build_field --tag)")
    parser.add_argument("--orbits", type=int, default=1000)
    parser.add_argument("--xlim", type=float, nargs=2, default=[6.0, 11.0],
                        metavar=("RMIN", "RMAX"))
    parser.add_argument("--ylim", type=float, nargs=2, default=[-4.5, 4.5],
                        metavar=("ZMIN", "ZMAX"))
    parser.add_argument("--modB-levels", type=float, nargs="+", default=None,
                        help="Custom |B| contour levels in T")
    parser.add_argument("--replot", action="store_true",
                        help="Skip tracing; load cached trace_*.h5 files")
    parser.add_argument("--no-modB", action="store_true",
                        help="Hide |B| contour overlay")
    parser.add_argument("--no-cube-marker", action="store_true",
                        help="Hide red cube initial-point markers")
    parser.add_argument("--no-line-marker", action="store_true",
                        help="Hide blue line initial-point markers")
    parser.add_argument("--dot-color", default=None,
                        help="Override field-line dot color (e.g. 'black'). "
                             "Default: viridis by line index")
    parser.add_argument("--dot-ms", type=float, default=0.3,
                        help="Field-line dot marker size (default: 0.6)")
    parser.add_argument("--symmetrize", action="store_true",
                        help="Mirror every Poincare dot about Z=0 "
                             "(restores stell-sym lost in the integrator).")
    parser.add_argument("--out", default="poincare_manav.png")
    args = parser.parse_args()

    global MODB_LEVELS
    if args.modB_levels is not None:
        MODB_LEVELS = np.asarray(args.modB_levels)

    os.makedirs("./jax_cache", exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    walls = np.load(f"{OUT_DIR}/walls_phi0.npz")

    print(f"[{time.strftime('%H:%M:%S')}] Loading equilibrium for LCFS...")
    fam = load(EQ_FILE)
    eq = fam[-1] if hasattr(fam, "__getitem__") else fam
    NFP_eq = int(eq.NFP)
    R0 = float(eq.get_axis().R_n[0])

    g = LinearGrid(rho=1.0, theta=360, zeta=PHI_PLOT, NFP=NFP_eq, endpoint=True)
    d = eq.compute(["R", "Z"], grid=g)
    lcfs = (np.asarray(d["R"]), np.asarray(d["Z"]))

    results = {}
    for tag in ("biotsavart", "nested"):
        if args.replot:
            cache_path = f"{OUT_DIR}/trace_{tag}_{args.tag}.h5"
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Loading cached trace: {cache_path}")
            trace_data, modB, bR, bZ = load_cached_trace(cache_path)
            results[tag] = (trace_data, modB, bR, bZ)
            continue

        path = f"{OUT_DIR}/field_{tag}_{args.tag}.h5"
        print(f"[{time.strftime('%H:%M:%S')}] Loading {path}...")
        spline, NFP, bR, bZ = load_spline_h5(path)
        print(f"  NFP={NFP}, bounds R={bR}, Z={bZ}")

        print(f"[{time.strftime('%H:%M:%S')}] Tracing {tag}...")
        t0 = time.time()
        trace_data = trace(spline, NFP, bR, bZ, args.orbits, R0)
        print(f"  trace done in {time.time() - t0:.1f}s")

        print(f"[{time.strftime('%H:%M:%S')}] Computing |B| for contours...")
        modB = compute_modB_plane(spline, PHI_PLOT, bR, bZ)
        results[tag] = (trace_data, modB, bR, bZ)

        # Save the trace + |B| plane for fast --replot iteration.
        Rf, Zf, Rb, Zb, r0, z0, n_cube, NFP_t = trace_data
        half = args.orbits // 2
        nphi_pts = NPHI_SECTIONS * half * NFP_t + 1
        phi_fwd = np.linspace(0.0, half * 2 * np.pi, nphi_pts)
        phi_bwd = np.linspace(0.0, -half * 2 * np.pi, nphi_pts)
        trace_path = f"{OUT_DIR}/trace_{tag}_{args.tag}.h5"
        with h5py.File(trace_path, "w") as f:
            f.create_dataset("r0", data=r0)
            f.create_dataset("z0", data=z0)
            f.create_dataset("phi_fwd", data=phi_fwd)
            f.create_dataset("phi_bwd", data=phi_bwd)
            f.create_dataset("R_fwd", data=Rf)
            f.create_dataset("Z_fwd", data=Zf)
            f.create_dataset("R_bwd", data=Rb)
            f.create_dataset("Z_bwd", data=Zb)
            R2, Z2, B_mag = modB
            f.create_dataset("modB_R", data=R2)
            f.create_dataset("modB_Z", data=Z2)
            f.create_dataset("modB", data=B_mag)
            f.attrs["NFP"] = NFP_t
            f.attrs["bounds_R"] = bR
            f.attrs["bounds_Z"] = bZ
            f.attrs["orbits"] = args.orbits
            f.attrs["nphi_sections"] = NPHI_SECTIONS
            f.attrs["n_cube_pts"] = n_cube
            f.attrs["foci_R"] = FOCI[0]
            f.attrs["foci_Z"] = FOCI[1]
            f.attrs["cube_l"] = CUBE_L
            f.attrs["n_cube_side"] = N_CUBE
            f.attrs["n_lines"] = N_LINES
            f.attrs["phi_plot"] = PHI_PLOT
            f.attrs["A_res_tag"] = args.tag
        print(f"  Saved trace: {trace_path}")

    # Plot bounds: user override (zoom), with spline-domain clamp.
    bR_spline = (
        min(results["biotsavart"][2][0], results["nested"][2][0]),
        max(results["biotsavart"][2][1], results["nested"][2][1]),
    )
    bZ_spline = (
        min(results["biotsavart"][3][0], results["nested"][3][0]),
        max(results["biotsavart"][3][1], results["nested"][3][1]),
    )
    bR_all = (max(args.xlim[0], bR_spline[0]),
              min(args.xlim[1], bR_spline[1]))
    bZ_all = (max(args.ylim[0], bZ_spline[0]),
              min(args.ylim[1], bZ_spline[1]))

    width = 190 / 25.4
    height = width * (11/15)
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharex=True, sharey=True)
    titles = {
        "biotsavart": "",
        "nested": "",
    }
    for ax, tag in zip(axes, ("biotsavart", "nested")):
        trace_data, modB, _, _ = results[tag]
        plot_panel(
            ax, trace_data, walls, lcfs, modB,
            bR_all, bZ_all, titles[tag],
            show_modB=not args.no_modB,
            show_cube=not args.no_cube_marker,
            show_line=not args.no_line_marker,
            dot_color=args.dot_color,
            dot_ms=args.dot_ms,
            symmetrize=args.symmetrize,
        )
    axes[0].set_xlabel('R [m]', fontsize=10)
    axes[0].set_ylabel('Z [m]', fontsize=10)
    axes[1].set_xlabel('R [m]', fontsize=10)
    axes[1].set_ylabel('Z [m]', fontsize=10)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=10)
    #fig.suptitle(
    #    f"Helios G1600-12-89 QA Bxdl25 — finite-β Poincare at φ=0 "
    #    f"(A_res={args.tag.lstrip('A') or '?'}, "
    #    f"{args.orbits} orbits)",
    #    y=0.99,
    #)
    fig.tight_layout()
    out_path = f"{OUT_DIR}/{args.out}"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

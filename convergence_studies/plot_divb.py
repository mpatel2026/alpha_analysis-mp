"""Compare raw div(B) on a (R, z) cross section between two ASCOT5 bfields.

Uses a5py's own `divb` derived quantity (computed from the same C splines
that orbit-following uses) rather than re-deriving the finite-difference
divergence by hand. Plots the two inputs side by side on a shared color
scale for direct comparison.
"""

import argparse
from pathlib import Path

import numpy as np
import unyt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a5py import Ascot

FILE1 = (
    "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
    "G1600-free-reopt_I_10cm-newwall_100k-mrk_mixed-bfield_p01cellarea_"
    "FO-mode_zero-profile_alpha-power_04172026.h5"
)
FILE2 = (
    "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
    "G1600-free-reopt_I_engineeringwall_100k-mrk_biot-bfield_p01cellarea_FO-mode_decay-profile_05082026.h5"
)
LABEL1 = "(a)"
LABEL2 = "(b)"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file1", default=FILE1, help="First ASCOT5 .h5 file")
    p.add_argument("--file2", default=FILE2, help="Second ASCOT5 .h5 file")
    p.add_argument("--phi", type=float, default=0.0,
                    help="Toroidal angle [deg] of the cross section")
    p.add_argument("--rmin", type=float, default=7.5,
                    help="Minimum R [m] of the evaluation window")
    p.add_argument("--rmax", type=float, default=10.0,
                    help="Maximum R [m] of the evaluation window")
    p.add_argument("--zmin", type=float, default=-3.5,
                    help="Minimum z [m] of the evaluation window")
    p.add_argument("--zmax", type=float, default=3.5,
                    help="Maximum z [m] of the evaluation window")
    p.add_argument("--nr", type=int, default=400)
    p.add_argument("--nz", type=int, default=400)
    p.add_argument("--rhomax", type=float, default=None,
                    help="Mask points with rho > rhomax")
    p.add_argument("--out", default=None,
                    help="Output PNG path (default: figures/divb_compare_phi<phi>.png)")
    return p.parse_args()


def stats(name, arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"  {name}: all NaN")
        return
    print(f"  {name}: min={np.min(finite):.3e} max={np.max(finite):.3e} "
          f"rms={np.sqrt(np.mean(finite**2)):.3e}")


def conformal_offset_curve(r, z, offset):
    """Offset a closed poloidal curve (r, z) outward by `offset` [m].

    Same normal-offset construction as a5py's
    ``import_desc_conformal_offset_wall`` template
    (a5py/templates/importdata.py), specialized to a single 2D poloidal
    contour instead of a full 3D (theta, zeta) surface: estimate the local
    tangent from neighboring points, rotate 90 degrees for the normal, and
    flip it outward using the direction away from the contour's centroid.
    """
    r_up, r_down = np.roll(r, -1), np.roll(r, 1)
    z_up, z_down = np.roll(z, -1), np.roll(z, 1)
    dr = r_up - r_down
    dz = z_up - z_down
    perp_r, perp_z = dz, -dr
    mag = np.hypot(perp_r, perp_z)
    mag = np.where(mag == 0, 1e-10, mag)
    unit_r, unit_z = perp_r / mag, perp_z / mag

    center_r, center_z = np.mean(r), np.mean(z)
    dot = unit_r * (r - center_r) + unit_z * (z - center_z)
    flip = np.where(dot < 0, -1, 1)
    unit_r, unit_z = unit_r * flip, unit_z * flip

    return r + offset * unit_r, z + offset * unit_z


def compute_divb(path, args, ax, overlay_rhocontour=True,
                  rho_style=None, overlay_wall=False, wall_style=None,
                  overlay_offsets=None, offset_style=None):
    """Open `path`, evaluate div(B) at phi=args.phi, overlay rho=1 on `ax`.

    `rho_style`/`wall_style` are optional kwarg dicts (e.g. color, label)
    forwarded to the respective a5py overlay call. `overlay_offsets` is an
    optional list of outward offsets [m] from the rho=1 (LCFS) contour,
    plotted with `offset_style` (label applied only to the first curve).

    Returns (r, z, divb_v, btype, qid).
    """
    rho_style = {"color": "black", **(rho_style or {})}
    offset_style = {"color": "tab:orange", **(offset_style or {})}
    a5 = Ascot(path)
    bgroup = a5.data.bfield.active
    btype = bgroup.get_type()
    qid = bgroup.get_qid()
    print(f"[{Path(path).name}] Active bfield input: type={btype}, qid={qid}, "
          f"desc={bgroup.get_desc()!r}")
    if not btype.startswith("B_"):
        raise RuntimeError(f"Unexpected bfield type {btype!r}")
    if btype not in ("B_STS", "B_3DS"):
        print(f"Warning: {btype} fields are analytic/2D and may be "
              "divergence-free by construction; div(B) map may be trivial.")

    bdata = bgroup.read()
    if "b_rmin" in bdata:
        rmin, rmax = bdata["b_rmin"], bdata["b_rmax"]
        zmin, zmax = bdata["b_zmin"], bdata["b_zmax"]
        nr_grid = int(np.squeeze(bdata["b_nr"]))
        nz_grid = int(np.squeeze(bdata["b_nz"]))
    else:
        rmin, rmax = bdata["rmin"], bdata["rmax"]
        zmin, zmax = bdata["zmin"], bdata["zmax"]
        nr_grid = int(np.squeeze(bdata["nr"]))
        nz_grid = int(np.squeeze(bdata["nz"]))
    rmin, rmax = float(np.squeeze(rmin)), float(np.squeeze(rmax))
    zmin, zmax = float(np.squeeze(zmin)), float(np.squeeze(zmax))

    dR = (rmax - rmin) / (nr_grid - 1)
    dz = (zmax - zmin) / (nz_grid - 1)

    # Stay one full grid cell inside the stored domain so we never sample
    # outside the spline's support (which returns NaN), then narrow further
    # to the user-requested R window.
    r_lo = max(rmin + dR, args.rmin)
    r_hi = min(rmax - dR, args.rmax)
    if r_lo >= r_hi:
        raise ValueError(
            f"Requested R range [{args.rmin}, {args.rmax}] does not overlap "
            f"the bfield's valid domain [{rmin + dR:.4f}, {rmax - dR:.4f}]"
        )
    z_lo = max(zmin + dz, args.zmin)
    z_hi = min(zmax - dz, args.zmax)
    if z_lo >= z_hi:
        raise ValueError(
            f"Requested z range [{args.zmin}, {args.zmax}] does not overlap "
            f"the bfield's valid domain [{zmin + dz:.4f}, {zmax - dz:.4f}]"
        )

    # Half-cell offset from the evaluation grid to the storage knots: this
    # samples away from points where the spline is pinned to the stored
    # value, giving an honest (rather than optimistic) view of the
    # interpolation error between knots.
    r = np.linspace(r_lo + dR / 2, r_hi - dR / 2, args.nr) * unyt.m
    z = np.linspace(z_lo + dz / 2, z_hi - dz / 2, args.nz) * unyt.m

    a5.input_init(bfield=True)
    try:
        qnts = ["divb"]
        if args.rhomax is not None:
            qnts.append("rho")
        result = a5.input_eval(r, args.phi * unyt.deg, z, 0 * unyt.s,
                                *qnts, grid=True)
        if args.rhomax is not None:
            divb, rho = result
        else:
            divb, rho = result, None
        divb_v = np.squeeze(divb[:, 0, :, 0]).v

        if rho is not None:
            rho = np.squeeze(rho[:, 0, :, 0]).v
            divb_v = np.where(rho > args.rhomax, np.nan, divb_v)

        print("div(B) [T/m]:")
        stats("raw", divb_v)

        if overlay_rhocontour:
            try:
                a5.input_plotrhocontour(rho=1.0, phi=args.phi * unyt.deg,
                                         axes=ax, **rho_style)
            except Exception as exc:
                print(f"Could not overlay rho=1 contour: {exc}")

        if overlay_offsets:
            try:
                ntheta = 360
                theta = np.linspace(0, 2 * np.pi, ntheta) * unyt.rad
                r_lcfs, z_lcfs = a5.input_rhotheta2rz(
                    np.ones(ntheta), theta, args.phi * unyt.deg, 0 * unyt.s)
                r_lcfs, z_lcfs = r_lcfs.to("m").v, z_lcfs.to("m").v
                for i, off in enumerate(overlay_offsets):
                    r_off, z_off = conformal_offset_curve(r_lcfs, z_lcfs, off)
                    style = dict(offset_style)
                    if i > 0:
                        style.pop("label", None)
                    ax.plot(np.append(r_off, r_off[0]),
                            np.append(z_off, z_off[0]), **style)
            except Exception as exc:
                print(f"Could not overlay conformal offset walls: {exc}")
    finally:
        a5.input_free(bfield=True)

    if overlay_wall:
        try:
            a5.input_plotwallcontour(phi=args.phi * unyt.deg, axes=ax,
                                      **(wall_style or {}))
        except Exception as exc:
            print(f"Could not overlay wall contour: {exc}")

    return r, z, divb_v, btype, qid


def main():
    args = parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5),
                                    layout="compressed")

    rho_style = {"color": "black", "label": "LCFS"}
    r1, z1, divb1, btype1, qid1 = compute_divb(args.file1, args, ax1,
                                                rho_style=rho_style)
    r2, z2, divb2, btype2, qid2 = compute_divb(args.file2, args, ax2,
                                                rho_style=rho_style)

    clim = [-0.01, 0.01]
    norm = matplotlib.colors.CenteredNorm(halfrange=max(abs(c) for c in clim))
    cmap = "seismic"

    def draw_panel(ax, r, z, divb_v):
        zm = np.ma.masked_invalid(divb_v)
        shading = "nearest" if zm.shape == (r.size, z.size) else "flat"
        pcm = ax.pcolormesh(r, z, zm.T, norm=norm, cmap=cmap, shading=shading)
        ax.patch.set(hatch="x", edgecolor=[0.9, 0.9, 0.9])
        ax.set_xlim(r[0], r[-1])
        ax.set_ylim(z[0], z[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R [m]")
        return pcm

    draw_panel(ax1, r1.v, z1.v, divb1)
    ax1.set_ylabel("z [m]")
    ax1.set_title(LABEL1)

    pcm2 = draw_panel(ax2, r2.v, z2.v, divb2)
    ax2.set_title(LABEL2)
    # input_plotrhocontour() unconditionally sets an "z [m]" ylabel on
    # whatever axes it's given; clear it since this label is shared with ax1.
    ax2.set_ylabel("")
    ax2.tick_params(labelleft=False)

    cbar = fig.colorbar(pcm2, ax=ax2)
    cbar.set_label(r"$\nabla\cdot B$ [T/m]")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=10)

    out = args.out or f"figures/divb_compare_phi{args.phi:g}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()

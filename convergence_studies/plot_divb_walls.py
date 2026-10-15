"""Compare raw div(B) between two ASCOT5 bfields, styled like the poincare-fig
comparison figure (alpha_scaling_paper_figures/poincare-fig/make_figure.py):
shared axes, grid, legend banner, and wall/LCFS overlays, but with the
poincare dot cloud replaced by the div(B) heatmap from plot_divb.py.

The wall/LCFS overlays are computed directly from each input .h5 file rather
than from the poincare-fig folder's cached walls_phi0.npz (sliced from a
different equilibrium/wall set that did not line up with these files):

- Actual (divertor/engineering) wall: each file's own `wall_3D` mesh sliced
  at phi=args.phi via a5py's `input_plotwallcontour`.
- LCFS: each file's own rho=1 contour via `input_plotrhocontour`.
- 10/20/30 cm conformal walls: LCFS offset outward along its local normal,
  using the same construction as a5py's
  `import_desc_conformal_offset_wall` template
  (a5py/templates/importdata.py), see `conformal_offset_curve` in
  plot_divb.py.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import unyt

from a5py import Ascot
from a5py.routines import plotting as a5plt
from plot_divb import FILE1, FILE2, LABEL1, LABEL2, compute_divb

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

RHO_STYLE = dict(color="red", lw=1.0, label="LCFS (rho=1)")
WALL_STYLE = dict(color="black", label="Divertor wall")
OFFSET_STYLE = dict(color="tab:orange", lw=1.0,
                     label="Conformal walls: 10, 20, 30 cm")
OFFSETS_M = [0.10, 0.20, 0.30]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file1", default=FILE1, help="First ASCOT5 .h5 file")
    p.add_argument("--file2", default=FILE2, help="Second ASCOT5 .h5 file")
    p.add_argument("--phi", type=float, default=0.0,
                    help="Toroidal angle [deg] of the cross section")
    p.add_argument("--rmin", type=float, default=7.5,
                    help="Minimum R [m] of the evaluation window")
    p.add_argument("--rmax", type=float, default=10.25,
                    help="Maximum R [m] of the evaluation window")
    p.add_argument("--zmin", type=float, default=-4.25,
                    help="Minimum z [m] of the evaluation window")
    p.add_argument("--zmax", type=float, default=4.25,
                    help="Maximum z [m] of the evaluation window")
    p.add_argument("--nr", type=int, default=400)
    p.add_argument("--nz", type=int, default=400)
    p.add_argument("--rhomax", type=float, default=None,
                    help="Mask points with rho > rhomax")
    p.add_argument("--wall-file", default=None,
                    help="ASCOT5 .h5 whose wall_3D is drawn on BOTH panels "
                         "(default: --file2, i.e. the Biot-Savart file)")
    p.add_argument("--out", default=None,
                    help="Output PNG path (default: figures/divb_walls_phi<phi>.png)")
    return p.parse_args()


def get_wall_lines(path, phi_deg):
    """Return this file's wall_3D cross section at phi=phi_deg as line
    segments (n, 2, 2), via a5py's wall_3D.getwallcontour()."""
    a5 = Ascot(path)
    return a5.data.wall.active.getwallcontour(phi=phi_deg * unyt.deg)


def plot_wall_lines(ax, lines, color="black", label=None):
    for i in range(len(lines)):
        ax.plot(lines[i, :, 0], lines[i, :, 1], color=color,
                label=label if i == 0 else None)


def main():
    args = parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(190 / 25.4, 190 / 25.4 * 11 / 15),
                                    sharex=True, sharey=True)

    r1, z1, divb1, btype1, qid1 = compute_divb(
        args.file1, args, ax1, overlay_rhocontour=True, rho_style=RHO_STYLE,
        overlay_wall=False,
        overlay_offsets=OFFSETS_M, offset_style=OFFSET_STYLE)
    r2, z2, divb2, btype2, qid2 = compute_divb(
        args.file2, args, ax2, overlay_rhocontour=True, rho_style=RHO_STYLE,
        overlay_wall=False,
        overlay_offsets=OFFSETS_M, offset_style=OFFSET_STYLE)

    # Use one file's actual wall on both panels for a consistent overlay
    # (defaults to file2, the Biot-Savart / engineering-wall file).
    wall_lines = get_wall_lines(args.wall_file or args.file2, args.phi)
    plot_wall_lines(ax1, wall_lines, **WALL_STYLE)
    plot_wall_lines(ax2, wall_lines, **WALL_STYLE)

    clim = [-0.01, 0.01]

    a5plt.mesh2d(r1.v, z1.v, divb1, diverging=True, cmap="seismic",
                 axesequal=True, xlabel="R [m]", ylabel="Z [m]",
                 clabel=r"$\nabla\cdot B$ [T/m]", clim=list(clim), axes=ax1)
    ax1.set_title(LABEL1)

    a5plt.mesh2d(r2.v, z2.v, divb2, diverging=True, cmap="seismic",
                 axesequal=True, xlabel="R [m]", ylabel="Z [m]",
                 clabel=r"$\nabla\cdot B$ [T/m]", clim=list(clim), axes=ax2)
    ax2.set_title(LABEL2)

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(args.rmin, args.rmax)
        ax.set_ylim(args.zmin, args.zmax)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=10)
    fig.tight_layout()

    out = args.out or f"figures/divb_walls_phi{args.phi:g}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()

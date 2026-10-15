"""Wall-load summary for one finished ASCOT5 run: tor-pol wall-load plot, VTK of
the wall mesh with loads, and power-loss / peak-load numbers.

Power to the wall is the sum over wall tiles of the deposited power
(weight * end energy of markers whose end condition is "wall"). Power loss is
that divided by a reference total power (--p-ref, MW).

Peak load is max(tile power / tile area). With few markers this is a noisy
quantity (a handful of markers on one tile), so the number of markers on the
peak tile and the tile area are reported too.

  python wallload_summary.py --input ISHW/<file>.h5 --label engwall-LCFSmark100k
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from a5py import Ascot

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", required=True, help="ASCOT5 file, relative to --dir or absolute")
parser.add_argument("--dir", default="/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/")
parser.add_argument("--label", required=True, help="short name used for output files")
parser.add_argument("--p-ref", type=float, default=191.0,
                    help="reference total power in MW that power loss is a fraction of")
parser.add_argument("--clim", type=float, nargs=2, default=[1e3, 6e6],
                    help="colour limits of the wall-load plot in W/m^2")
parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
args = parser.parse_args()

fn = args.input if os.path.isabs(args.input) else os.path.join(args.dir, args.input)
figdir = os.path.join(args.outdir, "figs")
vtkdir = os.path.join(args.outdir, "vtks")
os.makedirs(figdir, exist_ok=True)
os.makedirs(vtkdir, exist_ok=True)

a5 = Ascot(fn)
run = a5.data.active

# ---- numbers -----------------------------------------------------------------
def val(x, unit=None):
    """Plain float array, converting units first if x carries them."""
    if unit is not None and hasattr(x, "to"):
        x = x.to(unit)
    return np.asarray(getattr(x, "v", x), dtype=float)


ids, area, edepo, pdepo, _ = run.getwall_loads()
_, _, _, mload, _ = run.getwall_loads(weights=False)       # markers per tile
ids = val(ids).astype(int)
area = val(area, "m**2")
power_W = val(edepo, "W")
mload = val(mload)
load = power_W / area                                      # W/m^2 per wetted tile
ipk = int(np.argmax(load))

endcond = np.asarray(run.getstate("endcond", state="end"))
n_total = endcond.size
n_wall = int(mload.sum())
P_wall = power_W.sum() * 1e-6                              # MW
P_ini = float(np.sum(val(run.getstate("ekin", state="ini"), "J")
                     * val(run.getstate("weight", state="ini")))) * 1e-6

summary = dict(
    file=fn, label=args.label,
    n_markers=int(n_total), n_wall_hit=n_wall,
    frac_markers_wall=n_wall / n_total,
    initial_power_MW=P_ini,
    wall_power_MW=P_wall,
    wall_power_fraction_of_initial=P_wall / P_ini,
    p_ref_MW=args.p_ref,
    power_loss_percent_of_pref=100 * P_wall / args.p_ref,
    wetted_area_m2=float(area.sum()),
    n_wetted_tiles=int(area.size),
    peak_load_MW_m2=float(load[ipk] * 1e-6),
    peak_tile_id=int(ids[ipk]),
    peak_tile_area_m2=float(area[ipk]),
    peak_tile_markers=int(mload[ipk]),
    p99_load_MW_m2=float(np.percentile(load, 99) * 1e-6),
    mean_load_MW_m2=float(power_W.sum() / area.sum() * 1e-6),
)
with open(os.path.join(figdir, f"{args.label}_wallload_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
for k, v in summary.items():
    print(f"{k:32s} {v}")

# ---- VTK ---------------------------------------------------------------------
mesh = run.getwall_3dmesh()
vtkfn = os.path.join(vtkdir, f"{args.label}_wallload.vtk")
mesh.save(vtkfn)
print("saved", vtkfn)

# ---- tor-pol plot ------------------------------------------------------------
a5.input_init(bfield=True)
fig = plt.figure(figsize=(5.4, 4.8))   # the plot function forces a square axes box
ax = fig.add_subplot(111)
cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.1)
im = run.plotwall_torpol_new(qnt="eload", axes=ax, cmap=cc.cm.fire, cax=cax,
                             clim=list(args.clim))
ax.set_xlabel(r"$\phi$ [$\degree$]")
ax.set_ylabel(r"$\theta$ [$\degree$]")
mappable = im if im is not None else ax.collections[0]
fig.colorbar(mappable, cax=cax).set_label(r"$P_{surface}$ [W/m$^2$]")
ax.set_title(f"{args.label}: {P_wall:.2f} MW to wall, peak {load[ipk] * 1e-6:.2f} MW/m$^2$",
             fontsize=10)
pngfn = os.path.join(figdir, f"{args.label}_wall_torpol.png")
fig.savefig(pngfn, dpi=300, bbox_inches="tight")
print("saved", pngfn)

"""Four-panel figure: Poincare plots for both field methods (Biot-Savart and
mixed) followed by div(B) maps for the same two methods, all at phi=0.

Poincare panels use the cached traces in poincare-fig/out/ (same as
save_psirho.py). div(B) panels reuse compute_divb() from
convergence_studies/plot_divb.py. The engineering wall is mirrored about Z=0
on every panel so the divertor sits at the bottom.
"""

import sys
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_divb import FILE1 as DIVB_MIXED, FILE2 as DIVB_BIOT, compute_divb

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif'
})

poincare_dir = "poincare-fig/out/"
RLIM = (7.4, 10.3)
ZLIM = (-4.45, 3.8)
# div(B) window around the LCFS (as in figures/divb_compare_phi0.png); further
# clipped below to the region where both bfield grids are valid.
RLIM_DIVB = (7.5, 10.0)
ZLIM_DIVB = (-3.5, 3.5)
LCFS_STYLE = dict(color="black", lw=0.8, zorder=10)
OFFSET_STYLE = dict(color="tab:orange", lw=0.8, zorder=11)
WALL_STYLE = dict(color="darkgreen", lw=0.8, zorder=11)

fig = plt.figure(figsize=(190 / 25.4, 4.6), layout='constrained')
gs = fig.add_gridspec(1, 4)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3], sharex=ax3, sharey=ax3)
axes = [ax1, ax2, ax3, ax4]

# --- (a), (b): Poincare, Biot-Savart and mixed (nested) fields ---
walls = np.load(poincare_dir + "walls_phi0.npz")
for ax, tag in zip(axes[:2], ("biotsavart", "nested")):
    with h5py.File(poincare_dir + f"trace_{tag}_A256.h5", "r") as f:
        Rf, Zf = np.asarray(f["R_fwd"]), np.asarray(f["Z_fwd"])
        Rb, Zb = np.asarray(f["R_bwd"]), np.asarray(f["Z_bwd"])
        stride = int(f.attrs["nphi_sections"]) * int(f.attrs["NFP"])
    R = np.concatenate([Rf[0::stride][1:].ravel(), Rb[0::stride][1:].ravel()])
    Z = np.concatenate([Zf[0::stride][1:].ravel(), Zb[0::stride][1:].ravel()])
    ax.plot(R, Z, ls="", marker="o", ms=0.2, c="skyblue", rasterized=True)
    lcfs = walls["lcfs_CX"]
    ax.plot(lcfs[:, 0], lcfs[:, 1], label="LCFS", **LCFS_STYLE)
    for k, key in enumerate(("offset_30cm", "offset_10cm", "offset_20cm")):
        rz = walls[key]
        ax.plot(rz[:, 0], rz[:, 1],
                label="Conformal walls: 10, 20, 30 cm" if k == 0 else None,
                **OFFSET_STYLE)
    eng = walls["engineering"]
    ax.plot(eng[:, 0], -eng[:, 1], label="First wall with divertor cutout",
            **WALL_STYLE)

# --- (c), (d): div(B), Biot-Savart and mixed fields ---
divb_args = Namespace(phi=0.0, rmin=RLIM_DIVB[0], rmax=RLIM_DIVB[1],
                      zmin=ZLIM_DIVB[0], zmax=ZLIM_DIVB[1], nr=400, nz=400,
                      rhomax=None)
norm = matplotlib.colors.CenteredNorm(halfrange=10)
r_lo, r_hi, z_lo, z_hi = -np.inf, np.inf, -np.inf, np.inf
for ax, path in zip(axes[2:], (DIVB_BIOT, DIVB_MIXED)):
    r, z, divb, _, _ = compute_divb(path, divb_args, ax,
                                    rho_style=dict(LCFS_STYLE, lw=1.0))
    pcm = ax.pcolormesh(r.v, z.v, np.ma.masked_invalid(divb * 1e3).T, norm=norm,
                        cmap="seismic", shading="nearest", rasterized=True)
    r_lo, r_hi = max(r_lo, r.v[0]), min(r_hi, r.v[-1])
    z_lo, z_hi = max(z_lo, z.v[0]), min(z_hi, z.v[-1])

for ax, title in zip(axes, ("(a)", "(b)", "(c)", "(d)")):
    ax.set_aspect('equal')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('')
    ax.set_title(title)
for ax in axes[:2]:
    ax.grid(True, alpha=0.3)
ax1.set_xlim(RLIM)
ax1.set_ylim(ZLIM)
ax3.set_xlim(r_lo, r_hi)
ax3.set_ylim(z_lo, z_hi)
# Column widths follow each panel's data aspect so all four share one height.
w_ab = (RLIM[1] - RLIM[0]) / (ZLIM[1] - ZLIM[0])
w_cd = (r_hi - r_lo) / (z_hi - z_lo)
gs.set_width_ratios([w_ab, w_ab, w_cd, w_cd])
for ax in (ax2, ax4):
    ax.tick_params(labelleft=False)
ax1.set_ylabel('Z [m]')
ax3.set_ylabel('Z [m]', labelpad=0)

cbar = fig.colorbar(pcm, ax=axes[2:], shrink=0.8, ticks=[-10, -5, 0, 5, 10])
cbar.set_label(r'$\nabla\cdot B$ [mT/m]', labelpad=0)

h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=3)
plt.savefig("figs/poincare-divb.png", dpi=600)

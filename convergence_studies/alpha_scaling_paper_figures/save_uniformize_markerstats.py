"""Marker statistics for the uniformize-markers.py sampling change.

Plots the INITIAL MARKER distribution (unweighted -- where the simulation
spends its computational effort) and the INITIAL PARTICLE distribution
(weighted by marker weight, in particles/s -- the physical alpha birth
distribution) in rho, theta and phi, for the old AFSI sampling and the new
uniform sampling side by side.

The point of the pair: the marker row should become flat (that is the whole
purpose of uniformize-markers.py) while the particle row must stay put (the
physics must not move).

Unlike save_markerfigs.py this reads the marker INPUT rather than a run's
ini state, so it works before any simulation has been run.

Coordinates are ASCOT5's own, matching what save_markerfigs.py gets from
getstate():
  rho   -- from the bfield input
  theta -- geometric poloidal angle about the magnetic axis, i.e. the same
           atan2(z - axisz, r - axisr) that particle.c uses to fill
           particle_state.theta (the axis is evaluated per-marker, so it
           follows the axis excursion with phi)
  phi   -- the marker's own toroidal angle

Note uniformize-markers.py samples uniformly in *DESC* (rho, theta, zeta),
and DESC's theta is not ASCOT5's geometric theta, so the new theta row is
close to flat but not exactly flat. Pass --desc to additionally produce the
same figure in DESC coordinates, where uniformity is exact by construction.
"""
import sys
import os
import argparse

sys.path.insert(0, "/global/homes/m/mpatel26/alpha_analysis-mp")
import device_setup
device_setup.configure()

import numpy as np
import unyt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a5py import Ascot

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'font.family': 'sans-serif',
})

# salmon for markers, skyblue for particles -- same convention as
# save_markerfigs.py
C_MARKER = "salmon"
C_PARTICLE = "skyblue"
C_OLD = "0.35"

WIDTH_INCH = 190 / 25.4
NBINS = 20


def finite_only(rho, theta, phi, w, where=""):
    """Drop markers whose coordinates could not be evaluated.

    Markers sitting outside the bfield grid come back with rho = NaN (a
    fraction of a percent in some files), and a single NaN is enough to
    poison the histogram bin edges. They are excluded from the plots rather
    than being allowed to blank the figure.
    """
    good = (np.isfinite(rho) & np.isfinite(theta) & np.isfinite(phi)
            & np.isfinite(w))
    n_bad = int((~good).sum())
    if n_bad:
        print(f"[{where}] dropped {n_bad} of {good.size} markers "
              f"({100.0 * n_bad / good.size:.3f}%) with non-finite "
              "coordinates")
    return rho[good], theta[good], phi[good], w[good]


def marker_coords(a5, mrk):
    """(rho, theta[deg], phi[deg], weight) in ASCOT5's own coordinates."""
    r = np.asarray(mrk["r"]).astype(np.float64).ravel()
    phi = np.asarray(mrk["phi"]).astype(np.float64).ravel()
    z = np.asarray(mrk["z"]).astype(np.float64).ravel()
    w = np.asarray(mrk["weight"]).astype(np.float64).ravel()

    rho, axisr, axisz = a5.input_eval(
        r * unyt.m, phi * unyt.deg, z * unyt.m, 0 * unyt.s,
        "rho", "axisr", "axisz")
    rho = np.asarray(rho).astype(np.float64).ravel()
    axisr = np.asarray(axisr).astype(np.float64).ravel()
    axisz = np.asarray(axisz).astype(np.float64).ravel()

    theta = np.rad2deg(np.mod(np.arctan2(z - axisz, r - axisr), 2 * np.pi))
    return finite_only(rho, theta, np.mod(phi, 360.0), w, where="ASCOT5")


def desc_coords(eqf, mrk):
    """(rho, theta[deg], zeta[deg], weight) in DESC flux coordinates."""
    import desc.io as dscio
    from desc.equilibrium.coords import map_coordinates
    fam = dscio.load(eqf, file_format="hdf5")
    try:
        eq = fam[-1]
    except (TypeError, IndexError, KeyError):
        eq = fam
    r = np.asarray(mrk["r"]).astype(np.float64).ravel()
    phi = np.asarray(mrk["phi"]).astype(np.float64).ravel()
    z = np.asarray(mrk["z"]).astype(np.float64).ravel()
    w = np.asarray(mrk["weight"]).astype(np.float64).ravel()
    out = np.asarray(map_coordinates(
        eq, np.stack([r, np.deg2rad(phi), z], axis=1),
        inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf), tol=1e-8, maxiter=50),
        dtype=np.float64)
    return finite_only(out[:, 0],
                       np.rad2deg(np.mod(out[:, 1], 2 * np.pi)),
                       np.rad2deg(np.mod(out[:, 2], 2 * np.pi)),
                       w, where="DESC")


def panel(ax, xold, xnew, wold, wnew, edges, xlabel, ylabel, colour,
          zoom=False):
    """One histogram panel: old sampling as a grey outline, new as filled.

    zoom=True drops the zero baseline and fits the y-axis to the data. The
    angular particle distributions are only a ~10% modulation about their
    mean, so anchored at zero they read as flat even though the physical
    structure (elongation peaks near 90 and 270 deg) is really there.
    """
    delta = edges[1] - edges[0]
    ho, _ = np.histogram(xold, bins=edges, weights=wold)
    hn, _ = np.histogram(xnew, bins=edges, weights=wnew)
    ho, hn = ho / delta, hn / delta

    ax.stairs(hn, edges, fill=not zoom, color=colour, alpha=0.85,
              edgecolor="black", linewidth=0.8, label="uniform (new)")
    ax.stairs(ho, edges, fill=False, color=C_OLD, linewidth=1.2,
              linestyle="--", label="AFSI (old)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if zoom:
        lo = min(ho.min(), hn.min())
        hi = max(ho.max(), hn.max())
        pad = 0.12 * (hi - lo) if hi > lo else 0.1 * hi
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.4)


def make_figure(old, new, title, outfile, angle_label=("θ", "ϕ"),
                zoom_angles=False):
    """2 rows (markers / particles) x 3 cols (rho, theta, phi)."""
    (rho_o, th_o, ph_o, w_o) = old
    (rho_n, th_n, ph_n, w_n) = new

    rmax = max(rho_o.max(), rho_n.max())
    edges = {
        "rho": np.linspace(0.0, rmax, NBINS + 1),
        "theta": np.linspace(0.0, 360.0, NBINS + 1),
        "phi": np.linspace(0.0, 360.0, NBINS + 1),
    }
    thlab, phlab = angle_label

    fig = plt.figure(figsize=(WIDTH_INCH, WIDTH_INCH * 0.52),
                     constrained_layout=True)
    fig.suptitle(title, fontsize=10)
    subfigs = fig.subfigures(2, 1)

    subfigs[0].suptitle("initial MARKER distribution "
                        "(unweighted -- where the compute goes)", fontsize=9)
    a1, a2, a3 = subfigs[0].subplots(1, 3)
    subfigs[1].suptitle("initial PARTICLE distribution "
                        "(weighted -- the physics)", fontsize=9)
    a4, a5_, a6 = subfigs[1].subplots(1, 3)

    one_o = np.ones_like(w_o)
    one_n = np.ones_like(w_n)

    panel(a1, rho_o, rho_n, one_o, one_n, edges["rho"],
          r"$\rho_{ini}$", r"$M/d\rho$ [markers]", C_MARKER)
    panel(a2, th_o, th_n, one_o, one_n, edges["theta"],
          rf"${thlab}_{{ini}}$ [$^\circ$]",
          rf"$M/d{thlab}$ [markers/$^\circ$]", C_MARKER, zoom=zoom_angles)
    panel(a3, ph_o, ph_n, one_o, one_n, edges["phi"],
          rf"${phlab}_{{ini}}$ [$^\circ$]",
          rf"$M/d{phlab}$ [markers/$^\circ$]", C_MARKER, zoom=zoom_angles)

    panel(a4, rho_o, rho_n, w_o, w_n, edges["rho"],
          r"$\rho_{ini}$", r"$F/d\rho$ [1/s]", C_PARTICLE)
    panel(a5_, th_o, th_n, w_o, w_n, edges["theta"],
          rf"${thlab}_{{ini}}$ [$^\circ$]",
          rf"$F/d{thlab}$ [1/(s$\cdot^\circ$)]", C_PARTICLE, zoom=zoom_angles)
    panel(a6, ph_o, ph_n, w_o, w_n, edges["phi"],
          rf"${phlab}_{{ini}}$ [$^\circ$]",
          rf"$F/d{phlab}$ [1/(s$\cdot^\circ$)]", C_PARTICLE, zoom=zoom_angles)

    a1.legend(loc="upper left", framealpha=0.9)
    fig.savefig(outfile, dpi=600)
    print(f"[fig] wrote {outfile}")
    plt.close(fig)


def summarise(name, old, new):
    """Numbers behind the figure, so the plot can be checked not just eyed."""
    (rho_o, th_o, ph_o, w_o) = old
    (rho_n, th_n, ph_n, w_n) = new
    print(f"\n=== {name}")
    print(f"  markers      old {rho_o.size}   new {rho_n.size}")
    print(f"  total weight old {w_o.sum():.6e}   new {w_n.sum():.6e}   "
          f"ratio {w_n.sum()/w_o.sum():.10f}")
    print(f"  zero weight  old {int((w_o <= 0).sum())}   "
          f"new {int((w_n <= 0).sum())}")
    print("  marker-count flatness (rel. std across 20 bins, "
          "0 = perfectly uniform):")
    for lab, xo, xn, rng in (("rho", rho_o, rho_n, (0, max(rho_o.max(), rho_n.max()))),
                             ("theta", th_o, th_n, (0, 360)),
                             ("phi", ph_o, ph_n, (0, 360))):
        e = np.linspace(rng[0], rng[1], NBINS + 1)
        co, _ = np.histogram(xo, bins=e)
        cn, _ = np.histogram(xn, bins=e)
        print(f"    {lab:<6} old {co.std()/co.mean():.4f}   "
              f"new {cn.std()/cn.mean():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default="/pscratch/sd/m/mpatel26/ascot_h5s/"
                    "convergence_studies/ISHW/G1600-free-reopt_I_eng-wall_"
                    "100k-mrk_biot-bfield_p01cellarea_GC-mode_zero-profile_"
                    "09012026.h5")
    ap.add_argument("--old-qid", default="3903961626",
                    help="the AFSI (TAG) marker group")
    ap.add_argument("--new-qid", default=None,
                    help="the uniformized group (default: the active one)")
    ap.add_argument("--equil", default="/pscratch/sd/m/mpatel26/equil/"
                    "equil_Helios_G1600-12-89_QA2e-1_Bxdl25_free_L12_M12_"
                    "N20.h5")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figs"))
    ap.add_argument("--tag", default="100kGC")
    ap.add_argument("--zoom", action="store_true",
                    help="fit the angular panels' y-axis to the data instead "
                         "of anchoring at zero, so the ~10%% modulation is "
                         "visible")
    ap.add_argument("--desc", action="store_true",
                    help="also make the figure in DESC flux coordinates")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    a5 = Ascot(args.h5)
    new_grp = (a5.data.marker.active if args.new_qid is None
               else getattr(a5.data.marker, "q" + args.new_qid))
    old_grp = getattr(a5.data.marker, "q" + args.old_qid)
    print(f"[read] old = {old_grp.get_qid()} ({old_grp.get_desc()})")
    print(f"[read] new = {new_grp.get_qid()} ({new_grp.get_desc()})")
    mold, mnew = old_grp.read(), new_grp.read()

    a5.input_init(bfield=True)
    old_a = marker_coords(a5, mold)
    new_a = marker_coords(a5, mnew)
    a5.input_free()

    summarise("ASCOT5 coordinates", old_a, new_a)
    make_figure(old_a, new_a,
                "Marker vs particle sampling, ASCOT5 coordinates",
                os.path.join(
                    args.outdir,
                    f"uniformize_markerstats_{args.tag}_ascot"
                    f"{'_zoom' if args.zoom else ''}.png"),
                zoom_angles=args.zoom)

    if args.desc:
        old_d = desc_coords(args.equil, mold)
        new_d = desc_coords(args.equil, mnew)
        summarise("DESC coordinates", old_d, new_d)
        make_figure(old_d, new_d,
                    "Marker vs particle sampling, DESC flux coordinates",
                    os.path.join(
                        args.outdir,
                        f"uniformize_markerstats_{args.tag}_desc"
                        f"{'_zoom' if args.zoom else ''}.png"),
                    angle_label=("θ", "ζ"), zoom_angles=args.zoom)
    return 0


if __name__ == "__main__":
    sys.exit(main())

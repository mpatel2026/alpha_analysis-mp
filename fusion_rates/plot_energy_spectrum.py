#!/usr/bin/env python
"""Standalone 1D energy spectrum (F/dE) from a triton_markers.h5 file.

Written to answer a specific question precisely: which energy feature has
the higher PEAK dRate/dE, not just the higher TOTAL rate. Those are
different things -- a channel with more total rate but a WIDER energy
spread (e.g. beam-target, whose CM boost smears the triton energy over a
broad continuum) can have a LOWER peak density than a channel with less
total rate concentrated in a NARROW range (e.g. thermal/beam-beam, both
close to the ~1.01 MeV CM birth energy). Peak height in a density plot
depends on how concentrated the rate is, not on how much of it there is --
same distinction as the rho panel's "why does the peak exceed the total"
question in dd_fusion_rates.py.

The combined 2D (energy, pitch) panel in triton_marker_distributions.png
can't answer this cleanly by eye: a broad-but-tall-in-total feature gets
its color diluted twice over (across both energy AND pitch, if pitch is
also spread out), on top of whatever colormap/zoom choices were made. A
plain 1D projection over all pitch removes that ambiguity.
"""
import argparse
import h5py
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same house style as dd_fusion_rates.py, for a consistent look.
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
})


def load_markers(fn):
    """Read energy [eV] and weight [particles/s] from a triton_markers.h5.

    Parameters
    ----------
    fn : str
        Path to a HDF5 file written by dd_fusion_rates.write_markers_h5.

    Returns
    -------
    ekin_eV, weight : numpy.ndarray
    """
    with h5py.File(fn, "r") as f:
        ekin_units = f["energy"].attrs.get("units", "eV")
        energy = f["energy"][:]
        weight = f["weight"][:]
    if ekin_units != "eV":
        raise ValueError(
            f"Expected marker energy in eV, got units={ekin_units!r} -- "
            "add a conversion here before trusting the MeV axis below.")
    return energy, weight


def plot_energy_spectrum(ekin_eV, weight, fn, nbins=100):
    """Plot F/dE (weighted energy density) and print the tallest bins.

    Parameters
    ----------
    ekin_eV : array_like
        Marker energies [eV].
    weight : array_like
        Marker weights [particles/s].
    fn : str
        Output figure path.
    nbins : int, optional
        Number of histogram bins spanning the markers' own energy range.

    Returns
    -------
    centers_mev, density : numpy.ndarray
        Bin centers [MeV] and F/dE [particles/(s*MeV)], for anyone who
        wants to inspect the numbers directly instead of just the print.
    """
    ekin_eV = np.asarray(ekin_eV)
    weight = np.asarray(weight)
    ek_mev = ekin_eV / 1e6
    lo, hi = ek_mev.min(), ek_mev.max()
    edges = np.linspace(lo, hi, nbins + 1)
    delta = (hi - lo) / nbins
    H, _ = np.histogram(ek_mev, bins=edges, weights=weight)
    density = H / delta  # F/dE, not a raw per-bin sum -- see module docstring
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.bar(centers, density, width=delta, color="skyblue", edgecolor="black")
    ax.set_xlabel("E [MeV]")
    ax.set_ylabel("F/dE [particles / (MeV * s)]")
    ax.grid(axis="y", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fn)
    plt.close(fig)

    order = np.argsort(density)[::-1]
    print("Tallest F/dE bins (highest 10, for comparing peak heights "
          "directly -- not the same as comparing total rate per feature):")
    for i in order[:10]:
        print(f"  E={centers[i]:.3f} MeV: F/dE={density[i]:.4e} "
              f"particles/(s*MeV)")

    return centers, density


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("markers_h5",
                    help="Path to a triton_markers.h5 written by "
                         "dd_fusion_rates.py.")
    p.add_argument("--nbins", type=int, default=100,
                    help="Number of energy bins.")
    p.add_argument("--out", default=None,
                    help="Output figure path (default: "
                         "figs/energy_spectra/<markers basename>"
                         "_energy_spectrum.png next to this script).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.out:
        out = args.out
    else:
        # <fusion_rates>/figs/energy_spectra/<markers basename>_energy_spectrum.png
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "figs", "energy_spectra")
        os.makedirs(figdir, exist_ok=True)
        base = os.path.basename(args.markers_h5).rsplit(".h5", 1)[0]
        out = os.path.join(figdir, base + "_energy_spectrum.png")
    ekin, weight = load_markers(args.markers_h5)
    print(f"Loaded {ekin.size} markers from {args.markers_h5}")
    plot_energy_spectrum(ekin, weight, out, nbins=args.nbins)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot lost-marker histograms from saved .npy files.

Reads:
  - lost_marker_hist.npy             : object array of 5 histograms (lost markers)
  - lost_marker_hist_normalized.npy  : object array of 5 histograms (lost / total)
  - lost_marker_edges.npz            : dict {property_name: edges_array}

Histogram ordering in the .npy arrays: phi, rho, theta, ekin, pitch.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
})

# Order of histograms inside each .npy file (matches save_lost_marker_dist.py).
HIST_NAMES = ["phi", "rho", "theta", "ekin", "pitch"]

# Display labels for axes.
LABELS = {
    "phi": r"$\phi_{\mathrm{ini}}$ [deg]",
    "rho": r"$\rho_{\mathrm{ini}}$",
    "theta": r"$\theta_{\mathrm{ini}}$ [deg]",
    "ekin": r"$E_{\mathrm{kin,ini}}$ [eV]",
    "pitch": r"$\lambda_{\mathrm{ini}}$",
}


def load_hist(path):
    """Load an object array of histograms saved with np.save."""
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object and arr.ndim == 0:
        arr = arr.item()
    return [np.asarray(h, dtype=float).ravel() for h in arr]


def to_numeric(arr):
    """Convert unyt arrays (if present) to a plain 1D numpy array."""
    if hasattr(arr, "value"):
        arr = arr.value
    return np.asarray(arr, dtype=float).ravel()


def load_edges(path):
    """Load edge arrays from an npz archive."""
    edges_npz = np.load(path, allow_pickle=True)
    return {name: to_numeric(edges_npz[name]) for name in edges_npz.files}


def plot_hist(ax, counts, edges, xlabel, ylabel=None, label=None):
    """Plot a pre-binned histogram using bin edges."""
    counts = np.asarray(counts, dtype=float).ravel()
    edges = to_numeric(edges)

    if len(counts) != len(edges) - 1:
        raise ValueError(
            f"Histogram length {len(counts)} does not match "
            f"{len(edges) - 1} bins from {len(edges)} edges."
        )

    ax.stairs(counts, edges, fill=True, label=label, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_xlim(edges[0], edges[-1])
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.75)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hist", default="lost_marker_hist.npy")
    parser.add_argument("--hist-norm", default="lost_marker_hist_normalized.npy")
    parser.add_argument("--edges", default="lost_marker_edges.npz")
    parser.add_argument("--out", default="lost_marker_hist.png",
                        help="Output image path (empty string to skip saving)")
    parser.add_argument("--show", action="store_true", help="Display the figure")
    args = parser.parse_args()

    hist_raw = load_hist(args.hist)
    hist_norm = load_hist(args.hist_norm)
    edges = load_edges(args.edges)

    if len(hist_raw) != len(HIST_NAMES) or len(hist_norm) != len(HIST_NAMES):
        raise ValueError(
            f"Expected {len(HIST_NAMES)} histograms, got "
            f"{len(hist_raw)} (raw) and {len(hist_norm)} (normalized)."
        )

    missing = [name for name in HIST_NAMES if name not in edges]
    if missing:
        raise KeyError(
            f"Missing edge keys {missing}. Available keys: {list(edges)}"
        )

    ncols = len(HIST_NAMES)
    fig, axes = plt.subplots(2, ncols, figsize=(2.2 * ncols, 4.5), constrained_layout=True)

    rows = [
        ("Lost-marker fraction", hist_raw),
        ("Lost / total", hist_norm),
    ]

    for row_idx, (row_title, hist) in enumerate(rows):
        for col_idx, name in enumerate(HIST_NAMES):
            ax = axes[row_idx, col_idx]
            plot_hist(
                ax,
                hist[col_idx],
                edges[name],
                xlabel=LABELS[name],
                ylabel=row_title if col_idx == 0 else None,
            )
            if row_idx == 0:
                ax.set_title(name)

    if args.out:
        out_path = Path(args.out)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path.resolve()}")

    if args.show or not args.out:
        plt.show()


if __name__ == "__main__":
    main()

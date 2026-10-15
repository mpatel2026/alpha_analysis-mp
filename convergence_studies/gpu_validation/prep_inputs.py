#!/usr/bin/env python3
"""Make a small ASCOT5 input for the GPU validation runs.

Copies only the input groups (no results) of an existing simulation file,
then activates a random subset of the markers and options with a shorter
simulation time so that a run fits in the debug QOS. The distribution time
axis is reduced to one bin to keep the histograms small.

Run in the ascot-develop-mp environment, e.g.

    conda activate ascot-develop-mp
    export HDF5_USE_FILE_LOCKING=FALSE
    python prep_inputs.py SRC.h5 $PSCRATCH/gpu_validation/gc20k.h5 \
        --nmrk 20000 --tmax 0.05
"""
import argparse
import os

import h5py
import numpy as np
from a5py import Ascot

INPUT_GROUPS = ["bfield", "efield", "plasma", "neutral", "wall", "boozer",
                "mhd", "asigma", "options", "marker", "nbi"]


def copy_inputs(src, dst):
    """Copy input groups (with their 'active' attributes) to a new file."""
    with h5py.File(src, "r") as fs, h5py.File(dst, "w") as fd:
        for key, val in fs.attrs.items():
            fd.attrs[key] = val
        for grp in INPUT_GROUPS:
            if grp in fs:
                fs.copy(fs[grp], fd, name=grp)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="existing ASCOT5 HDF5 file")
    parser.add_argument("dst", help="validation input to create")
    parser.add_argument("--nmrk", type=int, default=20000,
                        help="number of markers to keep (default 20000)")
    parser.add_argument("--tmax", type=float, default=0.05,
                        help="simulation time limit [s] (default 0.05)")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for choosing the markers")
    parser.add_argument("--keep-dist-bins", action="store_true",
                        help="keep the original DIST_NBIN_TIME")
    parser.add_argument("--enable-dist", action="store_true",
                        help="turn on dist5D and rho5D (one time bin), with "
                             "R, z and momentum ranges that cover the markers")
    args = parser.parse_args()

    if os.path.exists(args.dst):
        raise SystemExit(f"{args.dst} exists, refusing to overwrite")
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    copy_inputs(args.src, args.dst)

    a5 = Ascot(args.dst)

    # Marker subset
    mrkgrp = a5.data.marker.active
    mrktype = mrkgrp.get_type()
    mrk = mrkgrp.read()
    n = int(mrk["n"])
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n, size=min(args.nmrk, n), replace=False))
    sub = {}
    for key, val in mrk.items():
        if isinstance(val, np.ndarray) and val.shape[:1] == (n,):
            sub[key] = val[idx]
        else:
            sub[key] = val
    sub["n"] = idx.size
    a5.data.create_input(mrktype, **sub, activate=True,
                         desc=f"GPUVAL{idx.size}")

    # Shorter simulation time, smaller histograms
    opt = a5.data.options.active.read()
    opt["ENDCOND_SIMTIMELIM"] = 1
    opt["ENDCOND_LIM_SIMTIME"] = args.tmax
    opt["ENDCOND_MAX_MILEAGE"] = args.tmax
    if args.enable_dist:
        # Bin ranges from the markers so that the histograms are not empty:
        # R and z with a margin for orbit widths, momenta up to the maximum
        # birth momentum (energy in eV, mass in amu as stored in the input).
        amu, qe = 1.66053906660e-27, 1.602176634e-19
        pmax = 1.1 * np.max(np.sqrt(2 * sub["mass"] * amu
                                    * sub["energy"] * qe))
        r, z = sub["r"], sub["z"]
        opt.update({
            "ENABLE_DIST_5D": 1, "ENABLE_DIST_RHO5D": 1,
            "DIST_MIN_R": float(np.min(r)) - 0.5,
            "DIST_MAX_R": float(np.max(r)) + 0.5, "DIST_NBIN_R": 25,
            "DIST_MIN_Z": float(np.min(z)) - 1.0,
            "DIST_MAX_Z": float(np.max(z)) + 1.0, "DIST_NBIN_Z": 25,
            "DIST_MIN_RHO": 0, "DIST_MAX_RHO": 1.0, "DIST_NBIN_RHO": 25,
            "DIST_MIN_THETA": 0, "DIST_MAX_THETA": 360, "DIST_NBIN_THETA": 50,
            "DIST_MIN_PHI": 0, "DIST_MAX_PHI": 360, "DIST_NBIN_PHI": 26,
            "DIST_MIN_PPA": -pmax, "DIST_MAX_PPA": pmax, "DIST_NBIN_PPA": 30,
            "DIST_MIN_PPE": 0, "DIST_MAX_PPE": pmax, "DIST_NBIN_PPE": 30,
            "DIST_MIN_CHARGE": 0, "DIST_MAX_CHARGE": 3, "DIST_NBIN_CHARGE": 1,
        })
    if not args.keep_dist_bins:
        opt["DIST_NBIN_TIME"] = 1
        opt["DIST_MIN_TIME"] = 0.0
        opt["DIST_MAX_TIME"] = max(args.tmax * 1.01, 1e-6)
    a5.data.create_input("opt", **opt, activate=True, desc="GPUVAL")

    print(f"Wrote {args.dst}: {idx.size} {mrktype} markers of {n}, "
          f"tmax {args.tmax} s, SIM_MODE {int(opt['SIM_MODE'])}, "
          f"ENABLE_ADAPTIVE {int(opt['ENABLE_ADAPTIVE'])}, "
          f"dist5D {int(opt.get('ENABLE_DIST_5D', 0))}, "
          f"rho5D {int(opt.get('ENABLE_DIST_RHO5D', 0))}")


if __name__ == "__main__":
    main()

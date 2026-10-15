#!/usr/bin/env python
"""Initial marker power for an ASCOT5 run: Sum(weight * ekin) over inistate.

Marker weight in ASCOT5 is a particle FLUX [particles/s], not a count
(doc/physics.rst: "The weight is ... a particle flux and has units
particles/s"). So Sum(weight * ekin) has units of energy/time = power [W]
-- the same convention that makes wall loads come out in Watts rather than
Joules ("wall loads are not in units of Joule but in units of Watts").

This is the injected power represented by the marker population at launch
(t=0), before any slowing-down/losses -- i.e. the source power, not the
power actually deposited in the plasma.
"""
import argparse

import h5py
import numpy as np

from a5py import Ascot


def resolve_run_name(filepath, run_arg):
    """Resolve a run group name ("run_<qid>") in the results group."""
    with h5py.File(filepath, "r") as f:
        if run_arg is None:
            qid = f["results"].attrs["active"]
            if isinstance(qid, bytes):
                qid = qid.decode("utf-8")
            name = qid if qid.startswith("run_") else "run_" + qid
        else:
            name = run_arg if run_arg.startswith("run_") else "run_" + run_arg
        if name not in f["results"]:
            raise KeyError(
                f"{name} not found in {filepath}:/results "
                f"(available: {list(f['results'].keys())})")
        return name


def initial_marker_power(a5, run_name):
    """Sum(weight * ekin) over the inistate -> total initial power.

    Parameters
    ----------
    a5 : Ascot
        Open Ascot instance.
    run_name : str
        Run group name, e.g. "run_1669360920".

    Returns
    -------
    power : unyt_quantity
        Total initial power [W].
    n : int
        Number of markers.
    """
    run = getattr(a5.data, run_name)
    weight = run.getstate("weight", state="ini")
    ekin = run.getstate("ekin", state="ini")
    power = np.sum(weight * ekin).to("W")
    return power, weight.size


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to the ASCOT5 HDF5 file.")
    p.add_argument("--run", default=None,
                    help="Run name or bare qid (default: file's active run).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    run_name = resolve_run_name(args.input, args.run)
    a5 = Ascot(args.input, create=False)
    power, n = initial_marker_power(a5, run_name)

    print(f"Input file: {args.input}")
    print(f"Run: {run_name}")
    print(f"Markers: {n}")
    print(f"Initial marker power: {power:.6e} ({power.to('MW'):.4f})")
    return power


if __name__ == "__main__":
    main()

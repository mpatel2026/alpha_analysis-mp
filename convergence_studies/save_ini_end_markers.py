"""Save ini/end marker positions from an ASCOT5 file.

Usage: python save_ini_end_markers.py /path/to/ascot.h5 [out_dir] [--run QID]
out_dir defaults to the home directory. --run selects a run by QID
(e.g. 0544571068 or run_0544571068); defaults to the active run.
"""
import os
import argparse
from a5py import Ascot
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("inp_file")
parser.add_argument("out_dir", nargs="?", default=os.path.expanduser("~"))
parser.add_argument("--run", default=None)
args = parser.parse_args()
inp_file, out_dir = args.inp_file, args.out_dir
fname = os.path.basename(inp_file)

# "gc" returns guiding-center quantities, "prt" particle quantities.
# Use "prt" if the run was a full gyro-orbit simulation.
mode = "gc"

# ascot quantity -> (name in output file, description)
qnts = {
    "ids":      ("ids",   "Marker ID"),
    "r":        ("R",     "Cylindrical major radius"),
    "z":        ("Z",     "Cylindrical height"),
    "phimod":   ("phi",   "Toroidal angle, wrapped to [0, 360) deg"),
    "rho":      ("rho",   "Normalized minor radius sqrt(psi_n)"),
    "thetamod": ("theta", "Poloidal angle, wrapped to [0, 360) deg"),
}

ascot = Ascot(inp_file)
if args.run is None:
    run = ascot.data.active
else:
    run = ascot.data["q" + args.run.replace("run_", "")]
run_name = "run_" + run.get_qid()

outname = os.path.join(out_dir, fname.replace(".h5", f"_{run_name}_ini-end_markers_{mode}.h5"))
with h5py.File(outname, "w") as f:
    f.attrs["source_file"] = os.path.abspath(inp_file)
    f.attrs["run"] = run_name
    f.attrs["mode"] = mode
    f.attrs["note"] = ("phi is shared between the (R, phi, Z) and "
                       "(rho, theta, phi) coordinate systems; all arrays are "
                       "sorted by marker ID, so index i is the same marker")
    for state in ["ini", "end"]:
        data = run.getstate(*qnts, mode=mode, state=state)
        for qnt, val in zip(qnts, data):
            name, desc = qnts[qnt]
            name = f"{name}_{state}"
            dset = f.create_dataset(name, data=np.asarray(val))
            dset.attrs["units"] = str(getattr(val, "units", "dimensionless"))
            dset.attrs["description"] = f"{desc} at {state}state"
            dset.attrs["ascot_qnt"] = qnt
        print(f"Saved {len(f['ids_' + state])} {state}state markers")

print(f"Wrote {outname}")

import device_setup
device_setup.configure()

import argparse
import json
import os

import numpy as np
from a5py import Ascot

parser = argparse.ArgumentParser(description="Replace the marker input of an existing ASCOT5 file with markers from a JSON.")
parser.add_argument("--ascot_dir", type=str, required=True, help="directory containing the ASCOT5 file")
parser.add_argument("--ascot_name", type=str, required=True, help="ASCOT5 file name (with or without .h5)")
parser.add_argument("--marker_file", type=str, required=True, help="marker JSON path")
args = parser.parse_args()

simname = args.ascot_name[:-3] if args.ascot_name.endswith(".h5") else args.ascot_name
ascot_path = os.path.join(args.ascot_dir, simname + ".h5")
if not os.path.isfile(ascot_path):
    raise SystemExit(f"ASCOT5 file not found: {ascot_path}")

with open(args.marker_file, "r") as f:
    marker_file = json.load(f)
# GC JSONs carry energy/pitch/zeta; full-orbit (prt) JSONs carry vr/vphi/vz.
mode = "prt" if "vr" in marker_file else "gc"
if mode == "prt":
    keys = ["id", "weight", "phi", "r", "z", "vr", "vphi", "vz", "mass",
            "charge", "time", "anum", "znum"]
else:
    keys = ["id", "weight", "energy", "pitch", "zeta", "phi", "r", "z", "mass",
            "charge", "time", "anum", "znum"]
marker = {k: np.array(marker_file[k]) for k in keys}
marker["ids"] = marker.pop("id")
marker["n"] = marker_file["n"][0]
for k in ("charge", "anum", "znum"):        # ASCOT stores these as i4
    marker[k] = marker[k].astype(int)

print("marker mode = ", mode)
print("nmrk = ", marker["n"])
print("ids.size = ", marker["ids"].size)

simulation = Ascot(ascot_path)
simulation.data.create_input(mode, **marker, activate=True)

marker = simulation.data.marker.active.read()


print(f"set up {marker['n']} markers for {simname}")
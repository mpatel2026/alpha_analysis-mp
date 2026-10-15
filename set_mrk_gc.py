import device_setup
device_setup.configure()

import h5py
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from a5py import Ascot
from scipy.optimize import curve_fit
import matplotlib.cm as cm

import desc.io as dscio
import desc.grid as dscg

import argparse
import json

# Step 2: Set up the argument parser
parser = argparse.ArgumentParser(description="Run ASCOT simulation for a specific sim name.")
parser.add_argument('--params', type=str, required=True, help='param file path')                           
parser.add_argument("--marker_file", type=str, required=True)   
# Step 3: Parse the arguments
args = parser.parse_args()
with open(args.params, "r") as f:
    config = json.load(f)

# Extract required parameters with default fallbacks
inp_dir = config.get("inp_dir", None)
out_dir = config.get("out_dir", None)
equil_name = config.get("equil_file")
simname = config["sim_name"]  # Required

with open(args.marker_file, "r") as f:
    marker_file = json.load(f)
marker = {}
marker["n"] = marker_file["n"][0]
marker["ids"] = np.array(marker_file["id"])
marker["weight"] = np.array(marker_file["weight"])
marker["energy"] = np.array(marker_file["energy"])
marker["pitch"] = np.array(marker_file["pitch"])
marker["zeta"] = np.array(marker_file["zeta"])
marker["phi"] = np.array(marker_file["phi"])
marker["r"] = np.array(marker_file["r"])
marker["z"] = np.array(marker_file["z"])
marker["mass"] = np.array(marker_file["mass"])
marker["charge"] = np.array(marker_file["charge"])
marker["time"] = np.array(marker_file["time"])
marker["anum"] = np.array(marker_file["anum"])
marker["znum"] = np.array(marker_file["znum"])
marker["time"] = np.array(marker_file["time"])

print("nmrk = ", marker["n"])
print("ids.size = ", marker["ids"].size)

simulation = Ascot(f'{out_dir}{simname}.h5')
simulation.data.create_input('gc', **marker, activate=True)

marker = simulation.data.marker.active.read()


print(f"set up {marker['n']} markers for {simname}")
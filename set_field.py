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

# 1. Set up the Argument Parser to only take the JSON parameters file
parser = argparse.ArgumentParser(description="Create input for Alpha Analysis simulations from a JSON config.")
parser.add_argument("--params", type=str, required=True, help="Path to the JSON configuration file")
args = parser.parse_args()

# 2. Load and parse the JSON file
with open(args.params, "r") as f:
    config = json.load(f)

# Extract required parameters with default fallbacks
inp_dir = config.get("inp_dir", None)
out_dir = config.get("out_dir", None)
equil_name = config.get("equil_file")
encircling_name = config.get("EC_file", None)
shaping_name = config.get("SC_file", None)
simname = config["sim_name"]  # Required

nmrk = config.get("nmrk")
wall_offset = config.get("wall_offset", None)
cell_area = config.get("cell_area", None)
rescale_FPP = config.get("FPP_power", None)
use_mixed_field = config.get("use_mixed_field", False)
collect_dist = config.get("collect_dist", False)
sol_profile = config.get("SOL_profile", None)
simmode = config.get("simmode", "gc")
wall_file = config.get("wall_file", None)

fn = inp_dir + "equil_G1600_DESC_fixed.h5"
fam = dscio.load(fn, file_format="hdf5")
try:  # if file is an EquilibriaFamily, use final Equilibrium
    G1600 = fam[-1]
except:  # file is already an Equilibrium
    G1600 = fam

fn = inp_dir + equil_name
fam = dscio.load(fn, file_format="hdf5")
try:  # if file is an EquilibriaFamily, use final Equilibrium
    eq = fam[-1]
except:  # file is already an Equilibrium
    eq = fam

grid = dscg.ConcentricGrid(L=G1600.L_grid, M=G1600.M_grid, N=G1600.N_grid,NFP=G1600.NFP, node_pattern="linear")
data_G1600 = G1600.compute(['a','|B|'], grid = grid)
minor_radius_G1600 = data_G1600['a']
print(f"minor radius G1600: {minor_radius_G1600}")
B_on_axis_G1600 = np.mean(data_G1600['|B|'])
print(f"B on axis G1600: {B_on_axis_G1600}")
G1600_aB = minor_radius_G1600 * B_on_axis_G1600

grid = dscg.ConcentricGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid,NFP=eq.NFP, node_pattern="linear")
data_eq = eq.compute(['a','|B|'], grid = grid)
minor_radius_eq = data_eq['a']
print(f"minor radius eq: {minor_radius_eq}")
B_on_axis_eq = np.mean(data_eq['|B|'])
print(f"B on axis eq: {B_on_axis_eq}")
eq_aB = minor_radius_eq * B_on_axis_eq

aB_rescale = G1600_aB / eq_aB 
B_rescale = aB_rescale # since we don't want to change a, all of the rescale must go into B

simulation = Ascot(f'{out_dir}{simname}.h5')
bfield = simulation.data.bfield.active.read()
br = bfield['br']
bz = bfield['bz']
bphi = bfield['bphi']

bfield['br'] = B_rescale * br
bfield['bz'] = B_rescale * bz
bfield['bphi'] = B_rescale * bphi

simulation.data.create_input('B_STS', **bfield, activate=True)

print(f"set up bfield for {simname} to have Helios equivalent p* at 3.5 MeV by scaling up B by {B_rescale:.3f}")
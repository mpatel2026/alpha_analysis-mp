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

# Step 2: Set up the argument parser
parser = argparse.ArgumentParser(description="Run ASCOT simulation for a specific sim name.")
parser.add_argument('--sim_name', type=str, required=True, help='The name of the simulation to process')
parser.add_argument("--inp_dir", type=str, default='/pscratch/sd/m/mpatel26/equil/')
parser.add_argument("--out_dir", type=str, default='/pscratch/sd/m/mpatel26/ascot_h5s/')
parser.add_argument("--equil_name", type=str, default="equil_Eos10_G3213_DESC_R325_B50_P12MWbroader_constJe_1p5Wself_free.h5")
# Step 3: Parse the arguments
args = parser.parse_args()
simname = args.sim_name
inp_dir = args.inp_dir
out_dir = args.out_dir
equil_name = args.equil_name

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
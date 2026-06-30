import numpy as np
from a5py import Ascot
import matplotlib.pyplot as plt
import unyt
import matplotlib.colors as colors

plt.rcParams.update({
    'font.size': 8,          # General font size
    'axes.labelsize': 8,     # x and y labels
    'axes.titlesize': 8,     # Title size
    'xtick.labelsize': 8,    # x-axis tick labels
    'ytick.labelsize': 8,    # y-axis tick labels
    'legend.fontsize': 8,     # Legend size
    'font.family': 'sans-serif' # A clean, professional serif font
})

height_inch = 75 / 25.4
width_inch = height_inch * 1.3

input_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"

a_offsetpsi_decay = Ascot(input_dir+ "G1600-engineeringwall-withtargets-1000kmrk-gc-biot-decay-04292026.h5")
a_offsetpsi_flat = Ascot(input_dir+ "G1600-free-reopt_I_engineering-wall_1000k-mrk_biot-bfield_p01cellarea__FLR-mode_flat-profile_05052026.h5")
a_offsetpsi_zero = Ascot(input_dir + "G1600-free-reopt_I_engineering-wall_1000k-mrk_biot-bfield_p01cellarea__FLR-mode_zero-profile_05052026.h5")

bfield_dict_offsetpsi_zero = a_offsetpsi_zero.data.bfield.active.read()
rmax_offsetpsi_zero = bfield_dict_offsetpsi_zero['b_rmax']
rmin_offsetpsi_zero = bfield_dict_offsetpsi_zero['b_rmin']
zmax_offsetpsi_zero = bfield_dict_offsetpsi_zero['b_zmax']
zmin_offsetpsi_zero = bfield_dict_offsetpsi_zero['b_zmin']
psi_offsetpsi_data_zero = bfield_dict_offsetpsi_zero['psi']
psi_sep_zero = bfield_dict_offsetpsi_zero['psi1']

bfield_dict_offsetpsi_decay = a_offsetpsi_decay.data.bfield.active.read()
rmax_offsetpsi_decay = bfield_dict_offsetpsi_decay['b_rmax']
rmin_offsetpsi_decay = bfield_dict_offsetpsi_decay['b_rmin']
zmax_offsetpsi_decay = bfield_dict_offsetpsi_decay['b_zmax']
zmin_offsetpsi_decay = bfield_dict_offsetpsi_decay['b_zmin']
psi_offsetpsi_data_decay = bfield_dict_offsetpsi_decay['psi']
psi_sep_decay = bfield_dict_offsetpsi_decay['psi1']

bfield_dict_offsetpsi_flat = a_offsetpsi_flat.data.bfield.active.read()
rmax_offsetpsi_flat = bfield_dict_offsetpsi_flat['b_rmax']
rmin_offsetpsi_flat = bfield_dict_offsetpsi_flat['b_rmin']
zmax_offsetpsi_flat = bfield_dict_offsetpsi_flat['b_zmax']
zmin_offsetpsi_flat = bfield_dict_offsetpsi_flat['b_zmin']
psi_offsetpsi_data_flat = bfield_dict_offsetpsi_flat['psi']
psi_sep_flat = bfield_dict_offsetpsi_flat['psi1']
max_rho_decay = np.sqrt(np.max(psi_offsetpsi_data_decay) / psi_sep_decay)

profiles_decay = a_offsetpsi_decay.data.plasma.active.read()
profiles_flat = a_offsetpsi_flat.data.plasma.active.read()
profiles_zero = a_offsetpsi_zero.data.plasma.active.read()


rho_decay = profiles_decay['rho']
edensity_decay = profiles_decay['edensity']
etemperature_decay = profiles_decay['etemperature'] / 1000
itemperature_decay = profiles_decay['itemperature'] / 1000


rho_flat = profiles_flat['rho']
edensity_flat = profiles_flat['edensity']
etemperature_flat = profiles_flat['etemperature'] / 1000
itemperature_flat = profiles_flat['itemperature'] / 1000
rho_zero = profiles_zero['rho']
edensity_zero = profiles_zero['edensity']
etemperature_zero = np.clip(profiles_zero['etemperature'] / 1000, 1e-3, None)
itemperature_zero = np.clip(profiles_zero['itemperature'] / 1000, 1e-3, None)
# Create masks
mask_decay = rho_decay < 1.3
mask_flat = rho_flat < 1.3
mask_zero = rho_zero < 1.3

# Apply mask to decay data
rho_decay = rho_decay[mask_decay]
edensity_decay = edensity_decay[mask_decay]
etemperature_decay = etemperature_decay[mask_decay]
itemperature_decay = itemperature_decay[mask_decay]

# Apply mask to flat data
rho_flat = rho_flat[mask_flat]
edensity_flat = edensity_flat[mask_flat]
etemperature_flat = etemperature_flat[mask_flat]
itemperature_flat = itemperature_flat[mask_flat]

rho_zero = rho_zero[mask_zero]
edensity_zero = edensity_zero[mask_zero]
etemperature_zero = etemperature_zero[mask_zero]
itemperature_zero = itemperature_zero[mask_zero]

# 2. Create the plot
# The first argument is the x-axis, the second is the y-axis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Setup the figure and dual axes
fig, ax1 = plt.subplots(figsize=(width_inch, height_inch))
ax2 = ax1.twinx()

# 2. Plot Density (Left - Red)
ax1.plot(rho_decay, edensity_decay, color='red', linestyle='--', linewidth=2)
ax1.plot(rho_decay, edensity_flat, color='red', linestyle=':', linewidth=2)
ax1.plot(rho_decay, edensity_zero, color='red', linestyle='-', linewidth=2)
ax1.set_ylabel(r'$n_e$ (1/m$^3$)', color='red')
ax1.set_xlabel(r'$\rho$', color='black')
ax1.tick_params(axis='y', labelcolor='red')

# 3. Plot Temperature (Right - Blue)
ax2.plot(rho_decay, itemperature_decay, color='blue', linestyle='--', linewidth=2)
ax2.plot(rho_decay, itemperature_flat, color='blue', linestyle=':', linewidth=2)
ax2.plot(rho_decay, itemperature_zero, color='blue', linestyle='-', linewidth=2)

ax2.set_yscale('log')
ax2.set_ylabel(r'$T_i$ (KeV)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.spines['right'].set_color('blue')
ax2.spines['left'].set_color('red')

# 4. Create Custom Legend (Proxy Artists)
# We create dummy lines in black to represent the styles globally
custom_lines = [
    Line2D([0], [0], color='black', lw=2, linestyle='-'),
    Line2D([0], [0], color='black', lw=2, linestyle='--'),
    Line2D([0], [0], color='black', lw=2, linestyle=':')
]

ax1.legend(custom_lines, ['Vacuum', 'Exponential Decay', 'Flat'], 
           loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig("figs/profile-figures.png", dpi=600)
plt.show()
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
width_inch = height_inch * (10/16)

input_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"

a_offsetpsi_decay = Ascot(input_dir+ "G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5")
a_offsetpsi_flat = Ascot(input_dir+ "G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_flat-profile_04082026.h5")
a_offsetpsi_zero = Ascot(input_dir + "G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_FLR-mode_new-wall_zero-profile_04092026.h5")

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

a_offsetpsi_decay.input_init(bfield = True)
a_offsetpsi_decay.input_init(plasma = True)

r_offsetpsi_decay = np.linspace(7.25, rmax_offsetpsi_decay + 0.2, 200)
z_offsetpsi_decay = np.linspace(zmin_offsetpsi_decay, zmax_offsetpsi_decay, 200)
r_1d_offsetpsi_decay = r_offsetpsi_decay.flatten()
z_1d_offsetpsi_decay = z_offsetpsi_decay.flatten()
z_2d_offsetpsi_decay, r_2d_offsetpsi_decay = np.meshgrid(z_offsetpsi_decay, r_offsetpsi_decay)
psi_offsetpsi_decay = a_offsetpsi_decay.input_eval(r_2d_offsetpsi_decay, 0, z_2d_offsetpsi_decay, 0, 'psi', grid = False).reshape(200,200).T
rho_offsetpsi_decay = np.sqrt(psi_offsetpsi_decay / 61)
max_rho = np.max(psi_offsetpsi_decay)
print(max_rho)
fig, (ax1) = plt.subplots(1, 1, figsize=(width_inch, height_inch))
phi_deg = 0

contours = [0, 10, 20, 30, 40, 50, 60]
contours_extended = [0, 10, 20, 30, 40, 50, 62, 64, 66, 68, 70, 72, 74]
contours_rho = np.linspace(0, 1.3, 14)
phi = unyt.unyt_quantity(phi_deg, 'degree')

#im2 = ax1.pcolormesh(r_1d_offsetpsi_decay, z_1d_offsetpsi_decay, rho_offsetpsi_decay, cmap='inferno', shading='auto')   
contours1 = ax1.contour(np.asarray(r_1d_offsetpsi_decay), np.asarray(z_1d_offsetpsi_decay), np.asarray(rho_offsetpsi_decay), levels=[1], colors='red', linewidths=1)
contours2 = ax1.contour(np.asarray(r_1d_offsetpsi_decay), np.asarray(z_1d_offsetpsi_decay), np.asarray(rho_offsetpsi_decay), levels=contours_rho, colors='black', linewidths=0.5)
ax1.clabel(contours2, inline=True, fontsize=4, fmt='%1.2f')
#cb2 = plt.colorbar(im2, ax=ax1)
#cb2.set_label('$\psi$ [Wb]')

ax1.set_xlabel(r'R [m]')
ax1.set_ylabel(r'Z [m]')

ax1.set_aspect('equal')
plt.tight_layout()
plt.savefig("figs/rho-figures.png", dpi=600)

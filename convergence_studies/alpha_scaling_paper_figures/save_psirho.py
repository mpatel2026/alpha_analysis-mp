import numpy as np
from a5py import Ascot
import matplotlib.pyplot as plt
import unyt
import matplotlib.colors as colors

plt.rcParams.update({
    'font.size': 12,          # General font size
    'axes.labelsize': 12,     # x and y labels
    'axes.titlesize': 12,     # Title size
    'xtick.labelsize': 12,    # x-axis tick labels
    'ytick.labelsize': 12,    # y-axis tick labels
    'legend.fontsize': 12,     # Legend size
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
fig = plt.figure(figsize=(190 / 25.4, 4.6), layout='constrained')
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.54])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[2])
ax2.tick_params(labelleft=False)
phi_deg = 0

contours = [0, 10, 20, 30, 40, 50, 60]
contours_extended = [0, 10, 20, 30, 40, 50, 62, 64, 66, 68, 70, 72, 74]
contours_rho = np.linspace(0, 1.3, 14)
phi = unyt.unyt_quantity(phi_deg, 'degree')

#im2 = ax1.pcolormesh(r_1d_offsetpsi_decay, z_1d_offsetpsi_decay, rho_offsetpsi_decay, cmap='inferno', shading='auto')   
contours1 = ax1.contour(np.asarray(r_1d_offsetpsi_decay), np.asarray(z_1d_offsetpsi_decay), np.asarray(rho_offsetpsi_decay), levels=[1], colors='red', linewidths=1)
contours2 = ax1.contour(np.asarray(r_1d_offsetpsi_decay), np.asarray(z_1d_offsetpsi_decay), np.asarray(rho_offsetpsi_decay), levels=contours_rho, colors='black', linewidths=0.5)
ax1.clabel(contours2, inline=True, fontsize=6, fmt='%1.2f')
#cb2 = plt.colorbar(im2, ax=ax1)
#cb2.set_label('$\psi$ [Wb]')

ax1.set_xlabel(r'R [m]')
ax1.set_ylabel(r'Z [m]')

ax1.set_aspect('equal')

# --- Poincare (right panel of poincare_fig.png), Helios wall flipped about Z=0 ---
import h5py
poincare_dir = "poincare-fig/out/"
with h5py.File(poincare_dir + "trace_nested_A256.h5", "r") as f:
    Rf, Zf = np.asarray(f["R_fwd"]), np.asarray(f["Z_fwd"])
    Rb, Zb = np.asarray(f["R_bwd"]), np.asarray(f["Z_bwd"])
    stride = int(f.attrs["nphi_sections"]) * int(f.attrs["NFP"])
walls = np.load(poincare_dir + "walls_phi0.npz")
R = np.concatenate([Rf[0::stride][1:].ravel(), Rb[0::stride][1:].ravel()])
Z = np.concatenate([Zf[0::stride][1:].ravel(), Zb[0::stride][1:].ravel()])
ax2.plot(R, Z, ls="", marker="o", ms=0.2, c="skyblue", rasterized=True)
lcfs = walls["lcfs_CX"]
ax2.plot(lcfs[:, 0], lcfs[:, 1], "r-", lw=0.8, label="DESC LCFS", zorder=10)
for k, key in enumerate(("offset_30cm", "offset_10cm", "offset_20cm")):
    rz = walls[key]
    ax2.plot(rz[:, 0], rz[:, 1], color="tab:orange", lw=0.8, zorder=11,
             label="Conformal walls: 10cm, 20, 30cm" if k == 0 else None)
eng = walls["engineering"]
ax2.plot(eng[:, 0], -eng[:, 1], color="black", lw=0.8, zorder=11, label="First wall with divertor cutout")
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(7.0, 10.5)
ax2.set_ylim(-4.5, 4.5)
ax2.set_xlabel(r'R [m]')
ax1.set_xlim(7.0, 10.5)
ax1.set_ylim(-4.5, 4.5)
ax1.set_title('(a)')
ax2.set_title('(b)')
h, l = ax2.get_legend_handles_labels()
fig.legend(h, l, loc='upper left', bbox_to_anchor=(1.03, 1.0), bbox_transform=ax2.transAxes, ncol=1, fontsize=10)

# --- Plasma profiles (from save_profileplot.py), engineering-wall runs ---
from matplotlib.lines import Line2D
p_decay = Ascot(input_dir + "G1600-engineeringwall-withtargets-1000kmrk-gc-biot-decay-04292026.h5").data.plasma.active.read()
p_flat = Ascot(input_dir + "G1600-free-reopt_I_engineering-wall_1000k-mrk_biot-bfield_p01cellarea__FLR-mode_flat-profile_05052026.h5").data.plasma.active.read()
p_zero = Ascot(input_dir + "G1600-free-reopt_I_engineering-wall_1000k-mrk_biot-bfield_p01cellarea__FLR-mode_zero-profile_05052026.h5").data.plasma.active.read()

mask_prof = p_decay['rho'] < 1.3
rho_prof = p_decay['rho'][mask_prof]
ax3_t = ax3.twinx()
for ax in (ax3, ax3_t):
    ax.set_box_aspect(0.9)
    ax.set_anchor('S')
for p, ls, clip in ((p_decay, '--', False), (p_flat, ':', False), (p_zero, '-', True)):
    ti = p['itemperature'] / 1000
    if clip:
        ti = np.clip(ti, 1e-3, None)
    ax3.plot(rho_prof, p['edensity'][p['rho'] < 1.3], color='red', linestyle=ls, linewidth=2)
    ax3_t.plot(rho_prof, ti[p['rho'] < 1.3], color='blue', linestyle=ls, linewidth=2)
ax3.set_ylabel(r'$n$ (1/m$^3$)', color='red')
ax3.set_xlabel(r'$\rho$')
ax3.tick_params(axis='y', labelcolor='red')
ax3.yaxis.get_offset_text().set_color('red')
ax3_t.set_yscale('log')
ax3_t.set_ylabel(r'$T$ (keV)', color='blue')
ax3_t.tick_params(axis='y', labelcolor='blue')
ax3_t.spines['right'].set_color('blue')
ax3_t.spines['left'].set_color('red')
custom_lines = [Line2D([0], [0], color='black', lw=2, linestyle=s) for s in ('-', '--', ':')]
ax3.legend(custom_lines, ['Vacuum', 'Exponential Decay', 'Flat'], loc='lower left', fontsize=9, handlelength=1.5)
ax3.set_title('(c)')

plt.savefig("figs/rho-figures.png", dpi=600)

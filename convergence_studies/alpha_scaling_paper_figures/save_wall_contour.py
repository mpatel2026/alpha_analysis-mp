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


input_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"  
lcfs = Ascot(input_dir + "G1600-free-reopt_I_lcfs-wall_1000k-mrk_biot-bfield_FLR-mode_FPP-power_03142026.h5")
new_wall10cm = Ascot(input_dir+ "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5")
new_wall30cm = Ascot(input_dir+ "G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5")
new_wall20cm = Ascot(input_dir+ "20cm-walltest.h5")
firstwall = Ascot(input_dir + "G1600-free-reopt_I_divertorwall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04262026.h5")
divertorwall = Ascot(input_dir + "G1600-engineeringwall-withtargets-1000kmrk-gc-biot-decay-04292026.h5")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

lcfs.input_plotwallcontour(phi = unyt.unyt_quantity(0, 'degree'), axes = ax1, color="green", label= r"LCFS $\phi=0\degree$")
lcfs.input_plotwallcontour(phi = unyt.unyt_quantity(90, 'degree'), axes = ax2, color="green", label= r"LCFS $\phi=90\degree$")

new_wall10cm.input_plotwallcontour(phi = unyt.unyt_quantity(0, 'degree'), axes = ax1, color="steelblue", label=r"$\Delta$ w 10cm")
new_wall10cm.input_plotwallcontour(phi = unyt.unyt_quantity(90, 'degree'), axes = ax2, color="steelblue", label=r"$\Delta$ w 10cm")

new_wall20cm.input_plotwallcontour(phi = unyt.unyt_quantity(0, 'degree'), axes = ax1, color="orange", label=r"$\Delta$ w 20cm")
new_wall20cm.input_plotwallcontour(phi = unyt.unyt_quantity(90, 'degree'), axes = ax2, color="orange", label=r"$\Delta$ w 20cm")

new_wall30cm.input_plotwallcontour(phi = unyt.unyt_quantity(0, 'degree'), axes = ax1, color="salmon", label=r"$\Delta$ w 30cm")
new_wall30cm.input_plotwallcontour(phi = unyt.unyt_quantity(90, 'degree'), axes = ax2, color="salmon", label=r"$\Delta$ w 30cm")

divertorwall.input_plotwallcontour(phi = unyt.unyt_quantity(0, 'degree'), axes = ax1, color="black", label="Engineering Wall")
divertorwall.input_plotwallcontour(phi = unyt.unyt_quantity(90, 'degree'), axes = ax2, color="black", label="Engineering Wall")




ax1.set_xlabel("R (m)")
ax1.set_ylabel("Z (m)")
ax1.set_xlim(4.5,10.5)
ax1.set_ylim(-4.5,4.5)
ax1.legend(loc="upper left")
ax2.set_xlabel("R (m)")
ax2.set_ylabel("")
ax2.set_xlim(4.5,10.5)
ax2.set_ylim(-4.5,4.5)
ax2.legend(loc="upper left")

plt.savefig("figs/wall_contours.pdf", dpi=600)
plt.show()
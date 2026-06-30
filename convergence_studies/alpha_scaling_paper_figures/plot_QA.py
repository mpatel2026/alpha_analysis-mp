import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import desc.io
from desc.plotting import plot_boundary, plot_qs_error

# 1. Enforce font size 10 universally
plt.rcParams.update({
    'font.size': 7.5,
    'axes.labelsize': 7.5,
    'axes.titlesize': 7.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 6.5,
    'font.family': 'sans-serif' # A clean, professional serif font
})

# 2. Dimensions: 60 mm tall, 120 mm wide
mm_to_inch = 25.4
fig_width = 120 / mm_to_inch   # ~4.72 inches
fig_height = 60 / mm_to_inch   # ~2.36 inches

# Generate a side-by-side subplot canvas (1 row, 2 columns)
# This inherently locks their y-axis dimensions to be identical and perfectly in-line
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# 3. Load the equilibrium file
eq_file = "/pscratch/sd/m/mpatel26/equil/equil_Helios_G1600-12-89_QA2e-1_Bxdl25_free_L12_M12_N20.h5"  # Replace with your actual file path
eq_family = desc.io.load(eq_file)
eq = eq_family[-1]


# ==========================================
# SUBPLOT 1: BOUNDARY CROSS-SECTIONS
# ==========================================
phi_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
custom_colors = ["#E66101", "#FDB863", "#B2ABD2", "#5E3C99"]
line_widths = [1.5, 1.5, 1.5, 1.5]  

# Route DESC output explicitly to ax1
plot_boundary(
    eq, 
    phi=phi_angles, 
    color=custom_colors, 
    lw=line_widths,
    plot_axis=True,
    ax=ax1
)

ax1.set_aspect('equal', adjustable='box')
ax1.yaxis.labelpad = -2

# Add your custom short-handle legend
legend_handles = [Line2D([0], [0], color=c, lw=w) for c, w in zip(custom_colors, line_widths)]
legend_labels = [r"$\phi = 0$", r"$\phi = \pi/4$", r"$\phi = \pi/2$", r"$\phi = 3\pi/4$"]
ax1.legend(
    legend_handles, 
    legend_labels, 
    loc="upper left",          # Anchors the top-left corner of the legend box...
    bbox_to_anchor=(1.02, 1.0), # ...to just outside the top-right edge of the plot.
    frameon=True,
    handlelength=1.0,
    handletextpad=0.5
)


# ==========================================
# SUBPLOT 2: fB QA ERROR PROFILE (%)
# ==========================================
# Route DESC output explicitly to ax2
plot_qs_error(
    eq,
    helicity=(1, 0),  
    fB=True,          
    fC=False,         
    fT=False,         
    log=False,        
    legend=False,
    ax=ax2
)

# Convert raw data fractions to % values and strip markers
all_y_values = []
for line in ax2.get_lines():
    new_y = line.get_ydata() * 100
    line.set_ydata(new_y)
    line.set_marker('')  
    all_y_values.extend(new_y)

# Recalculate and update the axis scaling limits dynamically
if all_y_values:
    ymin, ymax = min(all_y_values), max(all_y_values)
    padding = (ymax - ymin) * 0.1 if ymax != ymin else 0.05
    ax2.set_ylim(0, ymax + padding)

# Final formatting updates for Subplot 2
ax2.set_ylabel(r"$f_B$ (%)")
ax2.ticklabel_format(style='plain', axis='y')


# ==========================================
# GLOBAL SAVING ENGINE
# ==========================================
# Automatically keeps labels from clipping or smashing into neighboring subplots
plt.tight_layout()

output_filename = "figs/Helios-data.png"
fig.savefig(
    output_filename, 
    dpi=600, 
    bbox_inches="tight",
    pad_inches=0.01  # Clean, close cut on the image outer wall borders
)
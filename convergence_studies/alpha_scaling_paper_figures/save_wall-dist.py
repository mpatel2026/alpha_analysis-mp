import numpy as np
import unyt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from a5py import Ascot

# --- 1. Configuration & Plotting Setup ---
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif'
})

width_inch = 190 / 25.4
height_inch = width_inch * (8/10)
LOAD_THRESHOLD = 50000  # W/m^2

# --- 2. Load Data ---
inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
file_name = "G1600-engineeringwall-withtargets-1000kmrk-gc-biot-decay-04292026.h5"
#file_name = "G1600-free-reopt_I_engineeringwall_100k-mrk_biot-bfield_p01cellarea_FO-mode_zero-profile_05092026.h5"
biot = Ascot(inp_dir + file_name)

# --- 3. Full Data Extraction ---
end_conditions = ["none", "wall", "RHOMAX"]
res = biot.data.active.getstate("ids", "weight", "ekin", "pitch", "walltile", "time", state="end", endcond=end_conditions)
ids_end, w_end, ekin_end, pitch_end, walltile_end, time_end = res

mom = biot.data.active.getstate("pr", "pphi", "pz", "pnorm", "phi", state="end", endcond=end_conditions)
pr_end, pphi_end, pz_end, pnorm_end, phi_end = mom

area_all, nvec_all = biot.data.active.wall.area(normal=True)

# Function discovery for getwall_loads
if hasattr(biot, 'getwall_loads'): 
    load_func = biot.getwall_loads
else: 
    load_func = biot.data.active.getwall_loads

# --- 4. Filtering for All Wall Hits ---
hit_mask = (walltile_end > 0)
w_hit = w_end[hit_mask]
ekin_hit = ekin_end[hit_mask]
pitch_hit = pitch_end[hit_mask]
time_hit = time_end[hit_mask]
phi_hit = phi_end[hit_mask]
walltile_hit = walltile_end[hit_mask]

# --- 5. Incident Angle Calculation (Global) ---
punit_cyl = np.array([pr_end[hit_mask], pphi_end[hit_mask], pz_end[hit_mask]]) / pnorm_end[hit_mask]
punit_cart = np.array([
    punit_cyl[0] * np.cos(phi_hit) - punit_cyl[1] * np.sin(phi_hit),
    punit_cyl[0] * np.sin(phi_hit) + punit_cyl[1] * np.cos(phi_hit),
    punit_cyl[2]
])
nvec_hit = nvec_all[:, walltile_hit - 1]
dotprod = np.sum(nvec_hit * punit_cart, axis=0)
angles_hit = np.arccos(np.clip(dotprod, -1.0, 1.0)) * (180 / np.pi)
angles_hit[angles_hit > 90] = 180 - angles_hit[angles_hit > 90]

# --- 6. Identify Hotspot Tiles ---
# Calculate loads for ALL wall-hitting markers
ids_tiles, areas_tiles, loads_tiles, _, _ = load_func(weights=True, p_ids=ids_end[hit_mask])
wall_loads = loads_tiles / areas_tiles

# Identify tile IDs exceeding the threshold
hotspot_tile_ids = ids_tiles[wall_loads > LOAD_THRESHOLD]
# Create a marker-level mask: Does the marker's walltile exist in the hotspot list?
hotspot_marker_mask = np.isin(walltile_hit, hotspot_tile_ids)

# --- 7. Plotting Function ---
def plot_distributions(ekin, angles, weights, filename):
    # --- Configuration ---
    total_width_inch = 190 / 25.4
    
    # Increase right_margin to create space after the colorbar
    left_margin = 0.8
    right_margin = 0.8  # Increased from 0.2 to 0.7 inches
    bottom_margin = 0.5
    top_margin = 0.2
    wspace = 0.7        # Space between subplots
    cbar_pad = 0.1      # Gap between 3rd plot and colorbar
    cbar_width = 0.15   # Thickness of colorbar
    
    # Calculate square plot width based on total width and margins
    plot_width = (total_width_inch - left_margin - right_margin - (2 * wspace) - cbar_pad - cbar_width) / 3
    
    # Total Height to keep subplots square
    total_height_inch = bottom_margin + top_margin + plot_width

    fig = plt.figure(figsize=(total_width_inch, total_height_inch))
    
    # --- Axes Positions ---
    ax_pos = []
    for i in range(3):
        x_start = (left_margin + i * (plot_width + wspace)) / total_width_inch
        y_start = bottom_margin / total_height_inch
        w_norm = plot_width / total_width_inch
        h_norm = plot_width / total_height_inch # Square
        ax_pos.append([x_start, y_start, w_norm, h_norm])

    axs = [fig.add_axes(pos) for pos in ax_pos]

    # --- Data Processing ---
    e_vals = ekin.v if hasattr(ekin, 'v') else ekin
    a_vals = angles.v if hasattr(angles, 'v') else angles
    w_vals = weights.v if hasattr(weights, 'v') else weights
    
    # --- Print Requested Statistics ---
    total_particles = np.nansum(w_vals)
    low_energy_particles = np.nansum(w_vals[e_vals < 50000])
    low_angle_particles = np.nansum(w_vals[a_vals < 10])

    print(f"\n--- Statistics for: {filename} ---")
    print(f"Total particles (integrating w): {total_particles:.4e}")
    print(f"Particles with energy < 5x10^4:  {low_energy_particles:.4e}")
    print(f"Particles with angle < 10°:      {low_angle_particles:.4e}")
    print("-" * 50)
    
    e_min = 1000.0
    valid = (e_vals >= e_min) & (~np.isnan(e_vals))
    e_bins = np.logspace(np.log10(e_min), np.log10(np.nanmax(e_vals)), 40)
    
    # --- Plotting ---
    # 1D Energy
    axs[0].hist(e_vals[valid], bins=e_bins, weights=w_vals[valid], 
                color='skyblue', edgecolor='black', alpha=0.7, lw=0.5)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel(r"$E_{\mathrm{fin}}$ [eV]")
    axs[0].set_ylabel(r'F [1/s]')
    axs[0].grid(True, axis='y', which='major', linestyle='-', alpha=0.5)
    
    # 1D Angle
    axs[1].hist(a_vals[valid], bins=40, weights=w_vals[valid], 
                color='skyblue', edgecolor='black', alpha=0.7, lw=0.5)
    axs[1].set_xlabel(r"$\eta$ [$^{\circ}$]")
    axs[1].set_ylabel(r'F [1/s]')
    axs[1].grid(True, axis='y', which='major', linestyle='-', alpha=0.5)
    
    # 2D Energy vs Angle
    h = axs[2].hist2d(e_vals[valid], a_vals[valid], 
                      bins=[e_bins, 40], weights=w_vals[valid], 
                      cmap='viridis', cmin=1e-20, norm=LogNorm())
    axs[2].set_xscale('log')
    axs[2].set_xlabel(r"$E_{\mathrm{fin}}$ [eV]")
    axs[2].set_ylabel(r"$\eta$ [$^{\circ}$]")
    
    # --- Colorbar with explicit spacing ---
    cbar_x = ax_pos[2][0] + ax_pos[2][2] + (cbar_pad / total_width_inch)
    cax = fig.add_axes([cbar_x, ax_pos[2][1], cbar_width / total_width_inch, ax_pos[2][3]])
    fig.colorbar(h[3], cax=cax, label=r'F [1/s]')

    plt.savefig(filename, dpi=600)
    print(f"Saved: {filename}\n")
    plt.close()

# --- 8. Generate Figures ---
# Figure 1: Global Distributions
plot_distributions(
    ekin_hit, angles_hit, w_hit, "figs/engineering_wall_dist.png"
)

# Figure 2: Hotspot Distributions (> 50,000 W/m^2)
plot_distributions(
    ekin_hit[hotspot_marker_mask], 
    angles_hit[hotspot_marker_mask], 
    w_hit[hotspot_marker_mask],
    "figs/hotspot_wall_dist.png"
)

non_hotspot_mask = ~hotspot_marker_mask

plot_distributions(
    ekin_hit[non_hotspot_mask], 
    angles_hit[non_hotspot_mask], 
    w_hit[non_hotspot_mask],
    "figs/non_hotspot_wall_dist.png"
)

print(f"Hotspot Markers: {np.sum(hotspot_marker_mask)} out of {len(w_hit)} total hits.")
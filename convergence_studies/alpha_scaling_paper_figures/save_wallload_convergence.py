import numpy as np
import unyt
import matplotlib.pyplot as plt
from a5py import Ascot

# --- Plotting Parameters ---
plt.rcParams.update({
    'font.size': 10,          # General font size
    'axes.labelsize': 10,     # x and y labels
    'axes.titlesize': 10,     # Title size
    'xtick.labelsize': 10,    # x-axis tick labels
    'ytick.labelsize': 10,    # y-axis tick labels
    'legend.fontsize': 10,    # Legend size
    'font.family': 'sans-serif' # A clean, professional serif font
})

width_inch = 190 / 25.4
height_inch = width_inch * (6/10) # ~4.48 inches, perfect for 1x2 square subplots

# --- Data Loading ---
inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
p01 = Ascot(inp_dir + "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5")
p001 = Ascot(inp_dir + "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p001cellarea_FLR-mode_decay-profile_05082026.h5")
p005 = Ascot(inp_dir + "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p005cellarea_FLR-mode_decay-profile_05082026.h5")
p02 = Ascot(inp_dir + "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p02cellarea_FLR-mode_decay-profile_05082026.h5")

# List of your data objects
data_objects = {
    "p001": p001,
    "p005": p005,
    "p01": p01,
    "p02": p02
}

# --- Calculation Parameters ---
end_conditions = ["none", "wall", "RHOMAX"]
nmrk = 1000000

# List of group sizes to iterate through
marker_count_list = [100000, 200000, 500000, 1000000]

# Results container: [name][group_key]['metrics']
results = {}
num_trials_fixed = 10  # Exactly 10 data points for each count

# --- Main Processing Loop ---
for name, obj in data_objects.items():
    results[name] = {}
    print(f"--- Processing Method: {name} ---")
    
    # 1. BULK DATA FETCH: Load all necessary arrays at once into memory
    all_ids, all_econd, all_tile, all_weight = obj.data.active.getstate(
        "ids", "endcond", "walltile", "weight", state="end"
    )
    
    wtotal = np.sum(all_weight)
    
    # 2. VECTORIZED PRE-PROCESSING: Create a global mask for wall hits
    all_econd_str = np.char.lower(all_econd.astype(str))
    all_is_wall = (all_econd_str == "wall") | (all_econd == 1) | (all_tile > 0)

    # Function discovery
    if hasattr(obj, 'getwall_loads'):
        load_func = obj.getwall_loads
    elif hasattr(obj.data.active, 'getwall_loads'):
        load_func = obj.data.active.getwall_loads
    else:
        raise AttributeError("Could not find 'getwall_loads'.")

    for marker_count in marker_count_list:
        k_label = int(marker_count / 1000)
        group_key = f"{k_label}k_markers"
        results[name][group_key] = {} 

        trial_peak_loads = []
        trial_wetted_areas = []

        print(f"Processing {group_key} ({num_trials_fixed} trials)")

        for i in range(num_trials_fixed):
            # 3. FAST RANDOM SAMPLING: Pick indices instead of shuffling IDs
            sampled_indices = np.random.choice(nmrk, size=marker_count, replace=False)
            
            # 4. IN-MEMORY SLICING: Apply the random indices to our pre-fetched arrays
            sample_is_wall = all_is_wall[sampled_indices]
            wall_ids = all_ids[sampled_indices][sample_is_wall]
            
            # Sum the weights for this specific sample batch
            w_sub_sum = np.sum(all_weight[sampled_indices])
            
            # 5. Calculation
            if len(wall_ids) > 0:
                _, areas, loads, _, _ = load_func(weights=True, p_ids=wall_ids)
                
                reweight_factor = wtotal / w_sub_sum if w_sub_sum > 0 else 0

                if loads is not None and loads.size > 0:
                    p_load = np.max(loads / areas) * reweight_factor
                    w_area = np.sum(areas[loads > 0])
                else:
                    p_load, w_area = 0.0, 0.0
            else:
                p_load, w_area = 0.0, 0.0
                
            trial_peak_loads.append(p_load)
            trial_wetted_areas.append(w_area)

        # 6. Statistics calculation
        results[name][group_key]['metrics'] = {
            'Peak Wall Load Mean': np.mean(trial_peak_loads),
            'Peak Wall Load Std': np.std(trial_peak_loads),
            'Wetted Area Mean': np.mean(trial_wetted_areas),
            'Wetted Area Std': np.std(trial_wetted_areas),
            'Raw Peak Loads': trial_peak_loads, 
            'Raw Wetted Areas': trial_wetted_areas
        }

        print(f"  Peak Load: {results[name][group_key]['metrics']['Peak Wall Load Mean']:.4e} "
              f"± {results[name][group_key]['metrics']['Peak Wall Load Std']:.4e} W/m^2")
        print(f"  Wetted Area: {results[name][group_key]['metrics']['Wetted Area Mean']:.4f} "
              f"± {results[name][group_key]['metrics']['Wetted Area Std']:.4f} m^2")

# --- Plotting Section ---
print("--- Generating Plots ---")

# Define labels, x-axis values, and markers
label_map = {
    "p001": r'0.001 m$^2$',
    "p005": r'0.005 m$^2$',
    "p01":  r'0.01 m$^2$',
    "p02":  r'0.02 m$^2$'
}

marker_map = {
    "p001": 'o',  # Circle
    "p005": 's',  # Square
    "p01":  '^',  # Triangle Up
    "p02":  'D'   # Diamond
}

x_values = [int(count / 1000) for count in marker_count_list]

# Initialize figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_inch, height_inch))

# Loop through our processed dataset keys
for name, label in label_map.items():
    # Extract arrays for plotting
    p_load_mean, p_load_std = [], []
    w_area_mean, w_area_std = [], []
    
    for x in x_values:
        group_key = f"{x}k_markers"
        metrics = results[name][group_key]['metrics']
        
        p_load_mean.append(metrics['Peak Wall Load Mean'])
        p_load_std.append(metrics['Peak Wall Load Std'])
        
        w_area_mean.append(metrics['Wetted Area Mean'])
        w_area_std.append(metrics['Wetted Area Std'])

    # Convert lists to NumPy arrays and divide by 1,000,000 to convert W to MW
    p_load_mean_mw = np.array(p_load_mean) / 1e6
    p_load_std_mw = np.array(p_load_std) / 1e6

    # Fetch the specific marker for this cell area
    curve_marker = marker_map[name]

    # Plot 1: Peak Wall Load
    ax1.errorbar(
        x_values, p_load_mean_mw, yerr=p_load_std_mw,
        marker=curve_marker, linestyle='-', linewidth=2, capsize=5, label=label
    )
    
    # Plot 2: Wetted Area
    ax2.errorbar(
        x_values, w_area_mean, yerr=w_area_std,
        marker=curve_marker, linestyle='-', linewidth=2, capsize=5, label=label
    )

# Subplot 1 Customization
ax1.set_xlabel(r'N ($\times 10^3$ markers)')
ax1.set_ylabel(r'Peak $P_{surface}$ (MW/m$^2$)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_box_aspect(1) # Forces the plot to be square

# Subplot 2 Customization
ax2.set_xlabel(r'N ($\times 10^3$ markers)')
ax2.set_ylabel(r'$A_{wetted}$ (m$^2$)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax1.legend(title='Average Cell Area', loc='upper right')
ax2.set_box_aspect(1) # Forces the plot to be square

# Final Layout Adjustments & Save
plt.tight_layout()
plt.savefig("figs/wall-pload-convergence.png", dpi=600, bbox_inches='tight')
plt.show()
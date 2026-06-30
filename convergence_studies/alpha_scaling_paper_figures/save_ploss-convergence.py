import numpy as np
import unyt
import matplotlib.pyplot as plt
from a5py import Ascot

plt.rcParams.update({
    'font.size': 12,          # General font size
    'axes.labelsize': 12,     # x and y labels
    'axes.titlesize': 12,     # Title size
    'xtick.labelsize': 12,    # x-axis tick labels
    'ytick.labelsize': 12,    # y-axis tick labels
    'legend.fontsize': 12,     # Legend size
    'font.family': 'sans-serif' # A clean, professional serif font
})

width_inch = 190 / 25.4
height_inch = width_inch * (6/10)

inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
biot = Ascot(inp_dir + "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5")

np.random.seed(42)
# List of your data objects
data_objects = {
    "compute_method": biot,
}


# Common parameters to ensure consistency
end_conditions = ["none", "wall", "RHOMAX"]

nmrk = 1000000


# List of group sizes to iterate through
ngroups_list = [100, 50, 20, 10, 5, 2]

# Results container: [name][ngroup][batch_name/metrics]
results = {}

for name, obj in data_objects.items():
    results[name] = {}
    print(f"--- Processing Method: {name} ---")

    for ngroup in ngroups_list:
        # 1. Create a shuffled array of all marker IDs (1 to 100,000)
        # Using a fixed seed here ensures consistency across different 'names' if desired
        shuffled_ids = np.random.permutation(np.arange(1, nmrk + 1))
        
        k_val = int(nmrk / (ngroup * 1000))
        group_key = f"{k_val}k_markers"
        results[name][group_key] = {}
        
        step = nmrk // ngroup
        
        # 2. Slice the shuffled array into batches
        id_batches = {
            f"batch {i + 1}": shuffled_ids[i * step : (i + 1) * step].tolist()
            for i in range(ngroup)
        }

        # List to collect losses for metrics calculation
        temp_batch_losses = []

        for batch_name, batch_ids in id_batches.items():
            # ... Data extraction (e_end, w_end, etc.) ...
            e_end = np.array(obj.data.active.getstate("ekin", state="end", ids=batch_ids, endcond=end_conditions))
            w_end = np.array(obj.data.active.getstate("weight", state="end", ids=batch_ids, endcond=end_conditions))
            e_start = np.array(obj.data.active.getstate("ekin", state="ini", ids=batch_ids))
            w_start = np.array(obj.data.active.getstate("weight", state="ini", ids=batch_ids))

            # Calculation
            p_loss = (np.nansum(e_end * w_end) / np.nansum(e_start * w_start)) * 100
            
            # Store batch result and track in list
            results[name][group_key][batch_name] = p_loss
            temp_batch_losses.append(p_loss)

        # Calculate and store metrics for this ngroup
        results[name][group_key]['metrics'] = {
            'average_loss': np.mean(temp_batch_losses),
            'std_loss': np.std(temp_batch_losses)
        }
        

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(width_inch, height_inch))

# Prepare variables for the reference line anchor
first_x = None
first_y = None

for name in results.keys():
    k_marker_values = []
    std_losses = []
    
    for group_key, group_data in results[name].items():
        try:
            # Convert "10.0k_markers" -> 10.0 (kilomarkers)
            k_val = float(group_key.replace('k_markers', ''))
            std_val = group_data['metrics']['std_loss']
            
            k_marker_values.append(k_val)
            std_losses.append(std_val)
        except (ValueError, KeyError):
            continue
    
    sorted_idx = np.argsort(k_marker_values)
    x = np.array(k_marker_values)[sorted_idx]
    y = ((np.array(std_losses)[sorted_idx])  / 4.49) * 100
    
    # Store the first point of the first method to anchor the reference line
    if first_x is None and len(x) > 0:
        first_x, first_y = x[0], y[0]
    
    plt.plot(x, y, 'o', label="Simulation Standard Deviation", linewidth=2)

# --- Add 1/sqrt(N) Reference Line ---
if first_x is not None:
    # We use the X values from the plot (in k_markers)
    # The relationship is: y = C * (1 / sqrt(x))
    # To anchor it: C = first_y * sqrt(first_x)
    anchor_constant = first_y * np.sqrt(first_x)
    x_ref = np.linspace(min(k_marker_values), 1000, 100)
    y_ref = anchor_constant / np.sqrt(x_ref)
    
    plt.plot(x_ref, y_ref, '--', color='gray', alpha=0.7, label=r'$1/\sqrt{N}$ Convergence')


plt.xlabel(r'N ($\times 10^3$)')
plt.ylabel(r'$\sigma (\%)$')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("figs/ploss-convergence-figures.png", dpi=600)
plt.show()
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

import desc.io as dscio
import desc.plotting as dscplt

from a5py import Ascot
import os
import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc

plt.rcParams.update({
    'font.size': 10,          # General font size
    'axes.labelsize': 10,     # x and y labels
    'axes.titlesize': 10,     # Title size
    'xtick.labelsize': 10,    # x-axis tick labels
    'ytick.labelsize': 10,    # y-axis tick labels
    'legend.fontsize': 10,     # Legend size
    'font.family': 'sans-serif' # A clean, professional serif font
})

width = 190 / 25.4
height = width * (5/12)

def plot_walload(fn, name, fig=None):
    obj = Ascot(fn)
    wetted_area, peak_total = obj.data.active.getwall_figuresofmerit()
    peak_total_MW = peak_total.to('MW/m**2')
    obj.input_init(bfield=True)
    print(f"{name} - Wetted Area: {wetted_area:.2f} m^2, Peak Total Power Load: {peak_total_MW:.2f} MW/m^2")

    cols = 2
    
    if fig is None:
        fig = plt.figure(figsize=(width, height))
        n_plots = 0
    else:
        # Identify existing wall map axes
        main_axes = [ax for ax in fig.axes if ax.get_label() == 'wall_map']
        n_plots = len(main_axes)

    current_idx = n_plots + 1
    rows = int(np.ceil(current_idx / cols))
    fig.set_size_inches(width, rows * height)

    # 1. Setup Grid - Small wspace for less distance between columns
    gs = fig.add_gridspec(rows, cols, right=0.88, wspace=0.05, hspace=0.3)
    
    existing_maps = [ax for ax in fig.axes if ax.get_label() == 'wall_map']

    # 2. Re-align existing maps
    for i, old_ax in enumerate(existing_maps):
        old_ax.set_subplotspec(gs[i // cols, i % cols])

    # 3. Add NEW map subplot
    # Share y-axis with the very first map created to ensure alignment
    share_y_with = existing_maps[0] if existing_maps else None
    ax = fig.add_subplot(gs[(current_idx - 1) // cols, (current_idx - 1) % cols], 
                         sharey=share_y_with)
    ax.set_label('wall_map')

    # Handle Y-Label and Ticks for shared axis cleanliness
    if (current_idx - 1) % cols == 0:
        # Only label the y-axis for the first column
        ax.set_ylabel(r"$\theta$ [$\degree$]")
    else:
        # Hide tick labels for the second column to save space
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel("")

    # 4. Master Colorbar
    cax = next((a for a in fig.axes if a.get_label() == 'master_cbar'), None)
    if cax is None:
        cax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) 
        cax.set_label('master_cbar')

    # 5. Plotting
    im = obj.data.active.plotwall_torpol_new(qnt="eload", axes=ax, cmap=cc.cm.fire, cax=cax, clim=[1000, 6000000])
        # Handle Y-Label and Ticks for shared axis cleanliness
    if (current_idx - 1) % cols == 0:
        # Only label the y-axis for the first column
        ax.set_ylabel(r"$\theta$ [$\degree$]")
    else:
        # Hide tick labels for the second column to save space
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel("")

    ax.set_xlabel(r"$\phi$ [$\degree$]")
    
    mappable = im if im is not None else ax.collections[0]
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r"$P_{surface}$ [W/m$^2$]")

    # Explicitly avoid plt.tight_layout() as it resets the manual gridspec spacing
    fig.subplots_adjust(right=0.88, left=0.1, wspace=0.05)

    return fig


def alpha_postprocess(input_dir, filenames, simnames, eq_dir=None, eq_names=None):
    """
    Processes multiple Ascot and DESC files, generating, saving, and displaying plots.
    """
    
    # 1. Initialize "State" variables
    wall_fig = None

    colors = plt.cm.tab10.colors

    # 2. Iterate through Ascot simulations
    for i, (fn_tail, s_name) in enumerate(zip(filenames, simnames)):
            fn = os.path.join(input_dir, fn_tail)
            # Pick a color based on the index (loops back if more than 10 sims)
            current_color = colors[i % len(colors)]
            
            print(f"Processing Ascot: {s_name}...")
            
            # Grid Plots
            wall_fig = plot_walload(fn, name=s_name, fig=wall_fig)
        


    # 4. Save the Figures
    print("\nSaving figures...")
    
    output_folder = "figs/"
    os.makedirs(output_folder, exist_ok=True)

    if wall_fig:
        wall_fig.savefig(os.path.join(output_folder, "wall_torpol.png"), bbox_inches='tight', dpi=600)
        
        

    print("All plots saved successfully.")

    # 5. Display
    plt.show()
    return

inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
filenames = [
             #"G1600-free-reopt_I_LCFS-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_zero-profile_05182026.h5",
             "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5",
             #"G1600-free-reopt_I_20cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_05132026.h5",
             "G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5"
             ]

simnames = [
            #"LCFS",
            "10cm offset",
            #"20cm ofset",
            "30cm offset"
            ]

alpha_postprocess(inp_dir, filenames, simnames)

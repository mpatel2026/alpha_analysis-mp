import os
import numpy as np
import matplotlib.pyplot as plt
from a5py import Ascot
from matplotlib.lines import Line2D

# --- Plot Styling ---
plt.rcParams.update({
    'font.size': 12,          # General font size
    'axes.labelsize': 12,     # x and y labels
    'axes.titlesize': 12,     # Title size
    'xtick.labelsize': 12,    # x-axis tick labels
    'ytick.labelsize': 12,    # y-axis tick labels
    'legend.fontsize': 10,     # Legend size
    'font.family': 'sans-serif' # A clean, professional serif font
})

width_inch = 190 / 25.4
height_inch = width_inch * (5/6)

def plot_single_ploss(ax, fn, name, nmrk, color, linestyle):
    """
    Extracts data for a single simulation and plots it onto the provided axes.
    """
    print(f"Processing Ascot: {name}...")
    obj = Ascot(fn)
    end_conditions = ["wall", "RHOMAX"]

    # --- Data Extraction ---
    e_end = np.array(obj.data.active.getstate("ekin", state="end", endcond=end_conditions))
    w_end = np.array(obj.data.active.getstate("weight", state="end", endcond=end_conditions))
    e_start = np.array(obj.data.active.getstate("ekin", state="ini"))
    w_start = np.array(obj.data.active.getstate("weight", state="ini"))
    t_end = np.array(obj.data.active.getstate("time", state="end", endcond=end_conditions))

    # Filter out abnormalities
    try:
        marker = obj.data.marker.active.read()
        marker_energy = marker['energy']
        max_energy = max(marker_energy)

        outlier_end = np.where(e_end > max_energy)[0]
        outlier_start = np.where(e_start > max_energy)[0]

        e_end = np.delete(e_end, outlier_end)
        w_end = np.delete(w_end, outlier_end)
        e_start = np.delete(e_start, outlier_start)
        w_start = np.delete(w_start, outlier_start)
        t_end = np.delete(t_end, outlier_end)
        
    except Exception as e:
        pass
    
    p_end = e_end * w_end
    p_start = e_start * w_start

    # --- Calculation ---
    time_grid = np.logspace(-4, 0, 1000)
    p_start_sum = np.nansum(p_start)
    lost_p_time = [100 * np.nansum(p_end[t_end < t]) / p_start_sum for t in time_grid]

    constant = 0.08 * np.sqrt(50000)
    stddev = constant / np.sqrt(nmrk)
    lower_bound = lost_p_time - stddev
    upper_bound = lost_p_time + stddev
    
    # --- Plotting ---
    ax.plot(time_grid, lost_p_time, label=f"{name} (Final: {lost_p_time[-1]:.2f}%)", 
            color=color, linestyle=linestyle)
    ax.fill_between(
        time_grid, 
        lower_bound, 
        upper_bound, 
        color=color, 
        alpha=0.2,       # Adjust transparency
        linewidth=0      # Removes the thin border line around the shaded area
    )
    print(f"Power loss for {name}: {lost_p_time[-1]:.2f}%")


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
    
    # 1. Set up the figure and axes
    fig, ax = plt.subplots(1, 2, figsize=[10, 4], dpi=600)
    colors = plt.cm.tab10.colors

    # 2. Call the plotting function for each simulation sequentially
    
    # Engineering Walls
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_engineering-wall_1000k-mrk_biot-bfield_p01cellarea__FLR-mode_flat-profile_05052026.h5"),
        name=r"Engineering - Flat",
        nmrk=1000000,
        color=colors[0],
        linestyle=":"
    )
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-engineeringwall-withtargets-1000kmrk-gc-biot-decay-04292026.h5"),
        name=r"Engineering - Decay",
        nmrk=1000000,
        color=colors[0],
        linestyle="--"
    )
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_engineering-wall_1000k-mrk_biot-bfield_p01cellarea__FLR-mode_zero-profile_05052026.h5"),
        name=r"Engineering - Vacuum",
        nmrk=1000000,
        color=colors[0],
        linestyle="-"
    )

    # 10cm Walls
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_flat-profile_04082026.h5"),
        name=r"$\Delta$w 10cm - Flat",
        nmrk=1000000,
        color=colors[1],
        linestyle=":"
    )
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5"),
        name=r"$\Delta$w 10cm - Decay",
        nmrk=1000000,
        color=colors[1],
        linestyle="--"
    )
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_FLR-mode_new-wall_zero-profile_04092026.h5"),
        name=r"$\Delta$w 10cm - Vacuum",
        nmrk=1000000,
        color=colors[1],
        linestyle="-"
    )

    # 20cm Walls
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_20cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_flat-profile_05132026.h5"),
        name=r"$\Delta$w 20cm - Flat",
        nmrk=1000000,
        color=colors[2],
        linestyle=":"
    )
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_20cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_05132026.h5"),
        name=r"$\Delta$w 20cm - Decay",
        nmrk=1000000,
        color=colors[2],
        linestyle="--"
    )
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_20cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_zero-profile_05132026.h5"),
        name=r"$\Delta$w 20cm - Vacuum",
        nmrk=1000000,
        color=colors[2],
        linestyle="-"
    )
    # 30cm walls
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_flat-profile_04082026.h5"),
        name=r"$\Delta$w 30cm - Flat",
        nmrk=1000000,
        color=colors[4],
        linestyle=":"
    )
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5"),
        name=r"$\Delta$w 30cm - Decay",
        nmrk=1000000,
        color=colors[4],
        linestyle="--"
    )
    plot_single_ploss(
        ax=ax[0],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_FLR-mode_new-wall_zero-profile_04092026.h5"),
        name=r"$\Delta$w 30cm - Vacuum",
        nmrk=1000000,
        color=colors[4],
        linestyle="-"
    )

    #plot sim modes and bfields
    plot_single_ploss(
        ax=ax[1],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_10cm-newwall_100k-mrk_biot-bfield_p01cellarea_FLR-mode_zero-profile_alpha-power_04222026.h5"),
        name=r"$\Delta$w 10cm GC: Biot-Savart",
        nmrk=1000000,
        color=colors[0],
        linestyle="--"
    )
    # 30cm walls
    plot_single_ploss(
        ax=ax[1],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_10cm-newwall_100k-mrk_mixed-bfield_p01cellarea_FLR-mode_zero-profile_alpha-power_04222026.h5"),
        name=r"$\Delta$w 10cmGC: Nested Internal Flux Surfaces",
        nmrk=1000000,
        color=colors[1],
        linestyle="--"
    )
    plot_single_ploss(
        ax=ax[1],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_10cm-newwall_100k-mrk_biot-bfield_p01cellarea_FO-mode_zero-profile_alpha-power_04172026.h5"),
        name=r"$\Delta$w 10cmFO: Biot-Savart",
        nmrk=1000000,
        color=colors[0],
        linestyle=":"
    )
    plot_single_ploss(
        ax=ax[1],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_10cm-newwall_100k-mrk_mixed-bfield_p01cellarea_FO-mode_zero-profile_alpha-power_04172026.h5"),
        name=r"$\Delta$w 10cm FO: Nested Internal Flux Surfaces",
        nmrk=1000000,
        color=colors[1],
        linestyle=":"
    )


    #plot sim modes and bfields
    plot_single_ploss(
        ax=ax[1],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_LCFS-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_zero-profile_05182026.h5"),
        name=r"LCFS GC: Biot-Savart",
        nmrk=1000000,
        color=colors[2],
        linestyle="--"
    )
    # 30cm walls
    plot_single_ploss(
        ax=ax[1],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_LCFS-wall_1000k-mrk_mixed-bfield_p01cellarea_FLR-mode_zero-profile_06112026.h5"),
        name=r"LCFS GC: Nested Internal Flux Surfaces",
        nmrk=1000000,
        color=colors[3],
        linestyle="--"
    )
    plot_single_ploss(
        ax=ax[1],
            fn=os.path.join(inp_dir, "G1600-free-reopt_I_LCFS-wall_100k-mrk_biot-bfield_p01cellarea_FO-mode_zero-profile_06112026.h5"),
        name=r"LCFS FO: Biot-Savart",
        nmrk=1000000,
        color=colors[2],
        linestyle=":"
    )
    plot_single_ploss(
        ax=ax[1],
        fn=os.path.join(inp_dir, "G1600-free-reopt_I_LCFS-wall_100k-mrk_mixed-bfield_p01cellarea_FO-mode_zero-profile_06112026.h5"),
        name=r"LCFS FO: Nested Internal Flux Surfaces",
        nmrk=1000000,
        color=colors[3],
        linestyle=":"
    )



    # 3. Standard Formatting (Applied once to the whole axis)

    ax[0].set_xscale('log')
    ax[0].set_ylabel(r'$\mathcal{P}_{loss}$ [%]')
    ax[0].set_xlabel(r'$t$ [s]')
    custom_lines = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='-'),
        Line2D([0], [0], color=colors[2], lw=2, linestyle='-'),
        Line2D([0], [0], color=colors[4], lw=2, linestyle='-'),
        Line2D([0], [0], color='black', lw=2, linestyle='-'),
        Line2D([0], [0], color='black', lw=2, linestyle='--'),
        Line2D([0], [0], color='black', lw=2, linestyle=':'),
    ]

    ax[0].legend(custom_lines, [r'$\Delta \bar{x}$ 10cm w/ divertor', r'$\Delta$w 10cm', r'$\Delta$w 20cm', r'$\Delta$w 30cm', "Vacuum", "Decay", "Flat"], 
            loc='upper left', frameon=True)
    ax[0].grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()


    ax[1].set_xscale('log')
    ax[1].set_ylabel(r'$\mathcal{P}_{loss}$ [%]')
    ax[1].set_xlabel(r'$t$ [s]')
    custom_lines2 = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='-'),
        Line2D([0], [0], color=colors[2], lw=2, linestyle='-'),
        Line2D([0], [0], color=colors[3], lw=2, linestyle='-'),
        Line2D([0], [0], color='black', lw=2, linestyle='--'),
        Line2D([0], [0], color='black', lw=2, linestyle=':'),
    ]

    ax[1].legend(custom_lines2, [r'$\Delta$w 10cm: Biot-Savart', r'$\Delta$w 10cm: Nested Internal Flux Surfaces', r'LCFS: Biot-Savart', r'LCFS: Nested Internal Flux Surfaces', r'Guiding Center', r'Full Orbit'], 
            loc='upper left', frameon=True)
    ax[1].grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()

    # 4. Save and Display
    print("\nSaving figure...")
    output_folder = "figs/"
    os.makedirs(output_folder, exist_ok=True)
    
    fig.savefig(os.path.join(output_folder, "Ploss_figure.png"), bbox_inches='tight', dpi=600)
    print("Plot saved successfully.")

    plt.show()
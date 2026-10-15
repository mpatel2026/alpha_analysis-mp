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
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_closing

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

MIN_FLUX = 1  # W/m^2; cells below this are not plotted (and not counted as wetted)
HOT_FLUX = 1e6  # W/m^2; contour (in HOT_COLOR) encloses cells above this
HOT_COLOR = 'blue'

def draw_hot_contour(ax, coll, level, res=0.1, linewidth=0.3):
    """Outline the union of the plotted triangles whose value exceeds level.

    The triangles are rasterized onto a res-degree grid and the 0.5 level of
    the resulting mask is contoured.
    """
    vals = np.asarray(coll.get_array())
    tris = np.array([p.vertices[:3] for p, v in zip(coll.get_paths(), vals) if v > level])
    if tris.size == 0:
        return
    lo = tris.reshape(-1, 2).min(axis=0) - 3 * res
    hi = tris.reshape(-1, 2).max(axis=0) + 3 * res
    xg = np.arange(lo[0], hi[0] + res, res)
    yg = np.arange(lo[1], hi[1] + res, res)
    mask = np.zeros((yg.size, xg.size), dtype=bool)
    for t in tris:
        i0, i1 = np.searchsorted(xg, [t[:, 0].min(), t[:, 0].max()])
        j0, j1 = np.searchsorted(yg, [t[:, 1].min(), t[:, 1].max()])
        X, Y = np.meshgrid(xg[i0:i1], yg[j0:j1])
        if X.size:
            inside = MplPath(t).contains_points(np.c_[X.ravel(), Y.ravel()])
            mask[j0:j1, i0:i1] |= inside.reshape(X.shape)
        # Make sure even sub-pixel triangles are marked
        c = t.mean(axis=0)
        mask[np.argmin(abs(yg - c[1])), np.argmin(abs(xg - c[0]))] = True
    # Close 1-pixel slits between neighbouring triangles
    mask = binary_closing(mask, structure=np.ones((3, 3)))
    ax.contour(xg, yg, mask.astype(float), levels=[0.5], colors=HOT_COLOR, linewidths=linewidth)

def plot_walload(fn, name, fig=None, zoom=None):
    """zoom: optional ((phi_min, phi_max), (theta_min, theta_max)) in degrees; the zoomed
    region is also outlined with a dashed box on the first map."""
    obj = Ascot(fn)
    wetted, area, edepo, _, _ = obj.data.active.getwall_loads()
    flux = (edepo / area).to('W/m**2').v
    show = flux >= MIN_FLUX
    wetted_area = np.sum(area.v[show])
    peak_total_MW = flux.max() / 1e6
    obj.input_init(bfield=True)
    print(f"{name} - Wetted Area (>{MIN_FLUX:.0e} W/m^2): {wetted_area:.2f} m^2, "
          f"Peak Total Power Load: {peak_total_MW:.2f} MW/m^2")

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
    # (a zoomed panel gets its own y-axis, otherwise zooming it would zoom the first map too)
    share_y_with = existing_maps[0] if existing_maps and zoom is None else None
    ax = fig.add_subplot(gs[(current_idx - 1) // cols, (current_idx - 1) % cols], 
                         sharey=share_y_with)
    ax.set_label('wall_map')

    # Handle Y-Label and Ticks for shared axis cleanliness
    if (current_idx - 1) % cols == 0:
        # Only label the y-axis for the first column
        ax.set_ylabel(r"$\theta$ [$\degree$]")
    elif zoom is None:
        # Hide tick labels for the second column to save space
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel("")

    # 4. Master Colorbar
    cax = next((a for a in fig.axes if a.get_label() == 'master_cbar'), None)
    if cax is None:
        cax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) 
        cax.set_label('master_cbar')

    # 5. Plotting
    # Pass only cells above MIN_FLUX so the rest are left blank instead of drawn in the lowest color
    im = obj.data.active.plotwall_torpol_new(qnt=(wetted[show], flux[show]), axes=ax, cmap=cc.cm.fire,
                                             cax=cax, clim=[MIN_FLUX, 6000000])
    # Flip the map across theta = 0 (theta -> -theta)
    coll = ax.collections[-1]
    coll.set_verts([p.vertices[:3] * [1, -1] for p in coll.get_paths()])
    # Thicker line on the zoomed panel, where the hot regions are larger on the page
    draw_hot_contour(ax, ax.collections[-1], HOT_FLUX, linewidth=0.3 if zoom is None else 0.6)
        # Handle Y-Label and Ticks for shared axis cleanliness
    if (current_idx - 1) % cols == 0:
        # Only label the y-axis for the first column
        ax.set_ylabel(r"$\theta$ [$\degree$]")
    elif zoom is None:
        # Hide tick labels for the second column to save space
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel("")

    ax.set_xlabel(r"$\phi$ [$\degree$]")

    if zoom is not None:
        (x0, x1), (y0, y1) = zoom
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks(np.linspace(x0, x1, 5))
        ax.set_yticks(np.linspace(y0, y1, 5))
        ax.set_ylabel("")  # drop a5py's default label; it would collide with the left panel
        if existing_maps:
            existing_maps[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                 edgecolor='k', linestyle='--', linewidth=0.8))
    
    mappable = im if im is not None else ax.collections[0]
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r"$P_{surface}$ [W/m$^2$]")
    # Mark the contour level on the colorbar
    cbar.ax.axhline(HOT_FLUX, color=HOT_COLOR, linewidth=1.5)

    # Explicitly avoid plt.tight_layout() as it resets the manual gridspec spacing
    fig.subplots_adjust(right=0.88, left=0.1, wspace=0.05)

    return fig


def alpha_postprocess(input_dir, filenames, simnames, zooms=None, suffix="", eq_dir=None, eq_names=None):
    """
    Processes multiple Ascot and DESC files, generating, saving, and displaying plots.
    """
    
    # 1. Initialize "State" variables
    wall_fig = None

    colors = plt.cm.tab10.colors

    # 2. Iterate through Ascot simulations
    if zooms is None:
        zooms = [None] * len(filenames)
    for i, (fn_tail, s_name, zoom) in enumerate(zip(filenames, simnames, zooms)):
            fn = os.path.join(input_dir, fn_tail)
            # Pick a color based on the index (loops back if more than 10 sims)
            current_color = colors[i % len(colors)]
            
            print(f"Processing Ascot: {s_name}...")
            
            # Grid Plots
            wall_fig = plot_walload(fn, name=s_name, fig=wall_fig, zoom=zoom)
        


    # 4. Save the Figures
    print("\nSaving figures...")
    
    output_folder = "figs/"
    os.makedirs(output_folder, exist_ok=True)

    if wall_fig:
        wall_fig.savefig(os.path.join(output_folder, f"wall_torpol{suffix}.png"), bbox_inches='tight', dpi=600)
        
        

    print("All plots saved successfully.")

    # 5. Display
    plt.show()
    return

inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/ISHW/"
filenames = [
             #"G1600-free-reopt_I_LCFS-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_zero-profile_05182026.h5",
             "G1600-free-reopt_I_30cm_1M-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_10072026.h5",
             #"G1600-free-reopt_I_20cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_05132026.h5",
             #"G1600-free-reopt_I_30cm-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_04082026.h5"
             "G1600-free-reopt_I_eng-wall_1M-mrk_biot-bfield_p01cellarea_FLR-mode_decay-profile_10072026.h5",
             ]

simnames = [
            #"LCFS",
            "engwall decay",
            #"20cm ofset",
            #"30 cm"
            "engwall flat",
            ]

# None = full map; ((phi_min, phi_max), (theta_min, theta_max)) = zoomed panel
zooms = [None, ((30, 150), (-150, -30))]

alpha_postprocess(inp_dir, filenames, simnames, zooms=zooms)

# Engineering wall, LCFS-born markers (full map + zoom) -> figs/wall_torpol_engwall.png
engwall_fn = "G1600-free-reopt_I-engwall-LCFSmark1M_biot-bfield_p01cellarea_FO-mode_zero-profile_09302026.h5"
alpha_postprocess(inp_dir, [engwall_fn, engwall_fn], ["eng. wall", "eng. wall (zoom)"],
                  zooms=[None, ((30, 150), (-150, -30))], suffix="_engwall")

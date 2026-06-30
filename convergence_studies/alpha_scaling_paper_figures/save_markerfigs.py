import numpy as np
import unyt
import matplotlib.pyplot as plt
from a5py import Ascot

plt.rcParams.update({
    'font.size': 8,          # General font size
    'axes.labelsize': 8,     # x and y labels
    'axes.titlesize': 8,     # Title size
    'xtick.labelsize': 7,    # x-axis tick labels
    'ytick.labelsize': 7,    # y-axis tick labels
    'legend.fontsize': 8,     # Legend size
    'font.family': 'sans-serif' # A clean, professional serif font
})

input_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
ascot = Ascot(input_dir + "test_loss_sampling3.h5")
theta_init = ascot.data.active.getstate("thetamod", state="ini")
phi_init = ascot.data.active.getstate("phimod", state="ini")
rho_init = ascot.data.active.getstate("rho", state="ini")
weight_init = ascot.data.active.getstate("weight", state="ini")
Ekin_init = ascot.data.active.getstate("ekin", state="ini")
pitch_init = ascot.data.active.getstate("pitch", state="ini")
Ekin_init = Ekin_init.to("MeV")

end_conditions = ["none", "wall", "RHOMAX"]
lost_theta_init = ascot.data.active.getstate("thetamod", state="ini", endcond=end_conditions)
lost_phi_init = ascot.data.active.getstate("phimod", state="ini", endcond=end_conditions)
lost_rho_init = ascot.data.active.getstate("rho", state="ini", endcond=end_conditions)
lost_weight_init = ascot.data.active.getstate("weight", state="ini", endcond=end_conditions)
lost_Ekin_init = ascot.data.active.getstate("ekin", state="ini", endcond=end_conditions)
lost_pitch_init = ascot.data.active.getstate("pitch", state="ini", endcond=end_conditions)
lost_Ekin_init = lost_Ekin_init.to("MeV")



width_inch = 190 / 25.4
height_inch = width_inch * (5/20)

def plot_histogram(data, nbins, qnt, ax = None, weights = None):
    """
    Creates and returns a matplotlib figure showing a histogram of the input data.
    
    Parameters:
    data (list or np.array): The numerical data to be binned.
    nbins (int): The number of bins to use for the histogram.
    qnt (str): The quantity being plotted.
    ax (matplotlib.axes.Axes): The axis object to plot on.
    weights (list or np.array): Weights for each data point.
    
    Returns:
    matplotlib.axes.Axes: The resulting axis object.
    """
    # Create a figure and axis object
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
    
    # Plot the histogram
    # 'edgecolor' helps distinguish the bins visually
    data_range = np.max(data) - np.min(data)
    delta = data_range / nbins
    if data.units.is_dimensionless:
        unit_str = None
    elif str(data.units) in ['deg', 'degree', 'degrees']:
        unit_str = r'^\circ'
    else:
        unit_str = str(data.units)
    if weights is not None:
        # Applying your logic: weights divided by delta
        # colors: skyblue for particles, salmon for markers. 
        ax.hist(data, bins=nbins, color='salmon', edgecolor='black', weights=(weights/delta))
    else:
        ax.hist(data, bins=nbins, color='salmon', edgecolor='black')   
    # Dynamic X-label with unyt
    qnt_label = rf'${qnt}_{{ini}}$'

    if unit_str is None:
        ax.set_xlabel(qnt_label)
        #ax.set_ylabel(rf'$F/d{qnt} \ [1/s]$')
        ax.set_ylabel(rf'$M/d{qnt} \ [markers]$')
    elif unit_str == rf'^\circ':
        # This renders as: qnt_ini [°]
        ax.set_xlabel(rf'{qnt_label} [${unit_str}$]')
        #ax.set_ylabel(rf'$F/d{qnt} \ [1 / (s \cdot {unit_str})]$')
        ax.set_ylabel(rf'$M/d{qnt} \ [markers / {unit_str}]$')
    else:
        # This renders as: qnt_ini [unit]
        ax.set_xlabel(rf'{qnt_label} [{unit_str}]')
        #ax.set_ylabel(rf'$F/d{qnt} \ [1 / (s \cdot {unit_str})]$')
        ax.set_ylabel(rf'$M/d{qnt} \ [markers / {unit_str}]$')
    
    ax.grid(axis='y', alpha=0.75)
    
    return ax


def plot_hist2d(x_data, y_data, nbins, ax = None, weight = None):
    """
    Creates a 2D histogram (heatmap) of two variables.
    
    Parameters:
    x_data (array-like): Data for the x-axis.
    y_data (array-like): Data for the y-axis.
    nbins (int or [int, int]): Number of bins in each dimension.
    ax (matplotlib.axes.Axes): The axis object to plot on.
    weight (array-like): Weights for each data point.
    
    Returns:
    matplotlib.figure.Figure: The generated figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))

    else: 
        fig = ax.get_figure()
    
    # h is the 2D array of counts, xedges and yedges are the bin boundaries
    # image is the QuadMesh object used for the colorbar
    deltax = (np.max(x_data) - np.min(x_data)) / nbins
    deltay = (np.max(y_data) - np.min(y_data)) / nbins
    delta_area = deltay * deltay
    if weight is not None:
        h, xedges, yedges, image = ax.hist2d(
            x_data, 
            y_data, 
            bins=nbins, 
            cmap='inferno',  # viridis for particles, inferno for markers
            weights=(weight / delta_area),
            edgecolor='none',
            rasterized=True
        )
    else:
        h, xedges, yedges, image = ax.hist2d(
            x_data, 
            y_data, 
            bins=nbins, 
            cmap='inferno',  # viridis for particles, inferno for markers
            edgecolor='none',
            rasterized=True
        )

    # Add a colorbar to show the scale of the frequency
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.05)
    
    cbar.ax.yaxis.get_offset_text().set_horizontalalignment('right')
    cbar.ax.yaxis.get_offset_text().set_x(1.5) # Adjust 1.5 to push it further right if needed

    #cbar.set_label(r'F/(d$\lambda$dE) [1/(s $\cdot$ MeV)]', size=9)
    cbar.set_label(r'M/(d$\lambda$dE) [markers/(MeV)]', size=9)
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_xlabel(r'$\lambda_{ini}$')
    ax.set_ylabel(r'$E_{ini}$ (MeV)')
    
    return ax

def normalize_hist(data1, data2, nbins, qnt, ax=None, weights1=None, weights2=None):
    """
    Plots a 1D histogram showing the relative fraction of data1 over data2 per bin.
    Safely handles unyt arrays by extracting their underlying numerical values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))

    # Extract raw numpy arrays to bypass unyt unit enforcement during binning
    d1_raw = getattr(data1, 'value', np.asarray(data1))
    d2_raw = getattr(data2, 'value', np.asarray(data2))
    w1_raw = getattr(weights1, 'value', np.asarray(weights1)) if weights1 is not None else None
    w2_raw = getattr(weights2, 'value', np.asarray(weights2)) if weights2 is not None else None

    # Determine global bin edges across both datasets to keep bins perfectly aligned
    min_val = min(float(np.min(d1_raw)), float(np.min(d2_raw)))
    max_val = max(float(np.max(d1_raw)), float(np.max(d2_raw)))
    edges = np.linspace(min_val, max_val, nbins + 1)

    # Compute individual histograms using raw data
    hist1, _ = np.histogram(d1_raw, bins=edges, weights=w1_raw)
    hist2, _ = np.histogram(d2_raw, bins=edges, weights=w2_raw)

    # Safely compute the relative ratio (data1 / data2), avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(hist2 > 0, hist1 / hist2, 0.0)

    # Plot the relative fraction using a bar chart
    width = np.diff(edges)
    ax.bar(edges[:-1], ratio, width=width, align='edge', color='mediumpurple', edgecolor='black')

    # Replicate unyt dynamic labeling logic using the original array metadata
    if hasattr(data2, 'units') and not data2.units.is_dimensionless:
        if str(data2.units) in ['deg', 'degree', 'degrees']:
            unit_str = r'^\circ'
            ax.set_xlabel(rf'${qnt}_{{ini}}$ [${unit_str}$]')
        else:
            unit_str = str(data2.units)
            ax.set_xlabel(rf'${qnt}_{{ini}}$ [{unit_str}]')
    else:
        ax.set_xlabel(rf'${qnt}_{{ini}}$')

    ax.set_ylabel(r'Fraction ($M_{lost} / M_{all}$)')
    ax.grid(axis='y', alpha=0.75)

    return ax


def normalize_hist2d(x_data1, y_data1, x_data2, y_data2, nbins, ax=None, weight1=None, weight2=None):
    """
    Plots a 2D histogram heatmap showing the relative fraction of dataset 1 over dataset 2.
    Safely handles unyt arrays by extracting their underlying numerical values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
    else:
        fig = ax.get_figure()

    # Extract raw numpy arrays to bypass unyt unit enforcement during binning
    xd1_raw = getattr(x_data1, 'value', np.asarray(x_data1))
    yd1_raw = getattr(y_data1, 'value', np.asarray(y_data1))
    xd2_raw = getattr(x_data2, 'value', np.asarray(x_data2))
    yd2_raw = getattr(y_data2, 'value', np.asarray(y_data2))
    w1_raw = getattr(weight1, 'value', np.asarray(weight1)) if weight1 is not None else None
    w2_raw = getattr(weight2, 'value', np.asarray(weight2)) if weight2 is not None else None

    # Determine global boundaries for both dimensions
    x_min = min(float(np.min(xd1_raw)), float(np.min(xd2_raw)))
    x_max = max(float(np.max(xd1_raw)), float(np.max(xd2_raw)))
    y_min = min(float(np.min(yd1_raw)), float(np.min(yd2_raw)))
    y_max = max(float(np.max(yd1_raw)), float(np.max(yd2_raw)))

    if isinstance(nbins, int):
        xbins = ybins = nbins
    else:
        xbins, ybins = nbins

    xedges = np.linspace(x_min, x_max, xbins + 1)
    yedges = np.linspace(y_min, y_max, ybins + 1)

    # Compute 2D histograms using raw data
    h1, _, _ = np.histogram2d(xd1_raw, yd1_raw, bins=[xedges, yedges], weights=w1_raw)
    h2, _, _ = np.histogram2d(xd2_raw, yd2_raw, bins=[xedges, yedges], weights=w2_raw)

    # Safely divide maps
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(h2 > 0, h1 / h2, 0.0)

    # Draw the heatmap (transposed to match pcolormesh mesh orientation requirements)
    image = ax.pcolormesh(xedges, yedges, ratio.T, cmap='viridis', edgecolor='none', rasterized=True)

    # Add colorbar
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.05)
    cbar.ax.yaxis.get_offset_text().set_horizontalalignment('right')
    cbar.ax.yaxis.get_offset_text().set_x(1.5)
    cbar.set_label(r'Fraction ($M_{lost} / M_{all}$)', size=9)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel(r'$\lambda_{ini}$')
    
    if hasattr(y_data2, 'units') and str(y_data2.units) == 'MeV':
        ax.set_ylabel(r'$E_{ini}$ (MeV)')
    else:
        ax.set_ylabel(r'$E_{ini}$')

    return ax


def get_histogram_data(data, bin_edges, weights=None):
    """
    Computes the histogram of the input data using provided bin edges.
    
    Parameters:
    data (list or np.array): The numerical data to be binned.
    bin_edges (list or np.array): An array defining the bin edges.
    weights (list or np.array, optional): Weights for each data point.
    
    Returns:
    np.array: A 1D array of the values/counts in each bin.
    """
    if weights is not None:
        # Calculate delta assuming uniform bin widths
        delta = bin_edges[1] - bin_edges[0]
        adjusted_weights = weights / delta
    else:
        adjusted_weights = None

    # Calculate histogram using the provided bin edges
    # We use '_' to discard the returned edges since they are already an input
    hist_values, _ = np.histogram(data, bins=bin_edges, weights=adjusted_weights)
    
    return hist_values

"""
fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(width_inch, height_inch), constrained_layout=True)

ax1 = plot_histogram(rho_init, nbins=20, qnt="ρ", ax = ax1, weights=weight_init)
ax2 = plot_histogram(theta_init, nbins=20, qnt="θ", ax = ax2, weights=weight_init)
ax3 = plot_histogram(phi_init, nbins=20, qnt="ϕ", ax = ax3, weights=weight_init)
ax4 = plot_hist2d(pitch_init, Ekin_init, nbins=100, ax = ax4, weight=weight_init)

plt.savefig("figs/marker_init-figures.png", dpi=600)
plt.show()
"""

# 1. Create the main figure
fig = plt.figure(figsize=(width_inch, 2*height_inch), constrained_layout=True)

# 2. Split the figure into 2 rows of "subfigures"
subfigs = fig.subfigures(2, 1)

# 3. Setup Top Row
subfigs[0].suptitle("initial conditions of all markers", fontsize=14)
ax1, ax2, ax3, ax4 = subfigs[0].subplots(1, 4)

# 4. Setup Bottom Row (using raw string 'r' and \mathit for italics)
subfigs[1].suptitle(r"initial conditions of all $\mathit{lost}$ markers", fontsize=14)
ax5, ax6, ax7, ax8 = subfigs[1].subplots(1, 4)

# --- Your Plotting Code Remains the Same ---

ax1 = plot_histogram(rho_init, nbins=20, qnt="ρ", ax=ax1)
ax2 = plot_histogram(theta_init, nbins=20, qnt="θ", ax=ax2)
ax3 = plot_histogram(phi_init, nbins=20, qnt="ϕ", ax=ax3)
ax4 = plot_hist2d(pitch_init, Ekin_init, nbins=100, ax=ax4)

ax5 = plot_histogram(lost_rho_init, nbins=20, qnt="ρ", ax=ax5)
ax6 = plot_histogram(lost_theta_init, nbins=20, qnt="θ", ax=ax6)
ax7 = plot_histogram(lost_phi_init, nbins=20, qnt="ϕ", ax=ax7)
ax8 = plot_hist2d(lost_pitch_init, lost_Ekin_init, nbins=100, ax=ax8)

plt.savefig("figs/initial_conditions-of-all-vs-lost-markers.png", dpi=600)
plt.show()


fig = plt.figure(figsize=(width_inch, 2*height_inch), constrained_layout=True)

# 2. Split the figure into 2 rows of "subfigures"
subfigs = fig.subfigures(2, 1)

# 3. Setup Top Row
subfigs[0].suptitle("initial conditions of all markers", fontsize=14)
ax1, ax2, ax3, ax4 = subfigs[0].subplots(1, 4)

# 4. Setup Bottom Row (using raw string 'r' and \mathit for italics)
subfigs[1].suptitle(r"initial conditions of all particles", fontsize=14)
ax5, ax6, ax7, ax8 = subfigs[1].subplots(1, 4)

# --- Your Plotting Code Remains the Same ---

ax1 = plot_histogram(rho_init, nbins=20, qnt="ρ", ax=ax1)
ax2 = plot_histogram(theta_init, nbins=20, qnt="θ", ax=ax2)
ax3 = plot_histogram(phi_init, nbins=20, qnt="ϕ", ax=ax3)
ax4 = plot_hist2d(pitch_init, Ekin_init, nbins=100, ax=ax4)

ax5 = plot_histogram(rho_init, nbins=20, weights=weight_init, qnt="ρ", ax=ax5)
ax6 = plot_histogram(theta_init, nbins=20, weights=weight_init, qnt="θ", ax=ax6)
ax7 = plot_histogram(phi_init, nbins=20, weights=weight_init, qnt="ϕ", ax=ax7)
ax8 = plot_hist2d(pitch_init, Ekin_init, nbins=100, weight=weight_init, ax=ax8)
plt.savefig("figs/new_sampling_markers-vs-particles.png", dpi=600)
plt.show()

fig_ratio, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(width_inch, height_inch), constrained_layout=True)
fig_ratio.suptitle("Relative Fraction of Lost Markers / Total Initial Markers", fontsize=12)

# 1D Normalized Histograms (Unweighted representation)
normalize_hist(lost_rho_init, rho_init, nbins=20, qnt="ρ", ax=ax1)
normalize_hist(lost_theta_init, theta_init, nbins=20, qnt="θ", ax=ax2)
normalize_hist(lost_phi_init, phi_init, nbins=20, qnt="ϕ", ax=ax3)

# 2D Normalized Heatmap
normalize_hist2d(lost_pitch_init, lost_Ekin_init, pitch_init, Ekin_init, nbins=100, ax=ax4)

plt.savefig("figs/lost_marker_fraction.png", dpi=600)
plt.show()


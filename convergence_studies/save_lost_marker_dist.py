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
ascot = Ascot(input_dir + "G1600-free-reopt_I_10cm-wall_1000k-mrk_biot-bfield_FLR-mode_new-wall_zero-profile_04092026.h5")
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
import numpy as np

def get_histogram_data(data, bin_edges, weights=None):
    """
    Computes the histogram of the input data using provided bin edges.
    
    Parameters:
    data (list or np.array): The numerical data to be binned.
    bin_edges (list or np.array): An array defining the bin edges.
    weights (list or np.array, optional): Weights for each data point.
    
    Returns:
    np.array: A 1D array of the fractional values in each bin (summing to 1).
    """
    if weights is not None:
        # Calculate delta assuming uniform bin widths
        delta = bin_edges[1] - bin_edges[0]
        adjusted_weights = weights / delta
    else:
        adjusted_weights = None

    # Calculate histogram using the provided bin edges
    hist_values, _ = np.histogram(data, bins=bin_edges, weights=adjusted_weights)
    
    # Normalize the histogram to be a fraction of the whole (relative frequency)
    current_sum = np.sum(hist_values)
    if current_sum != 0:
        # Cast to float to avoid integer truncation, then divide by the total sum
        hist_values = hist_values.astype(float) / current_sum
    else:
        # Avoid ZeroDivisionError if the data/weights are entirely zero or empty
        print("Warning: Initial histogram sum is 0. Cannot normalize.")
            
    return hist_values

nR = 101
nthermal_vel = 10
nenergy = 50
npitch = 1


ascot.input_init(plasma=True)
pls = ascot.data.plasma.active.read()['etemperature'].max() * unyt.eV
mHe4 = 4.002602 * unyt.amu
Ealpha = 3.54 * unyt.MeV
vth_He4 = np.sqrt(2 * pls / mHe4).to('m/s')
vmax = nthermal_vel * vth_He4
Emax = Ealpha + 0.5 * mHe4 * vmax**2
Emin = Ealpha - 0.5 * mHe4 * vmax**2

rho_edges = np.linspace(1e-3, 0.99, nR)
theta_edges = np.linspace(0, 360, 2) * unyt.deg
phi_edges = np.linspace(0, 90, 2) * unyt.deg
ekin_edges = np.linspace(Emin.to('eV').value, Emax.to('eV').value, nenergy) * unyt.eV
pitch_edges = np.linspace(-1.0, 1.0, 2)  # Including endpoints.

rho_hist_lost = get_histogram_data(lost_rho_init, rho_edges)
theta_hist_lost = get_histogram_data(lost_theta_init, theta_edges)
phi_hist_lost = get_histogram_data(lost_phi_init, phi_edges)
ekin_hist_lost = get_histogram_data(Ekin_init, ekin_edges) # use total distribution of ekin
pitch_hist_lost = get_histogram_data(pitch_init, pitch_edges) # use total distribution of pitch

rho_hist_total = get_histogram_data(rho_init, rho_edges)
theta_hist_total = get_histogram_data(theta_init, theta_edges)
phi_hist_total = get_histogram_data(phi_init, phi_edges)
ekin_hist_total = get_histogram_data(Ekin_init, ekin_edges) # use total distribution of ekin
pitch_hist_total = get_histogram_data(pitch_init, pitch_edges) # use total distribution of pitch

rho_hist_normalized = np.divide(
    rho_hist_lost, rho_hist_total, 
    out=np.zeros_like(rho_hist_lost, dtype=float), 
    where=rho_hist_total != 0
)

theta_hist_normalized = np.divide(
    theta_hist_lost, theta_hist_total, 
    out=np.zeros_like(theta_hist_lost, dtype=float), 
    where=theta_hist_total != 0
)

phi_hist_normalized = np.divide(
    phi_hist_lost, phi_hist_total, 
    out=np.zeros_like(phi_hist_lost, dtype=float), 
    where=phi_hist_total != 0
)

ekin_hist_normalized = np.divide(
    ekin_hist_lost, ekin_hist_total, 
    out=np.zeros_like(ekin_hist_lost, dtype=float), 
    where=ekin_hist_total != 0
)

pitch_hist_normalized = np.divide(
    pitch_hist_lost, pitch_hist_total, 
    out=np.zeros_like(pitch_hist_lost, dtype=float), 
    where=pitch_hist_total != 0
)

#AFSI puts dist in order of phi, rho, theta, ekin, pitch
marker_hist = [phi_hist_normalized, rho_hist_normalized, theta_hist_normalized, ekin_hist_total, pitch_hist_total]
marker_edges = {
    "phi": phi_edges,
    "rho": rho_edges,
    "theta": theta_edges,
    "ekin": ekin_edges,
    "pitch": pitch_edges
}

np.save('lost_marker_hist_normalized.npy', np.array(marker_hist, dtype=object))
np.savez_compressed('lost_marker_edges.npz', **marker_edges)
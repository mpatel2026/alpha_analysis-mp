from pickle import TRUE
import unyt
import a5py
import alpha_analysis as aa
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from a5py import Ascot
from functools import reduce
from a5py.ascot5io.options import Opt
from a5py.ascot5io.dist import DistData


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

    
def main():

    # 2. Assign variables from arguments
    equil_path = "/pscratch/sd/m/mpatel26/equil/equil_G1600_DESC_fixed.h5"
    out_dir = "ascot_out"
    create = True
    nmrk = 10000

    # End conditions
    tmax = 2 * unyt.ms
    dt = 1 * unyt.us
    # Simulation mode settings
    mode = 'gc'

    h5file = os.path.join(out_dir, "test_preferential_loss_sampling.h5")
    if os.path.exists(h5file):
        os.remove(h5file)


    # 4. Initialize Simulation Input
    sim = aa.RunItem(
        str(equil_path), 
        path=out_dir, 
        sim_name="test_preferential_loss_sampling", 
        create=create,
        nR=200, nZ=200, nPhi=42
    )
    
    sim.run_afsi(nsymm=4, descfn=str(equil_path), mode='cylindrical')
    test = True
    if test == True:
        particledist = sim.afsi_dist.integrate(True, charge=np.s_[:], time=np.s_[:])
        
        print(f" >> Using AFSI distribution with loss sampling for marker generation.")
        
        """
        #Load and process the 1D histogram
        hist_data = np.load("lost_marker_hist_normalized.npy", allow_pickle=True)
        #normalize each array so that it integrates to equal 1
        normalized_hist_data = np.array([arr / np.sum(arr) for arr in hist_data], dtype=object)
        #Compute the 5D outer product using safe floating-point fractions first
        prob_grid = reduce(np.multiply.outer, [np.asarray(hist, dtype=float) for hist in normalized_hist_data])

        # Scale the final multidimensional grid by nmarkers
        marker_hist = prob_grid
        #marker_hist = particledist.distribution()
        
        markerdist = DistData(marker_hist, phi=particledist.abscissa_edges("phi"), rho=particledist.abscissa_edges("rho"), 
                            theta = particledist.abscissa_edges("theta"),
                            ekin=particledist.abscissa_edges("ekin"), pitch=particledist.abscissa_edges("xi"))
        
        
        
        """
        # Generate markerdist from a rho profile
        rho  = np.linspace(0, 1, 100)
        prob = np.ones((100,))
        prob = (1.0+rho)**3

        sim.a5.input_init(bfield=True, plasma=True)
        markerdist = sim.a5.markergen.rhoto5d(
            rho, prob, particledist.abscissa_edges("r"),
            particledist.abscissa_edges("phi"), particledist.abscissa_edges("z"),
            particledist.abscissa_edges("ekin"), particledist.abscissa_edges("xi"))
        sim.a5.input_free()
        
        
        
        mass = 4.002602 * unyt.amu
        charge = 2.0
        anum = 4
        znum = 2
        sim.a5.input_init(bfield=True, plasma=True)
        MarkerGenerator = a5py.routines.markergen.MarkerGenerator(sim.a5)
        mrk, mrkdist, prtdist = MarkerGenerator.generate(nmrk, mass, charge, anum, znum, particledist, markerdist=markerdist, mode=mode, minweight=1, return_dists=True)
        sim.a5.input_free()
        
        sim.mrk = mrk
        sim.a5.data.create_input(mode, **mrk, activate=True)
    
    else:
        sim.prepare_markers(
            descfn=str(equil_path), 
            nmarkers=nmrk, 
            mode=mode,
            adaptive=True, 
            flr_corrections=True,
            rhomax=None, 
            min_energy=200.0 * unyt.eV,
            enable_collisions=True,
            tmax=tmax, 
            marker_hist="lost_marker_hist_normalized.npy",
        )
        
    

    # Let's now generate the options.
    sim.opts = Opt.get_default()
    if mode == 'gc':
        sim.opts['SIM_MODE'] = 2  # Guiding-center
    else:
        sim.opts['SIM_MODE'] = 1  # Full-orbit

    sim.opts['ENABLE_COULOMB_COLLISIONS'] = 1 # Enable collisions.

    sim.opts['ENABLE_ADAPTIVE'] = 1 # Enable/Disable adaptive method (1/0).

    # In this case we update the adaptive options.
    ada_opts_def = {
        'ADAPTIVE_TOL_ORBIT': 1e-8,
        'ADAPTIVE_MAX_DRHO': 0.1,
        'ADAPTIVE_MAX_DPHI': 0.1,
    }

    for key in ada_opts_def:
        sim.opts[key] = ada_opts_def[key]
    


    sim.opts['ENABLE_FLR_LOSSES'] = 1 # Enable FLR corrections.

    sim.opts['ENABLE_ORBIT_FOLLOWING'] = 1 # Enable orbit following.
    
    sim.opts['ENABLE_MHD'] = 0 # Disable MHD.
    sim.opts['ENABLE_DIST_5D'] = 0 # disable here.
    if hasattr(sim.opts, '_OPT_ENABLE_RF'):
        sim.opts['ENABLE_RF'] = 0 # Disable RF.


    sim.opts["FIXEDSTEP_USE_USERDEFINED"] = 1 # We use a user-defined time step.
    Ealpha = 3.54e6 * unyt.eV
    # Let's guess the time step.
    sim.opts["FIXEDSTEP_USERDEFINED"] = dt.to('s').v

    # Final end conditions.
    sim.opts['ENDCOND_SIMTIMELIM'] = 1 # Flag for the simulation to finish at the MAX_MILEAGE.
    sim.opts['ENDCOND_WALLHIT'] = 1 # Stop simulation when a particle hits the wall.
    sim.opts['ENDCOND_RHOLIM'] = 0 # There is no limit on the rho.
    
    sim.opts['ENDCOND_ENERGYLIM'] = 1 # Enable energy limit end condition.

    min_energy = 200.0 * unyt.eV # eV
    thermal_factor = 2.0
    
    sim.opts["ENDCOND_MIN_ENERGY"] = min_energy.to('eV').v # Energy in eV.
    sim.opts["ENDCOND_MIN_THERMAL"] = thermal_factor # Multiplier for determining the thermal threshold.
    sim.opts['ENDCOND_MAX_MILEAGE'] = tmax.to('s').v # Set the maximum simulation time.
    sim.opts["ENDCOND_LIM_SIMTIME"] = tmax.to('s').v # Set the maximum simulation time.
    
    sim.opts['ENDCOND_MAX_RHO'] = 100.0

    # Orbit writing options.
    sim.opts['ENABLE_ORBITWRITE'] = 0
    sim.opts['ORBITWRITE_NPOINT'] = 1 # How many points to write for the orbit.

    # Writing the options.
    sim.a5.data.create_input('opt', **sim.opts, activate=True)



    #Rescale marker weights so that total power output matches helios FPP predictions
    new_mrk = sim.a5.data.marker.active.read()
    weight = new_mrk['weight']
    energy = new_mrk['energy']
    P_tot = np.dot(energy,weight)
    #convert units from ev/s to MW
    EV_S_TO_MW = 1.60218e-25
    P_tot = P_tot * EV_S_TO_MW

    #Calculate current total power from markers

    print(f"Total alpha power from markers before rescaling: {P_tot} MW")
    #rescale the weights so that the total power matches FPP predictions
    P_fpp = 958.0 # MW
    alpha_P = P_fpp * (3.5/17.1) # From fusion power get total power of just alphas
    rescale_P = alpha_P/P_tot
    new_weight = weight * rescale_P
    new_mrk['weight'] = new_weight
    
    #Create a new marker input which is the same as the original markers but with rescaled weights
    sim.a5.data.create_input(mode, **new_mrk, activate=True)
    
    new_P_alpha = np.dot(energy,new_weight)
    new_P_alpha = new_P_alpha * EV_S_TO_MW

    print(f"Rescaled marker weights so alpha power matches FPP predictions. Alpha Power: {new_P_alpha}")

    
    print(f"Simulation test_preferential_loss_sampling setup complete.")
    
    
    #Run the sim and save the distribution
    sim.a5.simulation_initinputs()
    mrk = sim.a5.data.marker.active.read()
    sim.a5.simulation_initmarkers(**mrk)

    # Options input can also be anything but here we just use the on ascot.h5
    opt = sim.a5.data.options.active.read()
    sim.a5.simulation_initoptions(**opt)
    print("starting simulation")
    vrun = sim.a5.simulation_run()
    print("simulation finished")
    theta_init = vrun.getstate("thetamod", state="ini")
    phi_init = vrun.getstate("phimod", state="ini")
    rho_init = vrun.getstate("rho", state="ini")
    weight_init = vrun.getstate("weight", state="ini")
    Ekin_init = vrun.getstate("ekin", state="ini")
    pitch_init = vrun.getstate("pitch", state="ini")
    Ekin_init = Ekin_init.to("MeV")

    width_inch = 190 / 25.4
    height_inch = width_inch * (5/20)


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
    plt.savefig("ascot_out/new_sampling_markers-vs-particles.png", dpi=600)
    plt.show()

    sim.a5.simulation_free(inputs=True, markers=True, diagnostics=True)

if __name__ == "__main__":
    main()

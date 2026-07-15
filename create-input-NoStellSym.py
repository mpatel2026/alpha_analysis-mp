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

def main():
    # 1. Set up the Argument Parser
    parser = argparse.ArgumentParser(description="Create input for Alpha Analysis simulations.")

    # File names
    parser.add_argument("--equil_name", type=str, default="equil_Helios_G1600-12-89_QA2e-1_Bxdl25_free_L12_M12_N20.h5")
    parser.add_argument("--encircling_name", type=str, default=None)
    parser.add_argument("--shaping_name", type=str, default=None)
    parser.add_argument("--sim_name", type=str, required=True, help="Name of the simulation (required)")

    # Simulation parameters
    parser.add_argument("--nmrk", type=int, default=100000, help="Number of markers")
    parser.add_argument("--wall_offset", type=float, default=10.0, help="Wall offset in cm")
    parser.add_argument("--cell_area", type=float, default=0.02, help="Cell area in m^2")

    # Directories
    parser.add_argument("--inp_dir", type=str, default='/pscratch/sd/m/mpatel26/equil/')
    parser.add_argument("--out_dir", type=str, default='/pscratch/sd/m/mpatel26/ascot_h5s/')
    parser.add_argument("--rescale_FPP", type=float, default=958.0, help="Rescale marker weights so that fusion power matches input")
    parser.add_argument('--use_mixed_field', type=bool, default=False, help="If true use eq.compute + Biot-Savart method for bfield. Otherwise Biot-Savart will be used everywhere.")
    parser.add_argument('--collect_dist', type=bool, default=False, help="If true, collect distribution function data during marker preparation. This can significantly increase memory usage and should only be used for testing.")
    parser.add_argument("--sol_profile", type=str, default='decay', help= "set to 'decay' for exponential decay sol plasma profile, 'flat' for sol profile to be lcfs profile, 'zero' to use the original alpha analysis template which has zeros outside")
    parser.add_argument("--simmode", type=str, default = 'gc', help="set to 'gc' for guiding center mode or 'prt' for full orbit mode")
    args = parser.parse_args()

    # 2. Assign variables from arguments
    equil_path = Path(args.inp_dir) / args.equil_name
    encircling_path = None
    if args.encircling_name is not None:
        encircling_path = str(Path(args.inp_dir) / "coils" / args.encircling_name)

    shaping_path = None
    if args.shaping_name is not None:
        shaping_path = str(Path(args.inp_dir) / "coils" / args.shaping_name)
        
    out_dir = args.out_dir
    create = True

    # 3. File Cleanup logic
    if create:
        file_path = Path(out_dir) / f"{args.sim_name}.h5"
        if file_path.exists():
            print(f"Removing existing file: {file_path}")
            file_path.unlink(missing_ok=True)

    # End conditions
    thermal_energy = 200.0 * unyt.eV
    tmax = 300 * unyt.ms
    # Simulation mode settings
    mode = args.simmode
    if mode == 'gc':
        adaptive = True 
        flr_corrections = True
    else:
        adaptive = False
        flr_corrections = False


    if args.collect_dist == True:
        dist_config = {
        # Distribution output
        #"ENABLE_DIST_5D":1, 
        "ENABLE_DIST_RHO5D":1,
        # (R,z) abscissae for the 5D distribution
        #"DIST_MIN_R":4.3,  "DIST_MAX_R":8.3, "DIST_NBIN_R":50,
        #"DIST_MIN_Z":-2.0, "DIST_MAX_Z":2.0, "DIST_NBIN_Z":50,
        # (rho, theta) abscissae for the rho5D distribution. Most of the time a single
        # theta slot is sufficient but please verify it in your case.
        "DIST_MIN_RHO"  :0, "DIST_MAX_RHO"  :1.0, "DIST_NBIN_RHO"  :100,
        "DIST_MIN_THETA":0, "DIST_MAX_THETA":360, "DIST_NBIN_THETA":1,
        # Single phi slot since this is not a stellarator.
        # These values are shared between other distributions
        "DIST_MIN_PHI":0,        "DIST_MAX_PHI":360,     "DIST_NBIN_PHI":1,
        # The momentum abscissae are shared by 5D distributions
        "DIST_MIN_PPA":-1.3e-19, "DIST_MAX_PPA":1.3e-19, "DIST_NBIN_PPA":100,
        "DIST_MIN_PPE":0,        "DIST_MAX_PPE":1.3e-19, "DIST_NBIN_PPE":50,
        # One time slot, the span doesn't matter as long as it covers the whole simulation time
        "DIST_MIN_TIME":0,       "DIST_MAX_TIME":1.0,    "DIST_NBIN_TIME":1,
        # One charge slot exactly at q=2 since we are simulating alphas
        "DIST_MIN_CHARGE":1,     "DIST_MAX_CHARGE":3,    "DIST_NBIN_CHARGE":1,
    }
        
    else: 
        dist_config = {}

    # 4. Initialize Simulation Input
    sim = aa.RunItem(
        str(equil_path), 
        path=out_dir, 
        sim_name=args.sim_name, 
        create=create,
        nR=200, nZ=200, nPhi=192, 
        wall_offset=args.wall_offset,
        use_stell_sym = False, 
        fn_shaping=shaping_path, 
        fn_encircling=encircling_path,
        cell_area=args.cell_area,
        #use_mixed_field=args.use_mixed_field,
        use_mixed_field=False,
        sol_profile=args.sol_profile
    )
    
    sim.run_afsi(nsymm=1, descfn=str(equil_path), mode='magnetic')

    sim.prepare_markers(
        descfn=str(equil_path), 
        nmarkers=args.nmrk, 
        mode=mode,
        adaptive=adaptive, 
        flr_corrections=flr_corrections,
        rhomax=None, 
        min_energy=thermal_energy,
        enable_collisions=True,
        tmax=tmax, 
        **dist_config
    )

    #Rescale marker weights so that total power output matches helios FPP predictions
    new_mrk = sim.a5.data.marker.active.read()
    weight = new_mrk['weight']
    if mode == 'gc':
        energy = new_mrk['energy']
        P_tot = np.dot(energy,weight)
        #convert units from ev/s to MW
        EV_S_TO_MW = 1.60218e-25
        P_tot = P_tot * EV_S_TO_MW

    if mode == 'prt':
        P_tot = 45.5 #MW  (Default MW from alpha analysis AFSI)
        
    #Calculate current total power from markers

    print(f"Total alpha power from markers before rescaling: {P_tot} MW")
    #rescale the weights so that the total power matches FPP predictions
    P_fpp = args.rescale_FPP # MW
    alpha_P = P_fpp * (3.5/17.1) # From fusion power get total power of just alphas
    rescale_P = alpha_P/P_tot
    new_weight = weight * rescale_P
    new_mrk['weight'] = new_weight
    
    #Create a new marker input which is the same as the original markers but with rescaled weights
    sim.a5.data.create_input(mode, **new_mrk, activate=True)
    
    if mode == 'gc':
        new_P_alpha = np.dot(energy,new_weight)
        new_P_alpha = new_P_alpha * EV_S_TO_MW
    if mode == 'prt':
        new_P_alpha = 45.5 * rescale_P #MW

    print(f"Rescaled marker weights so alpha power matches FPP predictions. Alpha Power: {new_P_alpha}")

    
    print(f"Simulation {args.sim_name} setup complete.")

if __name__ == "__main__":
    main()

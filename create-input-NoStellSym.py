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
import json

def main():
    # 1. Set up the Argument Parser to only take the JSON parameters file
    parser = argparse.ArgumentParser(description="Create input for Alpha Analysis simulations from a JSON config.")
    parser.add_argument("--params", type=str, required=True, help="Path to the JSON configuration file")
    args = parser.parse_args()

    # 2. Load and parse the JSON file
    with open(args.params, "r") as f:
        config = json.load(f)

    # Extract required parameters with default fallbacks
    inp_dir = config.get("inp_dir", None)
    out_dir = config.get("out_dir", None)
    equil_name = config.get("equil_file")
    encircling_name = config.get("EC_file", None)
    shaping_name = config.get("SC_file", None)
    sim_name = config["sim_name"]  # Required

    nmrk = config.get("nmrk")
    wall_offset = config.get("wall_offset", None)
    cell_area = config.get("cell_area", None)
    rescale_FPP = config.get("FPP_power", None)
    use_mixed_field = config.get("use_mixed_field", False)
    collect_dist = config.get("collect_dist", False)
    sol_profile = config.get("SOL_profile", None)
    simmode = config.get("simmode", "gc")
    wall_file = config.get("wall_file", None)

    # 3. Assign file paths
    equil_path = Path(inp_dir) / equil_name
    
    encircling_path = None
    if encircling_name is not None:
        encircling_path = str(Path(inp_dir) / "coils" / encircling_name)

    shaping_path = None
    if shaping_name is not None:
        shaping_path = str(Path(inp_dir) / "coils" / shaping_name)
        
    create = True

    # 4. File Cleanup logic
    if create:
        file_path = Path(out_dir) / f"{sim_name}.h5"
        if file_path.exists():
            print(f"Removing existing file: {file_path}")
            file_path.unlink(missing_ok=True)

    # End conditions
    thermal_energy = 200.0 * unyt.eV
    tmax = 300 * unyt.ms
    
    # Simulation mode settings
    mode = simmode
    if mode == 'gc':
        adaptive = True 
        flr_corrections = True
    else:
        adaptive = False
        flr_corrections = False

    if collect_dist == True:
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

    # 5. Initialize Simulation Input
    sim = aa.RunItem(
        str(equil_path), 
        path=out_dir, 
        sim_name=sim_name, 
        create=create,
        nR=200, nZ=200, nPhi=192, 
        wall_offset=wall_offset,
        use_stell_sym = False, 
        fn_shaping=shaping_path, 
        fn_encircling=encircling_path,
        cell_area=cell_area,
        use_mixed_field=use_mixed_field,
        sol_profile=sol_profile,
        fn_wall=wall_file
    )
    
    sim.run_afsi(nsymm=1, descfn=str(equil_path), mode='magnetic')

    sim.prepare_markers(
        descfn=str(equil_path), 
        nmarkers=nmrk, 
        mode=mode,
        adaptive=adaptive, 
        flr_corrections=flr_corrections,
        rhomax=None, 
        min_energy=thermal_energy,
        enable_collisions=True,
        tmax=tmax, 
        **dist_config
    )

    # Rescale marker weights so that total power output matches helios FPP predictions
    new_mrk = sim.a5.data.marker.active.read()
    weight = new_mrk['weight']
    if mode == 'gc':
        energy = new_mrk['energy']
        P_tot = np.dot(energy, weight)
        # convert units from ev/s to MW
        EV_S_TO_MW = 1.60218e-25
        P_tot = P_tot * EV_S_TO_MW

    if mode == 'prt':
        P_tot = 45.5  # MW  (Default MW from alpha analysis AFSI)
        
    # Calculate current total power from markers
    print(f"Total alpha power from markers before rescaling: {P_tot} MW")
    
    # rescale the weights so that the total power matches FPP predictions
    P_fpp = rescale_FPP  # MW
    alpha_P = P_fpp * (3.5 / 17.1)  # From fusion power get total power of just alphas
    rescale_P = alpha_P / P_tot
    new_weight = weight * rescale_P
    new_mrk['weight'] = new_weight
    
    # Create a new marker input which is the same as the original markers but with rescaled weights
    sim.a5.data.create_input(mode, **new_mrk, activate=True)
    
    if mode == 'gc':
        new_P_alpha = np.dot(energy, new_weight)
        new_P_alpha = new_P_alpha * EV_S_TO_MW
    if mode == 'prt':
        new_P_alpha = 45.5 * rescale_P  # MW

    print(f"Rescaled marker weights so alpha power matches FPP predictions. Alpha Power: {new_P_alpha}")
    print(f"Simulation {sim_name} setup complete.")

if __name__ == "__main__":
    main()
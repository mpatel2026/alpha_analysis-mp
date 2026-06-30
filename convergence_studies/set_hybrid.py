from a5py import Ascot
from a5py.ascot5io.options import Opt
import json
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description="Create input for Alpha Analysis simulations.")
    parser.add_argument("--sim_name", type=str, required=True, help="Name of the simulation (required)")

    args = parser.parse_args()
    inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
    sim_name = args.sim_name
    a = Ascot(inp_dir + f"{sim_name}.h5")
    opts = "../ascot_prep_notebooks/opts_jsons/opts_alpha_analysis_simmode3.json"
    opt = Opt.get_default()
    opts = json.load(open(opts))
    opt.update(opts)
    desc = {1: "full", 2: "Guiding center", 3: "Hybrid"}[opts["SIM_MODE"]]
    a.data.create_input("opt", **opt, desc=desc, activate=True)
    logging.info("Successfully set simulation options in ASCOT file.")  
    options = a.data.active.options.read()
    print("options: ", options)
    print("sim_mode: ", options["SIM_MODE"])


if __name__ == "__main__":
    main()

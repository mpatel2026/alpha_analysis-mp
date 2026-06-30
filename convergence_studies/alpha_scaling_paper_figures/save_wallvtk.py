import argparse
from a5py import Ascot

def main():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="Convert ASCOT 3D mesh to VTK format.")

    # 2. Add your arguments
    # We'll use --flags for clarity, but you can make them positional too
    parser.add_argument("-i", "--input", required=True, help="The input filename (e.g., input.h5)")
    parser.add_argument("-o", "--output", required=True, help="The output filename (e.g., mesh.vtk)")
    parser.add_argument("-d", "--dir", default="/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/", 
                        help="Base directory for input files")

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Use the arguments in your logic
    ascot = Ascot(args.dir + args.input)
    mesh = ascot.data.active.getwall_3dmesh()
    
    # Ensure the directory exists or just save to the path
    mesh.save("vtks/" + args.output)
    print(f"Successfully saved mesh to vtks/{args.output}")

if __name__ == "__main__":
    main()
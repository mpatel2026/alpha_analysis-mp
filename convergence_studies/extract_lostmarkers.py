from a5py import Ascot
import numpy as np

inp_dir = "/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"

biotsavart_lcfs = Ascot(inp_dir + "G1600-free-reopt_I_LCFS-wall_1000k-mrk_biot-bfield_p01cellarea_FLR-mode_zero-profile_05182026.h5")
end_conditions = ["wall"]
data = biotsavart_lcfs.data.active.getstate("r", "z", "phi", "vr", "vphi", "vz", "weight", "anum", "znum", "mass", "charge", "time", "zeta", 'ids',state="end", endcond=end_conditions)
keys = ["r", "z", "phi", "vr", "vphi", "vz", "weight", "anum", "znum", "mass", "charge", "time", 'ids']
data_dict = dict(zip(keys, data))
total_markers = len(data_dict['weight'])
target_size = 114976

if total_markers >= target_size:
    random_indices = np.random.choice(total_markers, size=target_size, replace=False)
    for key in data_dict:
        data_dict[key] = np.asarray(data_dict[key])[random_indices]
else:
    print(f"Warning: Only {total_markers} markers available, which is less than the target 100,000.")

data_dict['n']=len(data_dict['weight'])
print (f"Extracted {data_dict['n']} markers for the biotsavart_lcfs case.")



np.savez("lostmarkers_biotsavart_lcfs.npz", **data_dict)
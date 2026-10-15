"""Flip the sign of the active bfield input in an ASCOT5 file and save it as a
new active bfield input, leaving the original input untouched in the file."""

import device_setup
device_setup.configure()

from a5py import Ascot

FILENAME = "/pscratch/sd/m/mpatel26/ascot_h5s/Eos10-DDN/ascot_Eos10_G2000_DESC_R325_B50_P12MWbroader_25cmSC_1p45packing_10cmXpt_1250Bxdl_1Wself_DDNDivClaim_longerECs_2bins20-30cm_asymSC_free2.h5"
FIELD_COMPONENT_KEYS = ("br", "bphi", "bz")

simulation = Ascot(FILENAME)

old_qid = simulation.data.bfield.active.get_qid()
btype = simulation.data.bfield.active.get_type()
bfield = simulation.data.bfield.active.read()

print(f"Active bfield input: type={btype}, qid={old_qid}")
print(f"bfield dict keys: {list(bfield.keys())}")

for key in FIELD_COMPONENT_KEYS:
    bfield[key] = -1 * bfield[key]

simulation.data.create_input(btype, **bfield, activate=True)

new_qid = simulation.data.bfield.active.get_qid()
print(f"Created new bfield input with flipped {FIELD_COMPONENT_KEYS} "
      f"and activated it: type={btype}, qid={new_qid}")

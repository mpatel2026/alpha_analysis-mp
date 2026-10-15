"""Build a full-orbit ("prt") marker JSON from the markers that hit the wall in
an earlier guiding-center run.

The source run (e.g. the LCFS-wall simulation) traced GC markers to a simple
wall placed at the LCFS. This script takes the END state of every marker that
ended on the wall and turns each one into a full-orbit particle
(r, phi, z, vr, vphi, vz, ...) that is written as the INITIAL condition of a new
marker input, so a follow-up FO run can trace them through the SOL to the real
wall. The JSON is read by set_mrk.py (which detects the prt layout).

GC -> particle conversion (per marker, all from the GC end state):
  * the momentum vector (pr, pphi, pz) and velocity (vr, vphi, vz) come straight
    from a5py getstate(), which builds them from mu, ppar, B and the gyrophase
    zeta (relativistic, ASCOT's own convention); ekin gives the energy
  * particle position = GC position + (b x p) / (q B)  (the gyroradius vector)
  This reproduces the particle state ASCOT stores at the end of the run
  (rprt, prprt, ...) to ~0.05 mm for the ~94% of markers where that stored state
  is consistent. It is NOT read from the file because for ~6% of the wall-hit
  markers the stored particle momentum is stale (energy off by up to 6x).

Two modes:

  default        one output marker per lost marker, weights unchanged.
                 Lost power is conserved by construction.

  --nmrk N       multiply the lost markers up to exactly N markers. Every lost
                 marker is replicated k_i times (k_i = floor(N/n_lost) or +1,
                 the extras chosen at random), each copy carries weight
                 w_i/k_i, and each copy's position is displaced slightly
                 (Gaussian in R, Z and arc length along phi; velocity is kept,
                 with optional relative jitter of energy and pitch). All weights
                 are then multiplied by one global factor so
                     sum(w_new * E_new) == sum(w_lost * E_lost)
                 i.e. initial power of the new marker set == power that was
                 lost on the wall in the source run.

The final marker order is shuffled (fixed seed) unless --no-shuffle is given: in
source-ID order the file is ordered by birth position and energy, so contiguous
per-rank blocks would have very different costs.

Power is sum(weight * energy[eV]) * 1.602e-19 J/eV, weights in particles/s.

Run in the ascot-develop-mp conda env, e.g.
  python make_wall_marker_json.py --nmrk 1000000 --sigma-pos 0.005 \
      --out wall_markers_1M.json
"""
import argparse
import json
import numpy as np
from a5py import Ascot

EV = 1.602176634e-19
EV_TO_MW = EV * 1e-6
AMU = 1.66053906660e-27
C = 299792458.0
DEFAULT_H5 = ("/pscratch/sd/m/mpatel26/ascot_h5s/convergence_studies/"
              "G1600-free-reopt_I_LCFS-wall_1000k-mrk_biot-bfield_p01cellarea_"
              "FLR-mode_zero-profile_05182026.h5")

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--h5", default=DEFAULT_H5, help="source ASCOT5 file")
parser.add_argument("--run", default=None, help="run qid (default: active run)")
parser.add_argument("--out", default="wall_markers.json")
parser.add_argument("--nmrk", type=int, default=None,
                    help="multiply lost markers up to this many (>= n_lost). "
                         "Omit for one-to-one, unperturbed copy.")
parser.add_argument("--sigma-pos", type=float, default=0.005,
                    help="std of position displacement of copies in m (R, Z, "
                         "and arc length in phi). Default 5 mm.")
parser.add_argument("--sigma-energy", type=float, default=0.0,
                    help="relative std of energy jitter of copies")
parser.add_argument("--sigma-pitch", type=float, default=0.0,
                    help="std of pitch jitter of copies (pitch clipped to [-1,1]); "
                         "pitch is taken w.r.t. B at the GC end position")
parser.add_argument("--zero-time", action="store_true",
                    help="set marker start time to 0 instead of the wall-hit time")
parser.add_argument("--no-shuffle", action="store_true",
                    help="keep source-ID order (copies of a parent adjacent). By default "
                         "the final marker order is randomly permuted so contiguous "
                         "blocks handed to MPI ranks have similar cost.")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

a5 = Ascot(args.h5)
if args.run is not None:
    a5.data[args.run].activate()

# ---- GC end state of wall-hit markers (ASCOT base units made explicit) -------
# For GC end states getstate() itself builds the momentum/velocity vectors from
# mu, ppar, B and the gyrophase zeta, so nothing is recomputed here except the
# gyroradius offset of the particle position from the guiding center.
fields = ["r", "phi", "z", "ekin", "pitch", "ppar", "pnorm", "pr", "pphi", "pz",
          "vr", "vphi", "vz", "br", "bphi", "bz",
          "weight", "mass", "charge", "anum", "znum", "time", "ids"]
units = dict(r="m", phi="deg", z="m", ekin="eV", pitch="dimensionless",
             ppar="kg*m/s", pnorm="kg*m/s", pr="kg*m/s", pphi="kg*m/s", pz="kg*m/s",
             vr="m/s", vphi="m/s", vz="m/s", br="T", bphi="T", bz="T",
             weight="1/s", mass="amu", charge="e", time="s")
vals = a5.data.active.getstate(*fields, state="end", endcond=["wall"])
m = {}
for k, v in zip(fields, vals):
    if k in units and hasattr(v, "to"):
        v = v.to(units[k])
    m[k] = np.asarray(getattr(v, "v", v), dtype=float)
n_lost = m["weight"].size
if n_lost == 0:
    raise SystemExit("no wall-hit markers found")
bad = ~np.all([np.isfinite(m[k]) for k in fields if k != "ids"], axis=0)
if bad.any():
    raise SystemExit(f"{bad.sum()} wall-hit markers have non-finite state")

P_lost = np.sum(m["weight"] * m["ekin"]) * EV_TO_MW
print(f"wall-hit markers: {n_lost}")
print(f"lost power (sum w*E): {P_lost:.6g} MW   total weight: {m['weight'].sum():.6g} /s")

# Particle position = GC position + gyroradius vector (b x p)/(q B), components
# in the local (R, phi, z) basis at the GC.
P = np.array([m["pr"], m["pphi"], m["pz"]])
B = np.array([m["br"], m["bphi"], m["bz"]])
Bn = np.linalg.norm(B, axis=0)
bhat = B / Bn
gyro = np.cross(bhat.T, P.T).T / (m["charge"] * EV * Bn)     # [m]
print(f"max gyroradius = {np.max(np.linalg.norm(gyro, axis=0)):.4f} m")

out = dict(r=m["r"] + gyro[0], z=m["z"] + gyro[2],
           phi=m["phi"] + np.degrees(gyro[1] / m["r"]),
           vr=m["vr"], vphi=m["vphi"], vz=m["vz"], energy=m["ekin"],
           weight=m["weight"].copy(), mass=m["mass"], charge=m["charge"],
           anum=m["anum"], znum=m["znum"], time=m["time"].copy())

if args.nmrk is None:
    n = n_lost
else:
    if args.nmrk < n_lost:
        raise SystemExit(f"--nmrk {args.nmrk} < n_lost {n_lost}")
    k = np.full(n_lost, args.nmrk // n_lost)
    k[rng.choice(n_lost, size=args.nmrk - k.sum(), replace=False)] += 1
    idx = np.repeat(np.arange(n_lost), k)
    n = idx.size
    out = {key: v[idx].copy() for key, v in out.items()}
    out["weight"] = out["weight"] / k[idx]           # total weight conserved

    # optional velocity jitter: energy rescales |p|; pitch keeps the perpendicular
    # direction. Both relative to B at the GC end position.
    if args.sigma_energy > 0 or args.sigma_pitch > 0:
        Pc, bc, mk = P[:, idx], bhat[:, idx], m["mass"][idx] * AMU
        pn, pitch = m["pnorm"][idx], m["pitch"][idx]
        perp = Pc - m["ppar"][idx] * bc
        perp_norm = np.linalg.norm(perp, axis=0)
        if np.any(perp_norm == 0):
            raise SystemExit("marker with zero perpendicular momentum; cannot jitter")
        perp /= perp_norm
        if args.sigma_energy > 0:
            T = out["energy"] * (1 + rng.normal(0, args.sigma_energy, n))
            if np.any(T <= 0):
                raise SystemExit("energy jitter gave non-positive energy; reduce sigma")
            out["energy"] = T
            Tj = T * EV
            pn = np.sqrt(Tj * (Tj + 2 * mk * C**2)) / C      # relativistic |p|
        if args.sigma_pitch > 0:
            pitch = np.clip(pitch + rng.normal(0, args.sigma_pitch, n), -1, 1)
        Pc = pn * pitch * bc + pn * np.sqrt(1 - pitch**2) * perp
        gamma = 1 + out["energy"] * EV / (mk * C**2)
        out["vr"], out["vphi"], out["vz"] = Pc / (gamma * mk)

    # position displacement
    out["r"] = out["r"] + rng.normal(0, args.sigma_pos, n)
    out["z"] = out["z"] + rng.normal(0, args.sigma_pos, n)
    out["phi"] = out["phi"] + np.degrees(rng.normal(0, args.sigma_pos, n) / out["r"])
    if np.any(out["r"] <= 0):
        raise SystemExit("position jitter gave R <= 0; reduce sigma-pos")

    # one global factor: initial power of the new set == lost power of source run
    P_new = np.sum(out["weight"] * out["energy"]) * EV_TO_MW
    scale = P_lost / P_new
    out["weight"] = out["weight"] * scale
    print(f"copies per marker: {k.min()}-{k.max()}; power rescale factor {scale:.8f}")

if not args.no_shuffle:
    perm = rng.permutation(n)
    out = {key: v[perm] for key, v in out.items()}

if args.zero_time:
    out["time"][:] = 0.0
out["phi"] = np.mod(out["phi"], 360.0)      # source phi is unwrapped

P_out = np.sum(out["weight"] * out["energy"]) * EV_TO_MW
print(f"output markers: {n}   initial power: {P_out:.6g} MW "
      f"(rel. diff to lost: {(P_out - P_lost) / P_lost:.2e})")
vmag = np.sqrt(out["vr"]**2 + out["vphi"]**2 + out["vz"]**2)
print(f"|v| range: {vmag.min():.3e} - {vmag.max():.3e} m/s (c = {C:.3e})")

js = {
    "n": [int(n)],
    "id": list(range(1, n + 1)),
    "weight": out["weight"].tolist(),
    "r": out["r"].tolist(),
    "phi": out["phi"].tolist(),
    "z": out["z"].tolist(),
    "vr": out["vr"].tolist(),
    "vphi": out["vphi"].tolist(),
    "vz": out["vz"].tolist(),
    "mass": out["mass"].tolist(),
    "charge": out["charge"].astype(int).tolist(),
    "anum": out["anum"].astype(int).tolist(),
    "znum": out["znum"].astype(int).tolist(),
    "time": out["time"].tolist(),
}
with open(args.out, "w") as f:
    json.dump(js, f)
print("wrote", args.out)

"""Rewrite an ASCOT5 input file's markers to use uniform-in-(rho, theta, phi)
sampling, by construction rather than by resampling.

The markers ASCOT normally starts with are drawn proportional to the AFSI
alpha birth-rate distribution and given near-uniform weight, so hardly any
markers sit near the edge (in rho) or in the less-visited poloidal/toroidal
regions (theta, phi) where losses happen -- this matters especially for
non-axisymmetric devices (stellarators), where the loss physics genuinely
depends on theta and phi, not just rho.

An earlier version of this script bootstrap-resampled whole marker records
from the old set, binned in rho only. That can never be more uniform than
the old markers' own coverage -- bins the old set barely visited stay
noisy/lumpy no matter how much donor pooling is added. This version instead
constructs new marker POSITIONS directly and uniformly on a 3D (rho, theta,
phi) grid, independent of where any old marker happened to be, and then
computes each new marker's WEIGHT so the weighted (physical/particle)
distribution still matches the old one exactly.

How it works
------------
1. Build a uniform grid of NBINS_RHO x NBINS_THETA x NBINS_PHI cells over
   the same (rho, theta, phi) box the old markers span.
2. Split n_old markers exactly evenly across ALL cells (not just cells the
   old markers happened to visit), and within each cell draw its markers'
   (rho, theta, phi) *continuously and uniformly* inside that cell's own
   edges. This alone guarantees the marker distribution is uniform -- it
   does not depend on old-marker density anywhere.
3. Give every new marker in a rho slice weight = (old weighted rate in
   that slice) / (new markers in that slice). See "Weights" below for why
   the rate is shared per rho slice rather than per (rho, theta, zeta)
   cell.
4. Energy, pitch and time still need to come from somewhere physical, so
   (and only these) are bootstrapped from old markers in the same cell,
   weighted by their original weight, widening to neighboring cells
   (periodic in theta/phi) if a cell had too few old markers. This
   widening never touches position or weight, only these three fields.
5. Convert (rho, theta, zeta) to real-space (R, phi, Z) with DESC, using
   the same equilibrium the ASCOT5 input was built from. In DESC these
   *are* the equilibrium's own computational coordinates, so this direction
   is a plain forward evaluation of the R/Z Fourier-Zernike series -- exact,
   no root finding, and it cannot fail to converge.
6. n_new == n_old exactly, since step 2 splits n_old across cells with no
   remainder left over (asserted in code).

Coordinates
-----------
The flux coordinates come from the DESC equilibrium (`inp_dir` +
`equil_file` in the params JSON), not from ASCOT5:

* Going in, the old markers' (R, phi, Z) are mapped to (rho, theta, zeta)
  by DESC's root find (`map_coordinates`).
* Going out, the new markers' (rho, theta, zeta) are mapped to (R, phi, Z)
  by direct forward evaluation, which is where the real win is: an earlier
  version of this script hand-rolled a finite-difference Newton solve
  against ASCOT5's field evaluator for this step, and left ~1% of the new
  markers at NaN (R, Z) wherever it failed to converge. Those NaNs went
  straight into the marker input. The DESC forward map has no such failure
  mode.
* That Newton solve also needed a poloidal angle from ASCOT5, but
  ASCOT5's "theta" is the *Boozer* angle and requires a boozer input to be
  initialized, which this script never did -- so it had been silently
  falling back to a crude geometric arctan2 about the marker *centroid*
  (not even the magnetic axis). DESC's theta is a genuine, well-defined
  poloidal angle.
* DESC's rho is sqrt(normalized toroidal flux) where ASCOT5's is
  sqrt(normalized poloidal flux. For these equilibria the two agree to
  <0.002 in rho (correlation 1.000000 over the whole volume), which stands
  to reason since the ASCOT5 bfield was generated from this very DESC
  equilibrium -- so "uniform in rho" means the same thing either way.
* zeta is DESC's toroidal angle and equals the real geometric phi, so no
  inversion is involved in that coordinate at all. Internally this script
  works in radians and converts to degrees only when reading from / writing
  to the ASCOT5 file.

The magnetic field still comes from ASCOT5, not DESC: pitch has to be
defined against the field the simulation will actually integrate. So DESC
supplies geometry, ASCOT5 supplies B.

The new markers are written as a new, active marker group; the original
group is kept in the file, just marked inactive.

Weights
-------
Weight is assigned per rho slice, not per (rho, theta, zeta) cell.

Per-cell weighting is the obvious thing to do and it is what this script
did originally, but it has a fatal flaw: a cell no old marker landed in has
an old rate of exactly zero, so every new marker placed there gets weight
zero -- and ASCOT5 rejects weight <= 0 outright
(particle_input_to_state() raises ERR_MARKER_UNPHYSICAL). Those empty cells
are concentrated in the outer rho slices, because the alpha birth profile
is peaked and the old sample thins out towards the edge: with 25^3 cells,
a rho slice holding a few hundred old markers cannot fill 625 (theta, zeta)
cells. On the 100k G1600 cases this silently threw away ~14-16% of the new
markers, over 80% of them in the outermost rho slices -- precisely the
region the script exists to populate.

The fix rests on a measurement, not an assumption. Within a rho slice the
old weighted rate turns out to be *flat* in (theta, zeta):

  * per rho slice, the cell-to-cell scatter of the old rate matches the
    Poisson prediction for that slice's marker count to within a few
    percent, all the way from rho=0.04 to rho=0.9;
  * splitting the old markers into two independent halves and correlating
    the two (theta, zeta) shapes against each other gives -0.03, i.e. there
    is no reproducible structure to reproduce;
  * the theta and zeta marginals are uniform to 1.5-2%;
  * L1 distance to a flat shape is smaller than to any alternative tried,
    including the DESC Jacobian sqrt(g) -- so the rate does NOT follow cell
    volume, and the markers are simply uniform in (theta, zeta).

So per-cell weighting was not preserving real structure, it was copying the
old sample's counting noise into the new weights while also zeroing out the
edge. Sharing each rho slice's rate equally over its new markers preserves
the rho profile and the total weight exactly, adds no noise, and leaves no
marker at zero weight as long as every rho slice contains at least one old
marker (the script warns if one does not). Pass --weight-per-cell to get
the old behaviour back for comparison.

Marker types
------------
Both guiding-center ("gc") and full-orbit ("prt") inputs are accepted, and
the new group is written as whatever type the old one was. Everything above
is phrased in terms of energy and pitch, which is what "gc" markers carry
directly. A "prt" marker carries (vr, vphi, vz) instead, so on the way in
its energy and pitch are recovered from B at its own position, and on the
way out the bootstrapped energy/pitch are turned back into a velocity vector
using B at the marker's NEW position (plus the random gyro-angle). Copying
the donors' velocity components verbatim would be wrong: the new marker sits
where B points in a different direction, so the pitch this script works so
hard to preserve would not survive the move.

Usage
-----
    python uniformize-markers.py <path/to/file.h5>
"""
import device_setup
device_setup.configure()

import os
import sys

import numpy as np
import unyt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import argparse
import a5py
from a5py import Ascot
from a5py.physlib import energy_velocity, gamma_energy, vnorm_gamma
import desc.io as dscio
from desc.equilibrium.coords import map_coordinates

rng = np.random.default_rng()

# ---------------------------------------------------------------- settings
# Number of uniform bins per dimension of the (rho, theta, phi) grid. Total
# grid cells = NBINS_RHO * NBINS_THETA * NBINS_PHI. Unlike the bootstrap
# approach, position/weight fidelity does NOT degrade as this grows (no
# donor-pool starvation for position anymore) -- the trade-off is purely
# resolution vs. how many markers per cell you can afford (n_old / NCELLS).
NBINS_RHO = 25
NBINS_THETA = 25 
NBINS_PHI = 25

# Minimum number of distinct old-marker donors a grid cell should bootstrap
# energy/pitch/time from. If a cell's own donor count is below this, the
# donor pool widens to neighboring cells (periodic in theta/phi) until this
# floor is met. This ONLY affects energy/pitch/time -- position and weight
# are exact regardless of MIN_DONORS. Set to 0 or 1 to disable widening.
MIN_DONORS = 50

# Tolerance and iteration cap for DESC's (R, phi, Z) -> (rho, theta, zeta)
# root find on the OLD markers. The new markers' (rho, theta, zeta) ->
# (R, phi, Z) direction is a forward evaluation and needs neither.
DESC_TOL = 1e-8
DESC_MAXITER = 50

FIGDIR = "figures"

# Species the new markers are written as (alphas). The same mass is used to
# turn bootstrapped energies back into speeds for "prt" output, so that the
# energy the marker is given and the energy ASCOT5 will read back agree.
ALPHA_ANUM = 4
ALPHA_ZNUM = 2
ALPHA_MASS_AMU = 4.002602
ALPHA_CHARGE_E = 2.0


def load_desc_equilibrium(fn):
    """Load a DESC Equilibrium, taking the last member if the file holds an
    EquilibriaFamily (same convention as set_field.py / create-input.py).
    """
    fam = dscio.load(fn, file_format="hdf5")
    try:
        return fam[-1]
    except (TypeError, IndexError, KeyError):
        return fam


def rpz_to_flux(eq, r, phi_rad, z):
    """(R, phi, Z) -> (rho, theta, zeta) for the OLD markers, via DESC's
    root find. theta comes back wrapped to [0, 2*pi).

    Markers outside the equilibrium volume come back NaN and the caller
    drops them. They are only ever donors, so losing a few is harmless --
    unlike the old Newton solve, nothing that fails here can end up written
    to the marker input.
    """
    coords = np.stack([r, phi_rad, z], axis=1)
    out = np.asarray(map_coordinates(
        eq, coords, inbasis=("R", "phi", "Z"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
        tol=DESC_TOL, maxiter=DESC_MAXITER), dtype=np.float64)
    return out[:, 0], np.mod(out[:, 1], 2 * np.pi), out[:, 2]


def flux_to_rpz(eq, rho, theta, zeta):
    """(rho, theta, zeta) -> (R, phi, Z) for the NEW markers.

    These are DESC's own computational coordinates, so this is a direct
    forward evaluation of the equilibrium's R/Z series: exact, and with no
    root find there is nothing that can fail to converge. phi comes back
    equal to zeta.
    """
    coords = np.stack([rho, theta, zeta], axis=1)
    out = np.asarray(map_coordinates(
        eq, coords, inbasis=("rho", "theta", "zeta"),
        outbasis=("R", "phi", "Z")), dtype=np.float64)
    return out[:, 0], out[:, 1], out[:, 2]


def make_edges(x, nbins):
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def make_angle_edges(nbins):
    """Uniform bin edges over a full [0, 2*pi) period, for the angle-valued
    coordinates (theta, zeta).
    """
    edges = np.linspace(0.0, 2 * np.pi, nbins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def digitize_bin(x, edges, nbins):
    return np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)


def unravel_cell(b, nbins_theta, nbins_phi):
    i = b // (nbins_theta * nbins_phi)
    rem = b % (nbins_theta * nbins_phi)
    j = rem // nbins_phi
    k = rem % nbins_phi
    return i, j, k


def periodic_dist(a, b, n):
    """Distance between bin indices a and b on a periodic axis of n bins."""
    d = np.abs(a - b)
    return np.minimum(d, n - d)


def build_donor_pool(b, bin_rho, bin_theta, bin_phi, min_donors,
                     nbins_rho, nbins_theta, nbins_phi, positive):
    """Indices of old markers used as energy/pitch/time donors for target
    cell b (NOT used for position or weight -- see module docstring).

    Starts with donors strictly inside cell b. If that pool holds fewer than
    `min_donors`, widens to a neighborhood around the cell (periodic in
    theta/phi, since those are angles; not periodic in rho) until enough are
    available, or the whole array is covered. Returns (donor_indices,
    radius).

    `positive` is a boolean mask of old markers with weight > 0, and only
    those count toward `min_donors`. The donors are sampled with probability
    proportional to their weight, so a pool of nothing but zero-weight
    markers would divide by zero and hand NaN probabilities to rng.choice.
    That is not hypothetical: a previous run of this script can itself write
    zero-weight markers, and re-running on its own output used to crash here.
    """
    i0, j0, k0 = unravel_cell(b, nbins_theta, nbins_phi)
    radius = 0
    idx = np.flatnonzero((bin_rho == i0) & (bin_theta == j0) &
                         (bin_phi == k0) & positive)
    max_radius = max(nbins_rho, nbins_theta, nbins_phi)
    while idx.size < min_donors and radius < max_radius:
        radius += 1
        idx = np.flatnonzero(
            (np.abs(bin_rho - i0) <= radius) &
            (periodic_dist(bin_theta, j0, nbins_theta) <= radius) &
            (periodic_dist(bin_phi, k0, nbins_phi) <= radius) & positive)
    return idx, radius


def report_zero_weight(w_new, rho_new, nbins=10):
    """Print how many new markers carry zero weight, and where in rho they
    sit. Returns that count.

    A cell that no old marker landed in has an old weighted rate of exactly
    zero, so every new marker placed there gets weight zero. ASCOT5 refuses
    such markers outright -- particle_input_to_state() raises
    ERR_MARKER_UNPHYSICAL on `weight <= 0` for both the prt and gc paths --
    so they are silently dropped at the start of the run. They also cluster
    at large rho, which is precisely the region this script exists to
    populate, so a large count means the run is not doing its job.
    """
    zero = (w_new <= 0)
    n_zero = int(zero.sum())
    if n_zero == 0:
        print("[weight] no zero-weight markers")
        return 0

    print(f"\n[weight] WARNING: {n_zero} of {zero.size} new markers "
          f"({100.0 * zero.mean():.2f}%) have zero weight, because their "
          "(rho, theta, zeta) cell contains no old marker.")
    print("[weight] ASCOT5 rejects weight <= 0 (ERR_MARKER_UNPHYSICAL), so "
          f"only {int((~zero).sum())} markers would actually be simulated.")
    edges = np.linspace(rho_new.min(), rho_new.max(), nbins + 1)
    idx = np.clip(np.digitize(rho_new, edges) - 1, 0, nbins - 1)
    print("[weight]   rho range          markers   zero-weight")
    for i in range(nbins):
        tot = int((idx == i).sum())
        nz = int((zero & (idx == i)).sum())
        frac = 100.0 * nz / tot if tot else 0.0
        print(f"[weight]   {edges[i]:.3f} - {edges[i+1]:.3f}   {tot:8d}   "
              f"{nz:8d} ({frac:5.1f}%)")
    return n_zero


def eval_bvec(a5, r, phi, z, t):
    """(br, bphi, bz) [T] at the given cylindrical positions, as plain
    float64 arrays. Requires a5.input_init(bfield=True) to have been called.
    """
    br, bphi, bz = a5.input_eval(r * unyt.m, phi * unyt.deg, z * unyt.m,
                                 t * unyt.s, "br", "bphi", "bz")
    return (np.asarray(br).astype(np.float64).ravel(),
            np.asarray(bphi).astype(np.float64).ravel(),
            np.asarray(bz).astype(np.float64).ravel())


def donor_energy_pitch(a5, old, mrktype, r, phi, z, t):
    """Energy [eV] and pitch of the OLD markers, whatever their input type.

    Guiding-center markers carry these directly. Full-orbit ("prt") markers
    carry (vr, vphi, vz) instead, so energy and pitch are recovered with the
    same formulas a5py itself uses (Prt.eval_energy / Prt.eval_pitch),
    evaluating B at each donor's own position. Downstream code only ever
    sees energy/pitch, so there is a single shared code path from here on.
    """
    if mrktype == "gc":
        return (np.asarray(old["energy"]).astype(np.float64).ravel(),
                np.asarray(old["pitch"]).astype(np.float64).ravel())

    vr = np.asarray(old["vr"]).astype(np.float64).ravel()
    vphi = np.asarray(old["vphi"]).astype(np.float64).ravel()
    vz = np.asarray(old["vz"]).astype(np.float64).ravel()
    mass = np.asarray(old["mass"]).astype(np.float64).ravel() * unyt.amu

    v = np.sqrt(vr**2 + vphi**2 + vz**2)
    energy = energy_velocity(mass, v * unyt.m / unyt.s).to("eV")

    br, bphi, bz = eval_bvec(a5, r, phi, z, t)
    b = np.sqrt(br**2 + bphi**2 + bz**2)
    pitch = (vr * br + vphi * bphi + vz * bz) / (b * v)
    return np.asarray(energy).astype(np.float64).ravel(), pitch


def energy_pitch_to_velocity(a5, mass_amu, energy, pitch, zeta, r, phi, z, t):
    """(vr, vphi, vz) [m/s] for full-orbit markers, built at the markers' NEW
    positions.

    The donors' raw velocity components cannot simply be copied: the new
    marker sits somewhere else, where B points in a different direction, so
    a copied vector would have the wrong pitch -- exactly the quantity this
    script is trying to preserve. Instead the speed comes from the
    bootstrapped energy, the parallel/perpendicular split from the
    bootstrapped pitch, and the direction from the local field, with the
    gyro-angle zeta placing the perpendicular part around it.
    """
    mass = mass_amu * unyt.amu
    v = np.asarray(vnorm_gamma(gamma_energy(mass, energy * unyt.eV)).to("m/s")
                   ).astype(np.float64).ravel()

    br, bphi, bz = eval_bvec(a5, r, phi, z, t)
    bnorm = np.sqrt(br**2 + bphi**2 + bz**2)
    bhat = np.stack([br, bphi, bz]) / bnorm

    # Two unit vectors spanning the plane perpendicular to bhat. Crossing
    # with whichever basis axis bhat leans on least keeps this well
    # conditioned everywhere.
    axis = np.zeros_like(bhat)
    axis[np.argmin(np.abs(bhat), axis=0), np.arange(bhat.shape[1])] = 1.0
    e1 = np.cross(bhat, axis, axis=0)
    e1 /= np.linalg.norm(e1, axis=0)
    e2 = np.cross(bhat, e1, axis=0)

    vpar = pitch * v
    vperp = np.sqrt(np.maximum(1.0 - pitch**2, 0.0)) * v
    vvec = (vpar * bhat
            + vperp * (np.cos(zeta) * e1 + np.sin(zeta) * e2))
    return vvec[0], vvec[1], vvec[2]


def main():
    # 1. Set up the Argument Parser to only take the JSON parameters file
    parser = argparse.ArgumentParser(description="Create input for Alpha Analysis simulations from a JSON config.")
    parser.add_argument("--params", type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument("--weight-per-cell", action="store_true",
                        help="Weight each new marker by its own (rho, theta, "
                             "zeta) cell's old rate instead of by its rho "
                             "slice. Reproduces the old sample's counting "
                             "noise and gives zero weight to markers in "
                             "cells no old marker reached.")
    parser.add_argument("--allow-zero-weight", action="store_true",
                        help="Write the new marker group even if some "
                             "markers have zero weight. ASCOT5 rejects those "
                             "markers (ERR_MARKER_UNPHYSICAL), so by default "
                             "the script reports them and refuses to write.")
    args = parser.parse_args()

    # 2. Load and parse the JSON file
    with open(args.params, "r") as f:
        config = json.load(f)

    # Extract required parameters with default fallbacks
    ascot_dir = config.get("out_dir", None)
    simname = config["sim_name"]  # Required
    h5path = os.path.join(ascot_dir, f"{simname}.h5")

    # The DESC equilibrium supplies the flux coordinates (see module
    # docstring); it is the same equilibrium create-input.py built this
    # ASCOT5 file's bfield from.
    equil_path = os.path.join(config["inp_dir"], config["equil_file"])
    eq = load_desc_equilibrium(equil_path)
    print(f"[desc] loaded equilibrium {os.path.basename(equil_path)} "
          f"(NFP={eq.NFP}, L/M/N={eq.L}/{eq.M}/{eq.N})")

    a5 = Ascot(h5path)
    old = a5.data.marker.active.read()
    old_qid = a5.data.marker.active.get_qid()
    mrktype = a5.data.marker.active.get_type()
    n_old = int(old["n"])

    if mrktype not in ("gc", "prt"):
        raise NotImplementedError(
            f"active marker group is of type '{mrktype}'; only guiding-center"
            " ('gc') and full-orbit ('prt') inputs are supported")
    print(f"[read] active marker group qid={old_qid} is '{mrktype}' "
          f"(n={n_old})")

    r_old = np.asarray(old["r"]).astype(np.float64).ravel()
    phi_old = np.asarray(old["phi"]).astype(np.float64).ravel()
    z_old = np.asarray(old["z"]).astype(np.float64).ravel()
    t_old = np.asarray(old["time"]).astype(np.float64).ravel()
    w_old = np.asarray(old["weight"]).astype(np.float64).ravel()

    # -------------------------------------------------- flux coordinates
    # bfield must be initialized before the donor energy/pitch is worked out,
    # since the "prt" branch needs B at the old marker positions.
    a5.input_init(bfield=True)
    e_old, p_old = donor_energy_pitch(a5, old, mrktype, r_old, phi_old,
                                      z_old, t_old)

    # ASCOT5 stores phi in degrees; DESC works in radians throughout.
    phi_rad_old = np.deg2rad(phi_old)
    rho_old, theta_old, zeta_old = rpz_to_flux(eq, r_old, phi_rad_old, z_old)

    good = (np.isfinite(rho_old) & np.isfinite(theta_old)
            & np.isfinite(zeta_old))
    if not np.all(good):
        print(f"[desc] dropped {int((~good).sum())} of {good.size} old "
              "markers that DESC could not place in the equilibrium volume "
              "(donors only -- does not affect the new markers)")
        (r_old, phi_old, z_old, t_old, w_old, e_old, p_old, rho_old,
         theta_old, zeta_old) = (x[good] for x in (
             r_old, phi_old, z_old, t_old, w_old, e_old, p_old, rho_old,
             theta_old, zeta_old))
    n_old = r_old.size

    # -------------------------------------------------------- target grid
    # rho is bounded by the old markers' own span, but theta and zeta are
    # angles: give them their full periodic range rather than whatever the
    # sample happened to reach, so the grid covers the device.
    edges_rho = make_edges(rho_old, NBINS_RHO)
    edges_theta = make_angle_edges(NBINS_THETA)
    edges_phi = make_angle_edges(NBINS_PHI)

    bin_rho_old = digitize_bin(rho_old, edges_rho, NBINS_RHO)
    bin_theta_old = digitize_bin(theta_old, edges_theta, NBINS_THETA)
    bin_phi_old = digitize_bin(np.mod(zeta_old, 2 * np.pi),
                               edges_phi, NBINS_PHI)

    NCELLS = NBINS_RHO * NBINS_THETA * NBINS_PHI
    cell_old = (bin_rho_old * NBINS_THETA + bin_theta_old) * NBINS_PHI \
        + bin_phi_old

    # T[b] = old weighted rate in cell b -- the target every new marker's
    # weight must reproduce. Independent of how many old markers landed
    # there (could be zero -- that just means T[b] is legitimately zero).
    T = np.bincount(cell_old, weights=w_old, minlength=NCELLS)

    # ------------------------------------------- uniform target counts
    # Split n_old markers exactly evenly across ALL cells, including ones
    # the old markers never visited -- this (not resampling) is what
    # guarantees a uniform marker distribution.
    all_cells = np.arange(NCELLS)
    base, remainder = divmod(n_old, NCELLS)
    target_counts = np.full(NCELLS, base, dtype=int)
    if remainder:
        extra = rng.choice(NCELLS, size=remainder, replace=False)
        target_counts[extra] += 1
    n_new = int(target_counts.sum())
    assert n_new == n_old, "new marker count must equal old marker count"

    # ------------------------------------- draw uniform (rho,theta,zeta)
    # Every cell gets its target count, with positions drawn uniformly at
    # random *within that cell's own edges* -- not tied to any old marker.
    cell_of_new = np.repeat(all_cells, target_counts)  # sorted ascending
    i_new, j_new, k_new = unravel_cell(cell_of_new, NBINS_THETA, NBINS_PHI)
    rho_new = rng.uniform(edges_rho[i_new], edges_rho[i_new + 1])
    theta_new = rng.uniform(edges_theta[j_new], edges_theta[j_new + 1])
    zeta_new = rng.uniform(edges_phi[k_new], edges_phi[k_new + 1])

    # ------------------------------------------------------------ weights
    # The old weighted rate is flat in (theta, zeta) within a rho slice --
    # measured, not assumed; see the "Weights" section of the module
    # docstring. So a slice's rate is shared equally among the new markers
    # in it, and only the rho profile carries real structure.
    #
    # This is what removes zero-weight markers. Per-cell weighting (below)
    # hands weight zero to every marker in a cell no old marker reached,
    # and those cells are almost all of the outer rho slices, which is
    # exactly where this script is supposed to be adding markers.
    R = np.bincount(bin_rho_old, weights=w_old, minlength=NBINS_RHO)
    new_per_slice = np.bincount(i_new, minlength=NBINS_RHO)
    if args.weight_per_cell:
        w_new = T[cell_of_new] / target_counts[cell_of_new]
        print("[weight] per-cell weighting (--weight-per-cell)")
    else:
        w_new = R[i_new] / new_per_slice[i_new]
        print("[weight] rho-profile weighting: each rho slice's rate shared "
              "equally over its new markers")
    n_empty_slice = int((R <= 0).sum())
    if n_empty_slice:
        print(f"[weight] WARNING: {n_empty_slice} of {NBINS_RHO} rho slices "
              "hold no old marker at all, so their new markers get zero "
              "weight no matter how (theta, zeta) is handled")

    # ----------------------------------- energy/pitch/time from donors
    # Position and weight are already fully determined. Energy, pitch and
    # time still need real values, so borrow them from old markers in the
    # same cell (weighted bootstrap, widened to neighbors -- periodic in
    # theta/phi -- only if that cell had too few old markers). This never
    # touches position or weight, only these three fields.
    e_new = np.empty(n_new)
    p_new = np.empty(n_new)
    t_new = np.empty(n_new)
    positive_old = w_old > 0
    if not np.all(positive_old):
        print(f"[donor] {int((~positive_old).sum())} of {positive_old.size} "
              "old markers have zero weight and are excluded from the donor "
              "pools")
    if not np.any(positive_old):
        raise ValueError(
            "every old marker has zero weight -- nothing to bootstrap "
            "energy/pitch/time from")
    pos = 0
    for b in np.flatnonzero(target_counts):
        k = target_counts[b]
        idxs = np.arange(pos, pos + k)
        pos += k
        members, _ = build_donor_pool(b, bin_rho_old, bin_theta_old,
                                      bin_phi_old, MIN_DONORS,
                                      NBINS_RHO, NBINS_THETA, NBINS_PHI,
                                      positive_old)
        p = w_old[members]
        p = p / p.sum()
        pick = rng.choice(members, size=k, replace=True, p=p)
        e_new[idxs] = e_old[pick]
        p_new[idxs] = p_old[pick]
        t_new[idxs] = t_old[pick]

    gyro_new = rng.random(n_new) * 2 * np.pi  # gyro-phase, no physical
                                              # correlation with anything

    # ---------------------------------- (rho,theta,zeta) -> (R,phi,Z)
    # A forward evaluation of the DESC equilibrium, not a root find: every
    # marker gets an exact position and none can come out NaN.
    r_new, phi_rad_new, z_new = flux_to_rpz(eq, rho_new, theta_new, zeta_new)
    phi_new = np.rad2deg(phi_rad_new)
    n_bad = int((~np.isfinite(r_new) | ~np.isfinite(z_new)).sum())
    assert n_bad == 0, f"{n_bad} new markers got non-finite (R, Z) from DESC"


    # Full-orbit output needs velocity vectors built against the field at the
    # new positions, so this has to happen while bfield is still initialized.
    if mrktype == "prt":
        vr_new, vphi_new, vz_new = energy_pitch_to_velocity(
            a5, ALPHA_MASS_AMU, e_new, p_new, gyro_new, r_new, phi_new,
            z_new, t_new)
    a5.input_free()

    # ------------------------------------------------------------- write
    # Refuse to write markers ASCOT5 would silently throw away, unless the
    # caller has explicitly said they want them anyway.
    n_zero = report_zero_weight(w_new, rho_new)
    if n_zero and not args.allow_zero_weight:
        print("\n[write] ABORTED -- nothing written, the existing marker "
              "group is untouched.\n"
              "        Fix the sampling (a coarser grid, a smaller rho "
              "range, or an\n"
              "        estimate for empty cells), or pass "
              "--allow-zero-weight to write anyway.")
        return 1

    # Same marker type in, same marker type out.
    mrk = a5py.ascot5io.Marker.generate(mrktype, n=n_new)
    mrk["weight"][:] = w_new / unyt.s
    mrk["anum"][:] = ALPHA_ANUM
    mrk["znum"][:] = ALPHA_ZNUM
    mrk["mass"][:] = ALPHA_MASS_AMU * unyt.amu
    mrk["charge"][:] = ALPHA_CHARGE_E * unyt.e
    mrk["time"][:] = t_new * unyt.s
    mrk["r"][:] = r_new * unyt.m
    mrk["z"][:] = z_new * unyt.m
    mrk["phi"][:] = phi_new * unyt.deg
    if mrktype == "gc":
        mrk["energy"][:] = e_new * unyt.eV
        mrk["pitch"][:] = p_new
        mrk["zeta"][:] = gyro_new * unyt.rad
    else:
        mrk["vr"][:] = vr_new * unyt.m / unyt.s
        mrk["vphi"][:] = vphi_new * unyt.m / unyt.s
        mrk["vz"][:] = vz_new * unyt.m / unyt.s

    name = a5.data.create_input(mrktype, **mrk, desc="UNIFRHOTHETAPHI",
                                activate=True)
    print(f"\n[write] created and activated marker group {name} "
          f"(n={n_new})")
    print(f"[write] previous group qid={old_qid} kept, now inactive")
    return 0
    

if __name__ == "__main__":
    sys.exit(main())
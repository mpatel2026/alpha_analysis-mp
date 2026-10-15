#!/usr/bin/env python
"""Total DD fusion rate grid (thermal / beam-target / beam-beam), both DD
branches, from ASCOT5 steady-state deuterium distributions -- (R, phi, z)
only.

Runs over every case in the ``CASES`` table (all five Eos10-beams runs by
default; ``--cases`` selects a subset, ``--input`` overrides with a single
explicit file). For each case it prints that case's own channel x reaction
rate grid, and at the end one combined table covering all three channels of
all cases, plus a compact per-case summary with the neutron rate. Every
per-case output file carries the case label, so nothing overwrites anything.

Two things below are NOT optional simplifications -- they're bug/memory
fixes needed to get correct numbers out of AFSI at all:

  * ``afsi_rpz.beamthermal``/``beambeam`` -- ASCOT5's shipped
    ``a5.afsi.beamthermal``/``beambeam`` call ``afsi_run`` with 5 arguments
    while its ctypes signature takes 7; they raise as shipped. See
    afsi_rpz.py's docstring. (``a5.afsi.thermal`` works as shipped, but it
    can only use the file's plasma, so the thermal channel goes through
    ``afsi_rpz.thermal`` too -- the same computation, plus the plasma
    override described below.)
  * ``read_steadystate_beam`` -- reads dist5d one time-slice at a time via
    h5py instead of ``run.getdist("5d")``, which would peak at ~6 GB on
    this file's ~2.9 GB ordinate array. It also collapses the distribution
    to a single time/charge bin (the only bins AFSI's beam sampler ever
    reads), by SUMMING the selected time bins (all, by default).

    Marker weight in ASCOT5 is a particle FLUX, not a count -- units
    "particles/s" (doc/physics.rst). Each dist5d bin accumulates
    "weight * dt" per crossing, so an individual bin is already in units
    of particles. The steady-state population in a phase-space cell, by
    Little's Law for a continuously-injected population, is the injection
    rate times the TOTAL dwell time a marker spends in that cell over its
    ENTIRE simulated lifetime (birth to loss/thermalization) -- i.e. the
    SUM of that cell's dwell time across every time bin, not any single
    bin. (A single bin only captures dwell time during that slice of the
    marker's life; the full lifetime sum is what Little's Law calls for.)
    A clean check that this must be a sum, not an average: the physical
    population can't depend on how finely the diagnostic binned the time
    axis, and plain summation is invariant to that (re-binning the same
    span into more, smaller bins doesn't change the sum) while averaging
    is not (it shrinks as the bin count grows, for the same physical
    population) -- so summing is the only bin-resolution-invariant choice.

  * ``build_dd_plasma`` -- the input files' plasma is a 50/50 D-T mix
    (n_D = n_T = n_e/2). By default (``--plasma DD``) AFSI is instead given
    a pure-D plasma for all three channels: the file's own n_e, T_e, T_i
    and rho grid, with the single ion species D at n_D = n_e
    (quasineutral, Z=1). It is injected in memory via
    ``a5.input_init(plasma=<dict>)``; the file is never written. Relative
    to the file's D-T plasma this makes thermal DD 4x larger (n_D^2) and
    beam-target DD 2x larger (n_D); beam-beam doesn't see the plasma at
    all. ``--plasma file`` reproduces the old D-T numbers.

    What this does NOT change: the fast-D distribution itself. ASCOT5
    slowed the beam down on the file's D-T plasma, and that dist5d is used
    as-is. A pure-D background slows beam D a little differently
    (sum(n_i Z^2/A_i)/n_e is 0.50 for pure D vs 0.42 for 50/50 D-T), so a
    fully self-consistent DD answer would need the ASCOT5 slowing-down run
    redone with a D plasma.

Assumptions: steady state = the sum of the selected (default: all) dist5d
time bins, i.e. each marker's full simulated lifetime; thermal D density =
n_e (``--plasma DD``, default) or the file's "ni1" as stored
(``--plasma file``); rates are per second, full torus. If markers
are still fast (not yet slowed down/lost) at the end of the simulated
window, the sum underestimates the true population by missing dwell time
beyond that window -- read_steadystate_beam prints each bin's total so you
can check the last bin has decayed toward zero rather than staying large.

Also saves the DD_He3n (neutron-branch) reaction rate per (R, phi, z) cell,
per channel and summed, as ``DD_He3n_rate_distribution_<case>.npz`` (see
``save_rate_distribution`` for the arrays it holds).

Also generates triton (from DD_Tp: D+D -> T+p) markers for a follow-on run,
one marker set per case (``--skip-markers`` turns this off if only the rate
table is wanted).
ASCOT5 has no file-based path for this at all -- checked a5py for any
netCDF/xarray usage (none) and doc/tutorials/markergen.ipynb, whose actual
pattern is entirely in-memory, one Python session: compute an AFSI product
distribution, hand it straight to ``a5.markergen.generate(...)`` as
``particledist``. So that's what this script does with the combined
(summed across channels) triton distribution, then writes the resulting
marker set out as an HDF5 file, ``triton_markers_<case>.h5`` (units
recorded per field, since unyt units
aren't uniform across fields -- e.g. ``mrk["phi"]`` ends up in radians, not
the marker template's usual degrees -- and can't round-trip through plain
JSON on their own), plus a diagnostic figure of the weighted (particle-rate)
marker distribution across rho, theta, and phi,
``triton_marker_distributions_<case>.png``. With ``--plasma DD`` both
names get a ``_DDplasma`` suffix, so the earlier D-T outputs are kept.
Marker POSITIONS are sampled
uniformly in rho (see ``build_uniform_rho_markerdist``); the physical profile
is carried by the weights, and the numeric agreement against the AFSI source
is reported by ``print_source_vs_marker_check``.
"""
import argparse
import os
import sys
import warnings

# NERSC's $HOME (GPFS) frequently fails HDF5's file locking on write --
# "NetCDF: HDF error" on close being the usual symptom. Your own job
# scripts (run-sims-cpu.sh etc.) already work around this the same way.
# Must be set before h5py/netCDF4 load the HDF5 library, so before any of
# those imports below; setdefault so an already-exported value wins.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import unyt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import afsi_rpz  # noqa: E402  (local module, path bootstrap above)

from a5py import Ascot  # noqa: E402
from a5py.ascot5io.dist import Dist, DistData  # noqa: E402

# Plotting style copied from
# convergence_studies/alpha_scaling_paper_figures/save_markerfigs.py --
# the existing precedent in this repo for marker rho/theta/phi histograms.
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
})

INPUT_DIR = "/pscratch/sd/m/mpatel26/ascot_h5s/Eos10-beams/"

# The five Eos10-beams cases, in the order they are reported. The key is the
# short case label: it goes into every per-case output filename
# (triton_markers_<label>.h5, triton_marker_distributions_<label>.png) and
# into the combined rate table printed at the end. The value is the ASCOT5
# file, relative to INPUT_DIR.
CASES = {
    #"steadystate_00": "ascot_Eos_steadystate_00-RPZ.h5",
    "BP3_broader": "ascot_Eos10_G2000_OP3ProfileBP3_12MWbroader.h5"
    #"OP3Profile1":    "ascot_G2000_free_OP3Profile1.h5",
    #"OP3Profile2":    "ascot_G2000_free_OP3Profile2.h5",
    #"OP3Profile3":    "ascot_G2000_free_OP3Profile3.h5",
}
DEFAULT_INPUT = INPUT_DIR + CASES["BP3_broader"]
# $HOME is GPFS and unreliable for large HDF5 writes -- matching the rest
# of this repo's own convention (ascot_h5s/ etc. all live on pscratch,
# never under alpha_analysis-mp/). The marker file itself is small (tens
# of KB for a few thousand markers), but keep the default here anyway.
DEFAULT_OUTDIR = "/global/homes/m/mpatel26/alpha_analysis-mp/fusion_rates"
DD_REACTIONS = ("DD_He3n", "DD_Tp")
CHANNELS = ("thermal", "beam-target", "beam-beam")
TRITON_REACTION = "DD_Tp"
RATE_DIST_REACTION = "DD_He3n"  # the neutron branch; its (R,phi,z) rate is saved
TRITON_ANUM_ZNUM = (3, 1)  # triton: 3 amu, +1e


def resolve_run_name(filepath, run_arg):
    """Resolve a run group name ("run_<qid>") in the results group."""
    with h5py.File(filepath, "r") as f:
        if run_arg is None:
            qid = f["results"].attrs["active"]
            if isinstance(qid, bytes):
                qid = qid.decode("utf-8")
            name = qid if qid.startswith("run_") else "run_" + qid
        else:
            name = run_arg if run_arg.startswith("run_") else "run_" + run_arg
        if name not in f["results"]:
            raise KeyError(
                f"{name} not found in {filepath}:/results "
                f"(available: {list(f['results'].keys())})")
        return name


def build_dd_plasma(a5):
    """Pure-deuterium version of the file's active plasma, as a dict that
    ``a5.input_init(plasma=...)`` injects in memory.

    Keeps the file's rho grid, n_e, T_e, T_i and vtor; replaces the ion
    species with D alone at n_D = n_e (quasineutral, Z=1). D's mass is
    taken from the file's own D entry.

    Returns
    -------
    plasma : dict
        ``plasma_1D.write_hdf5``-format data (without ``fn``/``desc``).
    """
    pls = a5.data.plasma.active
    if pls.get_type() != "plasma_1D":
        raise ValueError(f"build_dd_plasma only handles plasma_1D, not "
                         f"{pls.get_type()}")
    p = pls.read()
    anum = np.asarray(p["anum"]).ravel()
    znum = np.asarray(p["znum"]).ravel()
    iD = np.flatnonzero((anum == 2) & (znum == 1))
    if iD.size != 1:
        raise ValueError(f"No unique D species in plasma input "
                         f"(anum={anum}, znum={znum})")
    mass_d = float(np.asarray(p["mass"]).ravel()[iD[0]])
    ne = np.ascontiguousarray(p["edensity"], dtype=float).reshape(-1, 1)
    return {
        "nrho": int(p["nrho"]),
        "nion": 1,
        "anum": np.array([[2]], dtype=np.int32),
        "znum": np.array([[1]], dtype=np.int32),
        "mass": np.array([[mass_d]]),
        "charge": np.array([[1]], dtype=np.int32),
        "rho": np.ascontiguousarray(p["rho"], dtype=float),
        "vtor": np.ascontiguousarray(p["vtor"], dtype=float),
        "edensity": ne,
        "etemperature": np.ascontiguousarray(p["etemperature"], dtype=float),
        "idensity": ne.copy(),  # (nrho, nion=1): n_D = n_e
        "itemperature": np.ascontiguousarray(p["itemperature"], dtype=float),
    }


def read_steadystate_beam(filepath, runname, tmin=None, tmax=None):
    """Sum of the selected dist5d time bins, as a particle-count DistData
    with a single time bin and a single charge bin, read without loading
    the full ordinate array into memory twice. See module docstring for
    why this is a SUM across bins (each marker's full simulated lifetime),
    not an average and not divided by the window duration.

    Prints each selected time bin's total inventory so you can check the
    population has decayed toward zero by the last bin (i.e. markers are
    mostly slowed down/lost by then) rather than being truncated mid-life.

    Returns
    -------
    beam : DistData
    n_total : float
        Steady-state fast-D inventory [particles] (sum over selected bins).
    duration : float
        Selected time window's span [s] (informational only -- not used
        as a divisor).
    """
    with h5py.File(filepath, "r") as f:
        g = f["results"][runname]["dist5d"]
        ndim = int(g["abscissa_ndim"][()].item())
        edges, names, units = [], [], []
        for i in range(ndim):
            d = g[f"abscissa_vec_0{i + 1}"]
            edges.append(d[:])
            names.append(d.attrs[f"name_0{i}"].decode("utf-8"))
            units.append(d.attrs[f"unit_0{i}"].decode("utf-8"))

        expected_order = ["r", "phi", "z", "ppar", "pperp", "time", "charge"]
        if names != expected_order:
            raise ValueError(
                f"Unexpected dist5d abscissa order {names}; expected "
                f"{expected_order}.")

        t_edges = edges[names.index("time")]
        it = np.arange(t_edges.size - 1)
        if tmin is not None:
            it = it[t_edges[:-1] >= tmin]
        if tmax is not None:
            it = it[t_edges[1:] <= tmax]
        if it.size == 0:
            raise ValueError(
                f"No dist5d time bins fall within [{tmin}, {tmax}] s "
                f"(available edges: {t_edges})")
        if not np.array_equal(it, np.arange(it[0], it[-1] + 1)):
            raise ValueError(
                "Selected time bins are not contiguous; tmin/tmax must "
                "select a single contiguous window.")

        # ordinate axes: (ordinate, r, phi, z, ppar, pperp, time, charge)
        acc = None
        per_bin_totals = []
        for i in it:
            s = np.sum(g["ordinate"][:, :, :, :, :, :, i, :], axis=(0, -1))
            per_bin_totals.append(float(s.sum()))
            acc = s if acc is None else acc + s

    duration = float(t_edges[it[-1] + 1] - t_edges[it[0]])
    if duration <= 0:
        raise ValueError(f"Non-positive time window duration: {duration} s")

    print("  dist5d per-time-bin inventory [particles] "
          "(last bin should have decayed toward zero -- if not, markers "
          "are still fast at the window's end and the sum below "
          "underestimates the true population):")
    for i, total in zip(it, per_bin_totals):
        print(f"    bin {i} [{t_edges[i]:.3f}, {t_edges[i + 1]:.3f}] s: "
              f"{total:.4e}")

    hist = acc  # SUM over selected time bins -> particles (each marker's
                # full simulated lifetime dwell time; see module docstring
                # -- do NOT average and do NOT divide by duration)

    abscissae = {}
    for n, e, u in zip(names, edges, units):
        if n in ("time", "charge"):
            continue
        abscissae[n] = e * unyt.Unit(u.replace(" ", "*"))
    abscissae["time"] = np.array([t_edges[it[0]], t_edges[it[-1] + 1]]) * unyt.s
    abscissae["charge"] = np.array([0.0, 2.0]) * unyt.e

    beam = DistData(hist[..., np.newaxis, np.newaxis] * unyt.particles,
                     **abscissae)

    rebuilt = beam.histogram().v.squeeze()
    if not np.allclose(rebuilt, hist, rtol=1e-8, atol=0.0):
        raise AssertionError(
            "Steady-state beam DistData did not round-trip its histogram "
            "-- aborting before handing this to AFSI.")

    return beam, float(hist.sum()), duration


def total_rate(prod1, prod2, label):
    """Sum a fusion-product pair's bins into a total rate [1/s], warning if
    a non-negligible fraction sits in the outermost momentum bins (AFSI
    silently drops product samples that land outside the grid).
    """
    r1 = float(prod1.histogram().v.sum())
    r2 = float(prod2.histogram().v.sum())
    if r1 > 0 and abs(r1 - r2) / r1 > 1e-6:
        warnings.warn(f"[{label}] product1/product2 totals disagree "
                       f"({r1:.4e} vs {r2:.4e}); expected identical.")
    for name, prod, total in (("product1", prod1, r1), ("product2", prod2, r2)):
        if total <= 0:
            continue
        hist = prod.histogram().v
        ippar = prod.abscissae.index("ppar")
        ipperp = prod.abscissae.index("pperp")
        edge = 0.0
        # pperp's lowest bin is the physical pperp = 0 boundary, not a grid
        # cutoff -- nothing can be lost below it, so it isn't checked.
        for dim, edge_indices in ((ippar, (0, hist.shape[ippar] - 1)),
                                  (ipperp, (hist.shape[ipperp] - 1,))):
            for edge_index in edge_indices:
                idx = [slice(None)] * hist.ndim
                idx[dim] = edge_index
                edge += hist[tuple(idx)].sum()
        if edge / total > 1e-3:
            warnings.warn(
                f"[{label}/{name}] {edge / total:.2%} of the rate sits in "
                "the outermost momentum bins -- --pmax is likely too "
                "small and rate is being lost.")
    return r1


def identify_triton(a5, reaction):
    """Return which product (1 or 2) of `reaction` is the triton, by mass
    and charge rather than assuming a fixed product order.

    Raises
    ------
    ValueError
        If neither product matches (anum, znum) = TRITON_ANUM_ZNUM.
    """
    _m1, _q1, _m2, _q2, mprod1, qprod1, mprod2, qprod2, _Q = \
        a5.afsi.reactions(reaction)
    for which, (m, q) in ((1, (mprod1, qprod1)), (2, (mprod2, qprod2))):
        anum = int(np.round(m.to("amu").v).item())
        znum = int(np.round(q.to("e").v).item())
        if (anum, znum) == TRITON_ANUM_ZNUM:
            return which
    raise ValueError(f"No triton product found for reaction {reaction}")


def sum_distdata(dists):
    """Sum several DistData objects sharing the same grid into one."""
    abscissae = {k: dists[0].abscissa_edges(k) for k in dists[0].abscissae}
    hist = sum(d.histogram() for d in dists)
    return DistData(hist, **abscissae)


def save_rate_distribution(a5, dists, reaction, fn, extra=None):
    """Save a reaction's spatial (R, phi, z) rate distribution, per
    channel and summed, as a .npz file.

    Each channel's AFSI product distribution is integrated over product
    momentum (and the single time/charge bin), leaving reactions/s in each
    spatial cell. Either product works: both are born in the same cell, so
    their spatial rates are identical.

    Saved arrays (all shaped (nr, nphi, nz) unless noted):

    * ``rate_<channel>`` -- reactions/s in each cell, for ``thermal``,
      ``beam_target`` and ``beam_beam``; ``rate_total`` is their sum.
    * ``volume`` -- cell volume [m^3], built exactly as AFSI built it, so
      ``rate_total / volume`` is the local reaction rate density
      [reactions/(m^3 s)].
    * ``rho`` -- rho at each cell centre (NaN outside the plasma).
    * ``r_edges`` [m], ``phi_edges`` [deg], ``z_edges`` [m] -- 1D bin edges.

    Parameters
    ----------
    a5 : Ascot
        Open Ascot instance (bfield must be loadable), for rho.
    dists : dict
        channel -> product DistData from AFSI, all on the same grid.
    reaction : str
        Reaction name, recorded in the file.
    fn : str
        Output .npz path.
    extra : dict, optional
        Further arrays to store in the same file, by name.
    """
    out = {}
    for channel, dist in dists.items():
        spatial = dist.integrate(
            copy=True, **{k: np.s_[:] for k in dist.abscissae
                          if k not in ("r", "phi", "z")})
        out["rate_" + channel.replace("-", "_")] = spatial.histogram().v
    out["rate_total"] = sum(out[f"rate_{c.replace('-', '_')}"]
                            for c in dists)

    grid = next(iter(dists.values()))
    rc, phic, zc, vol = afsi_rpz._spatial_grid(grid)
    # vol comes out as m^3*rad (R dphi dR dz, dphi in rad) -- i.e. m^3.
    out["volume"] = vol.to("m**3*rad").v
    a5.input_init(bfield=True)
    try:
        out["rho"] = a5.input_eval(rc.ravel(), phic.ravel(), zc.ravel(),
                                   0 * unyt.s, "rho").v.reshape(rc.shape)
    finally:
        a5.input_free(bfield=True)
    out["r_edges"] = grid.abscissa_edges("r").to("m").v
    out["phi_edges"] = grid.abscissa_edges("phi").to("deg").v
    out["z_edges"] = grid.abscissa_edges("z").to("m").v
    out["reaction"] = np.array(reaction)
    out.update(extra or {})

    np.savez_compressed(fn, **out)
    print(f"Wrote {fn} ({reaction} rate per (R,phi,z) cell, total "
          f"{out['rate_total'].sum():.4e} /s)")


def write_markers_h5(mrk, fn):
    """Dump a marker dict to HDF5.

    unyt arrays can't be stored directly, and different fields carry
    different (sometimes non-obvious) units -- see the note in
    ``generate_triton_markers`` about ``mrk["phi"]`` ending up in radians
    rather than the marker template's degrees default. So each field is
    written as a dataset ``/<field>`` of plain numeric values, with its
    unit string (if any) stored in the dataset's ``units`` attribute,
    rather than assuming a fixed convention. The marker count is stored in
    the root attribute ``n`` when the dict has an ``n`` entry.

    Parameters
    ----------
    mrk : dict
        Marker dict as returned by ``a5.markergen.generate``.
    fn : str
        Output file path.
    """
    with h5py.File(fn, "w") as f:
        for key, val in mrk.items():
            arr = np.asarray(val.v if hasattr(val, "v") else val)
            if arr.ndim == 0:
                f.attrs[key] = arr.item()
                continue
            ds = f.create_dataset(key, data=arr, compression="gzip")
            if hasattr(val, "units"):
                ds.attrs["units"] = str(val.units)


def build_uniform_rho_markerdist(a5, particledist, nrho=20, nsub=16,
                                  niter=50, seed=0):
    """Build a ``markerdist`` that makes marker COUNT uniform in rho, while
    leaving the source's structure in theta/phi/ekin/pitch untouched.

    ``markergen.generate`` samples marker COORDINATES from ``markerdist``
    but takes marker WEIGHTS from ``particledist``
    (``weight = cell_rate / markers_in_that_cell``). That stays unbiased
    for ANY ``markerdist``, provided it is nonzero wherever the source is
    -- so the weighted histogram still reproduces the AFSI source no
    matter how the marker positions are redistributed. Only the marker
    POSITIONS change; the physical profile then lives in the weights,
    which come out large in the core and small at the edge.

    The obvious approach -- divide each cell by the source's rho marginal
    at the cell CENTRE -- does not actually flatten rho on this grid,
    because a marker's rho comes from its RANDOM position within the cell
    and one cell spans ~0.19 in rho (~4 rho bins). What markers produce is
    the cell weights CONVOLVED with each cell's own rho spread. Dividing by
    S(rho_centre) only got the count spread from 14.8x down to 2.7x.

    So this builds the convolution explicitly: sub-sample each cell to get
    ``frac[cell, k]``, the fraction of that cell's random positions landing
    in rho bin k, then solve

        find md  such that  sum_cells md[cell] * frac[cell, k] = const_k

    with a multiplicative (EM / Lucy-Richardson style) update. It is pure
    numpy on an (ncells x nrho) array -- no marker generation in the loop.

    Only rho is flattened: the correction is a single scalar per SPATIAL
    cell, so the relative distribution over theta, phi, ekin and pitch
    within each cell is untouched (those already matched the source well).

    Parameters
    ----------
    a5 : Ascot
        Open Ascot instance (bfield must be loadable).
    particledist : DistData
        The AFSI source, (r,phi,z,...) basis -- any momentum basis
        (ppar/pperp or ekin/pitch) works, only the spatial axes matter here.
    nrho : int, optional
        Number of rho bins the marker count is flattened over.
    nsub : int, optional
        Random sub-samples per cell used to build the smearing kernel.
    niter : int, optional
        Multiplicative update iterations.
    seed : int, optional
        Seed for the sub-sampling, so the result is reproducible.

    Returns
    -------
    markerdist : DistData
        Same grid as ``particledist``, to be passed as ``markerdist=``.
    """
    spatial = particledist.integrate(
        copy=True, **{k: np.s_[:] for k in particledist.abscissae
                      if k not in ("r", "phi", "z")})
    rate = spatial.histogram().v  # (nr, nphi, nz)

    r_edges = spatial.abscissa_edges("r").to("m").v
    phi_edges = spatial.abscissa_edges("phi").to("deg").v
    z_edges = spatial.abscissa_edges("z").to("m").v

    # Sub-sample each cell, because a marker's rho is set by its RANDOM
    # position within the cell, not the cell centre -- and on this grid a
    # single cell spans ~0.19 in rho (~4 rho bins). Dividing whole cells by
    # S(rho_centre) therefore cannot flatten the final marker histogram:
    # what markers actually produce is the cell weights CONVOLVED with each
    # cell's own rho spread. So build that convolution explicitly and solve
    # for the weights that flatten the smeared result.
    ir, ip, iz = np.meshgrid(np.arange(r_edges.size - 1),
                              np.arange(phi_edges.size - 1),
                              np.arange(z_edges.size - 1), indexing="ij")
    r_lo = np.repeat(r_edges[ir].ravel(), nsub)
    phi_lo = np.repeat(phi_edges[ip].ravel(), nsub)
    z_lo = np.repeat(z_edges[iz].ravel(), nsub)
    dr = np.repeat(np.diff(r_edges)[ir].ravel(), nsub)
    dphi = np.repeat(np.diff(phi_edges)[ip].ravel(), nsub)
    dz = np.repeat(np.diff(z_edges)[iz].ravel(), nsub)

    rng = np.random.default_rng(seed)
    r_s = r_lo + dr * rng.random(r_lo.size)
    phi_s = phi_lo + dphi * rng.random(phi_lo.size)
    z_s = z_lo + dz * rng.random(z_lo.size)

    a5.input_init(bfield=True)
    try:
        rho_s = a5.input_eval(r_s * unyt.m, phi_s * unyt.deg, z_s * unyt.m,
                               0 * unyt.s, "rho").v
    finally:
        a5.input_free(bfield=True)

    ncells = rate.size
    rho_edges = np.linspace(0.0, 1.0, nrho + 1)
    ibin = np.digitize(rho_s, rho_edges) - 1
    ok = np.isfinite(rho_s) & (ibin >= 0) & (ibin < nrho)

    # frac[c, k] = fraction of cell c's sub-samples landing in rho bin k,
    # i.e. the smearing kernel that turns cell weights into a rho histogram.
    cell_of = np.repeat(np.arange(ncells), nsub)
    frac = np.zeros((ncells, nrho))
    np.add.at(frac, (cell_of[ok], ibin[ok]), 1.0)
    frac /= nsub

    # Solve for per-cell sampling weights md giving a flat marker count in
    # rho, via a multiplicative (EM / Lucy-Richardson style) update on
    # md: predicted_k = sum_c md_c frac_ck, pushed toward a flat target.
    # Pure numpy on (ncells x nrho) -- no marker generation needed.
    flat = rate.ravel().astype(float).copy()
    reachable = frac.sum(axis=1) > 0
    active = reachable & (flat > 0)
    md = np.where(active, flat, 0.0)
    if np.any(active) and md.sum() > 0:
        occupied = frac[active].sum(axis=0) > 0
        target = np.zeros(nrho)
        target[occupied] = 1.0 / occupied.sum()
        for _ in range(niter):
            pred = frac.T @ md
            total = pred.sum()
            if total <= 0:
                break
            ratio = np.ones(nrho)
            nz = pred > 0
            ratio[nz] = target[nz] / (pred[nz] / total)
            md = md * (frac @ ratio)
            md[~active] = 0.0

    # Cells the sub-sampling never placed inside [0,1] (outside the plasma)
    # would get md = 0, which would silently DROP whatever rate they carry,
    # since markergen can then never sample them. Their rate should be ~0
    # (no plasma -> no fusion), but fall back to proportional sampling there
    # rather than risk losing it.
    lost = (~active) & (rate.ravel() > 0)
    if np.any(lost):
        scale = np.median(md[md > 0]) / np.median(flat[md > 0]) \
            if np.any(md > 0) else 1.0
        md[lost] = rate.ravel()[lost] * scale

    factor = np.zeros_like(flat)
    nzr = flat > 0
    factor[nzr] = md[nzr] / flat[nzr]
    factor = factor.reshape(rate.shape)

    markerdist_hist = particledist.histogram() * factor[..., np.newaxis,
                                                         np.newaxis]
    return DistData(markerdist_hist,
                    **{k: particledist.abscissa_edges(k)
                       for k in particledist.abscissae})


def afsi_source_rho_theta_phi(a5, particledist, nsub=16, seed=0):
    """rho/theta/phi and rate for the AFSI source grid -- the ground truth
    the sampled markers should reproduce.

    Each (r,phi,z) cell is SUB-SAMPLED at ``nsub`` random points drawn
    uniformly within the cell (independently in r, phi, z), with the cell's
    rate split evenly among them, rather than dumping the whole cell's rate
    at its centre.

    That sub-sampling is what makes this a fair comparison against the
    markers. ``markergen.generate`` places each marker at a random position
    WITHIN its cell (its ``randomize()`` helper is uniform in r, phi, z --
    matched exactly here), so markers are smeared in rho by the cell's own
    spatial extent. This grid's cells are ~8 cm in R and ~12 cm in z, and
    measured directly on this equilibrium, rho varies by a median of 0.19
    ACROSS a single in-plasma cell -- nearly 4 diagnostic rho bins, with a
    within-cell std of ~0.054 (about one full bin). Evaluating only at cell
    centres therefore produces an artificially sharp, comb-like source
    curve that cannot match the (correctly) smeared marker histogram in
    rho, even when the markers are perfectly sampled. theta and phi are
    barely affected because both are near-uniform to begin with.

    (The measured centre-vs-cell-mean rho bias is only +0.003, i.e. this is
    a pure resolution/smearing effect, not a systematic offset.)

    Parameters
    ----------
    a5 : Ascot
        Open Ascot instance.
    particledist : DistData
        AFSI source distribution with (r,phi,z,...) abscissae; any
        non-spatial axes are integrated out here.
    nsub : int, optional
        Random sub-samples per spatial cell.
    seed : int, optional
        Seed for the sub-sampling, so the diagnostic is reproducible.

    Returns
    -------
    rho, theta_deg, phi_deg, rate : numpy.ndarray
        Flattened, one entry per sub-sample (ncells * nsub). ``rate``
        [particles/s] is the cell rate divided by ``nsub``.
    """
    spatial = particledist.integrate(
        copy=True, **{k: np.s_[:] for k in particledist.abscissae
                      if k not in ("r", "phi", "z")})
    rate = spatial.histogram().v  # shape (nr, nphi, nz)

    r_edges = spatial.abscissa_edges("r").to("m").v
    phi_edges = spatial.abscissa_edges("phi").to("deg").v
    z_edges = spatial.abscissa_edges("z").to("m").v

    # Lower edge and width of the cell each flattened entry belongs to,
    # in the same (r, phi, z) index order as `rate`.
    ir, ip, iz = np.meshgrid(np.arange(r_edges.size - 1),
                              np.arange(phi_edges.size - 1),
                              np.arange(z_edges.size - 1), indexing="ij")
    r_lo = np.repeat(r_edges[ir].ravel(), nsub)
    phi_lo = np.repeat(phi_edges[ip].ravel(), nsub)
    z_lo = np.repeat(z_edges[iz].ravel(), nsub)
    dr = np.repeat(np.diff(r_edges)[ir].ravel(), nsub)
    dphi = np.repeat(np.diff(phi_edges)[ip].ravel(), nsub)
    dz = np.repeat(np.diff(z_edges)[iz].ravel(), nsub)

    rng = np.random.default_rng(seed)
    r_s = r_lo + dr * rng.random(r_lo.size)
    phi_s = phi_lo + dphi * rng.random(phi_lo.size)
    z_s = z_lo + dz * rng.random(z_lo.size)

    a5.input_init(bfield=True)
    try:
        rho, axisr, axisz = a5.input_eval(
            r_s * unyt.m, phi_s * unyt.deg, z_s * unyt.m, 0 * unyt.s,
            "rho", "axisr", "axisz")
    finally:
        a5.input_free(bfield=True)
    theta = np.arctan2((z_s * unyt.m - axisz).to("m").v,
                        (r_s * unyt.m - axisr).to("m").v) * unyt.rad
    theta_deg = np.mod(theta.to("deg").v, 360.0)
    phi_deg = np.mod(phi_s, 360.0)

    rate_s = np.repeat(rate.ravel(), nsub) / nsub
    return rho.v, theta_deg, phi_deg, rate_s


def plot_triton_marker_distributions(rho, theta_deg, phi_deg, weight, fn,
                                      exi=None, nbins=20, exi_nbins=(60, 30)):
    """F/dx diagnostic figure: the WEIGHTED (particle-rate) marker
    distribution across rho, theta and phi, plus a 2D (energy, pitch)
    histogram -- markers only, no AFSI source overlay anywhere in this
    figure (the numeric marker-vs-AFSI-source agreement is reported
    separately by :func:`print_source_vs_marker_check`).

    Style follows the existing precedent in
    convergence_studies/alpha_scaling_paper_figures/save_markerfigs.py:
    one histogram per axes, normalized by bin width so it reads as a rate
    density rather than a raw per-bin sum.

    The 2D (energy, pitch) panel is at its own resolution (``exi_nbins``),
    independent of the (ppar,pperp) or (ekin,pitch) grid used for marker
    SAMPLING. That decoupling is possible because each marker's energy and
    pitch are continuous values (computed from its own sampled momentum,
    not tied to that cell's edges) -- so they can be re-histogrammed at
    any resolution after the fact.

    Parameters
    ----------
    rho : array_like
        Dimensionless rho at each marker's (r,phi,z).
    theta_deg : array_like
        Geometric poloidal angle (relative to the local magnetic axis) at
        each marker, wrapped to [0, 360) deg -- see generate_triton_markers
        for why this isn't ASCOT5's Boozer theta.
    phi_deg : array_like
        Toroidal angle at each marker, wrapped to [0, 360) deg.
    weight : array_like
        Marker weight [particles/s].
    fn : str
        Output figure path.
    exi : tuple (mrk_ekin_eV, mrk_pitch), optional
        Each marker's energy [eV] and pitch. Drives the 2D panel.
    nbins : int, optional
        Number of histogram bins per 1D panel.
    exi_nbins : (int, int), optional
        (energy bins, pitch bins) for the 2D panel.
    """
    ncols = 4 if exi is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.0), dpi=300)
    panels = (("rho", rho, r"$\rho$", ""),
              ("theta", theta_deg, r"$\theta$", r"$^\circ$"),
              ("phi", phi_deg, r"$\phi$", r"$^\circ$"))
    for col, (name, data, label, unit_str) in enumerate(panels):
        data = np.asarray(data)
        # nanmin/nanmax: rho is legitimately NaN for markers that land
        # outside the plasma; plain min/max would let a single NaN poison
        # the whole bin range.
        lo, hi = np.nanmin(data), np.nanmax(data)
        edges = np.linspace(lo, hi, nbins + 1)
        delta = (hi - lo) / nbins

        ax = axes[col]
        ax.hist(data, bins=edges, color="skyblue", edgecolor="black",
                weights=weight / delta)
        ax.set_xlabel(f"{label} [{unit_str}]" if unit_str else label)
        # Each panel divides by its OWN bin width (weight/delta), i.e.
        # this is a rate DENSITY (F/dx), not a per-bin rate -- so its peak
        # height is not bounded by the total rate, and can exceed it
        # whenever a sizeable fraction of the total falls within one
        # (possibly much-less-than-1-wide) bin. Say so explicitly, on
        # every panel (they do NOT share a y-scale -- each is its own
        # independently-scaled density).
        unit_suffix = f"/{unit_str}" if unit_str else ""
        ax.set_ylabel(f"F/d{label} [particles/s{unit_suffix}]")
        ax.grid(axis="y", alpha=0.5)

    if exi is not None:
        mrk_ekin, mrk_pitch = exi
        mrk_ekin = np.asarray(mrk_ekin)
        mrk_pitch = np.asarray(mrk_pitch)
        n_ek, n_pi = exi_nbins
        ek_lo, ek_hi = np.nanmin(mrk_ekin), np.nanmax(mrk_ekin)
        pi_lo, pi_hi = np.nanmin(mrk_pitch), np.nanmax(mrk_pitch)
        ekin_edges = np.linspace(ek_lo, ek_hi, n_ek + 1)
        pitch_edges = np.linspace(pi_lo, pi_hi, n_pi + 1)
        ek_mev = ekin_edges / 1e6

        ax = axes[3]
        H, _, _ = np.histogram2d(mrk_ekin, mrk_pitch,
                                  bins=[ekin_edges, pitch_edges],
                                  weights=weight)
        # Density, not a raw per-bin sum (same convention as the 1D panels
        # above): divide by the bin area so the colorbar's claimed units
        # (particles per MeV per lambda per second) are actually true.
        dE_mev = ek_mev[1] - ek_mev[0]
        dlambda = pitch_edges[1] - pitch_edges[0]
        H_density = H / (dE_mev * dlambda)
        mesh = ax.pcolormesh(ek_mev, pitch_edges, H_density.T, cmap="viridis",
                              shading="flat")
        fig.colorbar(mesh, ax=ax,
                     label=r"F/d$\lambda$dE [particles / (MeV * s)]")

        # The triton birth energy is narrow relative to the full grid
        # range, so most columns can be empty and the panel would read as
        # one bright stripe. Zoom to where the markers actually are.
        col_sum = H.sum(axis=1)
        occ = np.flatnonzero(col_sum > 0.01 * col_sum.max()) \
            if col_sum.max() > 0 else np.array([], dtype=int)
        if occ.size:
            pad = 0.5 * (ek_mev[1] - ek_mev[0])
            ax.set_xlim(max(ek_mev[occ[0]] - pad, ek_mev[0]),
                         min(ek_mev[occ[-1] + 1] + pad, ek_mev[-1]))
        ax.set_xlabel("E [MeV]")
        ax.set_ylabel(r"$\lambda$")

    fig.tight_layout()
    fig.savefig(fn)
    plt.close(fig)


def print_source_vs_marker_check(rho, theta_deg, phi_deg, weight, source):
    """Print rate-weighted mean/std of rho/theta/phi for the sampled
    markers vs. the AFSI source grid, as a quick numeric consistency check
    alongside the figure.
    """
    def wstats(x, w):
        # rho is legitimately NaN for grid cells / marker positions outside
        # the plasma (input_eval's documented behavior for points where the
        # flux-surface evaluation fails) -- drop those before averaging,
        # since a single NaN would otherwise contaminate the whole mean.
        x = np.asarray(x)
        w = np.asarray(w)
        finite = np.isfinite(x)
        if not np.all(finite):
            x, w = x[finite], w[finite]
        mean = np.average(x, weights=w)
        std = np.sqrt(np.average((x - mean) ** 2, weights=w))
        return mean, std

    print("  Weighted mean +/- std, markers vs. AFSI source:")
    for name, mdata, sdata in (("rho", rho, source[0]),
                                ("theta [deg]", theta_deg, source[1]),
                                ("phi [deg]", phi_deg, source[2])):
        mm, ms = wstats(mdata, weight)
        sm, ss = wstats(sdata, source[3])
        print(f"    {name:<12} markers: {mm:8.4f} +/- {ms:7.4f}   "
              f"source: {sm:8.4f} +/- {ss:7.4f}")


def generate_triton_markers(a5, triton_dist, triton_which, nmrk, outdir,
                             nekin=20, npitch=5, diag_nbins=20,
                             uniform_rho_bins=20, exi_nekin=60, exi_npitch=30,
                             case=None):
    """Sample triton markers from the combined triton source distribution,
    write them to JSON, and plot their rho/theta/phi/(energy,pitch)
    distributions.

    Mirrors doc/tutorials/markergen.ipynb's own pattern: collapse the
    distribution's time/charge axes, REBIN momentum from AFSI's fine native
    (ppar,pperp) grid to a coarser (ekin,pitch) grid via
    ``Dist.ppappe2ekinpitch`` (particle-conserving), then hand that to
    ``a5.markergen.generate`` as ``particledist``.

    The rebin matters for more than style: this triton distribution's full
    (r,phi,z,ppar,pperp) grid has 25*26*25*79*39 ~= 50 MILLION cells.
    ``markergen.generate`` samples markers by drawing from that full 5D
    grid and assigns each marker's weight as (that cell's rate) / (markers
    landed in that same cell) -- so any cell never drawn contributes
    nothing to the total. Rebinning to nekin*npitch (~1.2M cells at the
    defaults) makes each cell capture a much larger, more reliably-sampled
    share of the rate.

    NOTE: an earlier version of this function switched to sampling
    directly on the native (ppar,pperp) grid, based on a claimed pitch
    artifact in ``ppappe2ekinpitch``. That claim was WRONG -- it came from
    a flawed test (an "isotropic" source built without the 2*pi*pperp
    gyro-volume weighting that a genuinely isotropic 3D distribution
    requires; that missing factor alone biases naive (ppar,pperp)-uniform
    weighting toward pitch=+-1, independent of any rebin). Re-tested with
    a properly pperp-weighted isotropic source: ``ppappe2ekinpitch`` comes
    back nearly flat in pitch (ratio 1.00-1.3, not the fabricated 2x). So
    this function reverted to the rebinned approach, which gives
    meaningfully better sampling coverage at the same marker count.

    Parameters
    ----------
    a5 : Ascot
        The (already-open) Ascot instance the AFSI run used.
    triton_dist : DistData
        Combined (all-channel) triton source distribution, (r,phi,z,ppar,
        pperp,time,charge).
    triton_which : int
        1 or 2 -- which reaction product of TRITON_REACTION is the triton
        (from ``identify_triton``), used to fetch its mass/charge.
    nmrk : int
        Number of markers to generate.
    outdir : str
        Directory to write triton_markers.h5 and the diagnostic figure
        into.
    case : str, optional
        Case label appended to both output filenames
        (``triton_markers_<case>.h5``,
        ``triton_marker_distributions_<case>.png``), so runs over several
        input files in one directory do not overwrite each other. ``None``
        keeps the unsuffixed names.
    nekin, npitch : int, optional
        Energy/pitch bin counts for the pre-sampling rebin.
    diag_nbins : int, optional
        Number of rho/theta/phi histogram bins in the diagnostic figure
        (unrelated to nekin/npitch -- this is purely a plotting choice).
    exi_nekin, exi_npitch : int, optional
        Energy/pitch bin counts for the figure's 2D (energy, pitch) panel.
        Also unrelated to nekin/npitch: markers carry continuous energy and
        pitch values, so this panel can be rebinned at any resolution
        independent of the (coarser) sampling grid.
    """
    _m1, _q1, _m2, _q2, mprod1, qprod1, mprod2, qprod2, _Q = \
        a5.afsi.reactions(TRITON_REACTION)
    mass, charge = (mprod1, qprod1) if triton_which == 1 else (mprod2, qprod2)
    # a5.afsi.reactions() returns masses/charges as shape-(1,) arrays, not
    # scalars. That silently breaks Dist.ppappe2ekinpitch's internal
    # broadcasting -- verified directly: shape-mismatch (25,26,25,19,9) vs
    # (25,26,25,19,1,9) with shape-(1,) mass, works fine with a true
    # scalar. Extract scalars before using either downstream.
    mass = mass.item() * mass.units
    charge = charge.item() * charge.units
    anum, znum = TRITON_ANUM_ZNUM

    spatial_momentum = triton_dist.integrate(copy=True, time=np.s_[:],
                                              charge=np.s_[:])
    particledist = Dist.ppappe2ekinpitch(spatial_momentum, mass,
                                          ekin_edges=nekin, pitch_edges=npitch)
    total_rate = float(particledist.histogram().v.sum())
    total_rate_prerebin = float(spatial_momentum.histogram().v.sum())
    # ppappe2ekinpitch is a histogram rebin (particle-conserving by design,
    # per its own docstring), not an exact analytic remap -- edge effects
    # at the pitch=+-1 boundary can drop a small fraction (verified: ~0.4%
    # in a realistic synthetic test). 1% catches a real problem without
    # flagging that expected small loss.
    if total_rate_prerebin > 0 and \
            abs(total_rate - total_rate_prerebin) / total_rate_prerebin > 1e-2:
        warnings.warn(
            "ppappe2ekinpitch lost more than 1% of the total rate: "
            f"{total_rate_prerebin:.4e} before rebin vs {total_rate:.4e} "
            "after -- larger than the usual pitch-boundary edge loss; "
            "investigate before trusting the generated markers.")

    # Sample marker POSITIONS uniformly in rho (theta/phi structure within
    # each rho shell preserved), but take marker WEIGHTS from the true
    # source -- see build_uniform_rho_markerdist. Pass uniform_rho_bins=0
    # to fall back to sampling positions proportional to the source.
    if uniform_rho_bins:
        markerdist = build_uniform_rho_markerdist(a5, particledist,
                                                   nrho=uniform_rho_bins)
        print(f"  Marker positions sampled uniformly in rho "
              f"({uniform_rho_bins} rho bins); weights still taken from the "
              "AFSI source, so the weighted histogram is unchanged.")
    else:
        markerdist = None

    mrk = a5.markergen.generate(nmrk, mass, charge, anum, znum, particledist,
                                 markerdist=markerdist)
    mrk_total_weight_raw = float(np.sum(mrk["weight"].v))

    # Rescale all marker weights by a single constant so their total
    # exactly equals the true source total. This fixes NORMALIZATION
    # (the finite-sample under-coverage from drawing nmrk markers out of
    # r*phi*z*nekin*npitch cells -- see this function's docstring) but
    # NOT any residual SHAPE bias: if certain phase-space regions are
    # systematically over/under-sampled, this uniform factor doesn't move
    # rate between them. The printed source-vs-marker comparison
    # (print_source_vs_marker_check) is what checks the shape; this only
    # checks/fixes the total.
    if mrk_total_weight_raw > 0 and total_rate > 0:
        rescale = total_rate / mrk_total_weight_raw
        mrk["weight"] = mrk["weight"] * rescale
    else:
        rescale = 1.0
    mrk_total_weight = float(np.sum(mrk["weight"].v))

    coverage = mrk_total_weight_raw / total_rate if total_rate > 0 else float("nan")
    print(f"Generated {nmrk} triton markers: raw sampled total weight "
          f"{mrk_total_weight_raw:.4e} /s ({coverage:.1%} of the true "
          f"source total {total_rate:.4e} /s)")
    print(f"  Rescaled all weights by {rescale:.4g}x so the total matches "
          f"exactly ({mrk_total_weight:.4e} /s). This corrects the overall "
          "normalization only -- it does not fix any shape mismatch; see "
          "the source-vs-marker comparison printed below.")
    if coverage < 0.5:
        warnings.warn(
            f"Raw marker sampling only captured {coverage:.1%} of the true "
            "total before rescaling -- individual marker weights may carry "
            "large variance even though the total is now exact; consider "
            "--ntriton-markers larger or coarser --nekin/--npitch if the "
            "shape comparison below looks noisy.")

    suffix = f"_{case}" if case else ""
    # Data go to <outdir>/data/triton_markers/<DD|DT>_plasma/
    datadir = os.path.join(outdir, "data", "triton_markers",
                           "DD_plasma" if suffix.endswith("_DDplasma")
                           else "DT_plasma")
    os.makedirs(datadir, exist_ok=True)
    fn = os.path.join(datadir, f"triton_markers{suffix}.h5")
    write_markers_h5(mrk, fn)
    print(f"Wrote {fn}")
    print("  This is only the marker set -- bfield, plasma, wall, and "
          "options must still be added (e.g. via your create-input.py "
          "pipeline) before a run can use it.")

    # rho/theta at each marker's (r,phi,z). "theta" via input_eval requires
    # this run's boozer input, but that returns NaN unconditionally here
    # (verified directly: even phi=0, rho=0.76, well inside the plasma,
    # comes back NaN) -- the boozer group in this file is a very coarse
    # grid (npsi=6, nr=5) that doesn't evaluate for this equilibrium, not
    # something fixable from here. Using a geometric poloidal angle
    # instead: atan2(z - axisz(phi), r - axisr(phi)), relative to the
    # local magnetic axis at each marker's own phi (both axisr/axisz and
    # rho DO evaluate correctly everywhere, including stellarator-periodic
    # phi outside the field's raw storage window -- verified directly).
    # This is NOT ASCOT5's Boozer theta (which is itself not DESC's
    # theta -- see afsi5.py's thermal_from_desc docstring) -- it's a
    # plain geometric angle, fine for a diagnostic plot. Wrapped to
    # [0,360) deg to match the "thetamod"/"phimod" convention used
    # elsewhere in this repo (e.g. save_markerfigs.py).
    a5.input_init(bfield=True)
    try:
        rho, axisr, axisz = a5.input_eval(mrk["r"], mrk["phi"], mrk["z"],
                                           mrk["time"], "rho", "axisr", "axisz")
    finally:
        a5.input_free(bfield=True)
    theta = np.arctan2((mrk["z"] - axisz).to("m").v,
                        (mrk["r"] - axisr).to("m").v) * unyt.rad
    theta_deg = np.mod(theta.to("deg").v, 360.0)
    phi_deg = np.mod(mrk["phi"].to("deg").v, 360.0)

    # Same rho/theta/phi evaluation, but at the AFSI source grid's own
    # cell centers instead of marker positions -- the ground truth the
    # markers should reproduce once sampling noise averages out.
    source = afsi_source_rho_theta_phi(a5, particledist)
    print_source_vs_marker_check(rho.v, theta_deg, phi_deg, mrk["weight"].v,
                                  source)

    # Each marker's own (continuous) energy and pitch -- rebinned at a much
    # finer, independent resolution in the plot (see
    # plot_triton_marker_distributions), decoupled from the coarser
    # (ekin,pitch) grid used for sampling.
    exi = (mrk["energy"].to("eV").v,
           np.asarray(mrk["pitch"].v if hasattr(mrk["pitch"], "v")
                      else mrk["pitch"]))

    # Figures go to <outdir>/figs/triton_marker_distributions/<DT|DD>_plasma/
    figdir = os.path.join(outdir, "figs", "triton_marker_distributions",
                          "DD_plasma" if suffix.endswith("_DDplasma")
                          else "DT_plasma")
    os.makedirs(figdir, exist_ok=True)
    figfn = os.path.join(figdir, f"triton_marker_distributions{suffix}.png")
    plot_triton_marker_distributions(
        rho.v, theta_deg, phi_deg, mrk["weight"].v, figfn,
        exi=exi, nbins=diag_nbins, exi_nbins=(exi_nekin, exi_npitch))
    print(f"Wrote {figfn}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", default=list(CASES),
                    metavar="CASE",
                    help="Which of the known cases to run, by label "
                         f"(default: all of {list(CASES)}). Ignored when "
                         "--input is given.")
    p.add_argument("--input", default=None,
                    help="Run a single explicit ASCOT5 file instead of the "
                         "--cases table. Its case label is taken from the "
                         "filename.")
    p.add_argument("--plasma", choices=("DD", "file"), default="DD",
                    help="Thermal plasma AFSI uses: 'DD' (default) = pure D "
                         "at n_D = n_e with the file's n_e/T profiles; "
                         "'file' = the plasma stored in the input file "
                         "(50/50 D-T for the Eos10-beams cases).")
    p.add_argument("--skip-markers", action="store_true",
                    help="Only compute and tabulate the rates; skip triton "
                         "marker generation and its diagnostic figure.")
    p.add_argument("--run", default=None,
                    help="Run name or bare qid (default: file's active run).")
    p.add_argument("--nmc", type=int, default=1000,
                    help="AFSI Monte Carlo samples per spatial cell.")
    p.add_argument("--tmin", type=float, default=None)
    p.add_argument("--tmax", type=float, default=None)
    p.add_argument("--nppar", type=int, default=80)
    p.add_argument("--npperp", type=int, default=40)
    p.add_argument("--pmax", type=float, default=8.0e-20,
                    help="Product momentum grid half-width [kg m/s]. The "
                         "binding constraint is the HEAVY product: with a "
                         "1.2 MeV beam D, beam-target tritons reach 3.05 "
                         "MeV (p=7.0e-20) and He3 2.74 MeV (p=6.6e-20), so "
                         "the old 6.5e-20 (triton max 2.63 MeV) clipped the "
                         "high-energy tail. 8e-20 reaches 3.99 MeV for the "
                         "triton, with headroom. Protons/neutrons are light "
                         "and never the limit.")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--ntriton-markers", type=int, default=100000,
                    help="Number of triton markers to generate.")
    p.add_argument("--nekin", type=int, default=20,
                    help="Energy bins to rebin the triton momentum grid to "
                         "before sampling markers. Fewer bins = fewer "
                         "phase-space cells = better sampling coverage, but "
                         "the triton spectrum is NOT a single narrow line: "
                         "thermal sits at 1.01 MeV, beam-beam near 1.3-1.5 "
                         "MeV, and beam-target is a broad 0.2-3.0 MeV "
                         "continuum. 20 bins over 0-4 MeV (~200 keV) "
                         "separates those; 10 merges the thermal and "
                         "beam-beam features.")
    p.add_argument("--npitch", type=int, default=5,
                    help="Pitch bins for the same rebin. Fusion products are "
                         "born near-isotropically, so pitch is nearly flat "
                         "and coarse binning costs little.")
    p.add_argument("--diag-nbins", type=int, default=20,
                    help="Number of rho/theta/phi histogram bins in "
                         "triton_marker_distributions.png (unrelated to "
                         "--nekin/--npitch -- a plotting choice only).")
    p.add_argument("--exi-nekin", type=int, default=60,
                    help="Energy bins for the figure's 2D (energy, pitch) "
                         "panel. Markers carry continuous energy values, so "
                         "this is independent of (and can be much finer "
                         "than) the sampling grid.")
    p.add_argument("--exi-npitch", type=int, default=30,
                    help="Pitch bins for the same 2D panel.")
    p.add_argument("--uniform-rho-bins", type=int, default=20,
                    help="Sample marker positions uniformly in rho using "
                         "this many rho bins to estimate the source's rho "
                         "marginal (theta/phi structure within each rho "
                         "shell is preserved; weights still come from the "
                         "AFSI source). 0 disables it, reverting to "
                         "sampling positions proportional to the source.")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help="Where to write triton_markers.h5 and the "
                         "diagnostic figure (default: pscratch -- see "
                         "module docstring).")
    return p.parse_args(argv)


def print_rate_table(rates, prefix=""):
    """Print the channel x reaction rate grid for a single case."""
    header = f"{prefix}{'channel':<14}" \
        + "".join(f"{r:>16}" for r in DD_REACTIONS) + f"{'total DD':>16}"
    print(header)
    for channel in CHANNELS:
        row = f"{prefix}{channel:<14}"
        row_total = 0.0
        for reaction in DD_REACTIONS:
            v = rates[(reaction, channel)]
            row += f"{v:>16.4e}"
            row_total += v
        row += f"{row_total:>16.4e}"
        print(row)
    col_totals = f"{prefix}{'total':<14}"
    grand_total = 0.0
    for reaction in DD_REACTIONS:
        v = sum(rates[(reaction, channel)] for channel in CHANNELS)
        col_totals += f"{v:>16.4e}"
        grand_total += v
    col_totals += f"{grand_total:>16.4e}"
    print(col_totals)


def print_all_cases_table(all_rates, plasma):
    """Print the combined table: every channel of every case, both DD
    branches, plus per-case totals and a compact case-vs-channel summary.

    ``all_rates`` maps case label -> {(reaction, channel): rate [1/s]}, in
    the order the cases were run.
    """
    width = 20 + 14 + 3 * 16
    print()
    print("=" * width)
    print(f"DD FUSION RATES, ALL CASES [reactions/s], {plasma} plasma")
    print("=" * width)
    header = f"{'case':<20}{'channel':<14}" \
        + "".join(f"{r:>16}" for r in DD_REACTIONS) + f"{'total DD':>16}"
    print(header)
    print("-" * width)
    for label, rates in all_rates.items():
        for channel in CHANNELS:
            row = f"{label:<20}{channel:<14}"
            row_total = 0.0
            for reaction in DD_REACTIONS:
                v = rates[(reaction, channel)]
                row += f"{v:>16.4e}"
                row_total += v
            print(row + f"{row_total:>16.4e}")
        row = f"{label:<20}{'ALL CHANNELS':<14}"
        case_total = 0.0
        for reaction in DD_REACTIONS:
            v = sum(rates[(reaction, c)] for c in CHANNELS)
            row += f"{v:>16.4e}"
            case_total += v
        print(row + f"{case_total:>16.4e}")
        print("-" * width)

    # Compact summary: one row per case, one column per channel (total DD
    # over both branches), plus the neutron rate (the DD_He3n column, which
    # is the only branch that makes a neutron).
    print()
    print("SUMMARY: total DD rate by channel [reactions/s], and neutron rate")
    swidth = 20 + (len(CHANNELS) + 2) * 16
    print("-" * swidth)
    print(f"{'case':<20}" + "".join(f"{c:>16}" for c in CHANNELS)
          + f"{'total DD':>16}" + f"{'neutrons':>16}")
    print("-" * swidth)
    for label, rates in all_rates.items():
        row = f"{label:<20}"
        total = 0.0
        for channel in CHANNELS:
            v = sum(rates[(r, channel)] for r in DD_REACTIONS)
            row += f"{v:>16.4e}"
            total += v
        neutrons = sum(rates[("DD_He3n", c)] for c in CHANNELS)
        print(row + f"{total:>16.4e}" + f"{neutrons:>16.4e}")
    print("-" * swidth)
    print("Neutron rate = DD_He3n summed over all three channels "
          "(one neutron per DD_He3n reaction; DD_Tp makes none).")


def run_case(args, label, filepath):
    """Compute every (reaction, channel) rate for one case and, unless
    --skip-markers, generate that case's triton markers and figure.

    Returns {(reaction, channel): rate [1/s]}.
    """
    banner = "=" * 78
    print()
    print(banner)
    print(f"CASE {label}: {filepath}")
    print(banner)

    run_name = resolve_run_name(filepath, args.run)
    print(f"Run: {run_name}")
    # Tag every output file with the plasma assumption, so DD-plasma runs
    # never overwrite the earlier D-T-plasma outputs.
    tag = f"{label}_DDplasma" if args.plasma == "DD" else label

    beam, n_total, duration = read_steadystate_beam(
        filepath, run_name, tmin=args.tmin, tmax=args.tmax)
    print(f"Steady-state fast-D inventory: {n_total:.4e} particles "
          f"(time window {duration:.4g} s)")

    a5 = Ascot(filepath, create=False)
    if args.plasma == "DD":
        plasma = build_dd_plasma(a5)
        print(f"Plasma: pure D, n_D = n_e (core n_e = "
              f"{plasma['edensity'][0, 0]:.4e} m^-3), T_i/T_e from file")
    else:
        plasma = None
        print("Plasma: as stored in the file")
    ppar = args.pmax * np.linspace(-1.0, 1.0, args.nppar)
    pperp = args.pmax * np.linspace(0.0, 1.0, args.npperp)

    triton_which = identify_triton(a5, TRITON_REACTION)
    triton_dists = {}  # channel -> DistData, only for TRITON_REACTION
    rate_dists = {}  # channel -> DistData, only for RATE_DIST_REACTION

    rates = {}  # (reaction, channel) -> rate [1/s]
    for reaction in DD_REACTIONS:
        for channel in CHANNELS:
            clabel = f"{label}/{reaction}/{channel}"
            if channel == "thermal":
                afsi_fn = afsi_rpz.thermal
            elif channel == "beam-target":
                afsi_fn = afsi_rpz.beamthermal
            else:  # beam-beam
                afsi_fn = afsi_rpz.beambeam
            prod1, prod2 = afsi_fn(
                a5, reaction, beam, nmc=args.nmc,
                ppar1=ppar, pperp1=pperp, ppar2=ppar, pperp2=pperp,
                verbose=args.verbose, plasma=plasma)
            rates[(reaction, channel)] = total_rate(prod1, prod2, clabel)
            print(f"  {clabel}: {rates[(reaction, channel)]:.4e} /s")

            if reaction == TRITON_REACTION:
                triton_dists[channel] = prod1 if triton_which == 1 else prod2
            if reaction == RATE_DIST_REACTION:
                rate_dists[channel] = prod1

    # <outdir>/data/rate_distributions/<DD|DT>_plasma/
    ratedir = os.path.join(args.outdir, "data", "rate_distributions",
                           "DD_plasma" if str(tag).endswith("_DDplasma")
                           else "DT_plasma")
    os.makedirs(ratedir, exist_ok=True)
    save_rate_distribution(
        a5, rate_dists, RATE_DIST_REACTION,
        os.path.join(ratedir,
                     f"{RATE_DIST_REACTION}_rate_distribution_{tag}.npz"))

    if args.skip_markers:
        print("  --skip-markers: no triton markers or figure written.")
    else:
        combined_triton = sum_distdata(list(triton_dists.values()))
        generate_triton_markers(a5, combined_triton, triton_which,
                                 args.ntriton_markers, args.outdir,
                                 nekin=args.nekin, npitch=args.npitch,
                                 diag_nbins=args.diag_nbins,
                                 uniform_rho_bins=args.uniform_rho_bins,
                                 exi_nekin=args.exi_nekin,
                                 exi_npitch=args.exi_npitch,
                                 case=tag)

    print()
    print(f"{label}: total DD rate [reactions/s], {args.plasma} plasma")
    print_rate_table(rates, prefix="  ")
    return rates


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    if args.input is not None:
        # Single explicit file: label it after its basename, so its outputs
        # are still distinguishable from the table-driven ones.
        label = os.path.splitext(os.path.basename(args.input))[0]
        label = label[len("ascot_"):] if label.startswith("ascot_") else label
        selected = [(label, args.input)]
    else:
        unknown = [c for c in args.cases if c not in CASES]
        if unknown:
            raise SystemExit(
                f"Unknown case(s) {unknown}; known cases are {list(CASES)}.")
        selected = [(c, INPUT_DIR + CASES[c]) for c in args.cases]

    print(f"Running {len(selected)} case(s): "
          f"{', '.join(label for label, _ in selected)} "
          f"({args.plasma} plasma)")

    all_rates = {}
    for label, filepath in selected:
        all_rates[label] = run_case(args, label, filepath)

    print_all_cases_table(all_rates, args.plasma)
    return all_rates


if __name__ == "__main__":
    main()

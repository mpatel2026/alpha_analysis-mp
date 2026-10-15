#!/usr/bin/env python
"""DD neutron (DD_He3n) rate from a HYPOTHETICAL fast-D population, against
the steadystate_00 background plasma made pure deuterium.

Same AFSI machinery as dd_fusion_rates.py (the corrected ``afsi_rpz``
wrappers, ``build_dd_plasma``, ``save_rate_distribution``); the difference
is where the fast D comes from. Instead of ASCOT5's slowing-down dist5d, the
fast population is prescribed:

  * isotropic Maxwellian in velocity, with temperature falling linearly
    from 316 keV on axis to 140 keV at rho = 1 (``--tfast-axis-kev``,
    ``--tfast-edge-kev``; set both equal for a uniform temperature);
  * density n_fast(rho) = 3e18 * (1 - rho) m^-3 inside rho <= 1, zero
    outside;
  * uniform in theta and phi on each flux surface.

Why the (R, phi, z) conversion is needed: AFSI's beam sampler only reads a
beam histogram in the (R, phi, z) basis (see afsi_rpz.py), but the profile
is given in flux coordinates. ``deposit_fast_particles`` does the
conversion with the DESC equilibrium the steadystate_00 ASCOT5 input was
built from: it lays a fine tensor quadrature over (rho, theta, zeta), maps
every node forward to (R, phi, Z) with DESC, and adds that node's particle
count n_fast(rho) * |sqrt(g)| drho dtheta dzeta to the (R, phi, z) cell it
lands in. The cells are those of the steadystate_00 dist5d grid, so the
background, the cell volumes, and the product grid are exactly what
dd_fusion_rates.py uses for that case. (The DESC file's axis matches the
ASCOT5 field's to 1e-4 m, and its rho matches ASCOT5's to 0.002.)

Because T varies with rho and one (R, phi, z) cell spans a range of rho,
a cell's momentum distribution is a SUM of Maxwellians. The deposit keeps
each cell's particles split by the rho quadrature node they came from, and
each node's share is filled with the exact per-bin fractions of the
Maxwellian at that node's T, on one (ppar, pperp) grid spanning
+-``--pext`` thermal momenta of the HOTTEST temperature. AFSI samples each
bin at its CENTRE momentum, so the grid is kept much finer than the
thermal width of the COLDEST temperature too.

Channels (all DD_He3n, i.e. one neutron per reaction):
  * thermal     -- background D with itself (the pure-D steadystate_00
                   plasma; no fast D involved);
  * beam-target -- fast D against background D;
  * beam-beam   -- fast D with itself.

Also printed: an independent semi-analytic estimate for every channel. For
two Maxwellian D populations at T1 and T2, the relative velocity is itself
Maxwellian, so the rate density is n1 n2 <sigma v>(T_eff) (x 1/2 for a
population with itself), with T_eff = (T1 + T2) / 2 for equal masses. With
a sum of Maxwellians in each cell this becomes a sum over every pair of
components (fast-background: sum_k; fast-fast: 1/2 sum_k sum_l).
<sigma v> is integrated numerically from the SAME Bosch-Hale cross-section
AFSI uses (``boschhale_sigma`` in libascot), evaluated with the per-cell
densities, so it checks the Maxwellian construction and AFSI's Monte Carlo
sampling. It should agree to about a percent.

Output: ``DD_He3n_rate_distribution_hypFastD_T<axis>-<edge>keV_
steadystate_00_DDplasma.npz`` in --outdir, with the arrays that
``save_rate_distribution`` writes plus ``fast_density`` (cell-averaged
n_fast [m^-3]), ``fast_particles`` (per cell), ``fast_T_mean_keV``
(particle-weighted mean fast T per cell) and the profile parameters.
"""
import argparse
import ctypes
import os
import sys
import warnings

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import h5py
import numpy as np
import unyt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import afsi_rpz  # noqa: E402
from dd_fusion_rates import (  # noqa: E402
    CASES, CHANNELS, DEFAULT_OUTDIR, INPUT_DIR, build_dd_plasma,
    resolve_run_name, save_rate_distribution, total_rate)

from a5py import Ascot  # noqa: E402
from a5py.ascot5io.dist import DistData  # noqa: E402
from a5py.ascotpy.libascot import _LIBASCOT, AFSI_REACTIONS  # noqa: E402

CASE = "steadystate_00"
ASCOT_FN = INPUT_DIR + "ascot_Eos_steadystate_00-RPZ.h5"
DESC_FN = ("/pscratch/sd/m/mpatel26/equil/Eos10-beams/"
           "equil_Eos_G2000_DESC_Te_perturb_constantBeta_00.h5")
REACTION = "DD_He3n"


def read_grid_edges(filepath, runname):
    """(R, phi, z) cell edges of the case's dist5d grid, with units.

    Only the abscissae are read, not the (multi-GB) ordinate.
    """
    with h5py.File(filepath, "r") as f:
        g = f["results"][runname]["dist5d"]
        out = {}
        for i in range(int(g["abscissa_ndim"][()].item())):
            d = g[f"abscissa_vec_0{i + 1}"]
            name = d.attrs[f"name_0{i}"].decode("utf-8")
            unit = d.attrs[f"unit_0{i}"].decode("utf-8")
            if name in ("r", "phi", "z"):
                out[name] = d[:] * unyt.Unit(unit.replace(" ", "*"))
    return out["r"], out["phi"], out["z"]


def fast_density_profile(rho, n0):
    """n_fast(rho) = n0 * (1 - rho) inside the LCFS, zero outside [m^-3]."""
    rho = np.asarray(rho)
    return np.where(rho <= 1.0, n0 * (1.0 - rho), 0.0)


def fast_temperature_profile(rho, T_axis, T_edge):
    """Linear fast-D temperature, T_axis at rho = 0 to T_edge at rho = 1."""
    return T_axis + (T_edge - T_axis) * np.asarray(rho)


def deposit_fast_particles(desc_fn, r_edges, phi_edges, z_edges, n0,
                           nrho=64, ntheta=128, nzeta=720, zeta_chunk=8):
    """Map the flux-function density onto (R, phi, z) cells with DESC.

    Tensor midpoint quadrature over rho in (0, 1), theta in [0, 2 pi) and
    zeta over the FULL torus [0, 2 pi) (the dist5d grid spans 0-360 deg).
    Each node carries weight |sqrt(g)| drho dtheta dzeta [m^3] and
    n_fast(rho) * weight particles, and is binned by its DESC (R, phi, Z).
    Density is uniform in theta and zeta by construction.

    Parameters
    ----------
    desc_fn : str
        DESC equilibrium file.
    r_edges, phi_edges, z_edges : unyt_array
        Cell edges (phi in any angle unit).
    n0 : float
        On-axis fast density [m^-3].
    nrho, ntheta, nzeta : int, optional
        Quadrature resolution. The defaults give ~1 cm (rho), ~3 cm (theta,
        edge) and ~3 cm (zeta) node spacing, well below the ~8-12 cm cells.
    zeta_chunk : int, optional
        Toroidal planes per DESC compute call (bounds memory).

    Returns
    -------
    particles : ndarray, (nr, nphi, nz, nrho)
        Fast particles per cell, split by the rho quadrature node they came
        from (so a rho-dependent temperature can be applied per node).
    rho : ndarray, (nrho,)
        The rho quadrature nodes.
    plasma_vol : ndarray, (nr, nphi, nz)
        Plasma (rho <= 1) volume per cell [m^3].
    stats : dict
        Totals (and the parts landing outside the grid), plus DESC's own
        plasma volume for comparison.
    """
    import desc.io as dscio
    from desc.grid import LinearGrid

    eq = dscio.load(desc_fn)
    try:  # EquilibriaFamily -> final equilibrium
        eq = eq[-1]
    except TypeError:
        pass

    r_e = r_edges.to("m").v
    p_e = phi_edges.to("rad").v
    z_e = z_edges.to("m").v
    rho = (np.arange(nrho) + 0.5) / nrho
    zeta_all = 2.0 * np.pi * np.arange(nzeta) / nzeta
    dw = (1.0 / nrho) * (2.0 * np.pi / ntheta) * (2.0 * np.pi / nzeta)

    rho_edges = np.linspace(0.0, 1.0, nrho + 1)  # one node per bin
    shape = (r_e.size - 1, p_e.size - 1, z_e.size - 1)
    particles = np.zeros(shape + (nrho,))
    plasma_vol = np.zeros(shape)
    v_all = n_all = 0.0
    for i0 in range(0, nzeta, zeta_chunk):
        grid = LinearGrid(rho=rho, theta=ntheta,
                          zeta=zeta_all[i0:i0 + zeta_chunk], NFP=1, sym=False)
        # The grid is deliberately NFP=1 (full torus) on an NFP=2
        # equilibrium; DESC warns about that on every call.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unequal number of field periods")
            d = eq.compute(["R", "Z", "phi", "sqrt(g)"], grid=grid)
        R = np.asarray(d["R"])
        Z = np.asarray(d["Z"])
        phi = np.mod(np.asarray(d["phi"]), 2.0 * np.pi)
        w = np.abs(np.asarray(d["sqrt(g)"])) * dw
        rho_n = np.asarray(grid.nodes[:, 0])
        n = fast_density_profile(rho_n, n0) * w
        v_all += w.sum()
        n_all += n.sum()
        pts = np.column_stack([R, phi, Z])
        plasma_vol += np.histogramdd(pts, bins=(r_e, p_e, z_e), weights=w)[0]
        particles += np.histogramdd(np.column_stack([R, phi, Z, rho_n]),
                                    bins=(r_e, p_e, z_e, rho_edges),
                                    weights=n)[0]

    stats = {
        "V_quadrature": v_all, "V_desc": float(np.asarray(eq.compute("V")["V"])),
        "V_outside_grid": v_all - plasma_vol.sum(),
        "N_total": n_all, "N_outside_grid": n_all - particles.sum(),
    }
    return particles, rho, plasma_vol, stats


def maxwellian_bin_fractions(ppar_edges, pperp_edges, sigma):
    """Exact fraction of an isotropic Maxwellian in each (ppar, pperp) bin.

    With thermal momentum sigma = sqrt(m T), ppar is Gaussian(0, sigma) and
    pperp is Rayleigh(sigma) (2D Maxwellian radius), independently.
    """
    from scipy.special import erf
    cdf_par = 0.5 * (1.0 + erf(ppar_edges / (np.sqrt(2.0) * sigma)))
    cdf_perp = 1.0 - np.exp(-0.5 * (pperp_edges / sigma) ** 2)
    return np.outer(np.diff(cdf_par), np.diff(cdf_perp))


def build_fast_dist(particles, r_edges, phi_edges, z_edges, mass, T,
                    nppar, npperp, pext):
    """Fast-D DistData on the (R, phi, z) grid, a sum of isotropic
    Maxwellians in momentum, in the single-time/single-charge layout AFSI's
    beam sampler reads (same layout as ``read_steadystate_beam`` builds).

    Parameters
    ----------
    particles : ndarray, (nr, nphi, nz, nk)
        Particles per cell in each of nk temperature components.
    T : unyt_array, (nk,)
        Temperature of each component.

    Returns
    -------
    dist : DistData
    captured : float
        Smallest fraction of any component inside the momentum grid (the
        rest, beyond ``pext`` thermal momenta of the hottest component, is
        dropped). The hottest component is the one cut most.
    """
    sigma = np.sqrt((mass * T).to("kg**2*m**2/s**2").v)
    ppar = pext * sigma.max() * np.linspace(-1.0, 1.0, nppar + 1)
    pperp = pext * sigma.max() * np.linspace(0.0, 1.0, npperp + 1)
    frac = np.stack([maxwellian_bin_fractions(ppar, pperp, s)
                     for s in sigma])  # (nk, nppar, npperp)
    hist = np.tensordot(particles, frac, axes=([3], [0]))
    pu = unyt.Unit("kg*m/s")
    dist = DistData(hist[..., None, None] * unyt.particles,
                    r=r_edges, phi=phi_edges, z=z_edges,
                    ppar=ppar * pu, pperp=pperp * pu,
                    time=np.array([0.0, 1.0]) * unyt.s,
                    charge=np.array([0.0, 2.0]) * unyt.e)
    return dist, float(frac.sum(axis=(1, 2)).min())


def maxwellian_sigmav(reaction, mu, T_joule, E_grid=None):
    """<sigma v> [m^3/s] of two Maxwellian populations whose relative
    motion has effective temperature T (reduced mass ``mu`` [kg]), from
    AFSI's own Bosch-Hale cross-section:

        <sigma v> = sqrt(8 / (pi mu)) T^(-3/2) int sigma(E) E exp(-E/T) dE

    with E the centre-of-mass energy.
    """
    fun = _LIBASCOT.boschhale_sigma
    fun.restype = ctypes.c_double
    fun.argtypes = [ctypes.c_int, ctypes.c_double]
    ireac = {v: k for k, v in AFSI_REACTIONS.items()}[reaction]
    keV = 1e3 * unyt.qp.to("C").v
    if E_grid is None:
        E_grid = np.geomspace(0.5 * keV, 2.0e4 * keV, 20000)
    sig = np.array([fun(ireac, E) for E in E_grid])
    T = np.atleast_1d(T_joule).astype(float)
    integrand = sig[None, :] * E_grid[None, :] \
        * np.exp(-E_grid[None, :] / T[:, None])
    integral = np.trapezoid(integrand, E_grid, axis=1)
    return np.sqrt(8.0 / (np.pi * mu)) * T ** -1.5 * integral


def semi_analytic_rates(a5, plasma, grid, particles, T_fast, mass):
    """Per-channel total rate [1/s] from the two-Maxwellian formula, using
    the same per-cell densities AFSI sees (background at the cell centre,
    fast D cell-averaged), summed over every pair of Maxwellian components.

    ``particles`` is (nr, nphi, nz, nk) with component temperatures
    ``T_fast`` (nk,).
    """
    rc, phic, zc, vol = afsi_rpz._spatial_grid(grid)
    vol = vol.to("m**3*rad").v
    a5.input_init(bfield=True, plasma=plasma)
    try:
        nD, Ti = a5.input_eval(rc.ravel(), phic.ravel(), zc.ravel(),
                               0 * unyt.s, "ni1", "ti1")
    finally:
        a5.input_free(bfield=True, plasma=True)
    # NaN where the field can't evaluate rho -- AFSI skips those cells too.
    nD = np.nan_to_num(nD.to("m**-3").v).reshape(rc.shape)
    Ti = np.nan_to_num(Ti.to("J").v).reshape(rc.shape)
    mu = 0.5 * mass.to("kg").v
    Tk = T_fast.to("J").v
    nk = particles / vol[..., None]  # fast density per component [m^-3]

    # Tabulate <sigma v> against T_eff and interpolate (Ti varies by cell).
    # Ti is floored at 1e-3 of the hottest fast T: that only touches the
    # far edge (n ~ 1e15, T ~ 1 eV), where every rate is negligible anyway.
    Ti_in = Ti[nD > 0]
    lo = max(Ti_in.min() if Ti_in.size else Tk.min(), 1e-3 * Tk.max())
    hi = max(Ti_in.max() if Ti_in.size else Tk.max(), Tk.max())
    T_tab = np.geomspace(lo, hi, 400)
    sv_tab = maxwellian_sigmav(REACTION, mu, T_tab)
    sv = lambda T: np.interp(T, T_tab, sv_tab)  # noqa: E731
    Ti_safe = np.clip(Ti, T_tab[0], T_tab[-1])
    sv_ff = sv(0.5 * (Tk[:, None] + Tk[None, :]))  # (nk, nk)
    return {
        "thermal": float(np.sum(0.5 * nD ** 2 * sv(Ti_safe) * vol)),
        "beam-target": float(np.sum(
            nk * (nD * vol)[..., None]
            * sv(0.5 * (Tk + Ti_safe[..., None])))),
        "beam-beam": float(np.sum(
            0.5 * vol * np.einsum("...k,kl,...l->...", nk, sv_ff, nk))),
    }


class _GridEdges:
    """Minimal stand-in exposing ``abscissa_edges`` for the (R,phi,z)
    grid, so ``afsi_rpz._spatial_grid`` can build centres/volumes."""
    def __init__(self, r, phi, z):
        self._e = {"r": r, "phi": phi, "z": z}

    def abscissa_edges(self, k):
        return self._e[k]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tfast-axis-kev", type=float, default=316.0,
                   help="Fast-D temperature on axis [keV].")
    p.add_argument("--tfast-edge-kev", type=float, default=140.0,
                   help="Fast-D temperature at rho = 1 [keV]; linear in "
                        "rho between the two. Equal values = uniform T.")
    p.add_argument("--n0", type=float, default=0.5e18,
                   help="Fast-D density on axis [m^-3]; falls linearly "
                        "to 0 at rho = 1.")
    p.add_argument("--nmc", type=int, default=1000,
                   help="AFSI Monte Carlo samples per spatial cell.")
    p.add_argument("--nppar", type=int, default=80,
                   help="Fast-D ppar bins over +-pext thermal momenta. At "
                        "the defaults a bin is 0.125 thermal momenta of the "
                        "hottest (316 keV) and 0.19 of the coldest (140 "
                        "keV) component.")
    p.add_argument("--npperp", type=int, default=40,
                   help="Fast-D pperp bins over [0, pext] thermal momenta.")
    p.add_argument("--pext", type=float, default=5.0,
                   help="Fast-D momentum grid half-width, in thermal "
                        "momenta sqrt(m T) of the hottest component.")
    p.add_argument("--prod-nppar", type=int, default=80)
    p.add_argument("--prod-npperp", type=int, default=40)
    p.add_argument("--prod-pmax", type=float, default=1.5e-19,
                   help="Product momentum grid half-width [kg m/s]. Larger "
                        "than dd_fusion_rates.py's 8e-20: the 316 keV "
                        "Maxwellian's tail makes more energetic He3 "
                        "(1.5e-19 covers He3 to ~6.8 MeV).")
    p.add_argument("--nrho", type=int, default=64,
                   help="DESC quadrature points in rho.")
    p.add_argument("--ntheta", type=int, default=128,
                   help="DESC quadrature points in theta.")
    p.add_argument("--nzeta", type=int, default=720,
                   help="DESC quadrature points in zeta (full torus).")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    run_name = resolve_run_name(ASCOT_FN, None)
    r_e, phi_e, z_e = read_grid_edges(ASCOT_FN, run_name)
    grid = _GridEdges(r_e, phi_e, z_e)
    print(f"Case {CASE}: {ASCOT_FN} ({run_name})")
    print(f"DESC equilibrium: {DESC_FN}")

    a5 = Ascot(ASCOT_FN, create=False)
    plasma = build_dd_plasma(a5)
    print(f"Background: pure D, n_D = n_e (core n_e = "
          f"{plasma['edensity'][0, 0]:.4e} m^-3), T_i/T_e from file "
          f"(core T_i = {plasma['itemperature'][0, 0]:.4g} eV)")

    mass = a5.afsi.reactions(REACTION)[0]
    mass = mass.item() * mass.units
    # --- flux-coordinate profile -> (R, phi, z) cells, via DESC ---------
    particles_k, rho_k, plasma_vol, st = deposit_fast_particles(
        DESC_FN, r_e, phi_e, z_e, args.n0,
        nrho=args.nrho, ntheta=args.ntheta, nzeta=args.nzeta)
    T_k_keV = fast_temperature_profile(rho_k, args.tfast_axis_kev,
                                       args.tfast_edge_kev)
    T_k = (T_k_keV * unyt.keV).to("J")
    particles = particles_k.sum(axis=-1)
    _, _, _, vol = afsi_rpz._spatial_grid(grid)
    vol = vol.to("m**3*rad").v  # m^3 (dphi in rad)
    fast_density = particles / vol
    with np.errstate(invalid="ignore", divide="ignore"):
        T_mean_keV = np.where(particles > 0,
                              (particles_k * T_k_keV).sum(axis=-1)
                              / particles, np.nan)
    n_per_node = particles_k.sum(axis=(0, 1, 2))
    print(f"Fast D: T(rho) = {args.tfast_axis_kev:g} + "
          f"({args.tfast_edge_kev:g} - {args.tfast_axis_kev:g})*rho keV, "
          f"n(rho) = {args.n0:.3e}*(1 - rho) m^-3")
    print(f"  particle-weighted mean fast T "
          f"{(n_per_node * T_k_keV).sum() / n_per_node.sum():.2f} keV")
    print(f"  DESC quadrature plasma volume {st['V_quadrature']:.4f} m^3 "
          f"(DESC V = {st['V_desc']:.4f} m^3); outside the (R,phi,z) grid: "
          f"{st['V_outside_grid']:.3e} m^3")
    print(f"  total fast particles {st['N_total']:.4e} "
          f"(outside grid: {st['N_outside_grid']:.3e}); volume-averaged "
          f"density {st['N_total'] / st['V_quadrature']:.4e} m^-3")
    print(f"  max cell-averaged density {fast_density.max():.4e} m^-3 "
          f"(on-axis profile value {args.n0:.4e})")

    beam, captured = build_fast_dist(
        particles_k, r_e, phi_e, z_e, mass, T_k,
        args.nppar, args.npperp, args.pext)
    print(f"  Maxwellian fraction inside the +-{args.pext:g} sqrt(mT_max) "
          f"momentum grid (worst component): {captured:.6f}")

    # --- AFSI -----------------------------------------------------------
    ppar = args.prod_pmax * np.linspace(-1.0, 1.0, args.prod_nppar)
    pperp = args.prod_pmax * np.linspace(0.0, 1.0, args.prod_npperp)
    fns = {"thermal": afsi_rpz.thermal,
           "beam-target": afsi_rpz.beamthermal,
           "beam-beam": afsi_rpz.beambeam}
    rates, dists = {}, {}
    for channel in CHANNELS:
        prod1, prod2 = fns[channel](
            a5, REACTION, beam, nmc=args.nmc,
            ppar1=ppar, pperp1=pperp, ppar2=ppar, pperp2=pperp,
            verbose=args.verbose, plasma=plasma)
        rates[channel] = total_rate(prod1, prod2, f"{REACTION}/{channel}")
        dists[channel] = prod1
        print(f"  {REACTION}/{channel}: {rates[channel]:.4e} /s")

    ttag = f"T{args.tfast_axis_kev:g}-{args.tfast_edge_kev:g}keV"
    # <outdir>/data/rate_distributions/hypothetical_fastD/
    datadir = os.path.join(args.outdir, "data", "rate_distributions",
                           "hypothetical_fastD")
    os.makedirs(datadir, exist_ok=True)
    fn = os.path.join(datadir, f"{REACTION}_rate_distribution_"
                                   f"hypFastD_{ttag}_{CASE}_DDplasma.npz")
    save_rate_distribution(a5, dists, REACTION, fn, extra={
        "fast_density": fast_density, "fast_particles": particles,
        "fast_T_mean_keV": T_mean_keV,
        "fast_T_axis_keV": np.array(args.tfast_axis_kev),
        "fast_T_edge_keV": np.array(args.tfast_edge_kev),
        "fast_n0": np.array(args.n0)})

    # --- independent check ----------------------------------------------
    ref = semi_analytic_rates(a5, plasma, grid, particles_k, T_k, mass)
    # Sanity-check the <sigma v> integral against Bosch-Hale's own
    # reactivity fit, inside that fit's validity range (<= 100 keV).
    keV = 1e3 * unyt.qp.to("C").v
    mu = 0.5 * mass.to("kg").v
    fun = _LIBASCOT.boschhale_sigmav
    fun.restype = ctypes.c_double
    fun.argtypes = [ctypes.c_int, ctypes.c_double]
    ireac = {v: k for k, v in AFSI_REACTIONS.items()}[REACTION]
    for T in (10.0, 50.0):
        mine = maxwellian_sigmav(REACTION, mu, T * keV)[0]
        print(f"  <sigma v> check at {T:g} keV: integrated {mine:.4e}, "
              f"Bosch-Hale fit {fun(ireac, T):.4e} m^3/s")

    print()
    print(f"DD NEUTRON RATE (DD_He3n), hypothetical fast D "
          f"(T {args.tfast_axis_kev:g} -> {args.tfast_edge_kev:g} keV, "
          f"n0 = {args.n0:.2e} m^-3) on {CASE} pure-D background "
          "[neutrons/s]")
    print(f"{'channel':<14}{'AFSI':>14}{'semi-analytic':>16}{'AFSI/ref':>10}")
    for channel in CHANNELS:
        a, b = rates[channel], ref[channel]
        print(f"{channel:<14}{a:>14.4e}{b:>16.4e}"
              f"{(a / b if b > 0 else float('nan')):>10.4f}")
    print(f"{'total':<14}{sum(rates.values()):>14.4e}"
          f"{sum(ref.values()):>16.4e}")
    return rates, ref


if __name__ == "__main__":
    main()

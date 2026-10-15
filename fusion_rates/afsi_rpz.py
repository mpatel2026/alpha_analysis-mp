"""Corrected AFSI beam-target / beam-beam wrappers, (R, phi, z) only.

ASCOT5's shipped Python wrappers ``a5py.routines.afsi5.Afsi.beamthermal`` and
``Afsi.beambeam`` are broken: they call ``afsi_run`` with 5 positional
arguments while the registered ctypes ``argtypes`` declares 7
(``sim*, afsi_data*, int n, histogram* prod1, histogram* prod2, int verbose,
int save_output``), so they raise ``ArgumentError`` as shipped. This module
reimplements the call correctly, reusing ASCOT5's own private helper methods
(``Afsi._init_afsi_data``, ``Afsi._init_histogram``, ``Afsi._init_dist_5d``,
``Afsi._build_distdata``) rather than duplicating any ctypes plumbing.

Only the (R, phi, z) basis is supported here. AFSI's beam-target/beam-beam
sampler (``afsi_sample_beam_2d`` in src/afsi.c) indexes the beam histogram
with ``hist->strides[0]``, ``[1]``, ``[2]``, which are hardwired to the
R/PHI/Z axes -- a (rho, theta, phi) beam distribution would be silently
indexed with the wrong strides. Flux-coordinate beam-target/beam-beam rates
are out of scope; only ``a5.afsi.thermal`` (unmodified, from a5py) supports
both bases.

Crucially, ``save_output`` is always passed as ``0``: passing anything else
makes ``afsi_run`` write a new ``/results/afsi_<qid>`` group into
``sim->hdf5_in``, i.e. into the (large, shared) input file. Nothing in this
module writes to the HDF5 file.

Every wrapper takes an optional ``plasma`` dict, which replaces the file's
active plasma for that one call. It is handed to ``a5.input_init`` as a dict,
so it only lives in memory (see ``a5py.ascotpy.libproviders``) and the file
is never touched. That is also why ``thermal`` lives here even though
``a5.afsi.thermal`` works as shipped: that one hardwires
``input_init(plasma=True)``, so it can only use the plasma stored in the
file.
"""
import ctypes

import numpy as np
import unyt

from a5py.ascotpy.libascot import _LIBASCOT, AFSI_REACTIONS


def _spatial_grid(beam):
    """Cell centres and volumes for the beam's (R, phi, z) grid.

    Mirrors exactly what the (broken) ``Afsi.beamthermal``/``Afsi.beambeam``
    build, so AFSI's ``afsi->volshape`` loop and the product histogram
    strides line up with the beam distribution's own grid.

    Parameters
    ----------
    beam : :class:`a5py.ascot5io.dist.DistData`
        Beam distribution with ``r``, ``phi``, ``z`` abscissae.

    Returns
    -------
    rc, phic, zc : array_like
        Cell-centre R [m], phi [rad], z [m], each of shape
        (nr, nphi, nz) (note the meshgrid axis order below).
    vol : array_like
        Cell volume [m^3], same shape as ``rc``.
    """
    r = beam.abscissa_edges("r")
    z = beam.abscissa_edges("z")
    phi = beam.abscissa_edges("phi").to("rad")
    phic, rc, zc = np.meshgrid(0.5 * (phi[:-1] + phi[1:]),
                                0.5 * (r[:-1] + r[1:]),
                                0.5 * (z[:-1] + z[1:]))
    vol = rc * np.diff(r[:2]) * np.diff(z[:2]) * np.diff(phi[:2])
    return rc, phic, zc, vol


def _plasma_species_index(a5, anum, znum):
    """Return the plasma species index for the given (anum, znum).

    The result MUST stay a numpy integer (``np.int_``): ``Afsi._init_afsi_data``
    uses ``isinstance(react, np.int_)`` to decide whether a reactant is a
    thermal species (int index) or a beam distribution (ctypes struct
    pointer). A plain Python ``int`` would fail that check.

    Parameters
    ----------
    a5 : :class:`a5py.Ascot`
        Ascot instance with ``input_init(plasma=True)`` already called.
    anum : int
        Mass number of the species to look up.
    znum : int
        Charge number of the species to look up.

    Returns
    -------
    index : numpy.int_
        Index of the species among the plasma input's ion species.

    Raises
    ------
    ValueError
        If no plasma species matches (anum, znum).
    """
    nspec, _mass, _charge, anums, znums = a5.input_getplasmaspecies()
    for i in np.arange(nspec):
        if anum == anums[i] and znum == znums[i]:
            return i
    raise ValueError(
        f"Species anum={anum} znum={znum} not found in plasma input "
        f"(available anum/znum pairs: "
        f"{list(zip(np.asarray(anums).tolist(), np.asarray(znums).tolist()))})")


def _run(a5, reaction, react1, react2, mult, beam, nmc,
         ppar1, pperp1, ppar2, pperp2, verbose=1):
    """Call ``afsi_run`` with the correct 7 arguments and ``save_output=0``.

    Parameters
    ----------
    a5 : :class:`a5py.Ascot`
        Ascot instance with ``input_init(bfield=True, plasma=True)`` already
        called (bfield is required because AFSI evaluates rho at each cell
        even though the (R,phi,z) beam-target/beam-beam channels do not use
        it for volumes).
    reaction : str
        Reaction name, e.g. ``"DD_He3n"`` or ``"DD_Tp"``.
    react1, react2 : ctypes struct or numpy.int_
        Either a plasma species index (thermal reactant, from
        ``_plasma_species_index``) or a ``STRUCT_HIST`` built by
        ``a5.afsi._init_dist_5d`` (beam reactant).
    mult : float
        1.0 for two distinguishable populations (beam-target), 0.5 for a
        self-reaction (beam-beam with beam2=None).
    beam : :class:`a5py.ascot5io.dist.DistData`
        The beam distribution defining the (R, phi, z) grid AFSI computes
        products on. (Both react1/react2, if beam-type, must already share
        this same grid -- enforced by the public wrappers below.)
    nmc : int
        Number of Monte Carlo samples per spatial cell.
    ppar1, pperp1, ppar2, pperp2 : array_like
        Product momentum grid edges [kg m/s], plain numpy arrays (no units),
        matching the convention used throughout ``a5py.routines.afsi5``.
    verbose : int, optional
        AFSI verbosity level (passed straight to the C code).

    Returns
    -------
    prod1, prod2 : :class:`a5py.ascot5io.dist.DistData`
        Fusion product distributions. Bins are reactions/s (already the
        physical rate; no further normalization is needed).
    """
    _, _, _, _, _, qprod1, _, qprod2, _ = a5.afsi.reactions(reaction)
    ireaction = {v: k for k, v in AFSI_REACTIONS.items()}[reaction]
    q1 = np.round(qprod1.to("e").v)
    q2 = np.round(qprod2.to("e").v)

    rc, phic, zc, vol = _spatial_grid(beam)
    afsi = a5.afsi._init_afsi_data(react1=react1, react2=react2,
                                    reaction=ireaction, mult=mult,
                                    r=rc, phi=phic, z=zc, vol=vol)
    edges = (beam.abscissa_edges("r"),
              beam.abscissa_edges("phi").to("rad"),
              beam.abscissa_edges("z"))
    prod1 = a5.afsi._init_histogram(*edges, ppar1, pperp1, charge=q1, exi=False)
    prod2 = a5.afsi._init_histogram(*edges, ppar2, pperp2, charge=q2, exi=False)

    # NOTE: 7 arguments, save_output=0 -- this is the fix for the upstream
    # bug described in this module's docstring. Do NOT drop verbose/0 here.
    _LIBASCOT.afsi_run(ctypes.byref(a5._sim), ctypes.byref(afsi), nmc,
                        prod1, prod2, verbose, 0)

    return a5.afsi._build_distdata(prod1), a5.afsi._build_distdata(prod2)


def thermal(a5, reaction, beam, nmc=1000,
            ppar1=None, pperp1=None, ppar2=None, pperp2=None,
            verbose=1, plasma=None):
    """Thermonuclear fusion between two thermal (Maxwellian) species, on
    the same (R, phi, z) grid as ``beam``.

    Same computation as ``a5.afsi.thermal`` in the (R,phi,z)/(ppar,pperp)
    basis: identical cell centres and volumes, species lookup, and
    ``mult = 0.5`` when both reactants are the same species. The only
    difference is that the plasma can be overridden.

    Parameters
    ----------
    a5 : :class:`a5py.Ascot`
        Open Ascot instance (bfield and plasma inputs must be loadable).
    reaction : str
        Reaction name, e.g. ``"DD_He3n"`` or ``"DD_Tp"``.
    beam : :class:`a5py.ascot5io.dist.DistData`
        Used ONLY for its (R, phi, z) grid; its contents are ignored.
    nmc : int, optional
        Number of Monte Carlo samples per spatial cell.
    ppar1, pperp1, ppar2, pperp2 : array_like, optional
        Product momentum grid edges [kg m/s] for products 1 and 2.
    verbose : int, optional
        AFSI verbosity level.
    plasma : dict, optional
        Plasma data (``plasma_1D.write_hdf5`` format) replacing the file's
        active plasma for this call. ``None`` uses the file's.

    Returns
    -------
    prod1, prod2 : :class:`a5py.ascot5io.dist.DistData`
        Fusion product distributions (reactions/s per bin).
    """
    ppar1 = 1.3e-19 * np.linspace(-1., 1., 80) if ppar1 is None else ppar1
    pperp1 = np.linspace(0, 1.3e-19, 40) if pperp1 is None else pperp1
    ppar2 = 1.3e-19 * np.linspace(-1., 1., 80) if ppar2 is None else ppar2
    pperp2 = np.linspace(0, 1.3e-19, 40) if pperp2 is None else pperp2

    m1, q1, m2, q2, *_ = a5.afsi.reactions(reaction)
    a5.input_init(bfield=True, plasma=True if plasma is None else plasma)
    try:
        ispecies1 = _plasma_species_index(
            a5, np.round(m1.to("amu").v), np.round(q1.to("e").v))
        ispecies2 = _plasma_species_index(
            a5, np.round(m2.to("amu").v), np.round(q2.to("e").v))
        mult = 0.5 if ispecies1 == ispecies2 else 1.0
        return _run(a5, reaction, ispecies1, ispecies2, mult, beam, nmc,
                    ppar1, pperp1, ppar2, pperp2, verbose=verbose)
    finally:
        a5.input_free(bfield=True, plasma=True)


def beamthermal(a5, reaction, beam, nmc=1000,
                 ppar1=None, pperp1=None, ppar2=None, pperp2=None,
                 verbose=1, plasma=None):
    """Beam-target fusion: ``beam`` against the thermal background species.

    The thermal reactant's plasma species (density/temperature profile) is
    resolved automatically from the reaction's reactant masses/charges via
    ``a5.input_getplasmaspecies()`` -- the same logic the (broken) upstream
    ``Afsi.beamthermal`` uses, just with the corrected ``afsi_run`` call.

    Parameters
    ----------
    a5 : :class:`a5py.Ascot`
        Open Ascot instance (bfield and plasma inputs must be loadable).
    reaction : str
        Reaction name, e.g. ``"DD_He3n"`` or ``"DD_Tp"``.
    beam : :class:`a5py.ascot5io.dist.DistData`
        Fast-ion (beam) distribution with a *single* time bin and a *single*
        charge bin (AFSI's beam sampler ignores the time/charge axes and
        silently reads bin 0 of each -- see this module's docstring).
    nmc : int, optional
        Number of Monte Carlo samples per spatial cell.
    ppar1, pperp1, ppar2, pperp2 : array_like, optional
        Product momentum grid edges [kg m/s] for products 1 and 2. Defaults
        mirror ``a5py.routines.afsi5.Afsi.beamthermal``'s defaults.
    verbose : int, optional
        AFSI verbosity level.
    plasma : dict, optional
        Plasma data (``plasma_1D.write_hdf5`` format) replacing the file's
        active plasma for this call. ``None`` uses the file's.

    Returns
    -------
    prod1, prod2 : :class:`a5py.ascot5io.dist.DistData`
        Fusion product distributions (reactions/s per bin).
    """
    ppar1 = 1.3e-19 * np.linspace(-1., 1., 50) if ppar1 is None else ppar1
    pperp1 = np.linspace(0, 1.3e-19, 50) if pperp1 is None else pperp1
    ppar2 = 1.3e-19 * np.linspace(-1., 1., 50) if ppar2 is None else ppar2
    pperp2 = np.linspace(0, 1.3e-19, 50) if pperp2 is None else pperp2

    _m1, _q1, m2, q2, *_ = a5.afsi.reactions(reaction)
    a5.input_init(bfield=True, plasma=True if plasma is None else plasma)
    try:
        ispecies = _plasma_species_index(
            a5, np.round(m2.to("amu").v), np.round(q2.to("e").v))
        react1 = a5.afsi._init_dist_5d(beam)
        return _run(a5, reaction, react1, ispecies, 1.0, beam, nmc,
                    ppar1, pperp1, ppar2, pperp2, verbose=verbose)
    finally:
        a5.input_free(bfield=True, plasma=True)


def beambeam(a5, reaction, beam, nmc=1000,
             ppar1=None, pperp1=None, ppar2=None, pperp2=None,
             verbose=1, plasma=None):
    """Beam-beam fusion: ``beam`` against itself (self-reaction, mult=0.5).

    Parameters
    ----------
    a5 : :class:`a5py.Ascot`
        Open Ascot instance (bfield and plasma inputs must be loadable).
    reaction : str
        Reaction name, e.g. ``"DD_He3n"`` or ``"DD_Tp"``.
    beam : :class:`a5py.ascot5io.dist.DistData`
        Fast-ion (beam) distribution with a single time bin and a single
        charge bin (see :func:`beamthermal`).
    nmc : int, optional
        Number of Monte Carlo samples per spatial cell.
    ppar1, pperp1, ppar2, pperp2 : array_like, optional
        Product momentum grid edges [kg m/s] for products 1 and 2.
    verbose : int, optional
        AFSI verbosity level.
    plasma : dict, optional
        Plasma data replacing the file's active plasma for this call. The
        beam-beam rate itself doesn't depend on the plasma; this only keeps
        the initialized inputs consistent across channels.

    Returns
    -------
    prod1, prod2 : :class:`a5py.ascot5io.dist.DistData`
        Fusion product distributions (reactions/s per bin).
    """
    ppar1 = 1.3e-19 * np.linspace(-1., 1., 80) if ppar1 is None else ppar1
    pperp1 = np.linspace(0, 1.3e-19, 40) if pperp1 is None else pperp1
    ppar2 = 1.3e-19 * np.linspace(-1., 1., 80) if ppar2 is None else ppar2
    pperp2 = np.linspace(0, 1.3e-19, 40) if pperp2 is None else pperp2

    a5.input_init(bfield=True, plasma=True if plasma is None else plasma)
    try:
        react = a5.afsi._init_dist_5d(beam)
        return _run(a5, reaction, react, react, 0.5, beam, nmc,
                    ppar1, pperp1, ppar2, pperp2, verbose=verbose)
    finally:
        a5.input_free(bfield=True, plasma=True)

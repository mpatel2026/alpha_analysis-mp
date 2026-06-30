#!/usr/bin/env python
"""Build magnetic field arrays for Manav's two-method Poincare comparison.

Ports the field-construction half of Manav's `desc_field_extended` from
`ascotBfield.py`. Produces both pure (use_mixed_field=False) and mixed
(use_mixed_field=True) outputs in a single pass, sharing the expensive
PlasmaField construction.

CPU only. A_res=128 will OOM on an A100 80GB GPU.

Usage:
    python build_field.py                  # full run (A_RES=128, NPHI=50)
    python build_field.py --smoke          # quick sanity (A_RES=8, NPHI=2)
    python build_field.py --A-res 64       # fall back if 128 OOMs

Outputs (under out/):
    field_pure.h5, field_mixed.h5    # for tracing
    psi_grid.h5                       # for inspection
Each field .h5: R_1d, phi_1d, Z_1d, BR, Bphi, BZ (shape NR,Nphi,NZ),
plus attrs NFP, R0, bounds_R, bounds_Z, eq_Psi, A_res.
"""

import argparse
import os
import sys
import time

# Force CPU before any JAX imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.expanduser("~/DESC"))

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import NearestNDInterpolator, griddata

from desc import set_device

set_device("cpu")

from desc import grid as dscg
from desc.coils import MixedCoilSet
from desc.grid import QuadratureGrid
from desc.io import load
from desc.magnetic_fields import PlasmaField


# ── Config ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "out")

EQ_FILE = f"{DATA_DIR}/equil_Helios_G1600-12-89_QA2e-1_Bxdl25_free_L12_M12_N20.h5"
ENC_FILE = f"{DATA_DIR}/encircling_G1600-12-89_QA2e-1_Bxdl25.h5"
SHA_FILE = f"{DATA_DIR}/shaping_G1600-12-89_QA2e-1_Bxdl25.h5"

WALL_OFFSET = 1.0  # m. Manav uses 100 cm.
BFIELD_OFFSET = WALL_OFFSET * 1.2  # extra cushion for spline domain
NTHETA_BDRY = 360  # boundary mesh resolution for bbox
NPHI_BDRY = 200
L_RADIAL = 4
M_POLOIDAL = 4

# psi inside-LCFS buffer (Wb). Manav uses 1.0; same.
PSI_BUFFER = 1.0


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Quick sanity run: A_RES=8, NPHI=2, NR=NZ=40.")
    p.add_argument("--A-res", type=int, default=128)
    p.add_argument("--NR", type=int, default=200)
    p.add_argument("--NZ", type=int, default=200)
    p.add_argument("--NPHI", type=int, default=50)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--tag", default="",
                   help="Tag suffix for output filenames (e.g. 'A64').")
    return p.parse_args()


# ── Helpers ─────────────────────────────────────────────────────────────────
def fill_nans(arr):
    """Replace NaN cells using NearestNDInterpolator. Returns a new array."""
    bad = np.isnan(arr)
    if not bad.any():
        return arr
    good = ~bad
    interp = NearestNDInterpolator(
        np.transpose(np.nonzero(good)), arr[good]
    )
    out = arr.copy()
    out[bad] = interp(*np.nonzero(bad))
    return out


def lcfs_bbox(eq):
    """Return (rmin, rmax, zmin, zmax) for the LCFS over a fine (theta,phi) mesh."""
    g = dscg.LinearGrid(
        rho=1.0, theta=NTHETA_BDRY, zeta=NPHI_BDRY,
        NFP=1, sym=False, endpoint=True,
    )
    d = eq.compute(["R", "Z"], grid=g)
    R = np.asarray(d["R"])
    Z = np.asarray(d["Z"])
    return float(R.min()), float(R.max()), float(Z.min()), float(Z.max())


def save_field(path, R_1d, phi_1d, Z_1d, BR, Bphi, BZ,
               NFP, R0, bounds_R, bounds_Z, eq_Psi, A_res):
    with h5py.File(path, "w") as f:
        f.create_dataset("R_1d", data=R_1d)
        f.create_dataset("phi_1d", data=phi_1d)
        f.create_dataset("Z_1d", data=Z_1d)
        f.create_dataset("BR", data=BR)
        f.create_dataset("Bphi", data=Bphi)
        f.create_dataset("BZ", data=BZ)
        f.attrs["NFP"] = int(NFP)
        f.attrs["R0"] = float(R0)
        f.attrs["bounds_R"] = bounds_R
        f.attrs["bounds_Z"] = bounds_Z
        f.attrs["eq_Psi"] = float(eq_Psi)
        f.attrs["A_res"] = int(A_res)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.smoke:
        A_RES, NR, NZ, NPHI = 8, 40, 40, 2
        tag = "_smoke"
    else:
        A_RES, NR, NZ, NPHI = args.A_res, args.NR, args.NZ, args.NPHI
        tag = ("_" + args.tag) if args.tag else ""

    os.makedirs(args.out, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading eq + coils...")
    fam = load(EQ_FILE)
    eq = fam[-1] if hasattr(fam, "__getitem__") else fam
    encircling = load(ENC_FILE)
    shaping = load(SHA_FILE)
    if isinstance(encircling, (list, tuple)):
        encircling = encircling[-1]
    if isinstance(shaping, (list, tuple)):
        shaping = shaping[-1]
    coils = MixedCoilSet((encircling, shaping), check_intersection=False)

    NFP = int(eq.NFP)
    R0 = float(eq.get_axis().R_n[0])
    psi1 = float(eq.Psi)
    print(f"  NFP={NFP}, R0={R0:.4f}, eq.Psi={psi1:.4f}")
    print(f"  L={eq.L} M={eq.M} N={eq.N} | "
          f"L_grid={eq.L_grid} M_grid={eq.M_grid} N_grid={eq.N_grid}")

    print("  Computing LCFS bbox...")
    rmin_lcfs, rmax_lcfs, zmin_lcfs, zmax_lcfs = lcfs_bbox(eq)
    rmin = rmin_lcfs - BFIELD_OFFSET
    rmax = rmax_lcfs + BFIELD_OFFSET
    zmin = zmin_lcfs - BFIELD_OFFSET
    zmax = zmax_lcfs + BFIELD_OFFSET
    print(f"    LCFS bbox: R=[{rmin_lcfs:.3f}, {rmax_lcfs:.3f}] "
          f"Z=[{zmin_lcfs:.3f}, {zmax_lcfs:.3f}]")
    print(f"    Spline domain (+{BFIELD_OFFSET}m): "
          f"R=[{rmin:.3f}, {rmax:.3f}] Z=[{zmin:.3f}, {zmax:.3f}]")

    R_1d = np.linspace(rmin, rmax, NR)
    Z_1d = np.linspace(zmin, zmax, NZ)
    # One field period, endpoint=False (matches SplineMagneticField convention).
    phi_1d = np.linspace(0.0, 2 * np.pi / NFP, NPHI, endpoint=False)
    Z_2d, R_2d = np.meshgrid(Z_1d, R_1d)  # both shape (NR, NZ)

    # ── PlasmaField (built once) ───────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] "
          f"Building PlasmaField (A_res={A_RES})...")
    t0 = time.time()
    source_grid = QuadratureGrid(
        L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=NFP,
    )
    plasma = PlasmaField(
        eq,
        source_grid=source_grid,
        R_bounds=(rmin, rmax),
        Z_bounds=(zmin, zmax),
        A_res=A_RES,
        chunk_size=200,
    )
    print(f"  PlasmaField OK in {time.time() - t0:.1f}s")

    # ── JIT'd coil B-field call ────────────────────────────────────────────
    @jax.jit
    def coil_B(coords):
        return coils.compute_magnetic_field(
            coords, source_grid=None, chunk_size=10000,
        )

    r_jax = jnp.asarray(R_2d.ravel())
    z_jax = jnp.asarray(Z_2d.ravel())

    # ── Concentric grid (built once; mutate zeta per iteration) ────────────
    cgrid = dscg.ConcentricGrid(
        L=eq.L_grid * L_RADIAL, M=eq.M_grid * M_POLOIDAL, N=0,
        NFP=NFP, node_pattern="linear",
    )

    # ── Output buffers (NR, NPHI, NZ) ──────────────────────────────────────
    shape3d = (NR, NPHI, NZ)
    br_pure = np.zeros(shape3d)
    bphi_pure = np.zeros(shape3d)
    bz_pure = np.zeros(shape3d)
    br_mixed = np.zeros(shape3d)
    bphi_mixed = np.zeros(shape3d)
    bz_mixed = np.zeros(shape3d)
    psi_2d_all = np.zeros(shape3d)
    inside_frac = np.zeros(NPHI)

    # ── Per-phi loop ───────────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] "
          f"Building field on {NR}×{NPHI}×{NZ} grid...")
    t1 = time.time()

    for k, iphi in enumerate(phi_1d):
        if k % max(1, NPHI // 10) == 0:
            print(f"  phi {k+1}/{NPHI}  "
                  f"({(time.time()-t1):.1f}s elapsed)", flush=True)
        # 1. Coils
        phi_jax = jnp.full_like(r_jax, iphi)
        coords = jnp.column_stack([r_jax, phi_jax, z_jax])
        bcoil = np.asarray(coil_B(coords))
        br_c = bcoil[:, 0].reshape(NR, NZ)
        bphi_c = bcoil[:, 1].reshape(NR, NZ)
        bz_c = bcoil[:, 2].reshape(NR, NZ)

        # 2. Plasma (PlasmaField)
        bplasma = np.asarray(
            plasma.compute_magnetic_grid(R_1d, iphi, Z_1d, NFP)
        ).reshape(-1, 3)
        br_p = bplasma[:, 0].reshape(NR, NZ)
        bphi_p = bplasma[:, 1].reshape(NR, NZ)
        bz_p = bplasma[:, 2].reshape(NR, NZ)

        # 3. Eq concentric: R, Z, psi, B_R, B_phi, B_Z
        cgrid._nodes[:, 2] = iphi
        cdata = eq.compute(
            ["R", "Z", "psi", "B_R", "B_phi", "B_Z"], grid=cgrid,
        )
        cR = np.asarray(cdata["R"])
        cZ = np.asarray(cdata["Z"])
        cpsi = np.asarray(cdata["psi"]) * 2 * np.pi  # → Ψ·ρ²
        cBR = np.asarray(cdata["B_R"])
        cBphi = np.asarray(cdata["B_phi"])
        cBZ = np.asarray(cdata["B_Z"])

        psi_grid = griddata((cR, cZ), cpsi, (R_2d, Z_2d), fill_value=psi1)
        psi_2d_all[:, k, :] = psi_grid

        # 4. Pure: coil + plasma everywhere
        brP = br_c + br_p
        bphiP = bphi_c + bphi_p
        bzP = bz_c + bz_p

        # 5. Mixed: replace inside-LCFS with eq.compute B
        inside = psi_grid < (psi1 - PSI_BUFFER)
        inside_frac[k] = inside.mean()

        brM = brP.copy()
        bphiM = bphiP.copy()
        bzM = bzP.copy()
        if inside.any():
            brI = griddata((cR, cZ), cBR, (R_2d, Z_2d))
            bphiI = griddata((cR, cZ), cBphi, (R_2d, Z_2d))
            bzI = griddata((cR, cZ), cBZ, (R_2d, Z_2d))
            brM[inside] = brI[inside]
            bphiM[inside] = bphiI[inside]
            bzM[inside] = bzI[inside]

        # 6. Repair NaNs (Manav's approach)
        br_pure[:, k, :] = fill_nans(brP)
        bphi_pure[:, k, :] = fill_nans(bphiP)
        bz_pure[:, k, :] = fill_nans(bzP)
        br_mixed[:, k, :] = fill_nans(brM)
        bphi_mixed[:, k, :] = fill_nans(bphiM)
        bz_mixed[:, k, :] = fill_nans(bzM)

        if k > 0 and k % 25 == 0:
            jax.clear_caches()

    print(f"  Field loop done in {time.time() - t1:.1f}s")
    print(f"  Inside-LCFS fraction per phi: "
          f"min={inside_frac.min():.3f}, max={inside_frac.max():.3f}, "
          f"mean={inside_frac.mean():.3f}")

    # ── Save ───────────────────────────────────────────────────────────────
    bounds_R = (rmin, rmax)
    bounds_Z = (zmin, zmax)
    pure_path = f"{args.out}/field_pure{tag}.h5"
    mixed_path = f"{args.out}/field_mixed{tag}.h5"
    psi_path = f"{args.out}/psi_grid{tag}.h5"

    save_field(pure_path, R_1d, phi_1d, Z_1d,
               br_pure, bphi_pure, bz_pure,
               NFP, R0, bounds_R, bounds_Z, psi1, A_RES)
    save_field(mixed_path, R_1d, phi_1d, Z_1d,
               br_mixed, bphi_mixed, bz_mixed,
               NFP, R0, bounds_R, bounds_Z, psi1, A_RES)

    with h5py.File(psi_path, "w") as f:
        f.create_dataset("R_1d", data=R_1d)
        f.create_dataset("phi_1d", data=phi_1d)
        f.create_dataset("Z_1d", data=Z_1d)
        f.create_dataset("psi", data=psi_2d_all)
        f.attrs["psi1"] = psi1

    # |B| sanity print
    bmag_pure = np.sqrt(br_pure**2 + bphi_pure**2 + bz_pure**2)
    bmag_mixed = np.sqrt(br_mixed**2 + bphi_mixed**2 + bz_mixed**2)
    print(f"  |B| pure : min={bmag_pure.min():.2f}T "
          f"median={np.median(bmag_pure):.2f}T "
          f"max={bmag_pure.max():.2f}T")
    print(f"  |B| mixed: min={bmag_mixed.min():.2f}T "
          f"median={np.median(bmag_mixed):.2f}T "
          f"max={bmag_mixed.max():.2f}T")

    print(f"\nSaved: {pure_path}\n       {mixed_path}\n       {psi_path}")
    print(f"[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()

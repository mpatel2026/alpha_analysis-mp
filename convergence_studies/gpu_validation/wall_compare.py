#!/usr/bin/env python3
"""Compare power loss and wall loads of two ASCOT5 runs, with error bars.

Each argument is FILE.h5 (most recent run) or FILE.h5:run_XXXXXXXXXX.

    python wall_compare.py REF.h5 TEST.h5 [--mode gc|prt] [--nboot 200]

Reported per run:
- marker, particle and power loss fractions, with bootstrap 1-sigma errors;
- total deposited power, wetted area, peak load (a single-tile maximum, so
  noisy at finite marker counts);
- wall loads binned coarsely by tile centroid (phi x z). Each bin difference
  is compared with its Monte Carlo error sqrt(sum (w E)^2) from both runs.
"""
import argparse

import h5py
import numpy as np
from a5py import Ascot


def open_run(spec):
    fn, _, run = spec.partition(":")
    if not run:
        with h5py.File(fn, "r") as f:
            runs = [k for k in f["results"] if k.startswith("run_")]
            run = max(runs, key=lambda r: str(f["results"][r].attrs["date"]))
    a5 = Ascot(fn)
    return a5, a5.data[run], f"{fn.split('/')[-1]}:{run}"


def tile_centroids(a5):
    w = a5.data.wall.active.read()
    x = np.mean(w["x1x2x3"], axis=1)
    y = np.mean(w["y1y2y3"], axis=1)
    z = np.mean(w["z1z2z3"], axis=1)
    return np.mod(np.degrees(np.arctan2(y, x)), 360), z


def _v(x):
    """Plain array from a unyt array or an ndarray."""
    return np.asarray(getattr(x, "v", x))


def losses(run, mode, nboot, rng):
    ids0, w0, e0 = run.getstate("ids", "weight", "ekin", state="ini", mode=mode)
    ids1, w1, e1 = run.getstate("ids", "weight", "ekin", state="end",
                                mode=mode, endcond="wall")
    ids0, w0, e0 = _v(ids0), _v(w0), _v(e0.to("J"))
    o0 = np.argsort(ids0)
    ids0, w0, e0 = ids0[o0], w0[o0], e0[o0]
    # End-state weight and energy of wall-lost markers, aligned to ini order
    pos = np.searchsorted(ids0, _v(ids1))
    lost = np.zeros(ids0.size, bool)
    lost[pos] = True
    pout = np.zeros(ids0.size)
    wout = np.zeros(ids0.size)
    pout[pos] = _v(w1) * _v(e1.to("J"))
    wout[pos] = _v(w1)
    n = ids0.size
    pin = w0 * e0
    vals = np.array([lost.mean(), wout.sum() / w0.sum(), pout.sum() / pin.sum()])
    boot = np.empty((nboot, 3))
    for b in range(nboot):
        i = rng.integers(0, n, n)
        boot[b] = [lost[i].mean(), wout[i].sum() / w0[i].sum(),
                   pout[i].sum() / pin[i].sum()]
    return vals, boot.std(axis=0), pin.sum()


def binned_loads(run, phi_c, z_c, phi_edges, z_edges):
    ids, energy, weight = run.getstate(
        "walltile", "ekin", "weight", state="end", mode="prt", endcond="wall")
    p = energy.to("J").v * weight.v
    iphi = np.clip(np.digitize(phi_c[ids - 1], phi_edges) - 1, 0,
                   phi_edges.size - 2)
    iz = np.clip(np.digitize(z_c[ids - 1], z_edges) - 1, 0, z_edges.size - 2)
    shape = (phi_edges.size - 1, z_edges.size - 1)
    load = np.zeros(shape)
    var = np.zeros(shape)
    np.add.at(load, (iphi, iz), p)
    np.add.at(var, (iphi, iz), p**2)
    return load, var


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--mode", default="gc", choices=["gc", "prt"],
                    help="coordinates for ekin: gc for GC runs, prt for FO")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--nphi", type=int, default=36)
    ap.add_argument("--nz", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    res = []
    for spec in (args.ref, args.test):
        a5, run, label = open_run(spec)
        vals, err, ptot = losses(run, args.mode, args.nboot, rng)
        warea, peak = run.getwall_figuresofmerit()
        _, area, edepo, _, _ = run.getwall_loads()
        res.append(dict(a5=a5, run=run, label=label, vals=vals, err=err,
                        warea=float(warea.v), peak=float(peak.v),
                        pdep=float(np.sum(edepo).v), ptot=ptot))

    print(f"REF : {res[0]['label']}\nTEST: {res[1]['label']}\n")
    names = ["marker loss fraction", "particle loss fraction",
             "power loss fraction"]
    print(f"{'quantity':26s} {'ref':>18s} {'test':>18s} {'diff/sigma':>11s}")
    for k, name in enumerate(names):
        a, b = res[0]["vals"][k], res[1]["vals"][k]
        sa, sb = res[0]["err"][k], res[1]["err"][k]
        z = (b - a) / np.hypot(sa, sb)
        print(f"{name:26s} {100*a:8.3f}+-{100*sa:.3f} % "
              f"{100*b:8.3f}+-{100*sb:.3f} % {z:+10.2f}")
    for key, name, unit in [("pdep", "deposited power", "W"),
                            ("warea", "wetted area", "m2"),
                            ("peak", "peak load (1 tile)", "W/m2")]:
        a, b = res[0][key], res[1][key]
        print(f"{name:26s} {a:18.4e} {b:18.4e}   ratio {b/a:.4f}  [{unit}]")

    phi_c, z_c = tile_centroids(res[0]["a5"])
    phi_edges = np.linspace(0, 360, args.nphi + 1)
    z_edges = np.linspace(z_c.min(), z_c.max() + 1e-9, args.nz + 1)
    l0, v0 = binned_loads(res[0]["run"], phi_c, z_c, phi_edges, z_edges)
    l1, v1 = binned_loads(res[1]["run"], phi_c, z_c, phi_edges, z_edges)
    sig = np.sqrt(v0 + v1)
    used = sig > 0
    zb = (l1 - l0)[used] / sig[used]
    frac_top = np.sort(l0.ravel())[::-1]
    ntop = np.searchsorted(np.cumsum(frac_top) / frac_top.sum(), 0.9) + 1
    print(f"\nBinned wall loads ({args.nphi} phi x {args.nz} z bins on tile "
          f"centroids): {used.sum()} bins loaded")
    print(f"  bin differences / sigma: rms {np.sqrt(np.mean(zb**2)):.2f}, "
          f"max |z| {np.max(np.abs(zb)):.2f}, |z|>3 in {np.sum(np.abs(zb)>3)} "
          f"bins (expect rms ~1 and ~{0.0027*used.sum():.1f} bins >3 by chance)")
    print(f"  90% of the reference load sits in {ntop} bins")
    top = np.argsort(l0.ravel())[::-1][:5]
    print("  five most loaded bins (phi range, z range): ref vs test [W]")
    for t in top:
        i, j = np.unravel_index(t, l0.shape)
        s = np.sqrt(v0[i, j] + v1[i, j])
        print(f"    phi {phi_edges[i]:5.0f}-{phi_edges[i+1]:3.0f} deg, z "
              f"{z_edges[j]:+5.2f}..{z_edges[j+1]:+5.2f} m: {l0[i,j]:.3e} vs "
              f"{l1[i,j]:.3e}  ({(l1[i,j]-l0[i,j])/s:+.2f} sigma)")


if __name__ == "__main__":
    main()

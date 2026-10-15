#!/usr/bin/env python3
"""Compare ASCOT5 validation runs against a reference run.

Each file argument is FILE.h5 (the most recent run in it is used) or
FILE.h5:run_XXXXXXXXXX.

Bit-exact mode (--bitwise): marker end states, matched by id, must be
identical in every field except cputime. Use it for builds that should not
change results (e.g. LEGACY=1 build against the original code).

Statistical mode (default): end-condition fractions (binomial z-score), and
two-sample Kolmogorov-Smirnov tests of end energy, end time and rho, plus
the total weight in the distributions. Use it for changes that alter random
numbers or rounding, CPU vs GPU, and 1 vs 2 nodes.

    python compare_runs.py REF.h5 TEST.h5 [TEST2.h5 ...] [--bitwise]
"""
import argparse
import sys

import h5py
import numpy as np

ENDCONDS = [(0x1, "tlim"), (0x2, "emin"), (0x4, "therm"), (0x8, "wall"),
            (0x10, "rhomin"), (0x20, "rhomax"), (0x40, "polmax"),
            (0x80, "tormax"), (0x100, "cpumax"), (0x400, "neutr"),
            (0x800, "ioniz"), (0x1000, "flrwall")]
SKIP_BITWISE = {"cputime"}
C = 299792458.0
E = 1.602176634e-19


def open_run(spec):
    fn, _, run = spec.partition(":")
    f = h5py.File(fn, "r")
    if not run:
        runs = [k for k in f["results"] if k.startswith("run_")]
        run = max(runs, key=lambda r: str(f["results"][r].attrs["date"]))
    return f, f["results"][run], f"{fn.split('/')[-1]}:{run}"


def endstate(grp):
    es = grp["endstate"]
    data = {k: es[k][:].ravel() for k in es}
    order = np.argsort(data["ids"])
    return {k: v[order] for k, v in data.items()}


def ekin_gc(s):
    """Guiding-center kinetic energy [eV] from ppar, mu and |B|."""
    m = s["mass"] * 1.66053906660e-27
    bnorm = np.sqrt(s["br"]**2 + s["bphi"]**2 + s["bz"]**2)
    mu = s["mu"] * E  # stored in eV/T
    p2 = s["ppar"]**2 + 2 * m * mu * bnorm
    return (np.sqrt((m * C**2)**2 + p2 * C**2) - m * C**2) / E


def ks_2samp(a, b):
    """Two-sample KS statistic and asymptotic p-value (no scipy needed)."""
    a, b = np.sort(a), np.sort(b)
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    x = np.concatenate([a, b])
    d = np.max(np.abs(np.searchsorted(a, x, side="right") / a.size
                      - np.searchsorted(b, x, side="right") / b.size))
    ne = a.size * b.size / (a.size + b.size)
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    k = np.arange(1, 101)
    p = 2 * np.sum((-1)**(k - 1) * np.exp(-2 * (k * lam)**2))
    return d, float(np.clip(p, 0, 1))


def dist_totals(grp):
    out = {}
    for name in ("dist5d", "distrho5d", "distcom"):
        if name in grp:
            out[name] = float(np.sum(grp[name]["ordinate"][:]))
    return out


def compare_bitwise(ref, test):
    ok = True
    if not np.array_equal(ref["ids"], test["ids"]):
        print("  FAIL marker ids differ")
        return False
    # Aborted markers can hold uninitialized state data (e.g. markers rejected
    # at initialization for zero weight), so require the same markers to abort
    # with the same error, and compare fields only for the others.
    aborted = ref["errormsg"] != 0
    same = np.array_equal(aborted, test["errormsg"] != 0) and all(
        np.array_equal(ref[k][aborted], test[k][aborted])
        for k in ("errormsg", "errorline", "errormod"))
    if not same:
        print("  FAIL aborted markers or their errors differ")
        ok = False
    elif aborted.any():
        print(f"  note: {aborted.sum()} markers aborted with identical errors in "
              "both runs; excluded from the field comparison")
    rejected = aborted
    keep = ~rejected
    for key in sorted(ref):
        if key in SKIP_BITWISE or key not in test:
            continue
        a, b = ref[key][keep], test[key][keep]
        same = np.array_equal(a, b) if a.dtype.kind in "iub" else \
            np.array_equal(a, b, equal_nan=True)
        if not same:
            ok = False
            diff = np.nonzero(a != b)[0]
            md = np.max(np.abs(a[diff].astype(float) - b[diff].astype(float)))
            print(f"  FAIL {key:10s} {diff.size} markers differ, "
                  f"max |diff| {md:.3e}")
    print("  PASS all end-state fields identical" if ok else "")
    return ok


def compare_stats(ref, test, dref, dtest):
    ok = True
    n1, n2 = ref["ids"].size, test["ids"].size
    print(f"  {'end condition':14s} {'ref':>8s} {'test':>8s}   z")
    for bit, name in ENDCONDS:
        k1 = np.sum(ref["endcond"] & bit > 0)
        k2 = np.sum(test["endcond"] & bit > 0)
        if k1 == 0 and k2 == 0:
            continue
        p1, p2 = k1 / n1, k2 / n2
        pp = (k1 + k2) / (n1 + n2)
        z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) \
            if 0 < pp < 1 else 0.0
        flag = "  <-- |z|>3" if abs(z) > 3 else ""
        ok &= abs(z) <= 3
        print(f"  {name:14s} {p1:8.4f} {p2:8.4f} {z:6.2f}{flag}")
    n_abort = [np.sum(s["errormsg"] != 0) for s in (ref, test)]
    print(f"  {'aborted':14s} {n_abort[0]:8d} {n_abort[1]:8d}")
    ok &= n_abort[1] <= max(n_abort[0], 0)

    # Aborted markers may hold uninitialized data; use the others only
    r_ok, t_ok = ref["errormsg"] == 0, test["errormsg"] == 0
    rs = {k: v[r_ok] for k, v in ref.items()}
    ts = {k: v[t_ok] for k, v in test.items()}
    e1, e2 = ekin_gc(rs), ekin_gc(ts)
    for label, a, b in [
            ("end energy", e1, e2),
            ("end time", rs["time"], ts["time"]),
            ("end rho", rs["rho"], ts["rho"]),
            ("wall-hit time", rs["time"][rs["endcond"] & 0x8 > 0],
             ts["time"][ts["endcond"] & 0x8 > 0])]:
        d, p = ks_2samp(a, b)
        flag = "  <-- p<0.001" if p < 1e-3 else ""
        ok &= not (p < 1e-3)
        print(f"  KS {label:14s} D={d:.4f} p={p:.3f} "
              f"(mean {np.mean(a):.4g} vs {np.mean(b):.4g}){flag}")
    for name in dref:
        if name in dtest and dref[name] != 0:
            r = dtest[name] / dref[name]
            print(f"  {name} total weight ratio test/ref = {r:.4f}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref")
    parser.add_argument("tests", nargs="+")
    parser.add_argument("--bitwise", action="store_true")
    args = parser.parse_args()

    fref, gref, lref = open_run(args.ref)
    ref, dref = endstate(gref), dist_totals(gref)
    allok = True
    for spec in args.tests:
        ft, gt, lt = open_run(spec)
        print(f"\n{lt}  vs  reference {lref}")
        test = endstate(gt)
        if args.bitwise:
            ok = compare_bitwise(ref, test)
        else:
            ok = compare_stats(ref, test, dref, dist_totals(gt))
        print("  RESULT:", "PASS" if ok else "CHECK")
        allok &= ok
        ft.close()
    fref.close()
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-MPI-rank distribution footprint for an ASCOT5 input, and the
ntasks-per-node that follows from it on a Perlmutter CPU node.

Each rank calloc's its OWN full copy of every enabled histogram
(diag/dist_5D.c:50 etc.) and all its threads update it with #pragma omp atomic.
So ranks/node is really a NUMA-locality vs memory trade:

  Perlmutter CPU node = 2x EPYC 7763, 8 NUMA domains x 16 cores x ~64 GB.
  1 rank/node  -> histogram pages first-touched at random across 8 domains,
                  so ~7/8 of the atomic updates go to remote NUMA.
  8 ranks/node -> one rank per NUMA domain, every update local.
"""
import sys, h5py, numpy as np

GB = 1024**3
DOMAIN_GB, NDOMAIN = 64, 8
HEADROOM = 0.85          # leave room for splines/wall/plasma/marker states

def nbins(o, keys):
    n = 1
    for k in keys:
        n *= int(np.array(o[f"DIST_NBIN_{k}"]).ravel()[0])
    return n

DISTS = {
    "5D":    ("ENABLE_DIST_5D",    ["R","PHI","Z","PPA","PPE","TIME","CHARGE"]),
    "6D":    ("ENABLE_DIST_6D",    ["R","PHI","Z","PR","PPHI","PZ","TIME","CHARGE"]),
    "rho5D": ("ENABLE_DIST_RHO5D", ["RHO","THETA","PHI","PPA","PPE","TIME","CHARGE"]),
    "rho6D": ("ENABLE_DIST_RHO6D", ["RHO","THETA","PHI","PR","PPHI","PZ","TIME","CHARGE"]),
    "COM":   ("ENABLE_DIST_COM",   ["MU","EKIN","PTOR"]),
}

def main(path, qid=None):
    with h5py.File(path, "r") as h:
        og = h["options"]
        key = qid or og.attrs.get("active")
        if isinstance(key, bytes): key = key.decode()
        key = key if key in og else [k for k in og if str(key) in k][0]
        o = og[key]
        print(f"{path}\noptions group: {key}\n")
        total = 0
        for name, (flag, keys) in DISTS.items():
            if not int(np.array(o[flag]).ravel()[0]):
                continue
            n = nbins(o, keys)
            b = n * 8
            total += b
            dims = " x ".join(str(int(np.array(o[f'DIST_NBIN_{k}']).ravel()[0])) for k in keys)
            print(f"  dist{name:<6} {dims:<34} {n:>14,} bins  {b/GB:>8.2f} GB")
        print(f"  {'TOTAL PER RANK':<42}{'':>14}   {total/GB:>8.2f} GB\n")

        if total == 0:
            print("No distributions enabled -- ranks/node is unconstrained by memory.")
            return
        d = total / GB
        print(f"{'ranks/node':>10} {'cpus-per-task':>14} {'NUMA domains/rank':>18} {'mem/node':>10}  verdict")
        best = None
        for r in (8, 4, 2, 1):
            per_dom = d * r / NDOMAIN          # GB of histogram landing in each domain
            fits = per_dom <= DOMAIN_GB * HEADROOM
            if fits and best is None:
                best = r
            print(f"{r:>10} {256//r:>14} {NDOMAIN//r:>18} {d*r:>9.1f} GB  "
                  f"{'OK' if fits else 'EXCEEDS 64 GB/domain'}"
                  f"{'   <-- use this' if fits and best == r else ''}")
        print()
        if best == 8:
            print("=> 8 ranks/node: one per NUMA domain, all histogram atomics stay local.")
        elif best:
            print(f"=> {best} ranks/node is the finest split that fits; "
                  f"each rank still spans {NDOMAIN//best} NUMA domains, so some "
                  "atomic traffic stays remote.")
        else:
            print("=> Distribution does not fit even at 1 rank/node. Cut DIST_NBIN_TIME "
                  "or the momentum bins.")
        print(f"\n   #SBATCH --ntasks-per-node={best}\n   #SBATCH --cpus-per-task={256//best}\n"
              f"   export OMP_NUM_THREADS={256//best}; export OMP_PROC_BIND=close")

if __name__ == "__main__":
    main(*sys.argv[1:])

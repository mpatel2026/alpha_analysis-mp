#!/usr/bin/env python3
"""Pick --nodes / --ntasks-per-node for an ASCOT5 CPU run on Perlmutter.

Stock build assumptions: NSIMD=16, 1 rank per NUMA domain where memory allows.

Two constraints, both from the source:

  1. working_threads = n_markers // NSIMD
     particle_cycle_gc() hands each thread NSIMD markers on its first call and
     simulate_gc_adaptive does `while(n_running > 0)`, so any thread beyond
     that count exits without doing work.

  2. mpi_size must divide n_markers.
     mpi_my_particles() (mpi_interface.c:93) gives ranks 0..size-2 exactly
     n_tot//size markers and dumps the entire remainder on the last rank.
     Wall time is set by the slowest rank, so a bad divisor is a direct
     multiplier on runtime.
"""
import sys

NSIMD_DEFAULT = 16
LOGICAL_PER_NODE = 256
NUMA_DOMAINS = 8

def main(nmrk, nsimd=NSIMD_DEFAULT, rpn=NUMA_DOMAINS, maxnodes=512):
    nmrk, nsimd, rpn = int(nmrk), int(nsimd), int(rpn)
    tpr = LOGICAL_PER_NODE // rpn
    ceiling = nmrk // nsimd
    print(f"n_markers {nmrk}   NSIMD {nsimd}   ranks/node {rpn}   threads/rank {tpr}")
    print(f"thread ceiling = {nmrk} // {nsimd} = {ceiling}\n")
    rows = []
    for nodes in range(1, maxnodes + 1):
        R = nodes * rpn
        T = R * tpr
        if T > ceiling:
            break
        base = nmrk // R
        last = nmrk - (R - 1) * base
        rows.append((nodes, R, T, base, last, last / base))
    if not rows:
        print("Even 1 node exceeds the ceiling -- too few markers for this build.")
        return
    exact = [r for r in rows if r[4] == r[3]]
    best = max(exact, key=lambda r: r[2]) if exact else max(rows, key=lambda r: r[2] / r[5])
    print(f"{'nodes':>6} {'ranks':>6} {'threads':>8} {'mrk/rank':>9} {'last':>7} {'imbal':>7}")
    for r in rows[-16:]:
        mark = "  <== pick" if r == best else ""
        print(f"{r[0]:>6} {r[1]:>6} {r[2]:>8} {r[3]:>9} {r[4]:>7} {r[5]:>6.2f}x{mark}")
    n, R, T, base, last, imb = best
    print(f"\n   #SBATCH --nodes={n}")
    print(f"   #SBATCH --ntasks-per-node={rpn}")
    print(f"   #SBATCH --cpus-per-task={tpr}")
    print(f"\n   {T} threads = {100*T/ceiling:.0f}% of the {ceiling}-thread ceiling, "
          f"{base} markers/rank" + ("" if imb == 1 else f", last rank {imb:.2f}x overloaded"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__ + "\nusage: pick-nodes.py <n_markers> [NSIMD] [ranks_per_node]")
    main(*sys.argv[1:])

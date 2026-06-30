#!/usr/bin/env python
"""Slice the four wall VTK files at φ=0 and cache the (R, Z) polylines.

φ=0 is the half-plane (y=0, x≥0). We cut the polydata by the y=0 plane,
strip into polylines, drop the x<0 half (which is the φ=π slice), and
sort by angle about the magnetic-axis center for a clean closed loop.

Output: out/walls_phi0.npz  with one entry per wall:
    {'lcfs_CX': (R, Z) array shape (N, 2), 'engineering': ..., ...}
"""

import os

import numpy as np
import vtk
from vtk.util import numpy_support as ns

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "out")

WALLS = {
    "lcfs_CX": "G1600_lcfs_CX.vtk",
    "offset_10cm": "G1600_10cm_Biot_expdecay.vtk",
    "offset_20cm": "G1600_20cm_Biot_expdecay.vtk",
    "offset_30cm": "G1600_30cm_Biot_expdecay.vtk",
    "engineering": "G1600-engineeringwall-withtargets-200kmrk.vtk",
}

def read_polydata(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.Update()
    pd = reader.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise RuntimeError(f"VTK polydata empty for {path}")
    return pd


def slice_polylines(pd):
    """Cut by y=0 plane → strip → return list of polylines as (x,y,z) arrays.

    Each polyline preserves cell connectivity from the stripper, so non-convex
    walls (e.g. engineering wall with divertor targets) are handled correctly.
    Polylines are NaN-joined into a single (N, 3) array for easy plotting.
    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)
    plane.SetNormal(0.0, 1.0, 0.0)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(pd)
    cutter.GenerateTrianglesOff()
    cutter.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    out = stripper.GetOutput()
    all_pts = ns.vtk_to_numpy(out.GetPoints().GetData())  # (N, 3)

    polylines = []
    for cid in range(out.GetNumberOfCells()):
        cell = out.GetCell(cid)
        ids = cell.GetPointIds()
        idx = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        if len(idx) >= 2:
            polylines.append(all_pts[idx])
    return polylines


def filter_phi0(polylines):
    """Keep polylines whose mean x >= 0 (φ=0 hemisphere).

    Returns NaN-joined (R, Z) array. R = x, Z = z (y≈0 on the cut).
    """
    pieces = []
    for line in polylines:
        if line[:, 0].mean() < 0:
            continue
        R = line[:, 0]
        Z = line[:, 2]
        pieces.append(np.column_stack([R, Z]))
    if not pieces:
        return np.zeros((0, 2))
    nan_row = np.full((1, 2), np.nan)
    out = []
    for i, p in enumerate(pieces):
        if i > 0:
            out.append(nan_row)
        out.append(p)
    return np.vstack(out)


def slice_one(label, vtk_path):
    print(f"  [{label}] reading {os.path.basename(vtk_path)}...")
    pd = read_polydata(vtk_path)
    print(f"    {pd.GetNumberOfPoints():,} pts, "
          f"{pd.GetNumberOfCells():,} cells")
    polylines = slice_polylines(pd)
    print(f"    cut → {len(polylines)} polylines")
    rz = filter_phi0(polylines)
    finite = ~np.isnan(rz[:, 0])
    if finite.any():
        R = rz[finite, 0]
        Z = rz[finite, 1]
        print(f"    φ=0 hemisphere: {finite.sum():,} pts; "
              f"R∈[{R.min():.3f}, {R.max():.3f}] "
              f"Z∈[{Z.min():.3f}, {Z.max():.3f}]")
    return rz


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {}
    print("Slicing walls at φ=0:")
    for label, fname in WALLS.items():
        out[label] = slice_one(label, os.path.join(DATA_DIR, fname))
    out_path = os.path.join(OUT_DIR, "walls_phi0.npz")
    np.savez(out_path, **out)
    print(f"\nSaved: {out_path}")
    print("Keys:", list(out.keys()))


if __name__ == "__main__":
    main()

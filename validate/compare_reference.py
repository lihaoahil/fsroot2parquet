"""Compare the converter's columns with the ROOT reference CSV of ``calc_user_vars_ref.C``.

    python validate/compare_reference.py --ref ref.csv --root tree.root --tree NT --prefix "" \
        --pol-file POL.root --pol-hist hPol0

Recomputes the angles with ``fsroot2parquet.kinematics`` from the same tree (first N entries
of the CSV) and prints the largest absolute difference per column; exits 1 above ``--tol``.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import uproot

from fsroot2parquet.kinematics import kstar_hyperon_angles
from fsroot2parquet.polarization import find_bin_value

LABELS = {"beam": "PB", "kp": "P2", "pi0": "P3", "proton": "P1a", "pim": "P1b"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tree", default="ntFSGlueX_100000000_100001")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--pol-file", required=True)
    ap.add_argument("--pol-hist", required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()

    with open(a.ref) as f:
        rows = list(csv.DictReader(f))
    ref = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    n = len(rows)

    t = uproot.open(a.root)[a.tree]
    br = [f"{a.prefix}{c}{l}" for l in LABELS.values() for c in ("En", "Px", "Py", "Pz")]
    arr = t.arrays(br, entry_stop=n, library="np")
    p4 = {p: np.stack([arr[f"{a.prefix}{c}{l}"] for c in ("En", "Px", "Py", "Pz")], axis=1) for p, l in LABELS.items()}
    ang = kstar_hyperon_angles(p4["beam"], p4["kp"], p4["pi0"], p4["proton"], p4["pim"])
    h = uproot.open(a.pol_file)[a.pol_hist]
    pg = find_bin_value(np.asarray(h.axis().edges()), np.asarray(h.values()), ang["E_beam"])

    pairs = {"theta": ang["theta_V"], "phi": ang["phi_V"], "n0": ang["n_x"], "n1": ang["n_y"], "n2": ang["n_z"],
             "pGamma": pg, "bigPhi": ang["Phi_lab"]}
    worst = 0.0
    print(f"{n} entries of {a.root}")
    for k, v in pairs.items():
        d = np.abs(v - ref[k])
        if k in ("phi", "bigPhi"):                     # compare on the circle
            d = np.abs(np.angle(np.exp(1j * (v - ref[k]))))
        worst = max(worst, float(d.max()))
        print(f"  {k:7s} max |diff| = {d.max():.3e}")
    sys.exit(0 if worst <= a.tol else 1)


if __name__ == "__main__":
    main()

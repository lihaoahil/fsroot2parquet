"""Read the converted dataset back as NumPy arrays shaped for a JAX likelihood.

Nothing here imports JAX: ``jnp.asarray`` on the returned arrays is the whole hand-off.

The likelihood these arrays are built for is AmpTools' with each orientation's yield
profiled out (one LikelihoodCalculator per orientation, summed),

    ln L = sum_i lik_weight_i ln I(x_i; Phi_i, P_gamma_i)  -  sum_o N_o ln Z_o,
    Z_o  = (1 / N_acc) sum_j I(x_j; Phi_lab_j + polAngle_o, P_gamma_o(E_j)),

with ``lik_weight = +1`` for signal rows and ``-weight`` for sideband rows, and
N_o = sum of ``lik_weight`` over orientation o.  ``accmc_by_orientation`` builds the
per-orientation copies of the accepted MC that Z_o needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .kinematics import wrap_angle
from .polarization import PolarizationTable

POLARIZED = ("PARA_0", "PERP_45", "PERP_90", "PARA_135")
FIT_COLUMNS = ("theta_V", "phi_V", "theta_Y", "phi_Y", "Phi", "P_gamma", "lik_weight")


def _part(root: Path, sample: str, mass: str, tbin: str) -> Path:
    return Path(root) / f"sample={sample}" / f"mass={mass}" / f"tbin={tbin}" / "part-0.parquet"


def list_bins(root: str | Path, sample: str = "data") -> list[tuple[str, str]]:
    """(mass, tbin) labels present for ``sample``, sorted."""
    base = Path(root) / f"sample={sample}"
    return sorted((m.name.split("=", 1)[1], t.name.split("=", 1)[1])
                  for m in base.glob("mass=*") for t in m.glob("tbin=*"))


def read_polarization(root: str | Path) -> PolarizationTable:
    return PolarizationTable.from_arrow(pq.read_table(Path(root) / "polarization.parquet"))


def load_bin(root: str | Path, mass: str, tbin: str, sample: str = "data", columns=None,
             orientations=POLARIZED, roles=("data", "bkgnd")) -> dict[str, np.ndarray]:
    """One bin as a dict of 1-d arrays.  For ``sample='data'`` rows are filtered to ``orientations`` and ``roles``."""
    filters = None
    if sample == "data":
        filters = [("orientation", "in", list(orientations)), ("role", "in", list(roles))]
        if columns is not None:
            columns = list(dict.fromkeys(["orientation", "role", *columns]))
    table = pq.read_table(_part(root, sample, mass, tbin), columns=columns, filters=filters)
    out = {}
    for name in table.column_names:
        col = table[name].combine_chunks()
        if hasattr(col, "dictionary_decode"):
            col = col.dictionary_decode()
        out[name] = col.to_numpy(zero_copy_only=False)
    return out


def accmc_by_orientation(acc: dict[str, np.ndarray], pol: PolarizationTable,
                         orientations=POLARIZED) -> dict[str, np.ndarray]:
    """Stack one copy of the accepted MC per orientation, with that orientation's Phi and P_gamma(E_beam).

    Adds ``orientation`` (index into ``orientations``) so Z_o is a segment sum over it.
    """
    pieces = []
    for k, name in enumerate(orientations):
        n = len(acc["E_beam"])
        pieces.append({
            **acc,
            "Phi": wrap_angle(acc["Phi_lab"] + pol.pol_angle_rad(name)),
            "P_gamma": pol.p_gamma(name, acc["E_beam"]),
            "orientation": np.full(n, k, dtype=np.int32),
        })
    return {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}


def orientation_codes(names: np.ndarray, orientations=POLARIZED) -> np.ndarray:
    lookup = {n: i for i, n in enumerate(orientations)}
    return np.array([lookup[n] for n in names], dtype=np.int32)


def pad_bins(per_bin: list[dict[str, np.ndarray]], fields=FIT_COLUMNS) -> dict[str, np.ndarray]:
    """Pad K bins to (K, N_max).  ``mask`` is 1 on real rows; padded rows carry zeros (weight 0)."""
    n_max = max(len(b[fields[0]]) for b in per_bin)
    out = {f: np.zeros((len(per_bin), n_max), dtype=np.asarray(per_bin[0][f]).dtype) for f in fields}
    mask = np.zeros((len(per_bin), n_max))
    for k, b in enumerate(per_bin):
        n = len(b[fields[0]])
        for f in fields:
            out[f][k, :n] = b[f]
        mask[k, :n] = 1.0
    out["mask"] = mask
    return out

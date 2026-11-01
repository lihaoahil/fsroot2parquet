"""Per-orientation polarization: the fixed polAngle offsets and the P_gamma(E_beam) histograms."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa


@dataclass
class Orientation:
    name: str
    pol_angle_deg: float | None     # None for AMO: P_gamma = 0 and Phi = Phi_lab
    edges: np.ndarray | None = None  # P_gamma histogram bin edges (nbins + 1)
    values: np.ndarray | None = None  # P_gamma per bin (nbins)

    @property
    def polarized(self) -> bool:
        return self.pol_angle_deg is not None


def find_bin_value(edges: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``TH1::FindBin`` + ``GetBinContent`` with under/overflow -> 0, as in calcUserVars."""
    i = np.searchsorted(edges, x, side="right")          # bin number, 1-based; 0 = under, n+1 = over
    ok = (i >= 1) & (i <= len(values))
    return np.where(ok, values[np.clip(i - 1, 0, len(values) - 1)], 0.0)


@dataclass
class PolarizationTable:
    orientations: dict[str, Orientation] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def from_config(cls, cfg: dict) -> "PolarizationTable":
        import uproot
        pol_file = cfg.get("pol_file")
        f = uproot.open(pol_file) if pol_file else None
        out = {}
        for name, o in cfg["orientations"].items():
            ang = o.get("pol_angle_deg")
            hist = o.get("pol_hist")
            edges = values = None
            if hist is not None:
                h = f[hist]
                edges = np.asarray(h.axis().edges(), dtype=np.float64)
                values = np.asarray(h.values(), dtype=np.float64)
            out[name] = Orientation(name, None if ang is None else float(ang), edges, values)
        return cls(out, source=str(pol_file))

    def p_gamma(self, name: str, E_beam: np.ndarray) -> np.ndarray:
        o = self.orientations[name]
        if not o.polarized or o.values is None:
            return np.zeros_like(E_beam, dtype=np.float64)
        return find_bin_value(o.edges, o.values, E_beam)

    def pol_angle_rad(self, name: str) -> float:
        o = self.orientations[name]
        return 0.0 if not o.polarized else np.deg2rad(o.pol_angle_deg)

    def to_arrow(self) -> pa.Table:
        """Long table: one row per (orientation, E bin); AMO gets a single row with P_gamma = 0."""
        rows = {k: [] for k in ("orientation", "pol_angle_deg", "E_lo", "E_hi", "P_gamma")}
        for o in self.orientations.values():
            if o.edges is None:
                rows["orientation"].append(o.name); rows["pol_angle_deg"].append(o.pol_angle_deg)
                rows["E_lo"].append(None); rows["E_hi"].append(None); rows["P_gamma"].append(0.0)
                continue
            for lo, hi, v in zip(o.edges[:-1], o.edges[1:], o.values):
                rows["orientation"].append(o.name); rows["pol_angle_deg"].append(o.pol_angle_deg)
                rows["E_lo"].append(float(lo)); rows["E_hi"].append(float(hi)); rows["P_gamma"].append(float(v))
        return pa.table(rows).replace_schema_metadata({"source": self.source})

    @classmethod
    def from_arrow(cls, table: pa.Table) -> "PolarizationTable":
        d = table.to_pydict()
        out: dict[str, Orientation] = {}
        for name in dict.fromkeys(d["orientation"]):
            idx = [i for i, n in enumerate(d["orientation"]) if n == name]
            ang = d["pol_angle_deg"][idx[0]]
            if d["E_lo"][idx[0]] is None:
                out[name] = Orientation(name, ang)
                continue
            lo = np.array([d["E_lo"][i] for i in idx]); hi = np.array([d["E_hi"][i] for i in idx])
            out[name] = Orientation(name, ang, np.append(lo, hi[-1]), np.array([d["P_gamma"][i] for i in idx]))
        meta = table.schema.metadata or {}
        return cls(out, source=meta.get(b"source", b"").decode())

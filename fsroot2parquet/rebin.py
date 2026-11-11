"""Cut a converted dataset into new (mass, t) bins, keeping the layout ``load_bin`` reads.

The usual source is a ``--merge-tbins`` conversion (one t bin per mass bin), so any binning
inside its range is a filter on rows::

    python -m fsroot2parquet.rebin --src MERGED --out OUT --t-edges 0.1,0.2,0.3,...,2.5 [--mass-edges 0.84,0.89,0.94]

Rows go to the bin ``lo <= x < hi`` of ``x = -t_hyperon`` (``--t-column``) and of ``M_Kpi``.
For data and accepted MC ``t = t_hyperon`` (kinematic fit); the thrown MC is binned on its
own ``t_hyperon``, the variable the thrown skims were cut on.  New edges must lie inside the
source coverage; without ``--mass-edges`` the source mass bins are kept as they are.

Output: the same ``sample=*/mass=*/tbin=*/part-0.parquet`` files, ``polarization.parquet``
copied from the source, one manifest per bin and ``REBIN.json`` with the edges and source.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__

SAMPLES = ("data", "accmc", "genmc")


def _edges(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    e = np.array([float(x) for x in text.split(",")])
    if len(e) < 2 or np.any(np.diff(e) <= 0):
        raise ValueError(f"edges must increase: {text}")
    return e


def _label(lo: float, hi: float, digits: int) -> str:
    return f"{lo:.{digits}f}_{hi:.{digits}f}"


def _source_parts(src: Path, sample: str) -> list[tuple[Path, float, float, float, float, str]]:
    """(path, m_lo, m_hi, t_lo, t_hi, mass label) for every part of ``sample``."""
    out = []
    for p in sorted((src / f"sample={sample}").glob("mass=*/tbin=*/part-0.parquet")):
        m = p.parent.parent.name.split("=", 1)[1]
        t = p.parent.name.split("=", 1)[1]
        (m_lo, m_hi), (t_lo, t_hi) = (map(float, m.split("_")), map(float, t.split("_")))
        out.append((p, m_lo, m_hi, t_lo, t_hi, m))
    return out


def _covered(intervals: list[tuple[float, float]], lo: float, hi: float, tol: float = 1e-9) -> bool:
    """True if the union of ``intervals`` covers [lo, hi]."""
    x = lo
    for a, b in sorted(intervals):
        if a > x + tol:
            break
        x = max(x, b)
    return x >= hi - tol


def _assign(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index of ``lo <= x < hi``, -1 outside."""
    k = np.searchsorted(edges, x, side="right") - 1
    return np.where((k >= 0) & (k < len(edges) - 1), k, -1)


def rebin(src: str | Path, out: str | Path, t_edges: np.ndarray, mass_edges: np.ndarray | None = None,
          samples=SAMPLES, t_column: str = "t_hyperon", batch_size: int = 1_000_000) -> dict:
    src, out = Path(src), Path(out)
    t_edges = np.asarray(t_edges, dtype=float)
    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary: dict[tuple[str, str], dict] = {}

    for sample in samples:
        parts = _source_parts(src, sample)
        if not parts:
            raise FileNotFoundError(f"{src}: no sample={sample}")
        mass_bins = ([(_label(lo, hi, 3), lo, hi) for lo, hi in zip(mass_edges[:-1], mass_edges[1:])]
                     if mass_edges is not None else sorted({(m, lo, hi) for _, lo, hi, _, _, m in parts}))
        for _, m_lo, m_hi in mass_bins:                     # every new bin must be fully inside the source
            inside = [(t0, t1) for _, a, b, t0, t1, _ in parts if a <= m_lo + 1e-9 and b >= m_hi - 1e-9]
            if not inside or not _covered(inside, t_edges[0], t_edges[-1]):
                raise ValueError(f"sample={sample}: source does not cover mass {m_lo}-{m_hi}, -t {t_edges[0]}-{t_edges[-1]}")
        m_edges = np.array([mass_bins[0][1]] + [hi for _, _, hi in mass_bins])
        if mass_edges is None and any(abs(a[2] - b[1]) > 1e-9 for a, b in zip(mass_bins, mass_bins[1:])):
            raise ValueError("source mass bins are not contiguous; pass --mass-edges")
        tlabels = [_label(lo, hi, 2) for lo, hi in zip(t_edges[:-1], t_edges[1:])]

        writers: dict[tuple[int, int], pq.ParquetWriter] = {}
        rows: dict[tuple[int, int], dict] = {}
        for path, *_ in parts:
            pf = pq.ParquetFile(path)
            meta = dict(pf.schema_arrow.metadata or {})
            for batch in pf.iter_batches(batch_size=batch_size):
                tab = pa.Table.from_batches([batch])
                kt = _assign(-tab[t_column].to_numpy().astype(np.float64), t_edges)
                km = _assign(tab["M_Kpi"].to_numpy().astype(np.float64), m_edges)
                key = np.where((kt >= 0) & (km >= 0), km * len(tlabels) + kt, -1)
                for k in np.unique(key[key >= 0]):
                    im, it = divmod(int(k), len(tlabels))
                    sel = tab.take(np.flatnonzero(key == k))
                    if (im, it) not in writers:
                        mlab, mlo, mhi = mass_bins[im]
                        md = {**meta, b"mass": json.dumps([mlo, mhi]).encode(),
                              b"tbin": json.dumps([t_edges[it], t_edges[it + 1]]).encode(),
                              b"rebinned_from": json.dumps(str(src)).encode(), b"t_column": t_column.encode()}
                        dst = out / f"sample={sample}" / f"mass={mlab}" / f"tbin={tlabels[it]}" / "part-0.parquet"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        writers[im, it] = pq.ParquetWriter(dst.with_suffix(".tmp"), sel.schema.with_metadata(md),
                                                           compression="zstd")
                        rows[im, it] = {"rows": 0, "path": dst}
                        if sample == "data":
                            rows[im, it].update(sum_lik_weight=0.0, n_data=0, n_bkgnd=0)
                    writers[im, it].write_table(sel.replace_schema_metadata(writers[im, it].schema.metadata))
                    r = rows[im, it]
                    r["rows"] += sel.num_rows
                    if sample == "data":
                        role = np.asarray(sel["role"].to_pylist())
                        r["sum_lik_weight"] += float(np.sum(sel["lik_weight"].to_numpy()))
                        r["n_data"] += int(np.sum(role == "data")); r["n_bkgnd"] += int(np.sum(role == "bkgnd"))
        for key, w in writers.items():
            w.close()
            os.replace(rows[key]["path"].with_suffix(".tmp"), rows[key]["path"])
        for im, (mlab, _, _) in enumerate(mass_bins):
            for it, tlab in enumerate(tlabels):
                r = rows.get((im, it))
                if r is None:
                    print(f"warning: sample={sample} mass={mlab} tbin={tlab} is empty")
                    continue
                summary.setdefault((mlab, tlab), {})[sample] = {k: v for k, v in r.items() if k != "path"}

    for (mlab, tlab), s in summary.items():
        mp = out / "manifests" / f"{mlab}__{tlab}.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"mass": mlab, "tbin": tlab, "t_column": t_column, "rebinned_from": str(src),
                                  "created": created, "samples": s}, indent=1))
    shutil.copy2(src / "polarization.parquet", out / "polarization.parquet")
    info = {"created": created, "converter": f"fsroot2parquet {__version__}", "source": str(src),
            "t_column": t_column, "t_edges": t_edges.tolist(),
            "mass_edges": None if mass_edges is None else np.asarray(mass_edges).tolist(), "samples": list(samples)}
    if (src / "PROVENANCE.json").exists():
        info["source_provenance"] = json.loads((src / "PROVENANCE.json").read_text())
    (out / "REBIN.json").write_text(json.dumps(info, indent=1))
    return {"bins": {f"{m}__{t}": s for (m, t), s in summary.items()}, **info}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="converted dataset (usually a --merge-tbins one)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-edges", required=True, help="comma-separated -t edges, e.g. 0.1,0.2,0.3")
    ap.add_argument("--mass-edges", default=None, help="comma-separated M_Kpi edges (default: keep the source mass bins)")
    ap.add_argument("--t-column", default="t_hyperon")
    ap.add_argument("--samples", default=",".join(SAMPLES))
    a = ap.parse_args(argv)
    if Path(a.out).exists() and any(Path(a.out).iterdir()):
        raise SystemExit(f"{a.out} is not empty")
    res = rebin(a.src, a.out, _edges(a.t_edges), _edges(a.mass_edges),
                tuple(s.strip() for s in a.samples.split(",")), a.t_column)
    for b, s in res["bins"].items():
        print(b, {k: v["rows"] for k, v in s.items()})


if __name__ == "__main__":
    main()

"""Stage 2: the reduced store -> selected, weighted and binned Parquet, ready for a likelihood fit.

Everything an analysis varies is in the config: the cuts, the signal and sideband regions with their scale
factors, and the bin edges.  A systematic study is another config on the same store.

Sidebands follow FSRoot (``FSCut::makeCutSBWT``).  With sideband definitions j = 1..n, a row whose variables are
all in their signal regions is a ``data`` row with weight 1.  A row with the variables of a non-empty set S in
their sideband regions and the others in their signal regions is a ``bkgnd`` row with

    weight = - prod_{j in S} (- scale_j)          (scale_j for one sideband, - scale_j scale_k for two, ...)

and enters the likelihood with ``lik_weight = -weight``, as AmpTools subtracts its bkgnd sample.

The output has the layout of ``fsroot2parquet.convert`` and is read by ``fsroot2parquet.load``::

    polarization.parquet
    manifests/<mass>__<tbin>.json
    sample=data/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet      data and bkgnd rows of every orientation
    sample=accmc/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet
    sample=genmc/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet
    selection.json                                            the config, counts and timings

Usage::

    fsroot2parquet-select --config config/select.yaml [--store STORE] [--out OUT] [--samples data,accmc,genmc]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import __version__
from .convert import _combo_index, _to_table, _write
from .expr import all_of, evaluate
from .kinematics import wrap_angle
from .polarization import PolarizationTable


def read_store(store: Path, sample: str, periods=None) -> tuple[dict[str, np.ndarray], dict]:
    """All rows of one sample of the reduced store (optionally some periods) as columns, with the file metadata."""
    parts = sorted((store / f"sample={sample}").glob("period=*/part-0.parquet"))
    if periods:
        parts = [p for p in parts if p.parent.name.split("=", 1)[1] in periods]
    if not parts:
        raise FileNotFoundError(f"no reduced files for sample {sample!r} under {store}")
    tables = [pq.read_table(p, partitioning=None) for p in parts]      # the path keys are not columns
    meta = {k.decode(): json.loads(v) for k, v in (tables[0].schema.metadata or {}).items()}
    table = pa.concat_tables([t.replace_schema_metadata(None) for t in tables])
    cols = {name: table[name].to_numpy() for name in table.column_names}
    cols["period"] = np.concatenate([np.full(len(t), p.parent.name.split("=", 1)[1]) for t, p in zip(tables, parts)])
    return cols, meta


def sideband_weights(sidebands: list[dict], cols, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(in the signal region of every sideband variable, FSRoot CUTSBWT weight; 0 outside all regions).

    The signal and sideband regions of each variable must be disjoint: FSRoot would count a row in both in CUT and
    in CUTSBWT, which this split into data and bkgnd rows cannot represent, so such a row raises a ValueError.
    """
    signal = np.ones(n, dtype=bool)
    weight = np.zeros(n)
    regions = [(np.asarray(evaluate(s["signal"], cols), bool), np.asarray(evaluate(s["sideband"], cols), bool),
                float(s["scale"])) for s in sidebands]
    for s, (sig, side, _) in zip(sidebands, regions):
        both = int(np.count_nonzero(sig & side))
        if both:
            raise ValueError(f"sideband {s.get('name', s['signal'])!r}: {both} rows are in both the signal region "
                             f"{s['signal']!r} and the sideband region {s['sideband']!r}; the regions must be disjoint")
    for sig, _, _ in regions:
        signal &= sig
    for pattern in itertools.product((0, 1), repeat=len(regions)):
        if not any(pattern):
            continue
        mask, w = np.ones(n, dtype=bool), -1.0
        for (sig, side, scale), in_sideband in zip(regions, pattern):
            mask &= side if in_sideband else sig
            w *= -scale if in_sideband else 1.0
        weight[mask] += w
    return signal, weight


def _bins(spec: dict):
    """(column, sign, edges, label format) of one binning axis."""
    return spec["column"], float(spec.get("sign", 1)), [float(e) for e in spec["edges"]], spec.get("format", "{:.2f}")


def _bin_slices(cols, binning):
    """Yield (mass label, [lo, hi], t label, [lo, hi], row mask) for every bin; edges are exclusive, as in the skim."""
    mc, ms, me, mf = _bins(binning["mass"])
    tc, ts, te, tf = _bins(binning["t"])
    m, t = ms * cols[mc], ts * cols[tc]
    for mlo, mhi in zip(me[:-1], me[1:]):
        in_m = (m > mlo) & (m < mhi)
        for tlo, thi in zip(te[:-1], te[1:]):
            yield f"{mf.format(mlo)}_{mf.format(mhi)}", [mlo, mhi], f"{tf.format(tlo)}_{tf.format(thi)}", [tlo, thi], in_m & (t > tlo) & (t < thi)


def _take(cols, mask):
    return {k: v[mask] for k, v in cols.items()}


def select(cfg: dict, store: Path, out: Path, samples=("data", "accmc", "genmc"), periods=None) -> dict:
    t_all = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    pol = PolarizationTable.from_config(cfg)
    _write(pol.to_arrow(), out / "polarization.parquet")
    tree_values = {name: o.get("tree_value") for name, o in cfg["orientations"].items()}
    sidebands = cfg.get("sidebands") or []
    binning = cfg["binning"]
    summary = {"config": cfg, "store": str(store), "converter": f"fsroot2parquet {__version__}", "samples": {}, "bins": {}}
    part = lambda s, mass, tbin: out / f"sample={s}" / f"mass={mass}" / f"tbin={tbin}" / "part-0.parquet"

    for sample in samples:
        t0 = time.perf_counter()
        cols, store_meta = read_store(store, sample, periods)
        n_store = len(cols["run"])
        t_read = time.perf_counter() - t0
        kind = store_meta.get("kind", sample)
        meta = {"convention_version": store_meta.get("convention_version"), "converter": f"fsroot2parquet {__version__}",
                "selection": {k: cfg.get(k) for k in ("cuts", "sample_cuts", "sidebands", "binning")}}
        conditions = list((cfg.get("cuts") or []) if kind != "genmc" else []) + list((cfg.get("sample_cuts") or {}).get(sample, []))
        cols = _take(cols, all_of(conditions, cols, n_store))
        n_cut = len(cols["run"])
        active = [s for s in sidebands if sample in s.get("samples", ["data", "accmc"])] if kind != "genmc" else []
        signal, weight = sideband_weights(active, cols, n_cut)

        if kind == "data":
            # orientation from the tree value of the polarization angle, then Phi and P_gamma as the fit uses them
            cols["orientation"] = np.full(n_cut, "", dtype=object)
            for name, value in tree_values.items():
                cols["orientation"][cols[cfg.get("orientation_column", "pol_angle")] == value] = name
            cols["role"] = np.where(signal, "data", "bkgnd").astype(object)
            cols["weight"] = np.where(signal, 1.0, weight)
            cols["lik_weight"] = np.where(signal, 1.0, -weight)
            cols = _take(cols, (signal | (weight != 0)) & (cols["orientation"] != ""))
            cols["pol_angle_deg"] = np.full(len(cols["run"]), np.nan)
            cols["Phi"], cols["P_gamma"] = cols["Phi_lab"].copy(), np.zeros(len(cols["run"]))
            for name in pol.orientations:
                m = cols["orientation"] == name
                if pol.orientations[name].polarized:
                    cols["pol_angle_deg"][m] = pol.orientations[name].pol_angle_deg
                cols["Phi"][m] = wrap_angle(cols["Phi_lab"][m] + pol.pol_angle_rad(name))
                cols["P_gamma"][m] = pol.p_gamma(name, cols["E_beam"][m])
        elif kind == "accmc":
            cols = _take(cols, signal)                      # the MC is taken in the signal regions only
            cols["weight"] = np.ones(len(cols["run"]))
        counts = {"store_rows": n_store, "after_cuts": n_cut, "selected": len(cols["run"]), "bins": {}}

        for mass, m_edges, tbin, t_edges, mask in _bin_slices(cols, binning):
            rows = _take(cols, mask)
            n = len(rows["run"]) if "run" in rows else int(mask.sum())
            binmeta = meta | {"mass": m_edges, "tbin": t_edges}
            if kind == "data":
                # rows grouped as the skim wrote them (role, then orientation), with the combination index inside each
                order = np.lexsort((np.arange(n), [list(pol.orientations).index(o) for o in rows["orientation"]],
                                    rows["role"] != "data"))
                rows = {k: v[order] for k, v in rows.items()}
                rows["combo"] = np.zeros(n, dtype=np.int16)
                for role in ("data", "bkgnd"):
                    for name in pol.orientations:
                        m = (rows["role"] == role) & (rows["orientation"] == name)
                        rows["combo"][m] = _combo_index(rows["run"][m], rows["event"][m])
                first = ["role", "orientation", "run", "event", "combo", "weight", "lik_weight"]
                rows = {k: rows[k] for k in first + [k for k in rows if k not in first]}
                rows = {k: (v.astype(str) if v.dtype == object else v) for k, v in rows.items()}
                _write(_to_table(rows, binmeta, dictionary=("role", "orientation", "period")), part(sample, mass, tbin))
                info = {"rows": n, "signal": int((rows["role"] == "data").sum()), "sum_lik_weight": float(rows["lik_weight"].sum())}
            else:
                if kind == "accmc":
                    rows["combo"] = _combo_index(rows["run"], rows["event"])
                rows = {k: (v.astype(str) if v.dtype == object else v) for k, v in rows.items()}
                _write(_to_table(rows, binmeta, dictionary=("period",)), part(sample, mass, tbin))
                info = {"rows": n}
            counts["bins"][f"{mass}__{tbin}"] = info
            entry = summary["bins"].setdefault(f"{mass}__{tbin}", {"mass": m_edges, "tbin": t_edges, "samples": {}})
            entry["samples"][sample] = info
        counts["seconds"] = round(time.perf_counter() - t0, 2)
        counts["seconds_read"] = round(t_read, 2)
        summary["samples"][sample] = counts
        print(f"{sample}: {n_store} rows in the store -> {n_cut} after cuts -> {counts['selected']} selected, "
              f"{len(counts['bins'])} bins, {counts['seconds']:.1f} s (read {t_read:.1f})", flush=True)

    (out / "manifests").mkdir(exist_ok=True)
    for label, entry in summary["bins"].items():
        (out / "manifests" / f"{label}.json").write_text(json.dumps(entry | {"created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    summary["seconds"] = round(time.perf_counter() - t_all, 2)
    (out / "selection.json").write_text(json.dumps(summary, indent=1))
    print(f"total {summary['seconds']:.1f} s; wrote {out}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--store", help="reduced store (default: `store` of the config)")
    ap.add_argument("--out", help="output directory (default: `out` of the config)")
    ap.add_argument("--samples", default="data,accmc,genmc")
    ap.add_argument("--periods", help="comma-separated subset of the periods of the store")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    select(cfg, Path(args.store or cfg["store"]), Path(args.out or cfg["out"]),
           tuple(s.strip() for s in args.samples.split(",")), args.periods.split(",") if args.periods else None)


if __name__ == "__main__":
    main()

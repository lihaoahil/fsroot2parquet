"""FSRoot skims -> Parquet, one bin per call.

Layout written under ``--out``::

    polarization.parquet                                   orientation, pol_angle_deg, E_lo, E_hi, P_gamma
    manifests/<mass>__<tbin>.json                          sources, cycles, counts, weight sums, bin checks
    sample=data/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet   signal + sideband rows (role column)
    sample=accmc/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet  accepted phase-space MC, no orientation
    sample=genmc/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet  thrown MC, derived columns only (float32)

Usage::

    python -m fsroot2parquet.convert --config config/example.yaml --out OUT [--tbin 't_0.10*']
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import __version__
from .kinematics import CONVENTION_VERSION, kstar_hyperon_angles, wrap_angle
from .polarization import PolarizationTable

PARTICLES = ("beam", "kp", "pi0", "proton", "pim")
COMPONENTS = (("E", "En"), ("px", "Px"), ("py", "Py"), ("pz", "Pz"))
DERIVED = ("E_beam", "t", "t_hyperon", "M_Kpi", "M_ppi", "theta_V", "phi_V",
           "theta_Y", "phi_Y", "n_x", "n_y", "n_z", "Phi_lab")
GENMC_DERIVED = ("E_beam", "t", "M_Kpi", "M_ppi", "theta_V", "phi_V", "theta_Y", "phi_Y", "Phi_lab")
_EDGES = re.compile(r"_(-?[\d.]+)_(-?[\d.]+)$")


# --------------------------------------------------------------------------- reading

def _branches(labels: dict, prefix: str) -> list[str]:
    return [f"{prefix}{root}{labels[p]}" for p in PARTICLES for _, root in COMPONENTS]


def _four_vectors(arrs: dict, labels: dict, prefix: str) -> dict[str, np.ndarray]:
    return {p: np.stack([np.asarray(arrs[f"{prefix}{root}{labels[p]}"], dtype=np.float64) for _, root in COMPONENTS],
                        axis=1) for p in PARTICLES}


def _open_tree(path: Path, tree: str):
    """The highest cycle of ``tree``; also returns every cycle's entry count for the manifest."""
    import uproot
    f = uproot.open(path)
    cycles = sorted(int(k.split(";")[1]) for k in f.keys(cycle=True) if k.split(";")[0] == tree)
    if not cycles:
        raise KeyError(f"{path}: no tree {tree}")
    counts = {c: f[f"{tree};{c}"].num_entries for c in cycles}
    return f[f"{tree};{cycles[-1]}"], {"used_cycle": cycles[-1], "entries_by_cycle": counts}


def _file_info(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))}


def _combo_index(run: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Running count of rows sharing (run, event), in file order: 0 for the first combo."""
    order = np.lexsort((np.arange(len(run)), event, run))
    r, e = run[order], event[order]
    start = np.r_[True, (r[1:] != r[:-1]) | (e[1:] != e[:-1])]
    grp_start = np.maximum.accumulate(np.where(start, np.arange(len(r)), 0))
    out = np.empty(len(run), dtype=np.int16)
    out[order] = np.arange(len(r)) - grp_start
    return out


# --------------------------------------------------------------------------- tables

def _event_columns(p4: dict, extra_float32: bool = False) -> dict[str, np.ndarray]:
    ang = kstar_hyperon_angles(p4["beam"], p4["kp"], p4["pi0"], p4["proton"], p4["pim"])
    cols = {k: ang[k] for k in DERIVED}
    for p in PARTICLES:
        for i, (c, _) in enumerate(COMPONENTS):
            cols[f"{p}_{c}"] = p4[p][:, i].astype(np.float32) if extra_float32 else p4[p][:, i]
    return cols


def _read_sample(path: Path, tree: str, labels: dict, prefix: str, weight: dict | None = None) -> tuple[dict, dict]:
    t, cyc = _open_tree(path, tree)
    arrs = t.arrays(["Run", "Event"] + _branches(labels, prefix), library="np")
    cols = {"run": arrs["Run"].astype(np.int32), "event": arrs["Event"].astype(np.int64)}
    cols["combo"] = _combo_index(cols["run"], cols["event"])
    cols.update(_event_columns(_four_vectors(arrs, labels, prefix)))
    info = {"file": _file_info(path), **cyc}
    n = len(cols["run"])
    if weight is not None:
        wpath = Path(str(path) + weight["suffix"])
        wt, wcyc = _open_tree(wpath, tree + weight["tree_suffix"])
        w = wt[weight["branch"]].array(library="np").astype(np.float64)
        if len(w) != n:
            raise ValueError(f"{wpath}: {len(w)} weights for {n} rows")
        cols["weight"] = w
        info["weight_file"] = {**_file_info(wpath), **wcyc}
    else:
        cols["weight"] = np.ones(n)
    return cols, info


def _to_table(cols: dict, meta: dict, dictionary: tuple[str, ...] = ()) -> pa.Table:
    arrays = {k: (pa.array(v).dictionary_encode() if k in dictionary else pa.array(v)) for k, v in cols.items()}
    return pa.table(arrays).replace_schema_metadata({k: json.dumps(v) for k, v in meta.items()})


def _write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _bin_label(dirname: str) -> tuple[str, float, float]:
    m = _EDGES.search(dirname)
    if not m:
        raise ValueError(f"cannot read bin edges from {dirname}")
    return f"{m.group(1)}_{m.group(2)}", float(m.group(1)), float(m.group(2))


def _containment(x: np.ndarray, lo: float, hi: float) -> float:
    return float(np.mean((x >= lo) & (x < hi))) if len(x) else float("nan")


# --------------------------------------------------------------------------- one bin

def convert_bin(cfg: dict, pol: PolarizationTable, mass_dir: Path, tbin_dir: Path, out: Path,
                samples: tuple[str, ...]) -> dict:
    mass, m_lo, m_hi = _bin_label(mass_dir.name)
    tbin, t_lo, t_hi = _bin_label(tbin_dir.name)
    tree, labels, prefix = cfg["tree"], cfg["particles"], cfg["file_prefix"]
    meta = {"convention_version": CONVENTION_VERSION, "converter": f"fsroot2parquet {__version__}", "mass": [m_lo, m_hi], "tbin": [t_lo, t_hi],
            "kinematics_prefix": cfg.get("kinematics_prefix", ""), "particles": labels}
    manifest = {**meta, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "samples": {}}
    part = lambda s: out / f"sample={s}" / f"mass={mass}" / f"tbin={tbin}" / "part-0.parquet"
    kin = cfg.get("kinematics_prefix", "")

    if "data" in samples:
        t0 = time.time()
        pieces, infos = [], []
        for role, spec in cfg["data_files"].items():
            for name in pol.orientations:
                path = tbin_dir / spec["pattern"].format(prefix=prefix, orientation=name)
                if not path.exists():
                    infos.append({"role": role, "orientation": name, "missing": str(path)})
                    continue
                cols, info = _read_sample(path, tree, labels, kin, spec.get("weight"))
                n = len(cols["run"])
                sign = -1.0 if role == "bkgnd" else 1.0    # AmpTools: ln L = sum_data w ln I - sum_bkgnd w ln I
                cols["lik_weight"] = sign * cols["weight"]
                cols["pol_angle_deg"] = np.full(n, np.nan if not pol.orientations[name].polarized
                                                else pol.orientations[name].pol_angle_deg)
                cols["Phi"] = wrap_angle(cols["Phi_lab"] + pol.pol_angle_rad(name))
                cols["P_gamma"] = pol.p_gamma(name, cols["E_beam"])
                pieces.append({"role": np.full(n, role), "orientation": np.full(n, name), **cols})
                infos.append({"role": role, "orientation": name, "rows": n, "sum_weight": float(cols["weight"].sum()),
                              "sum_lik_weight": float(cols["lik_weight"].sum()), **info})
        cols = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        first = ["role", "orientation", "run", "event", "combo", "weight", "lik_weight", *DERIVED,
                 "pol_angle_deg", "Phi", "P_gamma"]
        cols = {k: cols[k] for k in first + [k for k in cols if k not in first]}
        _write(_to_table(cols, meta, dictionary=("role", "orientation")), part("data"))
        sig = cols["role"] == "data"
        manifest["samples"]["data"] = {
            "files": infos, "rows": len(cols["run"]), "seconds": round(time.time() - t0, 1),
            "bin_containment_signal": {
                "M_Kpi": _containment(cols["M_Kpi"][sig], m_lo, m_hi),
                "-t (beam - K*)": _containment(-cols["t"][sig], t_lo, t_hi),
                "-t (target - hyperon)": _containment(-cols["t_hyperon"][sig], t_lo, t_hi),
            },
        }

    if "accmc" in samples:
        t0 = time.time()
        path = tbin_dir / cfg["accmc_file"].format(prefix=prefix)
        cols, info = _read_sample(path, tree, labels, kin)
        if cfg.get("accmc_truth", True):
            t, _ = _open_tree(path, tree)
            truth = _event_columns(_four_vectors(t.arrays(_branches(labels, "MC"), library="np"), labels, "MC"))
            cols.update({f"mc_{k}": truth[k] for k in DERIVED})
        _write(_to_table(cols, meta), part("accmc"))
        manifest["samples"]["accmc"] = {"rows": len(cols["run"]), "seconds": round(time.time() - t0, 1), **info}

    if "genmc" in samples:
        t0 = time.time()
        path = tbin_dir / cfg["genmc_file"]
        t, cyc = _open_tree(path, tree)
        full = cfg.get("genmc_full", False)
        dst = part("genmc"); dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        writer, rows = None, 0
        for arrs in t.iterate(_branches(labels, "MC"), step_size=cfg.get("genmc_step", 500_000), library="np"):
            p4 = _four_vectors(arrs, labels, "MC")
            ang = kstar_hyperon_angles(p4["beam"], p4["kp"], p4["pi0"], p4["proton"], p4["pim"])
            chunk = {k: ang[k].astype(np.float32) for k in GENMC_DERIVED}
            if full:
                chunk.update({f"{p}_{c}": p4[p][:, i].astype(np.float32)
                              for p in PARTICLES for i, (c, _) in enumerate(COMPONENTS)})
            table = _to_table(chunk, meta)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk["E_beam"])
        writer.close()
        os.replace(tmp, dst)
        manifest["samples"]["genmc"] = {"rows": rows, "seconds": round(time.time() - t0, 1),
                                        "file": _file_info(path), **cyc}

    mpath = out / "manifests" / f"{mass}__{tbin}.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(mpath.read_text()) if mpath.exists() else {"samples": {}}
    manifest["samples"] = {**old.get("samples", {}), **manifest["samples"]}
    mpath.write_text(json.dumps(manifest, indent=1))
    return manifest


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mass", default="Mass_*", help="glob over mass-bin directories")
    ap.add_argument("--tbin", default="t_*", help="glob over t-bin directories")
    ap.add_argument("--samples", default="data,accmc,genmc")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pol = PolarizationTable.from_config(cfg)
    _write(pol.to_arrow(), out / "polarization.parquet")
    samples = tuple(s.strip() for s in args.samples.split(","))

    src = Path(cfg["source_root"])
    for mass_dir in sorted(p for p in src.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, args.mass)):
        for tbin_dir in sorted(p for p in mass_dir.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, args.tbin)):
            t0 = time.time()
            m = convert_bin(cfg, pol, mass_dir, tbin_dir, out, samples)
            counts = {s: v["rows"] for s, v in m["samples"].items() if s in samples}
            print(f"{mass_dir.name}/{tbin_dir.name}: {counts}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

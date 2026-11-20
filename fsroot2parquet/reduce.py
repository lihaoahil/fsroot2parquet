"""Stage 1: unpartitioned FSRoot trees -> a reduced, event-level Parquet store.

Each tree is read once.  Rows inside the *envelope* (the loosest cuts any later study will use) are kept with

  run, event                         identifiers; every combination of an event is kept
  the cut variables                  branches copied under the names of ``branches`` (chi2/dof, RF time, ...)
  the derived variables              ``variables``: masses of sets of particles, for cuts and sidebands
  the kinematic variables            ``two_vertex_variables`` of the reaction, helicity and Gottfried-Jackson
  the four-vectors                   <particle>_E, _px, _py, _pz of the particles of the reaction
  mc_<variable>                      the same kinematic variables from the thrown four-vectors (accepted MC)

No cut is applied at its analysis value, no sideband weight is computed and nothing is binned: that is stage 2
(``fsroot2parquet.selection``), which runs on this store in seconds.

Layout under ``--out``::

    reduce.json                                        the config, and per file: entries, rows kept, seconds
    sample=<name>/period=<label>/part-0.parquet

Usage::

    fsroot2parquet-reduce --config config/reduce.yaml --out STORE [--samples data,accmc] [--periods 2018_01]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import __version__
from .expr import FUNCTIONS, all_of, evaluate, names_of
from .kinematics import CONVENTION_VERSION, TARGET_MASS, mass2, rename, two_vertex_variables

COMPONENTS = (("E", "En"), ("px", "Px"), ("py", "Py"), ("pz", "Pz"))
GENMC_FLOAT64 = ("E_beam", "t", "t_recoil", "mass")      # binning variables keep full precision


def _p4_branches(labels: dict, prefix: str) -> list[str]:
    return [f"{prefix}{root}{label}" for label in labels.values() for _, root in COMPONENTS]


def _four_vectors(arrs: dict, labels: dict, prefix: str) -> dict[str, np.ndarray]:
    return {p: np.stack([np.asarray(arrs[f"{prefix}{root}{label}"], dtype=np.float64) for _, root in COMPONENTS], axis=1)
            for p, label in labels.items()}


def _mass_functions(p4: dict) -> dict:
    """mass(a, b, ...) and mass2(a, b, ...) of the summed four-vectors of the named particles."""
    total = lambda names: sum(p4[n] for n in names)
    return {"mass": lambda *names: np.sqrt(np.clip(mass2(total(names)), 0.0, None)), "mass2": lambda *names: mass2(total(names))}


class _Names(dict):
    """Particle names inside mass(...): each name evaluates to itself."""

    def __missing__(self, key):
        return key


def _kinematics(cfg: dict, p4: dict) -> dict:
    rx = cfg["reaction"]
    out = two_vertex_variables(p4[rx.get("beam", "beam")], [p4[p] for p in rx["upper"]], [p4[p] for p in rx["lower"]],
                               float(rx.get("target_mass", TARGET_MASS)), gj=cfg.get("gottfried_jackson", True))
    return rename(out, cfg.get("names", {}))


def _open_tree(path: Path, tree: str):
    """The highest cycle of ``tree`` and the entries of every cycle."""
    import uproot
    f = uproot.open(path)
    cycles = sorted(int(k.split(";")[1]) for k in f.keys(cycle=True) if k.split(";")[0] == tree)
    if not cycles:
        raise KeyError(f"{path}: no tree {tree}")
    return f[f"{tree};{cycles[-1]}"], {"used_cycle": cycles[-1], "entries_by_cycle": {c: f[f"{tree};{c}"].num_entries for c in cycles}}


def _split_conditions(conditions, cheap: set[str]) -> tuple[list, list]:
    """Conditions that need only the cheap columns (applied before the kinematics), and the others."""
    early = [c for c in conditions or [] if names_of(c) - set(FUNCTIONS) <= cheap]
    return early, [c for c in conditions or [] if c not in early]


def reduce_file(cfg: dict, sample: str, spec: dict, period: str, path: Path, out: Path) -> dict:
    """Reduce one tree file to ``out/sample=<sample>/period=<period>/part-0.parquet``; returns its manifest entry."""
    t_start = time.perf_counter()
    kind = spec.get("kind", sample)
    thrown = kind == "genmc"
    prefix = "MC" if thrown else cfg.get("kinematics_prefix", "")
    rx = cfg["reaction"]
    labels = cfg["particles"]
    needed = {p: labels[p] for p in [rx.get("beam", "beam"), *rx["upper"], *rx["lower"]]}       # written as four-vectors
    variables = {} if thrown else dict(cfg.get("variables") or {})
    branches = {} if thrown else dict(cfg.get("branches") or {}) | dict(spec.get("branches") or {})
    truth = kind == "accmc" and spec.get("truth", True)
    envelope = spec.get("envelope", cfg.get("envelope_genmc" if thrown else "envelope"))

    tree, cycles = _open_tree(path, cfg["tree"])
    read = ["Run", "Event", *branches.values(), *_p4_branches(labels if not thrown else needed, prefix)]
    if truth:
        read += _p4_branches(needed, "MC")
    read = list(dict.fromkeys(read))
    cheap = {"run", "event", *branches, *variables}
    early, late = _split_conditions(envelope, cheap)
    exact = set(rename(dict.fromkeys(GENMC_FLOAT64), cfg.get("names", {})))        # thrown columns kept in float64

    dst = out / f"sample={sample}" / f"period={period}" / "part-0.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    meta = {"convention_version": CONVENTION_VERSION, "converter": f"fsroot2parquet {__version__}", "sample": sample,
            "kind": kind, "period": period, "source": str(path), "config": cfg}
    writer, n_in, n_early, n_out = None, 0, 0, 0
    t_read = t_compute = t_write = 0.0
    it = tree.iterate(read, step_size=int(cfg.get("step", 2_000_000)), library="np")
    while True:
        t0 = time.perf_counter()
        arrs = next(it, None)
        t_read += time.perf_counter() - t0
        if arrs is None:
            break
        t0 = time.perf_counter()
        n = len(arrs["Run"])
        n_in += n
        cols = {"run": arrs["Run"].astype(np.int32), "event": arrs["Event"].astype(np.int64)}
        cols.update({name: np.asarray(arrs[branch]) for name, branch in branches.items()})
        p4 = _four_vectors(arrs, labels if not thrown else needed, prefix)
        fns = _mass_functions(p4)
        cols.update({name: evaluate(text, _Names(), fns) for name, text in variables.items()})
        keep = all_of(early, cols, n)                                # cheap conditions first
        n_early += int(keep.sum())
        cols = {k: v[keep] for k, v in cols.items()}
        p4 = {k: v[keep] for k, v in p4.items()}
        kin = _kinematics(cfg, p4)
        if thrown:
            kin = {k: v if k in exact else v.astype(np.float32) for k, v in kin.items()}
        cols.update(kin)
        late_keep = all_of(late, cols, len(cols["run"]))
        if truth:
            mc = _kinematics(cfg, {p: v[keep] for p, v in _four_vectors(arrs, needed, "MC").items()})
            cols.update({f"mc_{k}": v for k, v in mc.items()})
        if not thrown or spec.get("four_vectors", False):
            dtype = np.float32 if thrown else np.float64
            cols.update({f"{p}_{c}": p4[p][:, i].astype(dtype) for p in needed for i, (c, _) in enumerate(COMPONENTS)})
        cols = {k: v[late_keep] for k, v in cols.items()}
        n_out += len(cols["run"])
        t_compute += time.perf_counter() - t0
        t0 = time.perf_counter()
        table = pa.table({k: pa.array(v) for k, v in cols.items()})
        if writer is None:
            schema = table.schema.with_metadata({k: json.dumps(v) for k, v in meta.items()})
            writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        writer.write_table(table.cast(writer.schema))
        t_write += time.perf_counter() - t0
    if writer is None:
        raise ValueError(f"{path}: the tree is empty")
    writer.close()
    os.replace(tmp, dst)
    seconds = time.perf_counter() - t_start
    st = path.stat()
    return {"sample": sample, "period": period, "source": str(path), "source_bytes": st.st_size,
            "source_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)), **cycles,
            "entries": n_in, "after_cheap_cuts": n_early, "rows": n_out, "fraction_kept": n_out / max(n_in, 1),
            "output": str(dst), "output_bytes": dst.stat().st_size, "branches_read": len(read),
            "seconds": round(seconds, 1), "seconds_read": round(t_read, 1), "seconds_compute": round(t_compute, 1),
            "seconds_write": round(t_write, 1), "entries_per_second": round(n_in / seconds)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", help="comma-separated subset of the samples of the config")
    ap.add_argument("--periods", help="comma-separated subset of the period labels")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = args.samples.split(",") if args.samples else list(cfg["samples"])
    periods = args.periods.split(",") if args.periods else None
    mpath = out / "reduce.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"files": {}}
    manifest |= {"config": cfg, "convention_version": CONVENTION_VERSION, "converter": f"fsroot2parquet {__version__}"}
    t0 = time.perf_counter()
    for sample in samples:
        spec = cfg["samples"][sample]
        for period, path in spec["files"].items():
            if periods and str(period) not in periods:
                continue
            info = reduce_file(cfg, sample, spec, str(period), Path(path), out)
            manifest["files"][f"{sample}/{period}"] = info | {"created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            mpath.write_text(json.dumps(manifest, indent=1))
            print(f"{sample}/{period}: {info['entries']} entries -> {info['rows']} rows ({100 * info['fraction_kept']:.2f}%), "
                  f"{info['source_bytes'] / 1e9:.2f} GB -> {info['output_bytes'] / 1e6:.0f} MB, {info['seconds']:.0f} s "
                  f"(read {info['seconds_read']:.0f}, compute {info['seconds_compute']:.0f}, write {info['seconds_write']:.0f})",
                  flush=True)
    print(f"total {time.perf_counter() - t0:.0f} s; manifest {mpath}", flush=True)


if __name__ == "__main__":
    main()

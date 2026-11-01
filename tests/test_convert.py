"""End to end on a synthetic directory laid out like the FSRoot skims."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

uproot = pytest.importorskip("uproot")

from conftest import make_events
from fsroot2parquet.convert import main
from fsroot2parquet.load import accmc_by_orientation, list_bins, load_bin, pad_bins, read_polarization

TREE = "ntFSGlueX_100000000_100001"
PREFIX = "tree_pi0kplamb__B4_M7_M18"
CFG = yaml.safe_load((Path(__file__).parents[1] / "config" / "example.yaml").read_text())
LABELS = {"beam": "PB", "kp": "P2", "pi0": "P3", "proton": "P1a", "pim": "P1b", "lambda": "P1"}


def _branches(ev, prefix, run0=1):
    n = len(ev["beam"])
    out = {"Run": np.full(n, 30000 + run0, dtype=np.int32), "Event": (np.arange(n) // 2).astype(np.int64)}
    for k, lab in LABELS.items():
        for i, c in enumerate(("En", "Px", "Py", "Pz")):
            out[f"{prefix}{c}{lab}"] = ev[k][:, i]
    return out


def _write_tree(path, branches):
    with uproot.recreate(path) as f:
        f[TREE] = branches


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("src")
    tdir = root / "out" / "Mass_0.840_0.940" / "t_0.10_0.30"
    tdir.mkdir(parents=True)
    for k, o in enumerate(("PARA_0", "PERP_45", "PERP_90", "PARA_135", "AMO")):
        _write_tree(tdir / f"{PREFIX}_{o}_SIGNAL_SKIM.root", _branches(make_events(300, seed=10 + k), "", k))
        side = tdir / f"{PREFIX}_{o}_SIDEBANDS_SKIM.root"
        _write_tree(side, _branches(make_events(200, seed=20 + k), "", k))
        with uproot.recreate(str(side) + ".weight") as f:
            f[TREE + "_weight"] = {"weight": np.where(np.arange(200) % 2, 0.5, -0.25)}
    acc = make_events(1000, seed=3)
    _write_tree(tdir / f"{PREFIX}_PSMC_SIGNAL_SKIM.root", {**_branches(acc, ""), **_branches(acc, "MC")})
    _write_tree(tdir / "tree_thrown_PSMCGEN_SKIM.root", _branches(make_events(5000, seed=4), "MC"))
    edges = np.linspace(7.0, 12.0, 26)
    with uproot.recreate(root / "pol.root") as f:
        for o in CFG["orientations"].values():
            if o["pol_hist"]:
                f[o["pol_hist"]] = (np.linspace(0.1, 0.4, 25), edges)

    cfg = dict(CFG)
    cfg["source_root"] = str(root / "out")
    cfg["pol_file"] = str(root / "pol.root")
    cfg["genmc_step"] = 2000
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    out = root / "parquet"
    main(["--config", str(root / "cfg.yaml"), "--out", str(out)])
    return out


def test_layout_and_manifest(dataset):
    assert list_bins(dataset) == [("0.840_0.940", "0.10_0.30")]
    m = json.loads((dataset / "manifests" / "0.840_0.940__0.10_0.30.json").read_text())
    assert m["samples"]["data"]["rows"] == 5 * (300 + 200)
    assert m["samples"]["accmc"]["rows"] == 1000
    assert m["samples"]["genmc"]["rows"] == 5000


def test_data_columns(dataset):
    d = load_bin(dataset, "0.840_0.940", "0.10_0.30")
    assert set(d["orientation"]) == {"PARA_0", "PERP_45", "PERP_90", "PARA_135"}
    sig, bkg = d["role"] == "data", d["role"] == "bkgnd"
    np.testing.assert_array_equal(d["lik_weight"][sig], 1.0)
    np.testing.assert_array_equal(d["lik_weight"][bkg], -d["weight"][bkg])
    assert set(np.unique(d["combo"])) == {0, 1}                        # two rows per (run, event)
    for name, o in CFG["orientations"].items():
        if o["pol_angle_deg"] is None:
            continue
        sel = d["orientation"] == name
        dphi = np.angle(np.exp(1j * (d["Phi"][sel] - d["Phi_lab"][sel] - np.deg2rad(o["pol_angle_deg"]))))
        np.testing.assert_allclose(dphi, 0.0, atol=1e-12)
    assert np.all((d["P_gamma"] > 0.1) & (d["P_gamma"] < 0.4))
    amo = load_bin(dataset, "0.840_0.940", "0.10_0.30", orientations=("AMO",))
    np.testing.assert_array_equal(amo["P_gamma"], 0.0)


def test_accmc_expansion_and_padding(dataset):
    pol = read_polarization(dataset)
    acc = load_bin(dataset, "0.840_0.940", "0.10_0.30", sample="accmc")
    assert "mc_theta_V" in acc
    x = accmc_by_orientation(acc, pol)
    assert len(x["Phi"]) == 4 * 1000 and set(np.unique(x["orientation"])) == {0, 1, 2, 3}
    d = load_bin(dataset, "0.840_0.940", "0.10_0.30")
    p = pad_bins([d, d])
    assert p["theta_V"].shape == (2, len(d["theta_V"])) and p["mask"].sum() == 2 * len(d["theta_V"])

"""Stage 1 (reduce) and stage 2 (select) on a synthetic unpartitioned tree."""
from pathlib import Path

import numpy as np
import pytest
import yaml

uproot = pytest.importorskip("uproot")

from conftest import make_events
from fsroot2parquet.expr import all_of, evaluate
from fsroot2parquet.kinematics import kstar_hyperon_angles, two_vertex_variables
from fsroot2parquet.load import accmc_by_orientation, list_bins, load_bin, read_polarization
from fsroot2parquet.reduce import main as reduce_main
from fsroot2parquet.selection import main as select_main, sideband_weights

TREE = "ntFSGlueX_100000000_100001"
ROOT = Path(__file__).parents[1]
LABELS = {"beam": "PB", "kp": "P2", "pi0": "P3", "proton": "P1a", "pim": "P1b", "lambda": "P1"}


def _tree(n, seed, mc=False, data=False):
    """Branches of a tree: the reaction four-vectors, a Lambda with a smeared mass, and the cut variables."""
    rng = np.random.default_rng(seed)
    ev = make_events(n, seed)
    ev["lambda"] = ev["lambda"].copy()
    ev["lambda"][:, 0] += rng.normal(0, 0.012, n)                    # spreads M_Lambda over signal and sidebands
    out = {"Run": np.full(n, 30000, dtype=np.int32), "Event": (np.arange(n) // 2).astype(np.int64),
           "Chi2DOF": rng.uniform(0, 12, n), "EnUnusedSh": rng.exponential(0.1, n),
           "NumUnusedTracks": rng.integers(0, 3, n).astype(float), "ProdVz": rng.uniform(45, 85, n),
           "RFDeltaT": rng.choice([-8.016, -4.008, 0.0, 4.008, 8.016], n) + rng.normal(0, 0.3, n)}
    if data:
        out["PolarizationAngle"] = rng.choice([0.0, 45.0, 90.0, 135.0, -1.0], n)
    if mc:
        out["MCIsGenBeam"] = (rng.uniform(size=n) < 0.9).astype(float)
    for prefix in ("", "MC") if mc else ("",):
        for k, lab in LABELS.items():
            for i, c in enumerate(("En", "Px", "Py", "Pz")):
                out[f"{prefix}{c}{lab}"] = ev[k][:, i]
    return out, ev


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("two_stage")
    data, _ = _tree(40000, 1, data=True)
    acc, _ = _tree(20000, 2, mc=True)
    gen = make_events(30000, 3)
    for name, branches in (("data", data), ("acc", acc)):
        with uproot.recreate(root / f"{name}.root") as f:
            f[TREE] = branches
    with uproot.recreate(root / "gen.root") as f:
        f[TREE] = {"Run": np.full(30000, 30000, dtype=np.int32), "Event": np.arange(30000, dtype=np.int64),
                   **{f"MC{c}{lab}": gen[k][:, i] for k, lab in LABELS.items() for i, c in enumerate(("En", "Px", "Py", "Pz"))}}
    edges = np.linspace(7.0, 12.0, 26)
    red = yaml.safe_load((ROOT / "config" / "reduce_example.yaml").read_text())
    sel = yaml.safe_load((ROOT / "config" / "select_example.yaml").read_text())
    with uproot.recreate(root / "pol.root") as f:
        for o in sel["orientations"].values():
            if o["pol_hist"]:
                f[o["pol_hist"]] = (np.linspace(0.1, 0.4, 25), edges)
    red["step"] = 7000                                               # several chunks
    for s, name in (("data", "data"), ("accmc", "acc"), ("genmc", "gen")):
        red["samples"][s]["files"] = {"p1": str(root / f"{name}.root")}
    sel |= {"store": str(root / "store"), "out": str(root / "nominal"), "pol_file": str(root / "pol.root")}
    sel["binning"]["t"]["edges"] = [0.0, 0.5, 3.5]
    (root / "reduce.yaml").write_text(yaml.safe_dump(red))
    (root / "select.yaml").write_text(yaml.safe_dump(sel))
    reduce_main(["--config", str(root / "reduce.yaml"), "--out", str(root / "store")])
    select_main(["--config", str(root / "select.yaml")])
    return root, data, sel


def test_expressions():
    cols = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([3.0, 2.0, 1.0])}
    assert evaluate("(a > 1) & (b > 1)", cols).tolist() == [False, True, False]
    assert evaluate("a > 1 and b > 1 or a == 1", cols).tolist() == [True, True, False]
    assert evaluate("abs(a - b) < 1.5", cols).tolist() == [False, True, False]
    assert all_of(["a > 1", "b < 2"], cols, 3).tolist() == [False, False, True]
    for bad in ("__import__('os')", "a.real", "[a]", "lambda: a"):
        with pytest.raises((ValueError, SyntaxError, KeyError)):     # no builtins, no attributes, no containers
            evaluate(bad, cols)
    with pytest.raises(KeyError):
        evaluate("c > 0", cols)


def test_general_kinematics_are_the_kstar_hyperon_ones(events):
    ref = kstar_hyperon_angles(events["beam"], events["kp"], events["pi0"], events["proton"], events["pim"])
    gen = two_vertex_variables(events["beam"], [events["kp"], events["pi0"]], [events["proton"], events["pim"]])
    for old, new in (("M_Kpi", "mass"), ("theta_V", "theta"), ("phi_Y", "phi_B"), ("theta_Y_GJ", "theta_B_GJ"), ("t_hyperon", "t_recoil")):
        assert np.array_equal(ref[old], gen[new])
    single = two_vertex_variables(events["beam"], [events["kp"] + events["pi0"]], [events["proton"] + events["pim"]])
    assert "theta" not in single and "theta_B" not in single and np.allclose(single["mass"], gen["mass"])
    assert np.allclose(single["Phi_lab"], gen["Phi_lab"]) and np.allclose(single["t"], gen["t"])


def test_sideband_weights_follow_fsroot():
    """One sideband: +scale; two: -scale*scale; three: +scale^3; all in signal: a data row."""
    x = {"u": np.array([0, 1, 0, 1, 9]), "v": np.array([0, 0, 1, 1, 0])}
    sb = [{"signal": "u == 0", "sideband": "u == 1", "scale": 0.5}, {"signal": "v == 0", "sideband": "v == 1", "scale": 0.25}]
    signal, weight = sideband_weights(sb, x, 5)
    assert signal.tolist() == [True, False, False, False, False]
    assert weight.tolist() == [0.0, 0.5, 0.25, -0.125, 0.0]         # the last row is in no region


def test_overlapping_sideband_regions_are_rejected():
    """A row in both the signal and the sideband region of one variable has no data/bkgnd split."""
    x = {"u": np.array([0.0, 0.5, 2.0])}
    sb = [{"name": "u", "signal": "abs(u) < 1", "sideband": "abs(u) > 0.4", "scale": 0.5}]
    with pytest.raises(ValueError, match="'u': 1 rows are in both"):
        sideband_weights(sb, x, 3)


def test_selection_reproduces_a_direct_selection(stores):
    root, tree, sel = stores
    lam = np.stack([tree[f"{c}P1"] for c in ("En", "Px", "Py", "Pz")], axis=1)
    pi0 = np.stack([tree[f"{c}P3"] for c in ("En", "Px", "Py", "Pz")], axis=1)
    m = lambda p: np.sqrt(p[:, 0] ** 2 - (p[:, 1:] ** 2).sum(1))
    kin = kstar_hyperon_angles(*(np.stack([tree[f"{c}{LABELS[k]}"] for c in ("En", "Px", "Py", "Pz")], axis=1)
                                 for k in ("beam", "kp", "pi0", "proton", "pim")))
    common = ((tree["EnPB"] > 8.2) & (tree["EnPB"] < 8.8) & (tree["Chi2DOF"] < 5) & (tree["EnUnusedSh"] < 0.1)
              & (tree["NumUnusedTracks"] < 1) & (tree["ProdVz"] >= 51.2) & (tree["ProdVz"] <= 78.8) & (m(lam + pi0) > 2.0)
              & (kin["M_Kpi"] > 0.84) & (kin["M_Kpi"] < 0.94))
    dm, rf = np.abs(m(lam) - 1.115), np.abs(tree["RFDeltaT"])
    lam_sig, lam_sb, rf_sig, rf_sb = dm < 0.010, (dm > 0.015) & (dm < 0.035), rf < 2.004, rf > 6.012
    total = {"signal": 0, "lam": 0, "rf": 0, "both": 0}
    for tbin, (lo, hi) in (("0.00_0.50", (0.0, 0.5)), ("0.50_3.50", (0.5, 3.5))):
        inbin = common & (-kin["t_hyperon"] > lo) & (-kin["t_hyperon"] < hi)
        got = load_bin(root / "nominal", "0.840_0.940", tbin, orientations=("PARA_0", "PERP_45", "PERP_90", "PARA_135", "AMO"))
        w = np.round(got["weight"][got["role"] == "bkgnd"], 9)
        expect = {"signal": inbin & lam_sig & rf_sig, "lam": inbin & lam_sb & rf_sig, "rf": inbin & lam_sig & rf_sb, "both": inbin & lam_sb & rf_sb}
        assert int((got["role"] == "data").sum()) == int(expect["signal"].sum())
        assert int((w == 0.5).sum()) == int(expect["lam"].sum())
        assert int((w == round(1 / 6, 9)).sum()) == int(expect["rf"].sum())
        assert int((w == round(-0.5 / 6, 9)).sum()) == int(expect["both"].sum())
        assert np.allclose(got["lik_weight"], np.where(got["role"] == "data", 1.0, -got["weight"]))
        for k in total:
            total[k] += int(expect[k].sum())
    assert min(total.values()) > 5                                   # every kind of row occurs
    # the angles of the selected rows are those of the tree, and Phi carries the orientation's angle
    sig = got["role"] == "data"
    assert np.all((got["theta_V"] >= 0) & (got["theta_V"] <= np.pi)) and set(got["orientation"]) <= set(sel["orientations"])
    amo = got["orientation"] == "AMO"
    assert np.all(got["P_gamma"][amo] == 0) and np.allclose(got["Phi"][amo], got["Phi_lab"][amo]) and sig.any()


def test_store_layout_and_loader(stores):
    root, _, sel = stores
    import json
    manifest = json.loads((root / "store" / "reduce.json").read_text())
    for key in ("data/p1", "accmc/p1", "genmc/p1"):
        info = manifest["files"][key]
        assert info["rows"] <= info["entries"] and info["seconds"] >= 0 and Path(info["output"]).exists()
    assert manifest["files"]["data/p1"]["rows"] < 0.5 * manifest["files"]["data/p1"]["entries"]      # the envelope cuts
    assert list_bins(root / "nominal") == [("0.840_0.940", "0.00_0.50"), ("0.840_0.940", "0.50_3.50")]
    acc = load_bin(root / "nominal", "0.840_0.940", "0.50_3.50", sample="accmc")
    assert np.all(acc["mc_is_gen_beam"] == 1) and "mc_theta_V" in acc and np.all(np.abs(acc["M_Lambda"] - 1.115) < 0.010)
    stacked = accmc_by_orientation(acc, read_polarization(root / "nominal"))
    assert len(stacked["Phi"]) == 4 * len(acc["Phi_lab"])
    gen = load_bin(root / "nominal", "0.840_0.940", "0.50_3.50", sample="genmc")
    assert gen["theta_V"].dtype == np.float32 and gen["M_Kpi"].dtype == np.float64 and len(gen["M_Kpi"]) > 0
    summary = json.loads((root / "nominal" / "selection.json").read_text())
    assert summary["samples"]["data"]["selected"] == sum(b["samples"]["data"]["rows"] for b in summary["bins"].values())

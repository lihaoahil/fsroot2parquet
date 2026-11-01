import numpy as np

from fsroot2parquet.kinematics import boost_to_rest, kstar_hyperon_angles, mass2, wrap_angle
from fsroot2parquet.polarization import find_bin_value


def _angles(ev):
    return kstar_hyperon_angles(ev["beam"], ev["kp"], ev["pi0"], ev["proton"], ev["pim"])


def _rotate_z(ev, d):
    c, s = np.cos(d), np.sin(d)
    out = {}
    for k, p in ev.items():
        q = p.copy()
        q[:, 1], q[:, 2] = c * p[:, 1] - s * p[:, 2], s * p[:, 1] + c * p[:, 2]
        out[k] = q
    return out


def test_boost_to_own_rest_frame(events):
    lam = events["proton"] + events["pim"]
    r = boost_to_rest(lam, lam)
    np.testing.assert_allclose(r[:, 1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(r[:, 0], np.sqrt(mass2(lam)), rtol=1e-12)


def test_invariants(events):
    a = _angles(events)
    np.testing.assert_allclose(a["M_Kpi"], 0.8917, rtol=1e-9)
    np.testing.assert_allclose(a["M_ppi"], 1.115683, rtol=1e-9)
    np.testing.assert_allclose(a["t"], a["t_hyperon"], rtol=1e-9)   # exact 2 -> 2 kinematics
    np.testing.assert_allclose(a["n_x"] ** 2 + a["n_y"] ** 2 + a["n_z"] ** 2, 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.cos(a["theta_Y"]), a["n_z"], atol=1e-12)


def test_rotation_about_beam(events):
    """Decay angles are invariant under a rotation about the beam; Phi_lab = atan2(y_x, y_y) shifts by -delta."""
    a, d = _angles(events), 0.7
    b = _angles(_rotate_z(events, d))
    for k in ("theta_V", "theta_Y", "n_x", "n_y", "n_z"):
        np.testing.assert_allclose(b[k], a[k], atol=1e-10)
    for k in ("phi_V", "phi_Y"):
        np.testing.assert_allclose(wrap_angle(b[k] - a[k]), 0.0, atol=1e-10)
    np.testing.assert_allclose(wrap_angle(b["Phi_lab"] - a["Phi_lab"] + d), 0.0, atol=1e-10)


def test_flat_decays_give_flat_angles(events):
    """Isotropic decays in the parent rest frames -> cos(theta) uniform, <cos theta> ~ 0, phi spans the circle."""
    a = _angles(events)
    for k in ("theta_V", "theta_Y"):
        assert abs(np.mean(np.cos(a[k]))) < 0.06
    assert abs(np.mean(np.cos(a["phi_V"]))) < 0.06 and abs(np.mean(np.sin(a["phi_Y"]))) < 0.06


def test_find_bin_matches_th1_rules():
    edges = np.array([7.0, 8.0, 9.0, 10.0])
    values = np.array([0.1, 0.2, 0.3])
    x = np.array([6.9, 7.0, 7.5, 8.0, 9.999, 10.0, 11.0])
    np.testing.assert_array_equal(find_bin_value(edges, values, x), [0.0, 0.1, 0.1, 0.2, 0.3, 0.0, 0.0])

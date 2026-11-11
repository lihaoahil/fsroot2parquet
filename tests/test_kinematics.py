import numpy as np

from conftest import M
from fsroot2parquet.kinematics import _unit, boost_to_rest, kstar_hyperon_angles, mass2, wrap_angle
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
    for k in ("theta_V", "theta_Y", "n_x", "n_y", "n_z", "theta_V_GJ", "theta_Y_GJ", "n_x_GJ", "n_z_GJ"):
        np.testing.assert_allclose(b[k], a[k], atol=1e-10)
    for k in ("phi_V", "phi_Y", "phi_V_GJ", "phi_Y_GJ"):
        np.testing.assert_allclose(wrap_angle(b[k] - a[k]), 0.0, atol=1e-10)
    np.testing.assert_allclose(wrap_angle(b["Phi_lab"] - a["Phi_lab"] + d), 0.0, atol=1e-10)


def test_flat_decays_give_flat_angles(events):
    """Isotropic decays in the parent rest frames -> cos(theta) uniform, <cos theta> ~ 0, phi spans the circle."""
    a = _angles(events)
    for k in ("theta_V", "theta_Y"):
        assert abs(np.mean(np.cos(a[k]))) < 0.06
    assert abs(np.mean(np.cos(a["phi_V"]))) < 0.06 and abs(np.mean(np.sin(a["phi_Y"]))) < 0.06


def _kallen(x, y, z):
    return x * x + y * y + z * z - 2 * (x * y + y * z + z * x)


def test_gj_is_a_rotation_about_y(events):
    """GJ and helicity frames share y: the y components agree and the frames differ by one angle per event."""
    a = _angles(events)
    np.testing.assert_allclose(a["n_y_GJ"], a["n_y"], atol=1e-12)
    np.testing.assert_allclose(np.sin(a["theta_V_GJ"]) * np.sin(a["phi_V_GJ"]),
                               np.sin(a["theta_V"]) * np.sin(a["phi_V"]), atol=1e-12)
    np.testing.assert_allclose(a["n_x_GJ"] ** 2 + a["n_y_GJ"] ** 2 + a["n_z_GJ"] ** 2, 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.cos(a["theta_Y_GJ"]), a["n_z_GJ"], atol=1e-12)


def test_gj_crossing_angles_match_invariant_formula(events):
    """cos(omega) between helicity and GJ axes from s, t, u (the rho Delta formula, Delta -> Lambda, rho -> K*).

    Hyperon: GJ axis = target proton, at +omega from the helicity axis (positive x component);
    K*: GJ axis = beam, at -omega (negative x component).
    """
    beam, K, L = events["beam"], events["kp"] + events["pi0"], events["proton"] + events["pim"]
    tgt = np.zeros_like(beam); tgt[:, 0] = M["p"]
    s, mp = mass2(beam + tgt), M["p"]
    mV, mL = np.sqrt(mass2(K)), np.sqrt(mass2(L))

    t, u = mass2(tgt - L), mass2(tgt - K)
    cL = ((2 * mL**2 * (mp**2 + mV**2 - u) - (mp**2 + mL**2 - t) * (s - mV**2 - mL**2))
          / np.sqrt(_kallen(t, mp**2, mL**2) * _kallen(s, mV**2, mL**2)))
    zH = -_unit(boost_to_rest(K, L)[:, 1:]); zHG = _unit(boost_to_rest(tgt, L)[:, 1:])
    yH = _unit(np.cross(_unit(beam[:, 1:]), -_unit(K[:, 1:]))); xH = np.cross(yH, zH)
    np.testing.assert_allclose(np.sum(zH * zHG, axis=1), cL, atol=1e-10)
    assert np.all(np.sum(zHG * xH, axis=1) > 0)

    tV, uV = mass2(beam - K), mass2(beam - L)
    cV = ((2 * mV**2 * (mL**2 - uV) - (mV**2 - tV) * (s - mL**2 - mV**2))
          / np.sqrt(_kallen(tV, 0.0, mV**2) * _kallen(s, mL**2, mV**2)))
    z = -_unit(boost_to_rest(L, K)[:, 1:]); zG = _unit(boost_to_rest(beam, K)[:, 1:])
    y = _unit(np.cross(_unit(beam[:, 1:]), -_unit(L[:, 1:]))); x = np.cross(y, z)
    np.testing.assert_allclose(np.sum(z * zG, axis=1), cV, atol=1e-10)
    assert np.all(np.sum(zG * x, axis=1) < 0)

    # the stored GJ angles are the helicity ones rotated by omega about y
    a = _angles(events)
    w = np.arccos(cL)
    nz_rot = a["n_z"] * np.cos(w) + a["n_x"] * np.sin(w)
    np.testing.assert_allclose(a["n_z_GJ"], nz_rot, atol=1e-10)


def test_find_bin_matches_th1_rules():
    edges = np.array([7.0, 8.0, 9.0, 10.0])
    values = np.array([0.1, 0.2, 0.3])
    x = np.array([6.9, 7.0, 7.5, 8.0, 9.999, 10.0, 11.0])
    np.testing.assert_array_equal(find_bin_value(edges, values, x), [0.0, 0.1, 0.1, 0.2, 0.3, 0.0, 0.0])

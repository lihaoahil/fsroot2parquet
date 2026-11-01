"""Vectorized port of ``KStarHyperonExtended::calcUserVars`` (halld_sim AMPTOOLS_AMPS).

Four-vectors are (N, 4) arrays ordered (E, px, py, pz) in the lab.  Every line below
mirrors one line of the C++; the ROOT reference macro in ``validate/`` runs that C++
verbatim on the same trees so the two can be compared event by event.

Conventions (version ``CONVENTION_VERSION``):
  K* helicity frame : z = -(hyperon direction in the K* rest frame),
                      y = beam x (-hyperon) in the lab, x = y x z;
                      (theta_V, phi_V) are the polar angles of the K+ there.
  hyperon frame     : z = -(K* direction in the hyperon rest frame),
                      y = beam x (-K*) in the lab (= minus the K* frame's y), x = y x z;
                      n = (n_x, n_y, n_z) are the direction cosines of the proton,
                      theta_Y = arccos(n_z), phi_Y = atan2(n_y, n_x).
  beam azimuth      : Phi_lab = atan2(y . eps, beam_hat . (eps x y)) with eps = (1, 0, 0);
                      the fit uses Phi = polAngle [rad] + Phi_lab and
                      g1 = -P_gamma cos 2Phi, g2 = -P_gamma sin 2Phi.
"""
from __future__ import annotations

import numpy as np

CONVENTION_VERSION = "kstarhyperonextended-v1"

TARGET_MASS = 0.938272


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...i->...", a, b)


def mass2(p: np.ndarray) -> np.ndarray:
    return p[..., 0] ** 2 - _dot(p[..., 1:], p[..., 1:])


def boost_to_rest(p: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """``TLorentzRotation(-frame.BoostVector()) * p``: p seen in the rest frame of ``frame``."""
    b = frame[..., 1:] / frame[..., :1]
    b2 = _dot(b, b)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = _dot(b, p[..., 1:])
    # (gamma - 1) / b2 -> gamma^2 / (gamma + 1), which stays finite at b2 = 0
    coef = gamma**2 / (gamma + 1.0)
    out = np.empty_like(p)
    out[..., 0] = gamma * (p[..., 0] - bp)
    out[..., 1:] = p[..., 1:] + ((coef * bp - gamma * p[..., 0])[..., None]) * b
    return out


def kstar_hyperon_angles(beam, kp, pi0, proton, pim) -> dict[str, np.ndarray]:
    """userVars of KStarHyperonExtended plus the invariants used for binning.

    ``kp, pi0`` are the K* daughters (kIndices), ``proton, pim`` the hyperon daughters
    (lambdaIndices).  Returns a dict of (N,) float64 arrays.
    """
    beam, kp, pi0, proton, pim = (np.asarray(a, dtype=np.float64) for a in (beam, kp, pi0, proton, pim))
    resonance = kp + pi0
    hyperon = proton + pim

    hyperon_resonance = boost_to_rest(hyperon, resonance)
    p1_resonance = boost_to_rest(kp, resonance)
    kstar_hyperon = boost_to_rest(resonance, hyperon)
    y1_hyperon = boost_to_rest(proton, hyperon)

    beam_hat = _unit(beam[:, 1:])

    # K* helicity frame
    y = _unit(np.cross(beam_hat, -_unit(hyperon[:, 1:])))
    z = -_unit(hyperon_resonance[:, 1:])
    x = _unit(np.cross(y, z))
    a = p1_resonance[:, 1:]
    ax, ay, az = _dot(a, x), _dot(a, y), _dot(a, z)
    theta_V = np.arctan2(np.hypot(ax, ay), az)        # TVector3::Theta
    phi_V = np.arctan2(ay, ax)                         # TVector3::Phi

    # hyperon frame
    yH = _unit(np.cross(beam_hat, -_unit(resonance[:, 1:])))
    zH = -_unit(kstar_hyperon[:, 1:])
    xH = _unit(np.cross(yH, zH))
    q = y1_hyperon[:, 1:]
    qmag = np.linalg.norm(q, axis=-1)
    # cos(TVector3::Angle) = clamp(q . e / |q| |e|); the axes are unit vectors
    n_x = np.clip(_dot(q, xH) / qmag, -1.0, 1.0)
    n_y = np.clip(_dot(q, yH) / qmag, -1.0, 1.0)
    n_z = np.clip(_dot(q, zH) / qmag, -1.0, 1.0)

    eps = np.array([1.0, 0.0, 0.0])
    Phi_lab = np.arctan2(y @ eps, _dot(beam_hat, np.cross(eps, y)))

    target = np.zeros_like(beam)
    target[:, 0] = TARGET_MASS
    return {
        "E_beam": beam[:, 0],
        "t": mass2(beam - resonance),
        "t_hyperon": mass2(target - hyperon),
        "M_Kpi": np.sqrt(np.clip(mass2(resonance), 0.0, None)),
        "M_ppi": np.sqrt(np.clip(mass2(hyperon), 0.0, None)),
        "theta_V": theta_V,
        "phi_V": phi_V,
        "n_x": n_x,
        "n_y": n_y,
        "n_z": n_z,
        "theta_Y": np.arccos(n_z),
        "phi_Y": np.arctan2(n_y, n_x),
        "Phi_lab": Phi_lab,
    }


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Map to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(a)))

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
  Gottfried-Jackson (columns with suffix _GJ; same y axes as above, x = y x z):
    K* GJ frame     : z = beam direction in the K* rest frame (photon at the meson vertex);
                      (theta_V_GJ, phi_V_GJ) are the polar angles of the K+ there.
    hyperon GJ frame: z = target-proton direction in the hyperon rest frame (incoming
                      baryon at the baryon vertex, as for the Delta in rho Delta);
                      n_*_GJ are the proton direction cosines, n_y_GJ = n_y.
                      Both GJ axes lie in the production plane, so each frame differs from
                      its helicity frame by a rotation about y; Phi and Phi_lab are unchanged.
                      The GJ frames are made orthonormal (z_GJ projected orthogonal to y), which
                      matters only where the four-momenta do not balance (some thrown MC events).
  beam azimuth      : Phi_lab = atan2(y . eps, beam_hat . (eps x y)) with eps = (1, 0, 0);
                      the fit uses Phi = polAngle [rad] + Phi_lab and
                      g1 = -P_gamma cos 2Phi, g2 = -P_gamma sin 2Phi.
"""
from __future__ import annotations

import numpy as np

CONVENTION_VERSION = "kstarhyperonextended-v2"   # v2 = v1 + the _GJ columns (v1 columns unchanged)

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


def _polar(v, x, y, z):
    """(theta, phi) of v in the orthonormal frame (x, y, z)."""
    vx, vy, vz = _dot(v, x), _dot(v, y), _dot(v, z)
    return np.arctan2(np.hypot(vx, vy), vz), np.arctan2(vy, vx)


def _cosines(v, x, y, z):
    vmag = np.linalg.norm(v, axis=-1)
    return tuple(np.clip(_dot(v, e) / vmag, -1.0, 1.0) for e in (x, y, z))


GJ_SUFFIX = "_GJ"
_KSTAR_HYPERON_NAMES = {"mass": "M_Kpi", "mass_B": "M_ppi", "t_recoil": "t_hyperon", "theta": "theta_V", "phi": "phi_V",
                        "theta_B": "theta_Y", "phi_B": "phi_Y"}


def two_vertex_variables(beam, upper, lower, target_mass: float = TARGET_MASS, gj: bool = True) -> dict[str, np.ndarray]:
    """Kinematic variables of  beam + target -> X + B  for any reaction of this topology.

    ``upper`` is the list of four-vectors of the meson system X (one or two particles), ``lower`` that of the
    recoil system B (one or two).  The first particle of a two-particle vertex is its analyzer.  Returns

      E_beam, t = (beam - X)^2, t_recoil = (target - B)^2, mass, mass_B, Phi_lab         always
      theta, phi                    decay angles of X in its helicity frame               two-particle X
      theta_B, phi_B, n_x, n_y, n_z decay angles and direction cosines in the B frame     two-particle B
      the same with suffix _GJ in the Gottfried-Jackson frames                            with ``gj``

    The frames are those of the module docstring; every operation is the one of ``calcUserVars``.
    """
    beam = np.asarray(beam, dtype=np.float64)
    upper = [np.asarray(p, dtype=np.float64) for p in upper]
    lower = [np.asarray(p, dtype=np.float64) for p in lower]
    if not 1 <= len(upper) <= 2 or not 1 <= len(lower) <= 2:
        raise NotImplementedError("each vertex must have one or two particles")
    resonance = upper[0] + upper[1] if len(upper) == 2 else upper[0]
    hyperon = lower[0] + lower[1] if len(lower) == 2 else lower[0]
    beam_hat = _unit(beam[:, 1:])
    y = _unit(np.cross(beam_hat, -_unit(hyperon[:, 1:])))
    eps = np.array([1.0, 0.0, 0.0])
    target = np.zeros_like(beam)
    target[:, 0] = target_mass
    out = {
        "E_beam": beam[:, 0],
        "t": mass2(beam - resonance),
        "t_recoil": mass2(target - hyperon),
        "mass": np.sqrt(np.clip(mass2(resonance), 0.0, None)),
        "mass_B": np.sqrt(np.clip(mass2(hyperon), 0.0, None)),
        "Phi_lab": np.arctan2(y @ eps, _dot(beam_hat, np.cross(eps, y))),
    }
    if len(upper) == 2:
        z = -_unit(boost_to_rest(hyperon, resonance)[:, 1:])
        x = _unit(np.cross(y, z))
        a = boost_to_rest(upper[0], resonance)[:, 1:]
        ax, ay, az = _dot(a, x), _dot(a, y), _dot(a, z)
        out["theta"] = np.arctan2(np.hypot(ax, ay), az)        # TVector3::Theta
        out["phi"] = np.arctan2(ay, ax)                         # TVector3::Phi
        if gj:
            # x = y x z_GJ, then z = x x y: an orthonormal frame even when the four-momenta do not
            # balance exactly (thrown MC), identical to z_GJ when they do (data, kinematic fit)
            xG = _unit(np.cross(y, _unit(boost_to_rest(beam, resonance)[:, 1:])))
            zG = np.cross(xG, y)
            out["theta" + GJ_SUFFIX], out["phi" + GJ_SUFFIX] = _polar(a, xG, y, zG)
    if len(lower) == 2:
        yH = _unit(np.cross(beam_hat, -_unit(resonance[:, 1:])))
        zH = -_unit(boost_to_rest(resonance, hyperon)[:, 1:])
        xH = _unit(np.cross(yH, zH))
        q = boost_to_rest(lower[0], hyperon)[:, 1:]
        qmag = np.linalg.norm(q, axis=-1)
        # cos(TVector3::Angle) = clamp(q . e / |q| |e|); the axes are unit vectors
        n_x = np.clip(_dot(q, xH) / qmag, -1.0, 1.0)
        n_y = np.clip(_dot(q, yH) / qmag, -1.0, 1.0)
        n_z = np.clip(_dot(q, zH) / qmag, -1.0, 1.0)
        out.update({"n_x": n_x, "n_y": n_y, "n_z": n_z, "theta_B": np.arccos(n_z), "phi_B": np.arctan2(n_y, n_x)})
        if gj:
            xHG = _unit(np.cross(yH, _unit(boost_to_rest(target, hyperon)[:, 1:])))
            zHG = np.cross(xHG, yH)
            nx, ny, nz = _cosines(q, xHG, yH, zHG)
            out.update({"n_x" + GJ_SUFFIX: nx, "n_y" + GJ_SUFFIX: ny, "n_z" + GJ_SUFFIX: nz,
                        "theta_B" + GJ_SUFFIX: np.arccos(nz), "phi_B" + GJ_SUFFIX: np.arctan2(ny, nx)})
    return out


def rename(variables: dict, names: dict) -> dict:
    """Rename generic variables (mass, theta, theta_B, ...) by ``names``; a _GJ variable follows its base name."""
    def new(key):
        base, gj = (key[:-len(GJ_SUFFIX)], GJ_SUFFIX) if key.endswith(GJ_SUFFIX) else (key, "")
        return names.get(key, names.get(base, base) + gj)
    return {new(k): v for k, v in variables.items()}


def kstar_hyperon_angles(beam, kp, pi0, proton, pim, gj: bool = True) -> dict[str, np.ndarray]:
    """userVars of KStarHyperonExtended plus the invariants used for binning: ``two_vertex_variables`` for
    K* -> K+ pi0 and hyperon -> p pi-, under the names of the K* Lambda analysis.

    With ``gj`` the Gottfried-Jackson angles (``*_GJ``) are added.  Returns a dict of (N,) float64 arrays.
    """
    return rename(two_vertex_variables(beam, [kp, pi0], [proton, pim], gj=gj), _KSTAR_HYPERON_NAMES)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Map to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(a)))

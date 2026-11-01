"""Synthetic gamma p -> K*+ Lambda, K* -> K+ pi0, Lambda -> p pi- events (flat decays)."""
from __future__ import annotations

import numpy as np
import pytest

M = {"p": 0.938272, "Kp": 0.493677, "pi0": 0.134977, "pim": 0.139570, "Kstar": 0.8917, "Lambda": 1.115683}


def _boost(p, beta):
    """Boost (N, 4) four-vectors by velocity beta (N, 3) (rest frame -> frame moving with beta)."""
    b2 = np.sum(beta**2, axis=1)
    g = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(beta * p[:, 1:], axis=1)
    coef = np.where(b2 > 0, (g - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
    out = np.empty_like(p)
    out[:, 0] = g * (p[:, 0] + bp)
    out[:, 1:] = p[:, 1:] + ((coef * bp + g * p[:, 0])[:, None]) * beta
    return out


def _two_body(parent, m1, m2, rng):
    """Isotropic two-body decay of (N, 4) ``parent``; returns lab four-vectors of both daughters."""
    n = len(parent)
    M0 = np.sqrt(parent[:, 0] ** 2 - np.sum(parent[:, 1:] ** 2, axis=1))
    q = np.sqrt((M0**2 - (m1 + m2) ** 2) * (M0**2 - (m1 - m2) ** 2)) / (2 * M0)
    c = rng.uniform(-1, 1, n); s = np.sqrt(1 - c**2); ph = rng.uniform(-np.pi, np.pi, n)
    d = np.stack([s * np.cos(ph), s * np.sin(ph), c], axis=1) * q[:, None]
    a = np.column_stack([np.sqrt(m1**2 + q**2), d])
    b = np.column_stack([np.sqrt(m2**2 + q**2), -d])
    beta = parent[:, 1:] / parent[:, :1]
    return _boost(a, beta), _boost(b, beta)


def make_events(n: int = 2000, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    E = rng.uniform(8.2, 8.8, n)
    beam = np.column_stack([E, np.zeros(n), np.zeros(n), E])
    target = np.column_stack([np.full(n, M["p"]), np.zeros((n, 3))])
    W = beam + target
    # CM production: forward-peaked K*, random azimuth
    s = W[:, 0] ** 2 - W[:, 3] ** 2
    rs = np.sqrt(s)
    q = np.sqrt((s - (M["Kstar"] + M["Lambda"]) ** 2) * (s - (M["Kstar"] - M["Lambda"]) ** 2)) / (2 * rs)
    c = rng.uniform(0.7, 1.0, n); sn = np.sqrt(1 - c**2); ph = rng.uniform(-np.pi, np.pi, n)
    d = np.stack([sn * np.cos(ph), sn * np.sin(ph), c], axis=1) * q[:, None]
    kstar_cm = np.column_stack([np.sqrt(M["Kstar"] ** 2 + q**2), d])
    lam_cm = np.column_stack([np.sqrt(M["Lambda"] ** 2 + q**2), -d])
    beta = W[:, 1:] / W[:, :1]
    kstar, lam = _boost(kstar_cm, beta), _boost(lam_cm, beta)
    kp, pi0 = _two_body(kstar, M["Kp"], M["pi0"], rng)
    proton, pim = _two_body(lam, M["p"], M["pim"], rng)
    return {"beam": beam, "kp": kp, "pi0": pi0, "proton": proton, "pim": pim, "lambda": lam}


@pytest.fixture(scope="session")
def events():
    return make_events()

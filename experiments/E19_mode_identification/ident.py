"""Mode identification from a few sensors, for E19: peaks, their class, their shape.

Input: the Welch cross-spectral matrix G[i, j, f] of the kept sensors (E17's
convention, <conj(X_i) X_j>), their positions, and the deck's geometry.

Peaks.  Frequency domain decomposition: at each bin the largest eigenvalue s1(f) of
G(f) and its eigenvector u(f).  Candidates are the local maxima of log10 s1 inside the
band, with prominence >= `prominence` (tuned on the validation decks, never on test);
the MAX_PEAKS most prominent are kept.

Labels.  Four classes, as in E18: bend1, tors1, bend2, tors2 (character, then rank
by frequency).  Two labellers:

  rank        a peak is bending if its eigenvector moves the kept sensors on the two
              outer girders in phase, torsion otherwise; the lowest two of each
              character are class 1 and 2.  Needs nothing but the sensors.
  population  every assignment of four distinct peaks to the four classes (bend1 below
              bend2, tors1 below tors2) is scored by
                sum_c  log p(u_peak | class c)            the shape at the sensors
                + lam * log p(frequencies | geometry)     the frequencies
              and the best is kept.  p(u | c) is the angular central Gaussian of E18's
              class prior at the kept sensors, K_c = B_s Sigma_c B_s^T + tau_c^2 I:
              log p = -log det K_c - s log(u^H K_c^-1 u), blind to scale and phase as a
              mode shape must be.  The frequency term is a Gaussian on log f(bend1) and
              on log(f_c / f_bend1) for the other three, fitted to the K training decks
              nearest in geometry (same girder count; log L, log dy, balcony, bottom-beam
              position) - what a population says about a deck's frequency pattern
              before it is measured.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

CLASSES = ("bend1", "tors1", "bend2", "tors2")
BAND_HZ = (1.5, 20.0)
K_NEIGHBOURS = 100
MAX_PEAKS = 25            # the most prominent, so the assignment search stays small


def fdd(G: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Band-limited bins, s1 and the leading eigenvector at each bin."""
    keep = np.where((f >= BAND_HZ[0]) & (f <= BAND_HZ[1]))[0]
    Gb = np.moveaxis(G[:, :, keep], -1, 0)                        # (F, s, s), Hermitian
    w, v = np.linalg.eigh(Gb)
    return keep, w[:, -1], v[:, :, -1]


def peaks(G: np.ndarray, f: np.ndarray, prominence: float) -> list[dict]:
    keep, s1, U = fdd(G, f)
    idx, props = find_peaks(np.log10(np.maximum(s1, 1e-300)), prominence=prominence)
    idx = np.sort(idx[np.argsort(-props["prominences"])[:MAX_PEAKS]])
    out = []
    for i in idx:
        u = U[i]
        u = u * np.conj(u[np.argmax(np.abs(u))]) / np.abs(u[np.argmax(np.abs(u))])   # largest entry real > 0
        out.append({"bin": int(keep[i]), "f": float(f[keep[i]]), "s1": float(s1[i]), "u": u / np.linalg.norm(u)})
    return out


def character(u: np.ndarray, sxy: np.ndarray, ya: float, yb: float, x_tol: float, tol: float = 1.0) -> str:
    """'bend' if the kept sensors on the two outer girders move in phase, else 'tors'."""
    a = np.where(np.abs(sxy[:, 1] - ya) < tol)[0]
    b = np.where(np.abs(sxy[:, 1] - yb) < tol)[0]
    acc = 0.0
    for i in a:
        j = b[np.argmin(np.abs(sxy[b, 0] - sxy[i, 0]))]
        if abs(sxy[j, 0] - sxy[i, 0]) <= x_tol:
            acc += float(np.real(u[i] * np.conj(u[j])))
    return "bend" if acc >= 0 else "tors"


def assign_rank(pk: list[dict], chars: list[str]) -> dict[str, int]:
    out = {}
    for k in ("bend", "tors"):
        ids = sorted([i for i, c in enumerate(chars) if c == k], key=lambda i: pk[i]["f"])
        for r, i in enumerate(ids[:2]):
            out[f"{k}{r + 1}"] = i
    return out


def acg_loglik(u: np.ndarray, K: np.ndarray) -> float:
    """Complex angular central Gaussian log-density of direction u under covariance K."""
    sign, logdet = np.linalg.slogdet(K)
    q = float(np.real(np.conj(u) @ np.linalg.solve(K, u)))
    return float(-logdet - u.size * np.log(max(q, 1e-300)))


class FreqPrior:
    """Gaussian on log f(bend1) and log(f_c / f_bend1), from the K nearest training decks."""

    def __init__(self, table: dict, feats: dict):
        # table[sample] = {class: freq}, feats[sample] = (Ng, feature vector)
        self.samples = [s for s in table if all(c in table[s] for c in CLASSES)]
        self.ng = np.array([feats[s][0] for s in self.samples])
        Z = np.array([feats[s][1] for s in self.samples])
        self.mu_z, self.sd_z = Z.mean(0), Z.std(0) + 1e-12
        self.Z = (Z - self.mu_z) / self.sd_z
        F = np.array([[table[s][c] for c in CLASSES] for s in self.samples])
        self.logf1 = np.log(F[:, 0])
        self.logr = np.log(F[:, 1:] / F[:, :1])                    # tors1, bend2, tors2 over bend1

    def fit(self, ng: int, z: np.ndarray) -> dict:
        idx = np.where(self.ng == ng)[0]
        d = np.linalg.norm(self.Z[idx] - (z - self.mu_z) / self.sd_z, axis=1)
        nn = idx[np.argsort(d)[:K_NEIGHBOURS]]
        return {"m1": self.logf1[nn].mean(), "s1": self.logf1[nn].std() + 1e-3,
                "mr": self.logr[nn].mean(0), "sr": self.logr[nn].std(0) + 1e-3}

    @staticmethod
    def loglik(model: dict, f: dict[str, float]) -> float:
        lp = -0.5 * ((np.log(f["bend1"]) - model["m1"]) / model["s1"]) ** 2 - np.log(model["s1"])
        for k, c in enumerate(CLASSES[1:]):
            r = np.log(f[c] / f["bend1"])
            lp += -0.5 * ((r - model["mr"][k]) / model["sr"][k]) ** 2 - np.log(model["sr"][k])
        return float(lp)


def assign_population(pk: list[dict], shape_ll: np.ndarray, model: dict | None, lam: float) -> dict[str, int]:
    """Best assignment of four distinct peaks; shape_ll is (n_peaks, 4) in CLASSES order.

    Exhaustive over (bend1, bend2) with the best (tors1, tors2) for each found by one
    masked argmax, so the search is exact.  The score is FreqPrior.loglik term by term."""
    P = len(pk)
    if P < 4:
        return {}
    f = np.array([p["f"] for p in pk])
    lf = np.log(f)
    if lam and model is not None:
        g0 = -0.5 * ((lf - model["m1"]) / model["s1"]) ** 2 - np.log(model["s1"])
        R = lf[None, :] - lf[:, None]                                      # R[b1, x] = log(f_x / f_b1)
        T = [-0.5 * ((R - model["mr"][k]) / model["sr"][k]) ** 2 - np.log(model["sr"][k]) for k in range(3)]
    else:
        lam, g0, T = 0.0, np.zeros(P), [np.zeros((P, P))] * 3
    up = f[:, None] < f[None, :]
    best, arg = -np.inf, {}
    for b1 in range(P):
        ct = (shape_ll[:, 1] + lam * T[0][b1])[:, None] + (shape_ll[:, 3] + lam * T[2][b1])[None, :]
        for b2 in np.where(f > f[b1])[0]:
            valid = up.copy()
            valid[[b1, b2], :] = False
            valid[:, [b1, b2]] = False
            if not valid.any():
                continue
            k = int(np.argmax(np.where(valid, ct, -np.inf)))
            t1, t2 = divmod(k, P)
            sc = shape_ll[b1, 0] + shape_ll[b2, 2] + lam * (g0[b1] + T[1][b1, b2]) + ct[t1, t2]
            if sc > best:
                best, arg = sc, {"bend1": b1, "tors1": int(t1), "bend2": int(b2), "tors2": int(t2)}
    return arg

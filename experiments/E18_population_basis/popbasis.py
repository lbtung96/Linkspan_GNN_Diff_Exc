"""Population-basis fill-in for E18: what the linkspan population knows about mode shapes.

Every deck, synthetic or real, is described by the generator's parameters L_mm, Ng
and dy_mm, with girder lines at y = j * dy (j = 0..Ng-1), the hinge at x = 0 and the
front beam at x = L.  In normalised coordinates

    xi  = x / L                      0 at the hinge, 1 at the front beam
    eta = (y - y_c) / (W_g / 2)      -1 and +1 at the outer girders (W_g = (Ng-1) dy),
                                     beyond them on the deck edges and balconies

a mode shape is expanded in a tensor Legendre basis, P_i(2 xi - 1) Q_j(eta / ETA_SCALE),
i <= DEG_X, j <= DEG_Y.  The coefficients of every training shape (unit RMS over its
nodes) give the population prior: a zero-mean Gaussian over coefficients with covariance
Sigma = mean(a a^T), plus a nugget tau^2 (the mean per-node residual of the expansion).
The prior is sign-free, because a a^T is.

Filling in is kriging with that prior: for kept nodes K and any nodes T,

    fill_T = B_T Sigma B_K^T (B_K Sigma B_K^T + tau^2 I)^-1 values_K,

a linear operator W (n x s) like every imputer in E16, except that its weights can be
negative, so it can put a nodal line between sensors that move together.  The kept rows
are reset to their measured values, as every imputer does.

Priors are built per girder count, for two kinds of question:
  pooled     the four target modes together: needs no knowledge of which mode is which
  by class   one prior per mode class, defined from the shape itself rather than from the
             matcher's labels (which swap B2/T2 on Ng = 2 decks with a balcony): character
             (bending if the two outer girder lines move in phase, torsion otherwise) and
             rank by frequency among the target modes of that character, capped at 2:
             bend1, tors1, bend2, tors2.  Using it at a mode means knowing its class, which
             two sensors on opposite girders can tell (phase) together with peak order.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as LG

DEG_X, DEG_Y = 8, 3
ETA_SCALE = 2.5            # balconies reach |eta| ~ 2.3 on the narrowest decks
TOL_MM = 1.0
CLASSES = ("bend1", "tors1", "bend2", "tors2")
RIDGE = 1e-8               # relative, for the per-structure least-squares fit only


def norm_coords(coords: np.ndarray, L: float, ng: int, dy: float) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * (ng - 1) * dy
    return coords[:, 0] / L, (coords[:, 1] - half) / half


def design(xi: np.ndarray, eta: np.ndarray, dx: int = DEG_X, dyg: int = DEG_Y) -> np.ndarray:
    px = LG.legvander(2.0 * xi - 1.0, dx)
    py = LG.legvander(eta / ETA_SCALE, dyg)
    return (px[:, :, None] * py[:, None, :]).reshape(xi.size, -1)


def fit(B: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
    """Coefficients of v (unit RMS) in basis B, ridge-stabilised; and the mean squared residual."""
    v = v / np.sqrt(np.mean(v ** 2))
    G = B.T @ B
    a = np.linalg.solve(G + RIDGE * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), B.T @ v)
    return a, float(np.mean((v - B @ a) ** 2))


def line_correlation(values: np.ndarray, coords: np.ndarray, ya: float, yb: float, x_tol: float) -> float:
    """Uncentred correlation between two girder lines, nodes paired by x: > 0 in phase."""
    a = np.where(np.abs(coords[:, 1] - ya) < TOL_MM)[0]
    b = np.where(np.abs(coords[:, 1] - yb) < TOL_MM)[0]
    pa, pb = [], []
    for i in a:
        j = b[np.argmin(np.abs(coords[b, 0] - coords[i, 0]))]
        if abs(coords[j, 0] - coords[i, 0]) <= x_tol:
            pa.append(values[i]); pb.append(values[j])
    pa, pb = np.asarray(pa), np.asarray(pb)
    return float(pa @ pb / np.sqrt((pa @ pa) * (pb @ pb) + 1e-300))


def classes(shapes: np.ndarray, freqs: np.ndarray, coords: np.ndarray, ya: float, yb: float,
            x_tol: float) -> list[str]:
    """Class of each target mode (columns of shapes): character, then rank by frequency."""
    kind = ["bend" if line_correlation(shapes[:, m], coords, ya, yb, x_tol) >= 0 else "tors"
            for m in range(shapes.shape[1])]
    out = [""] * shapes.shape[1]
    for k in ("bend", "tors"):
        ms = sorted([m for m in range(shapes.shape[1]) if kind[m] == k], key=lambda m: freqs[m])
        for r, m in enumerate(ms):
            out[m] = f"{k}{min(r + 1, 2)}"
    return out


def operator(B: np.ndarray, known: np.ndarray, Sigma: np.ndarray, tau2: float) -> np.ndarray:
    """Kriging weights W (n, s); known rows reset to the identity."""
    Bk = B[known]
    K = Bk @ Sigma @ Bk.T + tau2 * np.eye(known.size)
    W = np.linalg.solve(K, Bk @ Sigma @ B.T).T
    W[known] = np.eye(known.size)
    return W

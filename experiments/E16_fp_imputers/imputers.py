"""Graph feature imputers for E16, whole-node (sensor) missingness.

Adapted from github.com/lbtung96/Feature-propagation, fpbench/imputers.py at
commit 9b7582d: Zero, Mean, NeighborMean, FP-rw, Harmonic, Tikhonov, PCFI.
`FP` is not here.  It is the production operator (phase3_utils.feature_propagation,
D^-1/2 A D^-1/2, 50 iterations, known rows reset), which the stage calls through
E14's `propagate` so that its rows reproduce s8.

Every method here except PCFI's inter-channel step is linear in the measured
values, so it is built once per sensor set as an (n, s) operator W - the method
run on the s unit vectors - and applied as W @ x[sensors].  That is the same map
as running the method on the data, and it keeps the time series (tens of
thousands of samples per node) out of the iterations.

Changes from the repository, for graphs of 36-168 nodes with whole nodes missing:
  * Harmonic and Tikhonov are solved directly (dense) instead of by batched CG at
    tol 1e-4: the same quantities without the solver tolerance.  Harmonic drops
    the eps regulariser, which only matters for components with no known node;
    every graph here is connected and has one.
  * FP-rw runs 50 iterations (the repository uses 40), so that it differs from
    the production FP in the normalisation only.
  * PCFI's inter-channel refinement mixes channels by their correlation across
    nodes.  It runs on PSD bins and on mode shapes.  On a time series the
    channels are time samples, where that correlation means nothing and the
    (T, T) matrix is out of reach, so time series get PCFI's diffusion only.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

METHODS = ("Zero", "Mean", "NeighborMean", "FP-rw", "Harmonic", "Tikhonov", "PCFI")
FP_RW_ITERATIONS = 50
TIKHONOV_LAMBDA = 1.0
PCFI_ALPHA, PCFI_BETA, PCFI_ITERATIONS = 0.9, 1.0, 100


def adjacency(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Dense A; duplicate edges sum, as the repository's coalesced sparse A does."""
    A = np.zeros((n, n))
    np.add.at(A, (edge_index[0], edge_index[1]), 1.0)
    return A


def _unit(n: int, known: np.ndarray) -> np.ndarray:
    X = np.zeros((n, known.size))
    X[known, np.arange(known.size)] = 1.0
    return X


def _hops_to_known(A: np.ndarray, known: np.ndarray) -> np.ndarray:
    d = shortest_path(csr_matrix(A), directed=False, unweighted=True, indices=known)
    return d.min(axis=0)


def operator(name: str, edge_index: np.ndarray, n: int, known: np.ndarray):
    """(W, xi): W maps the known rows (s, C) to all rows (n, C); xi is PCFI's confidence."""
    known = np.asarray(known, dtype=np.int64)
    A = adjacency(edge_index, n)
    deg = A.sum(axis=1)
    X = _unit(n, known)
    is_known = np.zeros(n, dtype=bool)
    is_known[known] = True
    unk = np.where(~is_known)[0]
    xi = None

    if name == "Zero":
        W = X
    elif name == "Mean":
        W = np.full((n, known.size), 1.0 / known.size)
        W[known] = X[known]
    elif name == "NeighborMean":
        num, den = A @ X, A @ is_known.astype(float)
        W = np.where(den[:, None] > 0, num / np.maximum(den, 1e-12)[:, None], 1.0 / known.size)
        W[known] = X[known]
    elif name == "FP-rw":
        P = A / np.maximum(deg, 1e-12)[:, None]
        W = X.copy()
        for _ in range(FP_RW_ITERATIONS):
            W = P @ W
            W[known] = X[known]
    elif name == "Harmonic":
        L = np.diag(deg) - A
        W = X.copy()
        W[unk] = np.linalg.solve(L[np.ix_(unk, unk)], A[np.ix_(unk, known)])
    elif name == "Tikhonov":
        dinv = np.where(deg > 0, deg, np.inf) ** -0.5
        L = np.eye(n) - dinv[:, None] * A * dinv[None, :]
        M = np.diag(is_known.astype(float))
        W = np.linalg.solve(M + TIKHONOV_LAMBDA * L, M @ X)   # known rows are not pinned
    elif name == "PCFI":
        xi = PCFI_ALPHA ** _hops_to_known(A, known)
        den = A @ xi
        W = X.copy()
        for _ in range(PCFI_ITERATIONS):
            num = A @ (xi[:, None] * W)
            W = np.where(den[:, None] > 0, num / np.maximum(den, 1e-12)[:, None], W)
            W[known] = X[known]
    else:
        raise ValueError(f"unknown imputer {name!r}")
    return W, xi


def apply(name: str, W: np.ndarray, xi, known: np.ndarray, xs: np.ndarray,
          refine: bool) -> np.ndarray:
    """Impute every row from the known rows xs (s, C).  `refine` = PCFI's channel step."""
    out = W @ np.asarray(xs, dtype=np.float64)
    if name != "PCFI" or not refine or PCFI_BETA <= 0 or out.shape[1] < 2:
        return out
    z = out - out.mean(axis=0, keepdims=True)
    z = z / np.maximum(np.linalg.norm(z, axis=0, keepdims=True), 1e-12)
    R = z.T @ z
    np.fill_diagonal(R, 0.0)
    R = R / np.maximum(np.abs(R).sum(axis=0, keepdims=True), 1e-12)
    refined = out + PCFI_BETA * (1.0 - xi)[:, None] * (out @ R)
    refined[known] = out[known]
    return refined

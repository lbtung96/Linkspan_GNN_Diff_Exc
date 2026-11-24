"""Shared utilities for Phase 3 real-structure pipeline.

Provides feature_propagation without depending on torch_scatter.
"""

from __future__ import annotations

import numpy as np
import torch

BANDPASS_F_LOW: float = 1.5   # Hz — lower passband edge for modal content filter
BANDPASS_F_HIGH: float = 20.0  # Hz — upper passband edge for modal content filter


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    f_low: float = BANDPASS_F_LOW,
    f_high: float = BANDPASS_F_HIGH,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter isolating the 4-mode frequency band.

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples)
    fs : sampling frequency in Hz
    f_low, f_high : passband edges in Hz (default 1.5–13.5 Hz)
    order : filter order; sosfiltfilt applies it forward+backward, so effective order is 2×order

    Returns
    -------
    ndarray, same shape and dtype as input
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * fs
    if f_low <= 0 or f_high >= nyq:
        raise ValueError(
            f"Passband [{f_low}, {f_high}] Hz is outside (0, {nyq:.1f}) Hz for fs={fs:.1f} Hz"
        )
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal).astype(signal.dtype)
    return np.apply_along_axis(
        lambda x: sosfiltfilt(sos, x), axis=-1, arr=signal
    ).astype(signal.dtype)


def _propagation_matrix(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Symmetric-normalized sparse adjacency (D^-0.5 A D^-0.5) via built-in PyTorch."""
    n_edges = edge_index.shape[1]
    edge_weight = torch.ones(n_edges, dtype=torch.float32, device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(n_nodes, dtype=torch.float32, device=edge_index.device)
    deg.scatter_add_(0, col, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    val = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(edge_index, val, (n_nodes, n_nodes))


def feature_propagation(
    x: torch.Tensor,
    mask: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
    n_iter: int = 50,
) -> torch.Tensor:
    """Diffuse features from known (mask=True) nodes to all other nodes.

    Replicates GraphSAGE_Modal_GNN.feature_propagation without torch_scatter.
    """
    out = torch.zeros_like(x)
    out[mask] = x[mask]
    adj = _propagation_matrix(edge_index, n_nodes)
    for _ in range(n_iter):
        out = torch.sparse.mm(adj, out)
        out[mask] = x[mask]
    return out


def load_training_freq_grid(
    # samples_phase2/ is empty (see CLAUDE.md); samples_phase2_bandpass/ carries a
    # byte-identical 400-bin grid (df = 0.09765625 Hz, 0 - 38.96484 Hz), verified 2026-08-31.
    synth_sample: str = "population_dataset_modes4/samples_phase2_bandpass/sample_0000_Ng2_phase1_gnn.npz",
    n_freq: int = 400,
) -> np.ndarray:
    """Return the 400-bin frequency grid used during synthetic training."""
    d = np.load(synth_sample)
    grid = d["freq_bins_hz"].astype(np.float64)
    assert grid.shape == (n_freq,), f"Expected {n_freq}-bin grid, got {grid.shape}"
    return grid


def load_graph_package(structure: str, graph_root: str = "phase3_real_graphs") -> dict:
    """Load graph_package.npz for a structure, return as plain dict of arrays."""
    from pathlib import Path
    path = Path(graph_root) / structure / "graph_package.npz"
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}

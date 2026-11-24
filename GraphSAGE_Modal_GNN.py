# -*- coding: utf-8 -*-
"""
GraphSAGE ModalGNN training script.

Supports two data sources:
1. legacy MATLAB object-array dataset
2. Phase-2A per-sample NPZ dataset under population_dataset_modes4/samples_phase2a/

Phase-2A contract:
- x_psd is already the normalized node PSD feature, shape [N, 400]
- y_mode_shape_abs is already the normalized absolute target, shape [N, 4]
- y_freq_hz is the graph-level natural-frequency target, shape [4]
- edge_attr_geom stores geometric edge attributes [length_mm, theta_rad]
- edge_attr_cpsd optionally stores CPSD-derived edge attributes [Re/Im coherency at B1,T1,B2,T2]
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific guard
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc

try:
    from torch_scatter import scatter_add
    TORCH_SCATTER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific guard
    scatter_add = None
    TORCH_SCATTER_IMPORT_ERROR = exc

try:
    import dgl
    import dgl.nn as dglnn
    from dgl.data import DGLDataset
    from dgl.dataloading import GraphDataLoader
    from dgl.nn import EdgeWeightNorm
    DGL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific guard
    dgl = None
    dglnn = None
    DGLDataset = object
    GraphDataLoader = None
    EdgeWeightNorm = None
    DGL_IMPORT_ERROR = exc


SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


DATA_SOURCE = "phase1_npz"
MODE_NAMES_PHASE1 = ("B1", "T1", "B2", "T2")
N_FREQ_TARGETS_PHASE2A = 4
DEFAULT_PHASE1_DIR = Path("population_dataset_modes4") / "samples_phase2a"
# The legacy MATLAB data source predates this repository and its dataset lives
# outside it, so there is no default any more: `--data-source legacy_mat` now
# requires an explicit `--mat-path`.  The consolidated pipeline never takes that
# branch - every run uses `phase1_npz`.
DEFAULT_MAT_PATH: Path | None = None
DEFAULT_RUNS_DIR = Path("runs_phase2")
# "cpsd_full" and "geom_orient" were patched in by the archive's fr_edge_modes.py (2026-09-28/29);
# they are library modes here, with the same arithmetic.
EDGE_FEATURE_MODES = ("geom", "zero", "cpsd", "cpsd_full", "geom_orient")
SHAPE_TARGET_MODES = ("abs", "signed")
SHAPE_LOSSES = ("mse", "signinv")
CPSD_EDGE_FEATURE_DIM = 8
CPSD_FULL_EDGE_FEATURE_DIM = 800
ORIENT_EDGE_FEATURE_DIM = 2


def edge_orientation(data: Any) -> np.ndarray:
    """(cos theta, sin theta) per directed edge, float32, from the stored edge angle edge_attr_geom[:, 1].

    No length, no phase: bounded in [-1, 1].  Shared by both architectures' "geom_orient" mode and by
    EdgeGAT's 802-d "cpsd_full_orient" mode."""
    theta = np.asarray(data["edge_attr_geom"], dtype=np.float64)[:, 1]
    return np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)


@dataclass
class TrainingConfig:
    data_source: str = DATA_SOURCE
    phase1_dir: Path = DEFAULT_PHASE1_DIR
    mat_path: Path | None = DEFAULT_MAT_PATH
    runs_dir: Path = DEFAULT_RUNS_DIR
    run_name: str | None = None
    expected_n_freq: int = 400
    expected_n_modes_out: int = 4
    batch_size: int = 16
    n_epoch: int = 200
    lr: float = 1e-3
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = SEED
    enc_hid: int = 64
    enc_out: int = 64
    gnn_hid: int = 64
    gnn_out: int = 64
    dec_hid: int = 64
    enable_freq_multitask: bool = True
    freq_loss_weight: float = 0.25
    freq_std_eps: float = 1e-6
    shuffle_train: bool = True
    max_samples: int | None = None
    save_last_checkpoint: bool = True
    show_plot: bool = False
    edge_feature_mode: str = "geom"
    shape_target_mode: str = "abs"
    edge_feat_dim: int = 2


@dataclass
class SplitArtifacts:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


@dataclass
class RunContext:
    run_dir: Path
    checkpoints_dir: Path
    eval_dir: Path
    predictions_dir: Path


@dataclass
class ModalGNNOutput:
    """Structured model outputs for the shared node/graph encoder.

    Attributes
    ----------
    pred_mode_shape_abs:
        Nodewise mode-shape prediction with shape [num_nodes_total, n_modes_out].
        For a batched DGL graph this is the concatenated node axis across the batch.
    pred_freq:
        Graph-level natural-frequency prediction with shape [batch_size, 4],
        aligned to [B1, T1, B2, T2].
    """

    pred_mode_shape_abs: torch.Tensor
    pred_freq: torch.Tensor


@dataclass
class FrequencyNormalizationStats:
    freq_mean_hz: np.ndarray
    freq_std_hz: np.ndarray


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GraphSAGEEdgeWeighted(nn.Module):
    def __init__(self, in_feats: int, hid_feats: int, out_feats: int):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, aggregator_type="pool")
        self.conv2 = dglnn.SAGEConv(hid_feats, hid_feats, aggregator_type="pool")
        self.conv3 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type="pool")
        self.norm = EdgeWeightNorm(norm="both")

    def forward(self, graph, node_h, edge_w):
        edge_w = edge_w.squeeze(-1)
        edge_w = torch.sigmoid(edge_w)
        edge_w = self.norm(graph, edge_w)

        h = self.conv1(graph, node_h, edge_weight=edge_w)
        h = F.relu(h)
        h = self.conv2(graph, h, edge_weight=edge_w)
        h = F.relu(h)
        h = self.conv3(graph, h, edge_weight=edge_w)
        return h


class ModalGNN(nn.Module):
    def __init__(
        self,
        *,
        n_freq: int,
        n_modes_out: int,
        enc_hid: int,
        enc_out: int,
        gnn_hid: int,
        gnn_out: int,
        dec_hid: int,
        edge_feat_dim: int = 2,
        n_freq_targets: int = N_FREQ_TARGETS_PHASE2A,
    ):
        super().__init__()
        self.node_encoder = MLP(n_freq, enc_hid, enc_out)
        self.edge_encoder = MLP(edge_feat_dim, enc_hid, 1)
        self.gnn = GraphSAGEEdgeWeighted(enc_out, gnn_hid, gnn_out)
        self.decoder = MLP(gnn_out, dec_hid, n_modes_out)
        self.freq_head = MLP(gnn_out, dec_hid, n_freq_targets)

    def forward(self, g, node_x, edge_x) -> ModalGNNOutput:
        """Run the shared encoder and return nodewise and graph-level predictions.

        Parameters
        ----------
        g:
            DGLGraph or batched DGLGraph.
        node_x:
            Node PSD features with shape [num_nodes_total, n_freq].
        edge_x:
            Edge geometric features with shape [num_edges_total, edge_feat_dim].

        Returns
        -------
        ModalGNNOutput
            `pred_mode_shape_abs` has shape [num_nodes_total, n_modes_out].
            `pred_freq` has shape [batch_size, 4] using mean pooling over node embeddings.
        """
        h_node = self.node_encoder(node_x)
        h_edge = self.edge_encoder(edge_x)
        h = self.gnn(g, h_node, h_edge)
        pred_mode_shape_abs = self.decoder(h)
        with g.local_scope():
            g.ndata["_pooled_h"] = h
            pooled_h = dgl.mean_nodes(g, "_pooled_h")
        pred_freq = self.freq_head(pooled_h)
        return ModalGNNOutput(
            pred_mode_shape_abs=pred_mode_shape_abs,
            pred_freq=pred_freq,
        )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    return value


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    return _normalize_json_value(asdict(config))


def _edge_feature_dim_for_mode(edge_feature_mode: str) -> int:
    if edge_feature_mode in {"geom", "zero"}:
        return 2
    if edge_feature_mode == "cpsd":
        return CPSD_EDGE_FEATURE_DIM
    if edge_feature_mode == "cpsd_full":
        return CPSD_FULL_EDGE_FEATURE_DIM
    if edge_feature_mode == "geom_orient":
        return ORIENT_EDGE_FEATURE_DIM
    raise ValueError(
        f"Unsupported edge_feature_mode={edge_feature_mode!r}. Expected one of {EDGE_FEATURE_MODES}."
    )


def _resolve_edge_attr_array(data: Any, edge_feature_mode: str) -> np.ndarray:
    if edge_feature_mode == "cpsd_full":
        a = np.asarray(data["edge_attr_cpsd_full"], dtype=np.float32)
        n_edges = np.asarray(data["edge_index"]).shape[1]
        if a.shape != (n_edges, CPSD_FULL_EDGE_FEATURE_DIM):
            raise ValueError(f"edge_attr_cpsd_full must have shape ({n_edges}, {CPSD_FULL_EDGE_FEATURE_DIM}), "
                             f"got {a.shape}")
        return a
    if edge_feature_mode == "geom_orient":
        return edge_orientation(data)
    edge_attr_geom = np.asarray(data["edge_attr_geom"], dtype=np.float32)
    if edge_feature_mode == "geom":
        return np.asarray(edge_attr_geom, dtype=np.float32)
    if edge_feature_mode == "zero":
        return np.zeros_like(np.asarray(edge_attr_geom, dtype=np.float32))
    if edge_feature_mode == "cpsd":
        if "edge_attr_cpsd" not in data:
            raise KeyError("Required edge field 'edge_attr_cpsd' not found for edge_feature_mode='cpsd'.")
        edge_attr_cpsd = np.asarray(data["edge_attr_cpsd"], dtype=np.float32)
        if edge_attr_cpsd.shape != (edge_attr_geom.shape[0], CPSD_EDGE_FEATURE_DIM):
            raise ValueError(
                f"edge_attr_cpsd must have shape ({edge_attr_geom.shape[0]}, {CPSD_EDGE_FEATURE_DIM}), got {edge_attr_cpsd.shape}."
            )
        return edge_attr_cpsd
    raise ValueError(
        f"Unsupported edge_feature_mode={edge_feature_mode!r}. Expected one of {EDGE_FEATURE_MODES}."
    )


def _shape_target_field_name(shape_target_mode: str) -> str:
    if shape_target_mode == "abs":
        return "y_mode_shape_abs"
    if shape_target_mode == "signed":
        return "y_mode_shape_signed"
    raise ValueError(
        f"Unsupported shape_target_mode={shape_target_mode!r}. Expected one of {SHAPE_TARGET_MODES}."
    )


def _resolve_shape_target_array(data: Any, shape_target_mode: str) -> np.ndarray:
    field_name = _shape_target_field_name(shape_target_mode)
    if field_name not in data:
        raise KeyError(f"Required shape target field '{field_name}' not found for shape_target_mode={shape_target_mode!r}.")
    return np.asarray(data[field_name], dtype=np.float32)


def _validate_phase1_npz_fields(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    expected_n_freq: int,
    expected_n_modes: int,
    shape_target_mode: str = "abs",
    edge_feature_mode: str = "geom",
) -> None:
    required_fields = [
        "graph_node_ids",
        "edge_index",
        "edge_attr_geom",
        "x_psd",
        "y_mode_shape_abs",
        "y_freq_hz",
        "freq_bins_hz",
        "target_mode_names",
    ]
    missing = [name for name in required_fields if name not in data]
    if missing:
        raise KeyError(f"{npz_path} is missing required Phase-1 fields: {missing}")

    graph_node_ids = np.asarray(data["graph_node_ids"], dtype=np.int64).reshape(-1)
    edge_index = np.asarray(data["edge_index"], dtype=np.int64)
    edge_attr_geom = np.asarray(data["edge_attr_geom"], dtype=np.float32)
    x_psd = np.asarray(data["x_psd"], dtype=np.float32)
    y_mode_shape = _resolve_shape_target_array(data, shape_target_mode)
    y_freq_hz = np.asarray(data["y_freq_hz"], dtype=np.float32).reshape(-1)
    freq_bins_hz = np.asarray(data["freq_bins_hz"], dtype=np.float32).reshape(-1)
    target_mode_names = np.asarray(data["target_mode_names"]).reshape(-1)

    if edge_index.shape[0] != 2:
        raise ValueError(f"{npz_path}: edge_index must have shape (2, E_dir), got {edge_index.shape}.")
    if edge_attr_geom.shape != (edge_index.shape[1], 2):
        raise ValueError(
            f"{npz_path}: edge_attr_geom must have shape ({edge_index.shape[1]}, 2), got {edge_attr_geom.shape}."
        )
    if edge_feature_mode == "cpsd":
        if "edge_attr_cpsd" not in data:
            raise KeyError(f"{npz_path}: missing required field 'edge_attr_cpsd' for edge_feature_mode='cpsd'.")
        edge_attr_cpsd = np.asarray(data["edge_attr_cpsd"], dtype=np.float32)
        if edge_attr_cpsd.shape != (edge_index.shape[1], CPSD_EDGE_FEATURE_DIM):
            raise ValueError(
                f"{npz_path}: edge_attr_cpsd must have shape ({edge_index.shape[1]}, {CPSD_EDGE_FEATURE_DIM}), got {edge_attr_cpsd.shape}."
            )
    if x_psd.shape != (graph_node_ids.size, expected_n_freq):
        raise ValueError(
            f"{npz_path}: x_psd must have shape ({graph_node_ids.size}, {expected_n_freq}), got {x_psd.shape}."
        )
    if y_mode_shape.shape != (graph_node_ids.size, expected_n_modes):
        raise ValueError(
            f"{npz_path}: {_shape_target_field_name(shape_target_mode)} must have shape ({graph_node_ids.size}, {expected_n_modes}), got {y_mode_shape.shape}."
        )
    if y_freq_hz.shape != (N_FREQ_TARGETS_PHASE2A,):
        raise ValueError(
            f"{npz_path}: y_freq_hz must have shape ({N_FREQ_TARGETS_PHASE2A},), got {y_freq_hz.shape}."
        )
    if freq_bins_hz.shape[0] != expected_n_freq:
        raise ValueError(
            f"{npz_path}: freq_bins_hz must have length {expected_n_freq}, got {freq_bins_hz.shape[0]}."
        )
    if target_mode_names.shape[0] != expected_n_modes:
        raise ValueError(
            f"{npz_path}: target_mode_names must have length {expected_n_modes}, got {target_mode_names.shape[0]}."
        )
    if np.any(edge_index < 0) or np.any(edge_index >= graph_node_ids.size):
        raise ValueError(f"{npz_path}: edge_index has values outside [0, {graph_node_ids.size - 1}].")


def _build_graph_from_phase1_npz(
    npz_path: Path,
    expected_n_freq: int,
    expected_n_modes: int,
    *,
    edge_feature_mode: str = "geom",
    shape_target_mode: str = "abs",
    return_freq_target: bool = False,
):
    with np.load(npz_path, allow_pickle=True) as data:
        _validate_phase1_npz_fields(
            data,
            npz_path,
            expected_n_freq,
            expected_n_modes,
            shape_target_mode=shape_target_mode,
            edge_feature_mode=edge_feature_mode,
        )

        graph_node_ids = np.asarray(data["graph_node_ids"], dtype=np.int64).reshape(-1)
        edge_index = np.asarray(data["edge_index"], dtype=np.int64)
        x_psd = np.asarray(data["x_psd"], dtype=np.float32)
        y_mode_shape = _resolve_shape_target_array(data, shape_target_mode)
        y_freq_hz = np.asarray(data["y_freq_hz"], dtype=np.float32).reshape(-1)
        edge_attr = _resolve_edge_attr_array(data, edge_feature_mode)
        target_mode_names = tuple(str(x) for x in np.asarray(data["target_mode_names"]).reshape(-1).tolist())
        sample_id = str(np.asarray(data["sample_id"]).item()) if "sample_id" in data else npz_path.stem

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=int(graph_node_ids.size))
    g.ndata["x"] = torch.tensor(x_psd, dtype=torch.float32)
    g.ndata["y"] = torch.tensor(y_mode_shape, dtype=torch.float32)
    g.edata["edge_x"] = torch.tensor(edge_attr, dtype=torch.float32)
    g.ndata["graph_node_id"] = torch.tensor(graph_node_ids, dtype=torch.int64)
    g.graph_data = {
        "npz_path": str(npz_path),
        "sample_identifier": npz_path.stem,
        "sample_id": sample_id,
        "target_mode_names": target_mode_names,
        "edge_feature_mode": edge_feature_mode,
        "shape_target_mode": shape_target_mode,
    }
    freq_target = torch.tensor(y_freq_hz, dtype=torch.float32)
    if return_freq_target:
        return g, freq_target
    return g


def build_phase1_graph_from_payload(
    payload: dict[str, Any],
    expected_n_freq: int,
    expected_n_modes: int,
    *,
    edge_feature_mode: str = "geom",
    shape_target_mode: str = "abs",
    return_freq_target: bool = False,
):
    graph_node_ids = np.asarray(payload["graph_node_ids"], dtype=np.int64).reshape(-1)
    edge_index = np.asarray(payload["edge_index"], dtype=np.int64)
    edge_attr_geom = np.asarray(payload["edge_attr_geom"], dtype=np.float32)
    edge_attr = _resolve_edge_attr_array(payload, edge_feature_mode)
    x_psd = np.asarray(payload["x_psd"], dtype=np.float32)
    y_mode_shape = _resolve_shape_target_array(payload, shape_target_mode)
    y_freq_hz = np.asarray(payload["y_freq_hz"], dtype=np.float32).reshape(-1)
    target_mode_names = tuple(str(x) for x in np.asarray(payload["target_mode_names"]).reshape(-1).tolist())

    if x_psd.shape != (graph_node_ids.size, expected_n_freq):
        raise ValueError(
            f"x_psd must have shape ({graph_node_ids.size}, {expected_n_freq}), got {x_psd.shape}."
        )
    if y_mode_shape.shape != (graph_node_ids.size, expected_n_modes):
        raise ValueError(
            f"{_shape_target_field_name(shape_target_mode)} must have shape ({graph_node_ids.size}, {expected_n_modes}), got {y_mode_shape.shape}."
        )
    if y_freq_hz.shape != (N_FREQ_TARGETS_PHASE2A,):
        raise ValueError(
            f"y_freq_hz must have shape ({N_FREQ_TARGETS_PHASE2A},), got {y_freq_hz.shape}."
        )
    if edge_index.shape != (2, edge_attr_geom.shape[0]):
        raise ValueError(
            f"edge_index must have shape (2, E_dir) with edge_attr_geom shape (E_dir, 2), got {edge_index.shape} and {edge_attr_geom.shape}."
        )
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"Resolved edge_attr row count {edge_attr.shape[0]} does not match edge count {edge_index.shape[1]}."
        )

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=int(graph_node_ids.size))
    g.ndata["x"] = torch.tensor(x_psd, dtype=torch.float32)
    g.ndata["y"] = torch.tensor(y_mode_shape, dtype=torch.float32)
    g.edata["edge_x"] = torch.tensor(edge_attr, dtype=torch.float32)
    g.ndata["graph_node_id"] = torch.tensor(graph_node_ids, dtype=torch.int64)
    g.graph_data = {
        "sample_identifier": str(payload.get("sample_identifier", "inference_case")),
        "sample_id": str(np.asarray(payload.get("sample_id", "inference_case")).item() if np.asarray(payload.get("sample_id", "inference_case")).shape == () else payload.get("sample_id", "inference_case")),
        "target_mode_names": target_mode_names,
        "edge_feature_mode": edge_feature_mode,
        "shape_target_mode": shape_target_mode,
    }
    freq_target = torch.tensor(y_freq_hz, dtype=torch.float32)
    if return_freq_target:
        return g, freq_target
    return g


class Phase1NPZDataset(DGLDataset):
    def __init__(self, sample_paths: list[Path], expected_n_freq: int, expected_n_modes: int, edge_feature_mode: str = "geom", shape_target_mode: str = "abs"):
        self.sample_paths = list(sample_paths)
        self.expected_n_freq = expected_n_freq
        self.expected_n_modes = expected_n_modes
        self.edge_feature_mode = edge_feature_mode
        self.shape_target_mode = shape_target_mode
        super().__init__(name="Phase1NPZDataset")

    def process(self):
        self.graphs = []
        self.labels = []
        self.sample_records = []
        for npz_path in self.sample_paths:
            g, y_freq_hz = _build_graph_from_phase1_npz(
                npz_path,
                self.expected_n_freq,
                self.expected_n_modes,
                edge_feature_mode=self.edge_feature_mode,
                shape_target_mode=self.shape_target_mode,
                return_freq_target=True,
            )
            self.graphs.append(g.to(DEVICE))
            self.labels.append(y_freq_hz)
            self.sample_records.append(g.graph_data)
        self.labels = torch.stack(self.labels, dim=0) if self.labels else torch.empty((0, N_FREQ_TARGETS_PHASE2A), dtype=torch.float32)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class MatlabModalDataset(DGLDataset):
    def __init__(
        self,
        mat_path: Path,
        case_indices: list[int],
        expected_n_freq: int,
        expected_n_modes: int,
    ):
        self.mat_path = Path(mat_path)
        self.case_indices = list(case_indices)
        self.expected_n_freq = expected_n_freq
        self.expected_n_modes = expected_n_modes
        super().__init__(name="MatlabModalDataset")

    def process(self):
        mat = sio.loadmat(self.mat_path)
        acc_magnitude = mat["acceleration_fft_abs_out"][:, 0]
        phi = np.abs(mat["modeshape_out"][:, 0])
        element = mat["element_out"][:, 0]
        element_L = mat["element_L_out"][:, 0]
        element_theta = mat["element_theta_out"][:, 0] / 180.0

        self.graphs = []
        self.labels = []
        self.sample_records = []

        for case_id in self.case_indices:
            ele = element[case_id] - 1
            src = np.concatenate([ele[:, 0], ele[:, 1]], axis=0)
            dst = np.concatenate([ele[:, 1], ele[:, 0]], axis=0)

            g = dgl.graph((src, dst))

            x = np.asarray(acc_magnitude[case_id][:, :self.expected_n_freq], dtype=np.float32)
            y = np.asarray(phi[case_id][:, :self.expected_n_modes], dtype=np.float32)

            if x.shape[1] != self.expected_n_freq:
                raise ValueError(f"Legacy MATLAB sample {case_id} has x shape {x.shape}; expected second dim {self.expected_n_freq}.")
            if y.shape[1] != self.expected_n_modes:
                raise ValueError(f"Legacy MATLAB sample {case_id} has y shape {y.shape}; expected second dim {self.expected_n_modes}.")

            g.ndata["x"] = torch.tensor(x, dtype=torch.float32)
            g.ndata["y"] = torch.tensor(y, dtype=torch.float32)

            L = torch.tensor(np.asarray(element_L[case_id][:, 0], dtype=np.float32), dtype=torch.float32)
            th = torch.tensor(np.asarray(element_theta[case_id][:, 0], dtype=np.float32), dtype=torch.float32)
            g.edata["edge_x"] = torch.cat([torch.cat([L, L]).unsqueeze(1), torch.cat([th, th]).unsqueeze(1)], dim=1)
            g.graph_data = {
                "sample_identifier": f"case_{case_id:04d}",
                "sample_id": str(case_id),
                "npz_path": str(self.mat_path),
                "target_mode_names": tuple(f"mode_{k + 1}" for k in range(self.expected_n_modes)),
            }

            self.graphs.append(g.to(DEVICE))
            self.labels.append(torch.zeros(N_FREQ_TARGETS_PHASE2A, dtype=torch.float32))
            self.sample_records.append(g.graph_data)

        self.labels = torch.stack(self.labels, dim=0) if self.labels else torch.empty((0, N_FREQ_TARGETS_PHASE2A), dtype=torch.float32)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def _validate_split_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    total = train_fraction + val_fraction + test_fraction
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"train/val/test fractions must sum to 1.0, got {train_fraction} + {val_fraction} + {test_fraction} = {total}."
        )
    if train_fraction <= 0.0:
        raise ValueError("train_fraction must be positive.")
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be nonnegative.")


def _compute_split_counts(n_total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray(fractions, dtype=float) * float(n_total)
    counts = np.floor(raw).astype(int)
    remainder = int(n_total - counts.sum())
    fractional = raw - counts

    for idx in np.argsort(-fractional):
        if remainder <= 0:
            break
        counts[idx] += 1
        remainder -= 1

    active = [i for i, frac in enumerate(fractions) if frac > 0.0]
    if n_total >= len(active):
        for idx in active:
            if counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[idx] += 1

    if counts.sum() != n_total:
        raise RuntimeError(f"Split counts do not sum to n_total={n_total}: {counts.tolist()}")
    return int(counts[0]), int(counts[1]), int(counts[2])


def _split_sequence(items: list[Any], fractions: tuple[float, float, float], seed: int) -> tuple[list[Any], list[Any], list[Any]]:
    n_total = len(items)
    if n_total == 0:
        raise ValueError("Cannot split an empty item list.")

    n_train, n_val, n_test = _compute_split_counts(n_total, fractions)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    shuffled = [items[int(i)] for i in perm]

    train_items = shuffled[:n_train]
    val_items = shuffled[n_train:n_train + n_val]
    test_items = shuffled[n_train + n_val:n_train + n_val + n_test]
    return train_items, val_items, test_items


def _write_split_file(path: Path, items: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def _create_run_context(config: TrainingConfig) -> RunContext:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"{config.data_source}_{timestamp}"
    run_dir = Path(config.runs_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "evaluation"
    predictions_dir = eval_dir / "test_predictions"

    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        eval_dir=eval_dir,
        predictions_dir=predictions_dir,
    )


def _save_run_config(run_context: RunContext, config: TrainingConfig) -> None:
    with (run_context.run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=2, sort_keys=True)


def _save_split_artifacts(run_context: RunContext, split_artifacts: SplitArtifacts) -> None:
    _write_split_file(run_context.run_dir / "train_split.txt", split_artifacts.train_ids)
    _write_split_file(run_context.run_dir / "val_split.txt", split_artifacts.val_ids)
    _write_split_file(run_context.run_dir / "test_split.txt", split_artifacts.test_ids)
    with (run_context.run_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_train": len(split_artifacts.train_ids),
                "n_val": len(split_artifacts.val_ids),
                "n_test": len(split_artifacts.test_ids),
                "train_ids": split_artifacts.train_ids,
                "val_ids": split_artifacts.val_ids,
                "test_ids": split_artifacts.test_ids,
            },
            f,
            indent=2,
            sort_keys=True,
        )


def _save_training_history_csv(run_context: RunContext, history_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss_total",
        "train_loss_shape",
        "train_loss_freq",
        "train_freq_mae_hz",
        "val_loss_total",
        "val_loss_shape",
        "val_loss_freq",
        "val_freq_mae_hz",
        "n_train_samples",
        "n_val_samples",
        "batch_size",
        "seed",
        "learning_rate",
        "n_epochs",
    ]
    with (run_context.run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainingConfig,
    epoch: int,
    train_loss: float,
    val_loss: float,
    freq_stats: FrequencyNormalizationStats | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config_to_dict(config),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "device": str(DEVICE),
    }
    if freq_stats is not None:
        payload["freq_mean_hz"] = np.asarray(freq_stats.freq_mean_hz, dtype=np.float32)
        payload["freq_std_hz"] = np.asarray(freq_stats.freq_std_hz, dtype=np.float32)
    torch.save(payload, path)


def _mean_squared_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _mean_absolute_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def mac(u, v):
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)
    num = np.abs(np.dot(u, v)) ** 2
    den = np.dot(u, u) * np.dot(v, v)
    return float(num / (den + 1e-30))


def _compute_sample_metrics(pred: np.ndarray, target: np.ndarray, mode_names: tuple[str, ...]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "mse": _mean_squared_error(pred, target),
        "mae": _mean_absolute_error(pred, target),
    }
    for mode_idx, mode_name in enumerate(mode_names):
        metrics[f"mse_{mode_name}"] = _mean_squared_error(pred[:, mode_idx], target[:, mode_idx])
        metrics[f"mae_{mode_name}"] = _mean_absolute_error(pred[:, mode_idx], target[:, mode_idx])
        metrics[f"mac_{mode_name}"] = mac(pred[:, mode_idx], target[:, mode_idx])
    return metrics


def compute_mode_shape_metrics(pred: np.ndarray, target: np.ndarray, mode_names: tuple[str, ...]) -> dict[str, Any]:
    return _compute_sample_metrics(pred, target, mode_names)


def compute_frequency_normalization_stats(
    freq_targets_hz: torch.Tensor,
    *,
    eps: float,
) -> FrequencyNormalizationStats:
    freq_np = np.asarray(freq_targets_hz.detach().cpu().numpy(), dtype=np.float32)
    if freq_np.ndim != 2 or freq_np.shape[1] != N_FREQ_TARGETS_PHASE2A:
        raise ValueError(
            f"Frequency targets must have shape [n_samples, {N_FREQ_TARGETS_PHASE2A}], got {freq_np.shape}."
        )
    freq_mean_hz = np.mean(freq_np, axis=0, dtype=np.float64).astype(np.float32)
    freq_std_hz = np.std(freq_np, axis=0, dtype=np.float64).astype(np.float32)
    freq_std_hz = np.maximum(freq_std_hz, np.float32(eps))
    return FrequencyNormalizationStats(freq_mean_hz=freq_mean_hz, freq_std_hz=freq_std_hz)


def _freq_stats_to_device(
    freq_stats: FrequencyNormalizationStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(freq_stats.freq_mean_hz, dtype=torch.float32, device=device)
    std = torch.tensor(freq_stats.freq_std_hz, dtype=torch.float32, device=device)
    return mean, std


def normalize_frequency_targets(
    freq_hz: torch.Tensor,
    freq_stats: FrequencyNormalizationStats,
) -> torch.Tensor:
    mean, std = _freq_stats_to_device(freq_stats, freq_hz.device)
    return (freq_hz - mean) / std


def denormalize_frequency_predictions(
    pred_freq_z: torch.Tensor,
    freq_stats: FrequencyNormalizationStats,
) -> torch.Tensor:
    mean, std = _freq_stats_to_device(freq_stats, pred_freq_z.device)
    return pred_freq_z * std + mean


def compute_frequency_metrics(
    pred_freq_hz: np.ndarray,
    target_freq_hz: np.ndarray,
    mode_names: tuple[str, ...],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "freq_mse_hz2": _mean_squared_error(pred_freq_hz, target_freq_hz),
        "freq_mae_hz": _mean_absolute_error(pred_freq_hz, target_freq_hz),
    }
    for mode_idx, mode_name in enumerate(mode_names):
        metrics[f"freq_mae_{mode_name}_hz"] = _mean_absolute_error(
            pred_freq_hz[:, mode_idx],
            target_freq_hz[:, mode_idx],
        )
    return metrics


def signinv_shape_loss(pred: torch.Tensor, target: torch.Tensor, batch_num_nodes) -> torch.Tensor:
    """Sign-invariant shape loss: sum over graphs and modes of min(||p - y||^2, ||p + y||^2), / numel.

    A mode shape is defined up to its sign, and the population's labels carry a sign convention the
    inputs cannot always reveal (EC T13-T16: the B1 label sign flips with Ng; torsion's sign is
    undetermined without a left/right cue).  MSE on such a target pulls the prediction to zero; this
    loss scores each graph's mode against whichever sign of the label is closer.  batch_num_nodes:
    graph.batch_num_nodes() of the batched graph, so each graph's nodes are summed separately.
    (Archive: p_common.signinv_shape_loss, installed in memory by p_common.install_signinv.)"""
    bnn = torch.as_tensor(batch_num_nodes, dtype=torch.int64, device=pred.device)
    if int(bnn.sum()) != pred.shape[0]:
        raise RuntimeError(f"batch has {pred.shape[0]} nodes, batch_num_nodes sums to {int(bnn.sum())}")
    seg = torch.repeat_interleave(torch.arange(bnn.numel(), device=pred.device), bnn)
    sp = torch.zeros(bnn.numel(), pred.shape[1], dtype=pred.dtype, device=pred.device).index_add_(0, seg, (pred - target) ** 2)
    sm = torch.zeros(bnn.numel(), pred.shape[1], dtype=pred.dtype, device=pred.device).index_add_(0, seg, (pred + target) ** 2)
    return torch.minimum(sp, sm).sum() / pred.numel()


def _compute_multitask_loss(
    outputs: ModalGNNOutput,
    shape_target: torch.Tensor,
    freq_target_hz: torch.Tensor,
    *,
    loss_fn: nn.Module,
    enable_freq_multitask: bool,
    freq_loss_weight: float,
    freq_stats: FrequencyNormalizationStats | None,
    shape_loss: str = "mse",
    batch_num_nodes=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Shape loss (loss_fn, or the sign-invariant loss when shape_loss == "signinv", which needs the
    batch's batch_num_nodes) plus freq_loss_weight x loss_fn on z-scored frequencies."""
    if shape_loss == "signinv":
        if batch_num_nodes is None:
            raise ValueError("shape_loss='signinv' needs batch_num_nodes (graph.batch_num_nodes())")
        loss_shape = signinv_shape_loss(outputs.pred_mode_shape_abs, shape_target, batch_num_nodes)
    elif shape_loss == "mse":
        loss_shape = loss_fn(outputs.pred_mode_shape_abs, shape_target)
    else:
        raise ValueError(f"Unsupported shape_loss={shape_loss!r}. Expected one of {SHAPE_LOSSES}.")
    zero = torch.zeros((), dtype=loss_shape.dtype, device=loss_shape.device)
    if not enable_freq_multitask:
        return loss_shape, loss_shape, zero, None
    if freq_stats is None:
        raise ValueError("Frequency multitask loss requires normalization stats.")
    freq_target_z = normalize_frequency_targets(freq_target_hz, freq_stats)
    loss_freq = loss_fn(outputs.pred_freq, freq_target_z)
    loss_total = loss_shape + float(freq_loss_weight) * loss_freq
    pred_freq_hz = denormalize_frequency_predictions(outputs.pred_freq, freq_stats)
    return loss_total, loss_shape, loss_freq, pred_freq_hz


def _save_frequency_normalization(run_context: RunContext, freq_stats: FrequencyNormalizationStats, config: TrainingConfig) -> None:
    with (run_context.run_dir / "frequency_normalization.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "freq_mean_hz": np.asarray(freq_stats.freq_mean_hz, dtype=np.float32).tolist(),
                "freq_std_hz": np.asarray(freq_stats.freq_std_hz, dtype=np.float32).tolist(),
                "freq_loss_weight": float(config.freq_loss_weight),
                "edge_feature_mode": config.edge_feature_mode,
        "shape_target_mode": config.shape_target_mode,
                "enable_freq_multitask": bool(config.enable_freq_multitask),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def build_model_from_config(config_like: dict[str, Any] | TrainingConfig) -> ModalGNN:
    if isinstance(config_like, TrainingConfig):
        cfg = config_to_dict(config_like)
    else:
        cfg = dict(config_like)

    return ModalGNN(
        n_freq=int(cfg["expected_n_freq"]),
        n_modes_out=int(cfg["expected_n_modes_out"]),
        enc_hid=int(cfg["enc_hid"]),
        enc_out=int(cfg["enc_out"]),
        gnn_hid=int(cfg["gnn_hid"]),
        gnn_out=int(cfg["gnn_out"]),
        dec_hid=int(cfg["dec_hid"]),
        edge_feat_dim=int(cfg.get("edge_feat_dim", _edge_feature_dim_for_mode(str(cfg.get("edge_feature_mode", "geom"))))),
        n_freq_targets=N_FREQ_TARGETS_PHASE2A,
    )


def load_trained_model_checkpoint(checkpoint_path: Path, device: torch.device | None = None) -> tuple[ModalGNN, dict[str, Any]]:
    ckpt_device = device if device is not None else DEVICE
    checkpoint = torch.load(Path(checkpoint_path), map_location=ckpt_device)
    model = build_model_from_config(checkpoint["config"]).to(ckpt_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _aggregate_metric_rows(metric_rows: list[dict[str, Any]], mode_names: tuple[str, ...]) -> dict[str, Any]:
    if not metric_rows:
        return {
            "n_test_samples": 0,
            "mse_mean": None,
            "mae_mean": None,
            "freq_mae_hz_mean": None,
            "freq_mse_hz2_mean": None,
        }

    summary: dict[str, Any] = {
        "n_test_samples": len(metric_rows),
        "mse_mean": float(np.mean([row["mse"] for row in metric_rows])),
        "mae_mean": float(np.mean([row["mae"] for row in metric_rows])),
        "freq_mae_hz_mean": float(np.mean([row["freq_mae_hz"] for row in metric_rows])),
        "freq_mse_hz2_mean": float(np.mean([row["freq_mse_hz2"] for row in metric_rows])),
    }
    for mode_name in mode_names:
        summary[f"mse_{mode_name}_mean"] = float(np.mean([row[f"mse_{mode_name}"] for row in metric_rows]))
        summary[f"mae_{mode_name}_mean"] = float(np.mean([row[f"mae_{mode_name}"] for row in metric_rows]))
        summary[f"mac_{mode_name}_mean"] = float(np.mean([row[f"mac_{mode_name}"] for row in metric_rows]))
        summary[f"freq_mae_{mode_name}_hz_mean"] = float(np.mean([row[f"freq_mae_{mode_name}_hz"] for row in metric_rows]))
    return summary


def _save_evaluation_outputs(
    run_context: RunContext,
    metric_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    mode_names: tuple[str, ...],
) -> None:
    with (run_context.eval_dir / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_normalize_json_value(summary), f, indent=2, sort_keys=True)

    fieldnames = ["sample_identifier", "sample_id", "npz_path", "mse", "mae"]
    for mode_name in mode_names:
        fieldnames.extend([f"mse_{mode_name}", f"mae_{mode_name}", f"mac_{mode_name}"])
    fieldnames.extend(["freq_mse_hz2", "freq_mae_hz"])
    for mode_name in mode_names:
        fieldnames.append(f"freq_mae_{mode_name}_hz")

    with (run_context.eval_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)


def _save_prediction_npz(
    path: Path,
    *,
    sample_identifier: str,
    sample_id: str,
    source_path: str,
    mode_names: tuple[str, ...],
    graph_node_ids: np.ndarray | None,
    prediction: np.ndarray,
    target: np.ndarray,
    pred_freq_hz: np.ndarray,
    target_freq_hz: np.ndarray,
    pred_freq_z: np.ndarray,
    target_freq_z: np.ndarray,
    metrics: dict[str, Any],
    shape_target_mode: str,
) -> None:
    payload = {
        "sample_identifier": np.asarray(sample_identifier),
        "sample_id": np.asarray(sample_id),
        "source_path": np.asarray(source_path),
        "target_mode_names": np.asarray(mode_names),
        "shape_target_mode": np.asarray(shape_target_mode),
        "target_field": np.asarray(_shape_target_field_name(shape_target_mode)),
        "prediction": np.asarray(prediction, dtype=np.float32),
        "target": np.asarray(target, dtype=np.float32),
        "pred_freq_hz": np.asarray(pred_freq_hz, dtype=np.float32),
        "target_freq_hz": np.asarray(target_freq_hz, dtype=np.float32),
        "pred_freq_z": np.asarray(pred_freq_z, dtype=np.float32),
        "target_freq_z": np.asarray(target_freq_z, dtype=np.float32),
        "graph_node_ids": np.asarray(graph_node_ids if graph_node_ids is not None else np.array([], dtype=np.int64), dtype=np.int64),
        "metrics_json": np.asarray(json.dumps(_normalize_json_value(metrics), sort_keys=True)),
    }
    np.savez(path, **payload)


def _build_phase1_datasets(config: TrainingConfig) -> tuple[Phase1NPZDataset, Phase1NPZDataset, Phase1NPZDataset, SplitArtifacts]:
    phase1_dir = Path(config.phase1_dir)
    if not phase1_dir.exists():
        raise FileNotFoundError(
            f"Phase-1 sample directory not found: {phase1_dir}. "
            "Generate Phase-2A NPZ files under population_dataset_modes4/samples_phase2a/ first."
        )

    sample_paths = sorted(phase1_dir.glob("*.npz"))
    if config.max_samples is not None:
        sample_paths = sample_paths[: config.max_samples]
    if not sample_paths:
        raise FileNotFoundError(f"No Phase-1 NPZ samples found in {phase1_dir}.")

    train_paths, val_paths, test_paths = _split_sequence(
        sample_paths,
        (config.train_fraction, config.val_fraction, config.test_fraction),
        config.seed,
    )
    split_artifacts = SplitArtifacts(
        train_ids=[p.name for p in train_paths],
        val_ids=[p.name for p in val_paths],
        test_ids=[p.name for p in test_paths],
    )

    print(f"Loaded Phase-2A sample list from {phase1_dir}")
    print(f"Number of samples loaded: {len(sample_paths)}")
    print(f"Train/val/test split: {len(train_paths)}/{len(val_paths)}/{len(test_paths)}")

    train_set = Phase1NPZDataset(train_paths, config.expected_n_freq, config.expected_n_modes_out, edge_feature_mode=config.edge_feature_mode, shape_target_mode=config.shape_target_mode)
    val_set = Phase1NPZDataset(val_paths, config.expected_n_freq, config.expected_n_modes_out, edge_feature_mode=config.edge_feature_mode, shape_target_mode=config.shape_target_mode)
    test_set = Phase1NPZDataset(test_paths, config.expected_n_freq, config.expected_n_modes_out, edge_feature_mode=config.edge_feature_mode, shape_target_mode=config.shape_target_mode)
    return train_set, val_set, test_set, split_artifacts


def _build_legacy_datasets(config: TrainingConfig) -> tuple[MatlabModalDataset, MatlabModalDataset, MatlabModalDataset, SplitArtifacts]:
    if config.mat_path is None:
        raise ValueError(
            "--data-source legacy_mat requires an explicit --mat-path. "
            "The legacy MATLAB dataset lives outside this repository and has no "
            "default; the consolidated pipeline uses --data-source phase1_npz."
        )
    if not config.mat_path.exists():
        raise FileNotFoundError(f"Legacy MAT dataset not found: {config.mat_path}")

    mat = sio.loadmat(config.mat_path)
    n_all = int(mat["acceleration_fft_abs_out"].shape[0])
    case_indices = list(range(n_all))
    if config.max_samples is not None:
        case_indices = case_indices[: config.max_samples]

    train_indices, val_indices, test_indices = _split_sequence(
        case_indices,
        (config.train_fraction, config.val_fraction, config.test_fraction),
        config.seed,
    )
    split_artifacts = SplitArtifacts(
        train_ids=[f"case_{idx:04d}" for idx in train_indices],
        val_ids=[f"case_{idx:04d}" for idx in val_indices],
        test_ids=[f"case_{idx:04d}" for idx in test_indices],
    )

    print(f"Loaded legacy MATLAB dataset from {config.mat_path}")
    print(f"Number of samples loaded: {len(case_indices)}")
    print(f"Train/val/test split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")

    train_set = MatlabModalDataset(config.mat_path, train_indices, config.expected_n_freq, config.expected_n_modes_out)
    val_set = MatlabModalDataset(config.mat_path, val_indices, config.expected_n_freq, config.expected_n_modes_out)
    test_set = MatlabModalDataset(config.mat_path, test_indices, config.expected_n_freq, config.expected_n_modes_out)
    return train_set, val_set, test_set, split_artifacts


def build_datasets(config: TrainingConfig):
    if config.data_source == "phase1_npz":
        return _build_phase1_datasets(config)
    if config.data_source == "legacy_mat":
        return _build_legacy_datasets(config)
    raise ValueError(f"Unsupported data_source={config.data_source!r}. Expected 'phase1_npz' or 'legacy_mat'.")


def evaluate_test_set(
    model: nn.Module,
    test_set,
    run_context: RunContext,
    *,
    config: TrainingConfig,
    freq_stats: FrequencyNormalizationStats | None,
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    mode_names: tuple[str, ...] | None = None

    model.eval()
    with torch.no_grad():
        for graph, freq_target_hz in test_set:
            outputs = model(graph, graph.ndata["x"], graph.edata["edge_x"])
            pred = outputs.pred_mode_shape_abs.detach().cpu().numpy()
            target = graph.ndata["y"].detach().cpu().numpy()
            freq_target_hz = freq_target_hz.to(DEVICE)
            if config.enable_freq_multitask:
                if freq_stats is None:
                    raise ValueError("Frequency multitask evaluation requires normalization stats.")
                pred_freq_hz_t = denormalize_frequency_predictions(outputs.pred_freq, freq_stats)
                target_freq_z_t = normalize_frequency_targets(freq_target_hz.unsqueeze(0), freq_stats)
                pred_freq_z = outputs.pred_freq.detach().cpu().numpy()
                target_freq_z = target_freq_z_t.detach().cpu().numpy()
                pred_freq_hz = pred_freq_hz_t.detach().cpu().numpy()
                target_freq_hz_np = freq_target_hz.unsqueeze(0).detach().cpu().numpy()
            else:
                pred_freq_z = np.zeros((1, N_FREQ_TARGETS_PHASE2A), dtype=np.float32)
                target_freq_z = np.zeros((1, N_FREQ_TARGETS_PHASE2A), dtype=np.float32)
                pred_freq_hz = np.zeros((1, N_FREQ_TARGETS_PHASE2A), dtype=np.float32)
                target_freq_hz_np = freq_target_hz.unsqueeze(0).detach().cpu().numpy()
            graph_node_ids = graph.ndata["graph_node_id"].detach().cpu().numpy() if "graph_node_id" in graph.ndata else None
            graph_data = getattr(graph, "graph_data", {})
            sample_identifier = str(graph_data.get("sample_identifier", "sample"))
            sample_id = str(graph_data.get("sample_id", sample_identifier))
            source_path = str(graph_data.get("npz_path", ""))
            mode_names = tuple(graph_data.get("target_mode_names", tuple(f"mode_{i + 1}" for i in range(target.shape[1]))))

            metrics = {
                **_compute_sample_metrics(pred, target, mode_names),
                **compute_frequency_metrics(pred_freq_hz, target_freq_hz_np, mode_names),
            }
            metric_rows.append(
                {
                    "sample_identifier": sample_identifier,
                    "sample_id": sample_id,
                    "npz_path": source_path,
                    **metrics,
                }
            )

            _save_prediction_npz(
                run_context.predictions_dir / f"{sample_identifier}_prediction.npz",
                sample_identifier=sample_identifier,
                sample_id=sample_id,
                source_path=source_path,
                mode_names=mode_names,
                graph_node_ids=graph_node_ids,
                prediction=pred,
                target=target,
                pred_freq_hz=pred_freq_hz,
                target_freq_hz=target_freq_hz_np,
                pred_freq_z=pred_freq_z,
                target_freq_z=target_freq_z,
                metrics=metrics,
                shape_target_mode=config.shape_target_mode,
            )

    if mode_names is None:
        mode_names = MODE_NAMES_PHASE1
    summary = _aggregate_metric_rows(metric_rows, mode_names)
    _save_evaluation_outputs(run_context, metric_rows, summary, mode_names)
    return summary


def train_one_run(config: TrainingConfig) -> dict[str, Any]:
    _validate_split_fractions(config.train_fraction, config.val_fraction, config.test_fraction)
    set_global_seed(config.seed)

    run_context = _create_run_context(config)
    _save_run_config(run_context, config)

    train_set, val_set, test_set, split_artifacts = build_datasets(config)
    _save_split_artifacts(run_context, split_artifacts)

    freq_stats: FrequencyNormalizationStats | None = None
    if config.enable_freq_multitask:
        if config.data_source != "phase1_npz":
            raise ValueError("Frequency multitask training is only supported with data_source='phase1_npz'.")
        freq_stats = compute_frequency_normalization_stats(train_set.labels, eps=config.freq_std_eps)
        _save_frequency_normalization(run_context, freq_stats, config)

    train_loader = GraphDataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        drop_last=False,
    )
    val_loader = GraphDataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = ModalGNN(
        n_freq=config.expected_n_freq,
        n_modes_out=config.expected_n_modes_out,
        enc_hid=config.enc_hid,
        enc_out=config.enc_out,
        gnn_hid=config.gnn_hid,
        gnn_out=config.gnn_out,
        dec_hid=config.dec_hid,
        edge_feat_dim=config.edge_feat_dim,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    sample_graph = train_set[0][0]
    sample_freq = train_set[0][1]
    print(
        "Sample debug:",
        {
            "nodes": int(sample_graph.num_nodes()),
            "edges": int(sample_graph.num_edges()),
            "x_shape": tuple(sample_graph.ndata["x"].shape),
            "y_shape": tuple(sample_graph.ndata["y"].shape),
            "y_freq_hz_shape": tuple(sample_freq.shape),
            "edge_x_shape": tuple(sample_graph.edata["edge_x"].shape),
            "source": getattr(sample_graph, "graph_data", {}),
        },
    )
    if freq_stats is not None:
        print(
            "Frequency normalization:",
            {
                "freq_mean_hz": freq_stats.freq_mean_hz.tolist(),
                "freq_std_hz": freq_stats.freq_std_hz.tolist(),
            },
        )

    debug_printed = False
    history_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    t0 = time.time()
    for epoch in range(config.n_epoch):
        model.train()
        train_loss_total = 0.0
        train_loss_shape = 0.0
        train_loss_freq = 0.0
        train_freq_mae_hz = 0.0

        for g_batch, freq_target_hz in train_loader:
            freq_target_hz = freq_target_hz.to(DEVICE)
            outputs = model(g_batch, g_batch.ndata["x"], g_batch.edata["edge_x"])
            pred = outputs.pred_mode_shape_abs
            target = g_batch.ndata["y"]

            if pred.shape != target.shape:
                raise RuntimeError(f"Model output shape {tuple(pred.shape)} does not match target shape {tuple(target.shape)}.")

            if not debug_printed:
                print(f"Batch debug x shape: {tuple(g_batch.ndata['x'].shape)}")
                print(f"Batch debug edge_x shape: {tuple(g_batch.edata['edge_x'].shape)}")
                print(f"Mode-shape output shape: {tuple(pred.shape)}")
                print(f"Frequency output shape: {tuple(outputs.pred_freq.shape)}")
                print(f"Target shape: {tuple(target.shape)}")
                print(f"Frequency target shape: {tuple(freq_target_hz.shape)}")
                print("Output/target shape match confirmed.")
                debug_printed = True

            loss_total, loss_shape, loss_freq, pred_freq_hz = _compute_multitask_loss(
                outputs,
                target,
                freq_target_hz,
                loss_fn=loss_fn,
                enable_freq_multitask=config.enable_freq_multitask,
                freq_loss_weight=config.freq_loss_weight,
                freq_stats=freq_stats,
            )
            opt.zero_grad()
            loss_total.backward()
            opt.step()
            train_loss_total += float(loss_total.detach().item())
            train_loss_shape += float(loss_shape.detach().item())
            train_loss_freq += float(loss_freq.detach().item())
            if pred_freq_hz is not None:
                train_freq_mae_hz += float(torch.mean(torch.abs(pred_freq_hz - freq_target_hz)).detach().item())

        train_loss_total /= max(1, len(train_loader))
        train_loss_shape /= max(1, len(train_loader))
        train_loss_freq /= max(1, len(train_loader))
        train_freq_mae_hz /= max(1, len(train_loader))

        model.eval()
        val_loss_total = 0.0
        val_loss_shape = 0.0
        val_loss_freq = 0.0
        val_freq_mae_hz = 0.0
        with torch.no_grad():
            for g_batch, freq_target_hz in val_loader:
                freq_target_hz = freq_target_hz.to(DEVICE)
                outputs = model(g_batch, g_batch.ndata["x"], g_batch.edata["edge_x"])
                pred = outputs.pred_mode_shape_abs
                loss_total, loss_shape, loss_freq, pred_freq_hz = _compute_multitask_loss(
                    outputs,
                    g_batch.ndata["y"],
                    freq_target_hz,
                    loss_fn=loss_fn,
                    enable_freq_multitask=config.enable_freq_multitask,
                    freq_loss_weight=config.freq_loss_weight,
                    freq_stats=freq_stats,
                )
                val_loss_total += float(loss_total.detach().item())
                val_loss_shape += float(loss_shape.detach().item())
                val_loss_freq += float(loss_freq.detach().item())
                if pred_freq_hz is not None:
                    val_freq_mae_hz += float(torch.mean(torch.abs(pred_freq_hz - freq_target_hz)).detach().item())
        val_loss_total /= max(1, len(val_loader))
        val_loss_shape /= max(1, len(val_loader))
        val_loss_freq /= max(1, len(val_loader))
        val_freq_mae_hz /= max(1, len(val_loader))

        history_row = {
            "epoch": epoch,
            "train_loss_total": train_loss_total,
            "train_loss_shape": train_loss_shape,
            "train_loss_freq": train_loss_freq,
            "train_freq_mae_hz": train_freq_mae_hz,
            "val_loss_total": val_loss_total,
            "val_loss_shape": val_loss_shape,
            "val_loss_freq": val_loss_freq,
            "val_freq_mae_hz": val_freq_mae_hz,
            "n_train_samples": len(train_set),
            "n_val_samples": len(val_set),
            "batch_size": config.batch_size,
            "seed": config.seed,
            "learning_rate": config.lr,
            "n_epochs": config.n_epoch,
        }
        history_rows.append(history_row)
        print(
            f"epoch {epoch:4d} | "
            f"train total {train_loss_total:.6e} shape {train_loss_shape:.6e} freq {train_loss_freq:.6e} freq_mae_hz {train_freq_mae_hz:.6e} | "
            f"valid total {val_loss_total:.6e} shape {val_loss_shape:.6e} freq {val_loss_freq:.6e} freq_mae_hz {val_freq_mae_hz:.6e}"
        )

        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            best_epoch = epoch
            _save_checkpoint(
                run_context.checkpoints_dir / "best_model.pt",
                model,
                config,
                epoch,
                train_loss_total,
                val_loss_total,
                freq_stats=freq_stats,
            )

    train_time_s = time.time() - t0
    print(f"Training time: {train_time_s:.1f} s")

    if config.save_last_checkpoint and history_rows:
        _save_checkpoint(
            run_context.checkpoints_dir / "last_model.pt",
            model,
            config,
            config.n_epoch - 1,
            float(history_rows[-1]["train_loss_total"]),
            float(history_rows[-1]["val_loss_total"]),
            freq_stats=freq_stats,
        )

    _save_training_history_csv(run_context, history_rows)
    test_summary = evaluate_test_set(model, test_set, run_context, config=config, freq_stats=freq_stats)

    run_summary = {
        "data_source": config.data_source,
        "run_dir": str(run_context.run_dir),
        "device": str(DEVICE),
        "seed": config.seed,
        "n_train_samples": len(train_set),
        "n_val_samples": len(val_set),
        "n_test_samples": len(test_set),
        "enable_freq_multitask": bool(config.enable_freq_multitask),
        "freq_loss_weight": float(config.freq_loss_weight),
        "edge_feature_mode": config.edge_feature_mode,
        "shape_target_mode": config.shape_target_mode,
        "freq_mean_hz": None if freq_stats is None else freq_stats.freq_mean_hz.tolist(),
        "freq_std_hz": None if freq_stats is None else freq_stats.freq_std_hz.tolist(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss_total": history_rows[-1]["train_loss_total"] if history_rows else None,
        "final_train_loss_shape": history_rows[-1]["train_loss_shape"] if history_rows else None,
        "final_train_loss_freq": history_rows[-1]["train_loss_freq"] if history_rows else None,
        "final_val_loss_total": history_rows[-1]["val_loss_total"] if history_rows else None,
        "final_val_loss_shape": history_rows[-1]["val_loss_shape"] if history_rows else None,
        "final_val_loss_freq": history_rows[-1]["val_loss_freq"] if history_rows else None,
        "train_time_s": train_time_s,
        "test_summary": test_summary,
    }
    with (run_context.run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_normalize_json_value(run_summary), f, indent=2, sort_keys=True)

    if config.show_plot:
        if MATPLOTLIB_IMPORT_ERROR is not None:
            raise RuntimeError(
                "show_plot=True requires matplotlib in the active environment. "
                f"Original error: {MATPLOTLIB_IMPORT_ERROR}"
            )
        plt.figure()
        plt.plot(np.log10(np.array([row["train_loss_total"] for row in history_rows]) + 1e-30), label="train")
        plt.plot(np.log10(np.array([row["val_loss_total"] for row in history_rows]) + 1e-30), label="valid")
        plt.xlabel("Epoch")
        plt.ylabel("log10(MSE)")
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "run_context": run_context,
        "run_summary": run_summary,
        "history_rows": history_rows,
    }


def get_propagation_matrix(edge_index, n_nodes):
    if scatter_add is None:
        raise RuntimeError(
            "torch_scatter is required for feature propagation but is not available. "
            f"Original error: {TORCH_SCATTER_IMPORT_ERROR}"
        )
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
    val = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    adj = torch.sparse.FloatTensor(edge_index, values=val, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj


def feature_propagation(x, mask, edge_index, n_nodes, n_iter=50):
    out = torch.zeros_like(x)
    out[mask] = x[mask]

    adj = get_propagation_matrix(edge_index, n_nodes)
    for _ in range(n_iter):
        out = torch.sparse.mm(adj, out)
        out[mask] = x[mask]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE ModalGNN on legacy MAT data or Phase-2A NPZ samples.")
    parser.add_argument("--data-source", choices=["phase1_npz", "legacy_mat"], default=DATA_SOURCE)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--mat-path", type=Path, default=DEFAULT_MAT_PATH)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-freq", type=int, default=400)
    parser.add_argument("--n-modes-out", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freq-loss-weight", type=float, default=0.25)
    parser.add_argument("--disable-freq-multitask", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--edge-feature-mode", choices=EDGE_FEATURE_MODES, default="geom")
    parser.add_argument("--shape-target-mode", choices=SHAPE_TARGET_MODES, default="abs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if DGL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "DGL failed to import in this Python environment. "
            f"Original error: {DGL_IMPORT_ERROR}"
        )

    n_modes_out = args.n_modes_out
    if n_modes_out is None:
        n_modes_out = 4 if args.data_source == "phase1_npz" else 3

    config = TrainingConfig(
        data_source=args.data_source,
        phase1_dir=args.phase1_dir,
        mat_path=args.mat_path,
        runs_dir=args.runs_dir,
        run_name=args.run_name,
        expected_n_freq=args.n_freq,
        expected_n_modes_out=n_modes_out,
        batch_size=args.batch_size,
        n_epoch=args.epochs,
        lr=args.lr,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
        enable_freq_multitask=not args.disable_freq_multitask,
        freq_loss_weight=args.freq_loss_weight,
        show_plot=args.show_plot,
        edge_feature_mode=args.edge_feature_mode,
        shape_target_mode=args.shape_target_mode,
        edge_feat_dim=_edge_feature_dim_for_mode(args.edge_feature_mode),
    )

    if config.data_source == "phase1_npz":
        print("Phase-1 mode names:", MODE_NAMES_PHASE1)
    elif config.enable_freq_multitask:
        print("Disabling frequency multitask mode for legacy_mat data source.")
        config.enable_freq_multitask = False

    result = train_one_run(config)
    print(f"Run directory: {result['run_context'].run_dir}")
    print(f"Best validation loss: {result['run_summary']['best_val_loss']:.6e}")
    print(f"Test summary: {result['run_summary']['test_summary']}")


if __name__ == "__main__":
    main()

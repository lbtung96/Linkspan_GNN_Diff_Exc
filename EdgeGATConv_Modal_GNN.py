from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    from dgl.dataloading import GraphDataLoader
    from dgl.nn import EdgeGATConv
    DGL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    dgl = None
    GraphDataLoader = None
    EdgeGATConv = None
    DGL_IMPORT_ERROR = exc

from GraphSAGE_Modal_GNN import (
    DEVICE,
    FrequencyNormalizationStats,
    MODE_NAMES_PHASE1,
    N_FREQ_TARGETS_PHASE2A,
    SplitArtifacts,
    RunContext,
    _aggregate_metric_rows,
    _compute_multitask_loss,
    _normalize_json_value,
    _save_checkpoint,
    _save_evaluation_outputs,
    _save_frequency_normalization,
    _save_prediction_npz,
    _save_run_config,
    _save_split_artifacts,
    _save_training_history_csv,
    _split_sequence,
    _validate_split_fractions,
    compute_frequency_metrics,
    compute_frequency_normalization_stats,
    denormalize_frequency_predictions,
    edge_orientation,
    normalize_frequency_targets,
    set_global_seed,
)

# Library modes since the 2026-11-24 restructure (the archive patched them in memory):
#   geom_orient        (cos, sin) of the stored edge angle, 2-d       [fr_edge_modes, 2026-09-28]
#   zero               zeros of shape (E, 2): no edge information     [fr_edge_modes, 2026-09-29]
#   cpsd_full_orient   800-d complex coherency + the 2-d orientation  [p_common.install_cohorient, T16]
EDGE_FEATURE_MODES = ("cpsd_full", "cpsd_resonance", "geom_orient", "zero", "cpsd_full_orient")
# Node input.  "psd": the row-normalised Welch PSD x_psd (every network).  "amp": sqrt of the
# un-normalised PSD, deck nodes scaled by the deck-wide maximum and support nodes by their own row
# maximum (EC T16's amplitude arm only; archive: p_common.install_amp).
NODE_FEATURE_MODES = ("psd", "amp")
NODE_FEATURE_MODE = "psd"   # module default read by build_phase2a_graph_from_payload; set_node_feature_mode()


def set_node_feature_mode(mode: str) -> None:
    """Select the node input for every graph this module builds from now on (training and inference).
    The pipeline records the mode in each run's config.json (fr_train) and sets it from there."""
    global NODE_FEATURE_MODE
    if mode not in NODE_FEATURE_MODES:
        raise ValueError(f"Unsupported node_feature_mode={mode!r}. Expected one of {NODE_FEATURE_MODES}.")
    NODE_FEATURE_MODE = mode


def amp_node_features(x_psd: np.ndarray, row_scale: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """The "amp" node input: sqrt(x_psd * row scale); deck rows / their joint maximum, support rows
    (degree 1) / their own maximum.  float32."""
    P = np.asarray(x_psd, np.float64) * np.asarray(row_scale, np.float64).reshape(-1, 1)
    A = np.sqrt(np.maximum(P, 0.0))
    deg = np.bincount(np.asarray(edge_index)[0], minlength=A.shape[0])
    sup = deg == 1
    out = np.empty_like(A)
    out[~sup] = A[~sup] / A[~sup].max()
    out[sup] = A[sup] / np.maximum(A[sup].max(axis=1, keepdims=True), 1e-300)
    return out.astype(np.float32)
SHAPE_TARGET_MODES = ("abs", "signed")
DEFAULT_DATASET_ROOT = Path("population_dataset_modes4") / "samples_phase2a"
DEFAULT_RUN_ROOT = Path("runs_phase2") / "EdgeGATConv"
# DEFAULT_DATASET_ROOT = Path("population_dataset_modes4") / "samples_phase2a"
# DEFAULT_RUN_ROOT = Path("experiments_ablation") / "runs"


@dataclass
class TrainingConfig:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    run_root: Path = DEFAULT_RUN_ROOT
    run_name: str | None = None
    epochs: int = 100
    batch_size: int = 16
    hidden_dim: int = 32
    edge_hidden_dim: int = 32
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 123
    split_seed: int = 123
    max_samples: int | None = None
    train_frac: float = 0.7
    
    val_frac: float = 0.15
    test_frac: float = 0.15
    normalize_inputs: bool = False
    shape_target_mode: str = "signed"
    edge_feature_mode: str = "cpsd_full"
    freq_loss_weight: float = 0.25
    disable_freq_multitask: bool = False
    enable_freq_multitask: bool = True
    freq_std_eps: float = 1e-6
    allow_zero_in_degree: bool = True
    expected_n_modes_out: int = 4
    node_feat_dim: int = 0
    edge_feat_dim: int = 0


@dataclass
class ModalGNNOutput:
    pred_mode_shape_abs: torch.Tensor
    pred_freq: torch.Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeGATBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_hidden_dim: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
        allow_zero_in_degree: bool,
    ):
        super().__init__()
        head_out_dim = max(1, math.ceil(hidden_dim / max(1, num_heads)))
        self.conv = EdgeGATConv(
            hidden_dim,
            edge_hidden_dim,
            head_out_dim,
            num_heads,
            feat_drop=dropout,
            attn_drop=attn_dropout,
            residual=True,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.proj = nn.Linear(num_heads * head_out_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, node_h: torch.Tensor, edge_h: torch.Tensor) -> torch.Tensor:
        h_new = self.conv(graph, node_h, edge_h).flatten(1)
        h_new = self.proj(h_new)
        h_new = F.elu(h_new)
        h_new = self.dropout(h_new)
        return self.norm(node_h + h_new)


class EdgeGATModalGNN(nn.Module):
    def __init__(
        self,
        *,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        edge_hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
        n_modes_out: int,
        allow_zero_in_degree: bool,
    ):
        super().__init__()
        self.node_encoder = MLP(node_feat_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.edge_encoder = MLP(edge_feat_dim, edge_hidden_dim, edge_hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                EdgeGATBlock(
                    hidden_dim=hidden_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    allow_zero_in_degree=allow_zero_in_degree,
                )
                for _ in range(num_layers)
            ]
        )
        self.node_decoder = MLP(hidden_dim, hidden_dim, n_modes_out, dropout=dropout)
        self.freq_head = MLP(hidden_dim, hidden_dim, N_FREQ_TARGETS_PHASE2A, dropout=dropout)

    def forward(self, graph: dgl.DGLGraph, node_x: torch.Tensor, edge_x: torch.Tensor) -> ModalGNNOutput:
        node_h = self.node_encoder(node_x)
        edge_h = self.edge_encoder(edge_x)
        for layer in self.layers:
            node_h = layer(graph, node_h, edge_h)
        pred_mode_shape_abs = self.node_decoder(node_h)
        with graph.local_scope():
            graph.ndata["_pooled_h"] = node_h
            pooled_h = dgl.mean_nodes(graph, "_pooled_h")
        pred_freq = self.freq_head(pooled_h)
        return ModalGNNOutput(pred_mode_shape_abs=pred_mode_shape_abs, pred_freq=pred_freq)


def _shape_target_field_name(shape_target_mode: str) -> str:
    if shape_target_mode == "abs":
        return "y_mode_shape_abs"
    if shape_target_mode == "signed":
        return "y_mode_shape_signed"
    raise ValueError(f"Unsupported shape_target_mode={shape_target_mode!r}.")


def _edge_attr_field_name(edge_feature_mode: str) -> str:
    if edge_feature_mode == "cpsd_full":
        return "edge_attr_cpsd_full"
    if edge_feature_mode == "cpsd_resonance":
        return "edge_attr_cpsd_resonance"
    raise ValueError(f"Unsupported edge_feature_mode={edge_feature_mode!r}.")


def _resolve_edge_attr_array(data: Any, edge_feature_mode: str) -> np.ndarray:
    if edge_feature_mode == "geom_orient":
        return edge_orientation(data)
    if edge_feature_mode == "zero":
        return np.zeros((np.asarray(data["edge_index"]).shape[1], 2), dtype=np.float32)
    if edge_feature_mode == "cpsd_full_orient":
        return np.hstack([_resolve_edge_attr_array(data, "cpsd_full"), edge_orientation(data)]).astype(np.float32)
    field_name = _edge_attr_field_name(edge_feature_mode)
    if field_name in data:
        return np.asarray(data[field_name], dtype=np.float32)
    if edge_feature_mode == "cpsd_resonance" and "edge_attr_cpsd" in data:
        return np.asarray(data["edge_attr_cpsd"], dtype=np.float32)
    raise KeyError(f"Required edge field '{field_name}' not found for edge_feature_mode={edge_feature_mode!r}.")


def _resolve_shape_target_array(data: Any, shape_target_mode: str) -> np.ndarray:
    field_name = _shape_target_field_name(shape_target_mode)
    if field_name not in data:
        raise KeyError(f"Required target field '{field_name}' not found.")
    return np.asarray(data[field_name], dtype=np.float32)


def _validate_phase2a_npz(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    *,
    expected_n_modes: int,
    edge_feature_mode: str,
    shape_target_mode: str,
) -> tuple[int, int]:
    required = ["graph_node_ids", "edge_index", "x_psd", "y_freq_hz", "target_mode_names", "freq_bins_hz"]
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"{npz_path} missing required fields: {missing}")

    graph_node_ids = np.asarray(data["graph_node_ids"], dtype=np.int64).reshape(-1)
    edge_index = np.asarray(data["edge_index"], dtype=np.int64)
    x_psd = np.asarray(data["x_psd"], dtype=np.float32)
    edge_attr = _resolve_edge_attr_array(data, edge_feature_mode)
    y_mode_shape = _resolve_shape_target_array(data, shape_target_mode)
    y_freq_hz = np.asarray(data["y_freq_hz"], dtype=np.float32).reshape(-1)
    if edge_index.shape[0] != 2:
        raise ValueError(f"{npz_path}: edge_index must have shape (2, E_dir), got {edge_index.shape}.")
    if x_psd.shape[0] != graph_node_ids.size:
        raise ValueError(f"{npz_path}: x_psd row count {x_psd.shape[0]} does not match node count {graph_node_ids.size}.")
    if y_mode_shape.shape != (graph_node_ids.size, expected_n_modes):
        raise ValueError(f"{npz_path}: target shape mismatch {y_mode_shape.shape}.")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(f"{npz_path}: edge feature row count {edge_attr.shape[0]} does not match edge count {edge_index.shape[1]}.")
    if y_freq_hz.shape != (N_FREQ_TARGETS_PHASE2A,):
        raise ValueError(f"{npz_path}: y_freq_hz must have shape ({N_FREQ_TARGETS_PHASE2A},), got {y_freq_hz.shape}.")
    return int(x_psd.shape[1]), int(edge_attr.shape[1])


def build_phase2a_graph_from_payload(
    payload: dict[str, Any],
    *,
    edge_feature_mode: str,
    shape_target_mode: str,
    node_feature_mode: str | None = None,
) -> tuple[dgl.DGLGraph, torch.Tensor]:
    """node_feature_mode: "psd" or "amp"; None = the module's NODE_FEATURE_MODE (set_node_feature_mode)."""
    node_feature_mode = NODE_FEATURE_MODE if node_feature_mode is None else node_feature_mode
    if node_feature_mode == "amp":
        payload = dict(payload)
        payload["x_psd"] = amp_node_features(payload["x_psd"], payload["x_psd_row_scale"], payload["edge_index"])
    elif node_feature_mode != "psd":
        raise ValueError(f"Unsupported node_feature_mode={node_feature_mode!r}. Expected one of {NODE_FEATURE_MODES}.")
    graph_node_ids = np.asarray(payload["graph_node_ids"], dtype=np.int64).reshape(-1)
    edge_index = np.asarray(payload["edge_index"], dtype=np.int64)
    x_psd = np.asarray(payload["x_psd"], dtype=np.float32)
    edge_attr = _resolve_edge_attr_array(payload, edge_feature_mode)
    y_mode_shape = _resolve_shape_target_array(payload, shape_target_mode)
    y_freq_hz = np.asarray(payload["y_freq_hz"], dtype=np.float32).reshape(-1)

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=int(graph_node_ids.size))
    g.ndata["x"] = torch.tensor(x_psd, dtype=torch.float32)
    g.ndata["y"] = torch.tensor(y_mode_shape, dtype=torch.float32)
    g.ndata["graph_node_id"] = torch.tensor(graph_node_ids, dtype=torch.int64)
    g.edata["edge_x"] = torch.tensor(edge_attr, dtype=torch.float32)
    sample_id_raw = payload.get("sample_id", "inference_case")
    sample_id = str(np.asarray(sample_id_raw).reshape(())) if np.asarray(sample_id_raw).shape == () else str(sample_id_raw)
    g.graph_data = {
        "sample_identifier": str(payload.get("sample_identifier", "inference_case")),
        "sample_id": sample_id,
        "target_mode_names": tuple(str(x) for x in np.asarray(payload["target_mode_names"]).reshape(-1).tolist()),
        "edge_feature_mode": edge_feature_mode,
        "shape_target_mode": shape_target_mode,
    }
    return g, torch.tensor(y_freq_hz, dtype=torch.float32)


class Phase2AEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, sample_paths: list[Path], *, edge_feature_mode: str, shape_target_mode: str, expected_n_modes: int):
        self.sample_paths = list(sample_paths)
        self.edge_feature_mode = edge_feature_mode
        self.shape_target_mode = shape_target_mode
        self.expected_n_modes = expected_n_modes
        self.graphs: list[dgl.DGLGraph] = []
        self.labels: torch.Tensor = torch.empty((0, N_FREQ_TARGETS_PHASE2A), dtype=torch.float32)
        self.sample_records: list[dict[str, Any]] = []
        self.node_feat_dim: int | None = None
        self.edge_feat_dim: int | None = None
        self._load()

    def _load(self) -> None:
        labels: list[torch.Tensor] = []
        for npz_path in self.sample_paths:
            with np.load(npz_path, allow_pickle=True) as data:
                node_feat_dim, edge_feat_dim = _validate_phase2a_npz(
                    data,
                    npz_path,
                    expected_n_modes=self.expected_n_modes,
                    edge_feature_mode=self.edge_feature_mode,
                    shape_target_mode=self.shape_target_mode,
                )
                graph, y_freq_hz = build_phase2a_graph_from_payload(
                    {name: data[name] for name in data.files},
                    edge_feature_mode=self.edge_feature_mode,
                    shape_target_mode=self.shape_target_mode,
                )
                graph.graph_data["npz_path"] = str(npz_path)
                graph.graph_data["sample_identifier"] = npz_path.stem
                self.graphs.append(graph)
                labels.append(y_freq_hz)
                self.sample_records.append(graph.graph_data)
                if self.node_feat_dim is None:
                    self.node_feat_dim = node_feat_dim
                    self.edge_feat_dim = edge_feat_dim
                elif self.node_feat_dim != node_feat_dim or self.edge_feat_dim != edge_feat_dim:
                    raise ValueError(
                        f"Inconsistent feature dimensions detected: expected node/edge {self.node_feat_dim}/{self.edge_feat_dim}, "
                        f"got {node_feat_dim}/{edge_feat_dim} in {npz_path}."
                    )
        self.labels = torch.stack(labels, dim=0) if labels else torch.empty((0, N_FREQ_TARGETS_PHASE2A), dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[dgl.DGLGraph, torch.Tensor]:
        return self.graphs[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.graphs)


def _compute_input_stats(train_set: Phase2AEdgeDataset) -> dict[str, np.ndarray]:
    node_stack = np.concatenate([g.ndata["x"].numpy() for g in train_set.graphs], axis=0)
    edge_stack = np.concatenate([g.edata["edge_x"].numpy() for g in train_set.graphs], axis=0)
    return {
        "node_mean": node_stack.mean(axis=0, dtype=np.float64).astype(np.float32),
        "node_std": np.maximum(node_stack.std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1e-6)),
        "edge_mean": edge_stack.mean(axis=0, dtype=np.float64).astype(np.float32),
        "edge_std": np.maximum(edge_stack.std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1e-6)),
    }


def _apply_input_stats(dataset: Phase2AEdgeDataset, stats: dict[str, np.ndarray]) -> None:
    node_mean = torch.tensor(stats["node_mean"], dtype=torch.float32)
    node_std = torch.tensor(stats["node_std"], dtype=torch.float32)
    edge_mean = torch.tensor(stats["edge_mean"], dtype=torch.float32)
    edge_std = torch.tensor(stats["edge_std"], dtype=torch.float32)
    for graph in dataset.graphs:
        graph.ndata["x"] = (graph.ndata["x"] - node_mean) / node_std
        graph.edata["edge_x"] = (graph.edata["edge_x"] - edge_mean) / edge_std


def _save_input_normalization(run_context: RunContext, stats: dict[str, np.ndarray]) -> None:
    payload = {key: np.asarray(value, dtype=np.float32).tolist() for key, value in stats.items()}
    with (run_context.run_dir / "input_normalization.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _create_run_context(run_root: Path, run_name: str | None) -> RunContext:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_name = run_name or f"edgegat_{timestamp}"
    run_dir = Path(run_root) / actual_name
    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "evaluation"
    predictions_dir = eval_dir / "test_predictions"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(run_dir=run_dir, checkpoints_dir=checkpoints_dir, eval_dir=eval_dir, predictions_dir=predictions_dir)


def build_datasets(config: TrainingConfig) -> tuple[Phase2AEdgeDataset, Phase2AEdgeDataset, Phase2AEdgeDataset, SplitArtifacts]:
    sample_paths = sorted(Path(config.dataset_root).glob("*.npz"))
    if config.max_samples is not None:
        sample_paths = sample_paths[: config.max_samples]
    if not sample_paths:
        raise FileNotFoundError(f"No NPZ samples found in {config.dataset_root}")

    train_paths, val_paths, test_paths = _split_sequence(
        sample_paths,
        (config.train_frac, config.val_frac, config.test_frac),
        config.split_seed,
    )
    train_set = Phase2AEdgeDataset(train_paths, edge_feature_mode=config.edge_feature_mode, shape_target_mode=config.shape_target_mode, expected_n_modes=config.expected_n_modes_out)
    val_set = Phase2AEdgeDataset(val_paths, edge_feature_mode=config.edge_feature_mode, shape_target_mode=config.shape_target_mode, expected_n_modes=config.expected_n_modes_out)
    test_set = Phase2AEdgeDataset(test_paths, edge_feature_mode=config.edge_feature_mode, shape_target_mode=config.shape_target_mode, expected_n_modes=config.expected_n_modes_out)
    split_artifacts = SplitArtifacts(
        train_ids=[path.name for path in train_paths],
        val_ids=[path.name for path in val_paths],
        test_ids=[path.name for path in test_paths],
    )
    return train_set, val_set, test_set, split_artifacts


def _compute_sample_metrics(pred: np.ndarray, target: np.ndarray, mode_names: tuple[str, ...]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
    }
    for mode_idx, mode_name in enumerate(mode_names):
        pred_mode = pred[:, mode_idx]
        target_mode = target[:, mode_idx]
        metrics[f"mse_{mode_name}"] = float(np.mean((pred_mode - target_mode) ** 2))
        metrics[f"mae_{mode_name}"] = float(np.mean(np.abs(pred_mode - target_mode)))
        denom = float(np.dot(pred_mode, pred_mode) * np.dot(target_mode, target_mode))
        metrics[f"mac_{mode_name}"] = float((abs(np.dot(pred_mode, target_mode)) ** 2) / (denom + 1e-30))
    return metrics


def evaluate_test_set(
    model: EdgeGATModalGNN,
    test_set: Phase2AEdgeDataset,
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
            graph = graph.to(DEVICE)
            freq_target_hz = freq_target_hz.to(DEVICE)
            outputs = model(graph, graph.ndata["x"], graph.edata["edge_x"])
            pred = outputs.pred_mode_shape_abs.detach().cpu().numpy()
            target = graph.ndata["y"].detach().cpu().numpy()
            if config.disable_freq_multitask:
                pred_freq_z = np.zeros((1, N_FREQ_TARGETS_PHASE2A), dtype=np.float32)
                target_freq_z = np.zeros((1, N_FREQ_TARGETS_PHASE2A), dtype=np.float32)
                pred_freq_hz = np.zeros((1, N_FREQ_TARGETS_PHASE2A), dtype=np.float32)
                target_freq_hz_np = np.zeros((1, N_FREQ_TARGETS_PHASE2A), dtype=np.float32)
            else:
                if freq_stats is None:
                    raise ValueError("Frequency normalization stats are required for multitask evaluation.")
                pred_freq_hz_t = denormalize_frequency_predictions(outputs.pred_freq, freq_stats)
                target_freq_z_t = normalize_frequency_targets(freq_target_hz.unsqueeze(0), freq_stats)
                pred_freq_z = outputs.pred_freq.detach().cpu().numpy()
                target_freq_z = target_freq_z_t.detach().cpu().numpy()
                pred_freq_hz = pred_freq_hz_t.detach().cpu().numpy()
                target_freq_hz_np = freq_target_hz.unsqueeze(0).detach().cpu().numpy()

            graph_node_ids = graph.ndata["graph_node_id"].detach().cpu().numpy()
            graph_data = getattr(graph, "graph_data", {})
            sample_identifier = str(graph_data.get("sample_identifier", "sample"))
            sample_id = str(graph_data.get("sample_id", sample_identifier))
            source_path = str(graph_data.get("npz_path", ""))
            mode_names = tuple(graph_data.get("target_mode_names", MODE_NAMES_PHASE1))
            metrics = {
                **_compute_sample_metrics(pred, target, mode_names),
                **compute_frequency_metrics(pred_freq_hz, target_freq_hz_np, mode_names),
            }
            metric_rows.append({"sample_identifier": sample_identifier, "sample_id": sample_id, "npz_path": source_path, **metrics})
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
    _validate_split_fractions(config.train_frac, config.val_frac, config.test_frac)
    set_global_seed(config.seed)

    run_context = _create_run_context(config.run_root, config.run_name)
    train_set, val_set, test_set, split_artifacts = build_datasets(config)
    config.node_feat_dim = int(train_set.node_feat_dim or 0)
    config.edge_feat_dim = int(train_set.edge_feat_dim or 0)
    _save_run_config(run_context, config)
    _save_split_artifacts(run_context, split_artifacts)

    if config.normalize_inputs:
        stats = _compute_input_stats(train_set)
        _save_input_normalization(run_context, stats)
        _apply_input_stats(train_set, stats)
        _apply_input_stats(val_set, stats)
        _apply_input_stats(test_set, stats)

    freq_stats: FrequencyNormalizationStats | None = None
    if not config.disable_freq_multitask:
        freq_stats = compute_frequency_normalization_stats(train_set.labels, eps=config.freq_std_eps)
        _save_frequency_normalization(run_context, freq_stats, config)

    train_loader = GraphDataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = GraphDataLoader(val_set, batch_size=config.batch_size, shuffle=False, drop_last=False)

    model = EdgeGATModalGNN(
        node_feat_dim=config.node_feat_dim,
        edge_feat_dim=config.edge_feat_dim,
        hidden_dim=config.hidden_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        n_modes_out=config.expected_n_modes_out,
        allow_zero_in_degree=config.allow_zero_in_degree,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    history_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    debug_printed = False
    t0 = time.time()

    for epoch in range(config.epochs):
        model.train()
        train_loss_total = 0.0
        train_loss_shape = 0.0
        train_loss_freq = 0.0
        train_freq_mae_hz = 0.0
        for g_batch, freq_target_hz in train_loader:
            g_batch = g_batch.to(DEVICE)
            freq_target_hz = freq_target_hz.to(DEVICE)
            outputs = model(g_batch, g_batch.ndata["x"], g_batch.edata["edge_x"])
            if not debug_printed:
                print(
                    "Batch debug:",
                    {
                        "node_x": tuple(g_batch.ndata["x"].shape),
                        "edge_x": tuple(g_batch.edata["edge_x"].shape),
                        "pred_shape": tuple(outputs.pred_mode_shape_abs.shape),
                        "pred_freq": tuple(outputs.pred_freq.shape),
                    },
                )
                debug_printed = True
            loss_total, loss_shape, loss_freq, pred_freq_hz = _compute_multitask_loss(
                outputs,
                g_batch.ndata["y"],
                freq_target_hz,
                loss_fn=loss_fn,
                enable_freq_multitask=not config.disable_freq_multitask,
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
                g_batch = g_batch.to(DEVICE)
                freq_target_hz = freq_target_hz.to(DEVICE)
                outputs = model(g_batch, g_batch.ndata["x"], g_batch.edata["edge_x"])
                loss_total, loss_shape, loss_freq, pred_freq_hz = _compute_multitask_loss(
                    outputs,
                    g_batch.ndata["y"],
                    freq_target_hz,
                    loss_fn=loss_fn,
                    enable_freq_multitask=not config.disable_freq_multitask,
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
        history_rows.append(
            {
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
                "n_epochs": config.epochs,
            }
        )
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

    _save_checkpoint(
        run_context.checkpoints_dir / "last_model.pt",
        model,
        config,
        config.epochs - 1,
        float(history_rows[-1]["train_loss_total"]),
        float(history_rows[-1]["val_loss_total"]),
        freq_stats=freq_stats,
    )
    train_time_s = time.time() - t0
    _save_training_history_csv(run_context, history_rows)
    test_summary = evaluate_test_set(model, test_set, run_context, config=config, freq_stats=freq_stats)
    run_summary = {
        "architecture": "edgegatconv",
        "dataset_root": str(config.dataset_root),
        "run_dir": str(run_context.run_dir),
        "device": str(DEVICE),
        "seed": config.seed,
        "split_seed": config.split_seed,
        "n_train_samples": len(train_set),
        "n_val_samples": len(val_set),
        "n_test_samples": len(test_set),
        "node_feat_dim": config.node_feat_dim,
        "edge_feat_dim": config.edge_feat_dim,
        "edge_feature_mode": config.edge_feature_mode,
        "shape_target_mode": config.shape_target_mode,
        "enable_freq_multitask": not config.disable_freq_multitask,
        "freq_loss_weight": float(config.freq_loss_weight),
        "normalize_inputs": bool(config.normalize_inputs),
        "allow_zero_in_degree": bool(config.allow_zero_in_degree),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss_total": history_rows[-1]["train_loss_total"],
        "final_train_loss_shape": history_rows[-1]["train_loss_shape"],
        "final_train_loss_freq": history_rows[-1]["train_loss_freq"],
        "final_val_loss_total": history_rows[-1]["val_loss_total"],
        "final_val_loss_shape": history_rows[-1]["val_loss_shape"],
        "final_val_loss_freq": history_rows[-1]["val_loss_freq"],
        "train_time_s": train_time_s,
        "freq_mean_hz": None if freq_stats is None else freq_stats.freq_mean_hz.tolist(),
        "freq_std_hz": None if freq_stats is None else freq_stats.freq_std_hz.tolist(),
        "test_summary": test_summary,
    }
    with (run_context.run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_normalize_json_value(run_summary), f, indent=2, sort_keys=True)
    return {"run_context": run_context, "run_summary": run_summary}


def build_model_from_config(config_like: dict[str, Any] | TrainingConfig) -> EdgeGATModalGNN:
    cfg = dict(asdict(config_like)) if isinstance(config_like, TrainingConfig) else dict(config_like)
    return EdgeGATModalGNN(
        node_feat_dim=int(cfg["node_feat_dim"]),
        edge_feat_dim=int(cfg["edge_feat_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        edge_hidden_dim=int(cfg.get("edge_hidden_dim", cfg["hidden_dim"])),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg.get("dropout", 0.1)),
        attn_dropout=float(cfg.get("attn_dropout", 0.1)),
        n_modes_out=int(cfg.get("expected_n_modes_out", 4)),
        allow_zero_in_degree=bool(cfg.get("allow_zero_in_degree", True)),
    )


def load_trained_model_checkpoint(checkpoint_path: Path, device: torch.device | None = None) -> tuple[EdgeGATModalGNN, dict[str, Any]]:
    ckpt_device = device if device is not None else DEVICE
    checkpoint = torch.load(Path(checkpoint_path), map_location=ckpt_device, weights_only=False)
    model = build_model_from_config(checkpoint["config"]).to(ckpt_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the parallel Phase-2A EdgeGATConv signed CPSD branch.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--edge-hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--shape-target-mode", choices=SHAPE_TARGET_MODES, default="signed")
    parser.add_argument("--edge-feature-mode", choices=EDGE_FEATURE_MODES, default="cpsd_full")
    parser.add_argument("--freq-loss-weight", type=float, default=0.25)
    parser.add_argument("--disable-freq-multitask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if DGL_IMPORT_ERROR is not None:
        raise RuntimeError(f"DGL failed to import: {DGL_IMPORT_ERROR}")
    config = TrainingConfig(
        dataset_root=args.dataset_root,
        run_root=args.run_root,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        split_seed=args.split_seed,
        max_samples=args.max_samples,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        normalize_inputs=args.normalize_inputs,
        shape_target_mode=args.shape_target_mode,
        edge_feature_mode=args.edge_feature_mode,
        freq_loss_weight=args.freq_loss_weight,
        disable_freq_multitask=args.disable_freq_multitask,
        enable_freq_multitask=not args.disable_freq_multitask,
    )
    result = train_one_run(config)
    print(f"Run directory: {result['run_context'].run_dir}")
    print(f"Best validation loss: {result['run_summary']['best_val_loss']:.6e}")
    print(f"Test summary: {result['run_summary']['test_summary']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from EdgeGATConv_Modal_GNN import (
    DEVICE,
    build_phase2a_graph_from_payload,
    load_trained_model_checkpoint,
)
from GraphSAGE_Modal_GNN import compute_mode_shape_metrics, denormalize_frequency_predictions, set_global_seed
from infer_phase1_unseen_structure import normalize_json_value, write_json
from infer_phase2a_unseen_structure import (
    assemble_unseen_phase2a_payload,
    load_frequency_normalization,
    plot_mode_shape_comparison,
    plot_mode_shape_differences,
    save_frequency_outputs,
    save_mode_shape_metrics,
    save_raw_outputs,
)


def load_input_normalization(run_dir: Path) -> dict[str, np.ndarray] | None:
    path = run_dir / "input_normalization.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}


def apply_input_normalization(graph, stats: dict[str, np.ndarray] | None) -> None:
    if stats is None:
        return
    graph.ndata["x"] = (graph.ndata["x"] - torch.tensor(stats["node_mean"], dtype=torch.float32)) / torch.tensor(
        stats["node_std"], dtype=torch.float32
    )
    graph.edata["edge_x"] = (graph.edata["edge_x"] - torch.tensor(stats["edge_mean"], dtype=torch.float32)) / torch.tensor(
        stats["edge_std"], dtype=torch.float32
    )


def run_inference_case(
    *,
    run_dir: Path,
    output_root: Path,
    case_name: str,
    seed: int,
    ng: int | None,
    checkpoint_path: Path | None,
    shape_target_mode: str,
    edge_feature_mode: str,
) -> Path:
    output_dir = output_root / case_name
    output_dir.mkdir(parents=True, exist_ok=False)

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else run_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, checkpoint = load_trained_model_checkpoint(ckpt_path, device=DEVICE)
    checkpoint_config = dict(checkpoint["config"])
    if checkpoint_config.get("shape_target_mode", "signed") != shape_target_mode:
        raise ValueError("Explicit shape_target_mode does not match checkpoint config.")
    if checkpoint_config.get("edge_feature_mode", "cpsd_full") != edge_feature_mode:
        raise ValueError("Explicit edge_feature_mode does not match checkpoint config.")

    payload, metadata = assemble_unseen_phase2a_payload(seed=seed, ng=ng)
    graph, freq_target_hz = build_phase2a_graph_from_payload(
        payload,
        edge_feature_mode=edge_feature_mode,
        shape_target_mode=shape_target_mode,
    )
    apply_input_normalization(graph, load_input_normalization(run_dir))
    graph = graph.to(DEVICE)

    freq_stats = load_frequency_normalization(checkpoint_path=ckpt_path, checkpoint=checkpoint, run_dir=run_dir)
    with torch.no_grad():
        outputs = model(graph, graph.ndata["x"], graph.edata["edge_x"])
        pred_mode_shape = outputs.pred_mode_shape_abs.detach().cpu().numpy()
        pred_freq_z = outputs.pred_freq.detach().cpu().numpy()
        pred_freq_hz = denormalize_frequency_predictions(outputs.pred_freq, freq_stats).detach().cpu().numpy()

    target_mode_shape = graph.ndata["y"].detach().cpu().numpy()
    target_freq_hz = freq_target_hz.unsqueeze(0).detach().cpu().numpy()
    coords_xy = np.asarray(payload["coords_xy_mm"], dtype=np.float32)
    mode_names = tuple(str(x) for x in np.asarray(payload["target_mode_names"]).reshape(-1).tolist())
    mode_shape_metrics = compute_mode_shape_metrics(pred_mode_shape, target_mode_shape, mode_names)
    save_mode_shape_metrics(output_dir, mode_shape_metrics, mode_names)
    frequency_summary = save_frequency_outputs(
        output_dir,
        pred_freq_z=pred_freq_z,
        pred_freq_hz=pred_freq_hz,
        mode_names=mode_names,
        target_freq_hz=target_freq_hz,
    )
    save_raw_outputs(
        output_dir,
        payload=payload,
        pred_mode_shape_abs=pred_mode_shape,
        target_mode_shape_abs=target_mode_shape,
        pred_freq_z=pred_freq_z,
        pred_freq_hz=pred_freq_hz,
        target_freq_hz=target_freq_hz,
        mode_shape_metrics=mode_shape_metrics,
        frequency_summary=frequency_summary,
    )
    plot_mode_shape_comparison(output_dir, coords_xy, pred_mode_shape, target_mode_shape, mode_names)
    plot_mode_shape_differences(output_dir, coords_xy, pred_mode_shape, target_mode_shape, mode_names)

    summary = {
        "case_name": case_name,
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "edge_feature_mode": edge_feature_mode,
        "shape_target_mode": shape_target_mode,
        "seed": seed,
        "ng": metadata["sampled_params"]["Ng"],
        "mode_names": list(mode_names),
        "sample_identifier": str(payload.get("sample_identifier", "unseen_inference_case")),
        "n_nodes": int(pred_mode_shape.shape[0]),
        "n_modes": int(pred_mode_shape.shape[1]),
        "pred_freq_z": np.asarray(pred_freq_z, dtype=np.float32).reshape(-1).tolist(),
        "pred_freq_hz": np.asarray(pred_freq_hz, dtype=np.float32).reshape(-1).tolist(),
        "target_freq_hz": np.asarray(target_freq_hz, dtype=np.float32).reshape(-1).tolist(),
        "mode_shape_metrics": mode_shape_metrics,
        "frequency_summary": frequency_summary,
        "metadata": normalize_json_value(metadata),
    }
    write_json(output_dir / "summary.json", summary)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unseen-structure inference for the EdgeGAT Phase-2A branch.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--case-name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--ng", type=int, default=None)
    parser.add_argument("--shape-target-mode", choices=["signed", "abs"], default="signed")
    parser.add_argument("--edge-feature-mode", choices=["cpsd_full", "cpsd_resonance"], default="cpsd_full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    run_dir = Path(args.run_dir)
    output_root = args.output_dir or Path("experiments") / "E1_architecture_ablation" / "inference" / run_dir.name
    output_dir = run_inference_case(
        run_dir=run_dir,
        output_root=output_root,
        case_name=args.case_name,
        seed=args.seed,
        ng=args.ng,
        checkpoint_path=args.checkpoint_path,
        shape_target_mode=args.shape_target_mode,
        edge_feature_mode=args.edge_feature_mode,
    )
    print(f"Inference output directory: {output_dir}")


if __name__ == "__main__":
    main()

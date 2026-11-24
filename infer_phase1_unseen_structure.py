from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from GraphSAGE_Modal_GNN import (
    DEVICE,
    DGL_IMPORT_ERROR,
    build_phase1_graph_from_payload,
    compute_mode_shape_metrics,
    load_trained_model_checkpoint,
    set_global_seed,
)
from generate_population_dataset_modes4 import (
    MODE_MATCH_CONFIG,
    MODE_NAMES,
    NG_CHOICES,
    N_CANDIDATE_MODES,
    REF_DIR,
    REF_FILE_PATTERN,
    REF_SENSOR_TOL_X,
    REF_X_FRACS,
    _candidate_channels_from_solve,
    _candidate_metadata_from_solve,
    assemble_phase1_sample_payload,
    build_solver_params,
    compute_S,
    load_reference_library_for_ng,
    run_solver,
    sample_params_for_ng,
    to_json,
    validate_sampled_params,
)
from standalone_mode_matching_mmi import match_target_modes


DEFAULT_INFERENCE_DIR = Path("inference_phase1")


def normalize_json_value(value: Any) -> Any:
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
    if isinstance(value, list):
        return [normalize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_json_value(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(normalize_json_value(payload), f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_checkpoint_path(run_dir: Path | None, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path
    if run_dir is None:
        raise ValueError("Provide either --checkpoint-path or --run-dir.")
    return run_dir / "checkpoints" / "best_model.pt"


def choose_ng(rng: np.random.Generator, ng: int | None) -> int:
    if ng is not None:
        return int(ng)
    return int(rng.choice(np.asarray(NG_CHOICES, dtype=int)))


def sample_valid_structure(rng: np.random.Generator, ng: int, max_attempts: int = 200) -> dict[str, Any]:
    for _ in range(max_attempts):
        sampled = sample_params_for_ng(ng, rng)
        if validate_sampled_params(sampled):
            return sampled
    raise RuntimeError(f"Failed to generate a valid sampled structure for Ng={ng} after {max_attempts} attempts.")


def assemble_unseen_phase1_payload(seed: int, ng: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    chosen_ng = choose_ng(rng, ng)
    sampled = sample_valid_structure(rng, chosen_ng)

    solver_params = build_solver_params(sampled, n_candidate_modes=N_CANDIDATE_MODES)
    solve_res = run_solver(solver_params, ref_x_fracs=REF_X_FRACS, tol_x=REF_SENSOR_TOL_X)
    ref_lib = load_reference_library_for_ng(REF_DIR, int(sampled["Ng"]))

    match_res = match_target_modes(
        phi_u3_candidates=solve_res.phi_u3_rep,
        freq_candidates_hz=solve_res.freq_all_hz,
        ref_lib=ref_lib,
        config=MODE_MATCH_CONFIG,
        candidate_channels=_candidate_channels_from_solve(solve_res, MODE_MATCH_CONFIG.signature_config),
        candidate_participation=solve_res.participation,
        candidate_metadata=_candidate_metadata_from_solve(solve_res),
    )

    selected_idx = np.asarray(match_res.selected_idx, dtype=int).reshape(-1)
    if selected_idx.shape != (4,):
        raise ValueError(f"Expected 4 matched modes for inference, got {selected_idx.shape}.")

    freqs_4 = np.asarray(solve_res.freq_all_hz[selected_idx], dtype=float)
    phi_u3_4 = np.asarray(solve_res.phi_u3_rep[:, selected_idx], dtype=float)
    S4 = np.asarray(compute_S(phi_u3_4), dtype=float)

    ref_path = REF_DIR / REF_FILE_PATTERN.format(Ng=int(sampled["Ng"]))
    payload = assemble_phase1_sample_payload(
        sample_id=-1,
        sampled=sampled,
        solve_res=solve_res,
        match_res=match_res,
        matcher_config_json=json.dumps(asdict(MODE_MATCH_CONFIG), sort_keys=True),
        reference_file=str(ref_path),
        freqs_4=freqs_4,
        phi_u3_4=phi_u3_4,
        S4=S4,
    )
    payload["sample_identifier"] = np.asarray("unseen_inference_case")

    metadata = {
        "sampled_params": sampled,
        "solver_params": solver_params,
        "matched_mode_indices": selected_idx,
        "matched_freq_hz": freqs_4,
        "mode_names": list(MODE_NAMES),
        "match_result_json": json.loads(match_res.to_json()),
        "reference_file": str(ref_path),
    }
    return payload, metadata


def save_metrics(output_dir: Path, metrics: dict[str, Any], mode_names: tuple[str, ...]) -> None:
    write_json(output_dir / "metrics.json", metrics)
    rows = []
    for mode_name in mode_names:
        rows.append(
            {
                "mode_name": mode_name,
                "mode_shape_mse": metrics[f"mse_{mode_name}"],
                "mode_shape_mae": metrics[f"mae_{mode_name}"],
                "mode_shape_mac": metrics[f"mac_{mode_name}"],
            }
        )
    write_csv(output_dir / "metrics_per_mode.csv", rows, ["mode_name", "mode_shape_mse", "mode_shape_mae", "mode_shape_mac"])


def save_raw_outputs(output_dir: Path, payload: dict[str, Any], prediction: np.ndarray, target: np.ndarray, metrics: dict[str, Any]) -> None:
    np.savez(
        output_dir / "raw_outputs.npz",
        prediction=np.asarray(prediction, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        x_psd=np.asarray(payload["x_psd"], dtype=np.float32),
        graph_node_ids=np.asarray(payload["graph_node_ids"], dtype=np.int64),
        coords_xy_mm=np.asarray(payload["coords_xy_mm"], dtype=np.float32),
        coords_xyz0_mm=np.asarray(payload["coords_xyz0_mm"], dtype=np.float32),
        matched_mode_indices=np.asarray(payload["matched_mode_indices"], dtype=np.int64),
        matched_freq_hz=np.asarray(payload["matched_freq_hz"], dtype=np.float32),
        target_mode_names=np.asarray(payload["target_mode_names"]),
        metrics_json=np.asarray(json.dumps(normalize_json_value(metrics), sort_keys=True)),
    )
    np.savez(output_dir / "assembled_phase1_sample.npz", **payload)


def plot_mode_shape_comparison(output_dir: Path, coords_xy: np.ndarray, prediction: np.ndarray, target: np.ndarray, mode_names: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(nrows=len(mode_names), ncols=2, figsize=(9.5, 3.2 * len(mode_names)), squeeze=False)
    for mode_idx, mode_name in enumerate(mode_names):
        pred_col = prediction[:, mode_idx]
        target_col = target[:, mode_idx]
        vmin = float(min(np.min(pred_col), np.min(target_col)))
        vmax = float(max(np.max(pred_col), np.max(target_col)))

        for col_idx, (title, values) in enumerate((("FE Target", target_col), ("GNN Prediction", pred_col))):
            ax = axes[mode_idx, col_idx]
            sc = ax.scatter(coords_xy[:, 0], coords_xy[:, 1], c=values, cmap="viridis", s=28, vmin=vmin, vmax=vmax)
            ax.set_title(f"{mode_name} {title}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Unseen Structure: FE vs GNN Absolute Mode Shapes", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "mode_shapes_compare_all.png", dpi=180)
    plt.close(fig)


def plot_mode_shape_differences(output_dir: Path, coords_xy: np.ndarray, prediction: np.ndarray, target: np.ndarray, mode_names: tuple[str, ...]) -> None:
    diff = prediction - target

    fig, axes = plt.subplots(nrows=1, ncols=len(mode_names), figsize=(4.2 * len(mode_names), 3.6), squeeze=False)
    for mode_idx, mode_name in enumerate(mode_names):
        ax = axes[0, mode_idx]
        diff_col = diff[:, mode_idx]
        diff_lim = float(max(np.max(np.abs(diff_col)), 1e-8))
        sc = ax.scatter(coords_xy[:, 0], coords_xy[:, 1], c=diff_col, cmap="coolwarm", s=30, vmin=-diff_lim, vmax=diff_lim)
        ax.set_title(f"{mode_name} Pred - FE")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Unseen Structure: Mode Shape Differences", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "mode_shapes_difference_all.png", dpi=180)
    plt.close(fig)


def run_inference_case(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int,
    ng: int | None,
) -> dict[str, Any]:
    if DGL_IMPORT_ERROR is not None:
        raise RuntimeError(f"DGL failed to import in the active environment: {DGL_IMPORT_ERROR}")

    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=False)

    model, checkpoint = load_trained_model_checkpoint(checkpoint_path, device=DEVICE)
    checkpoint_config = checkpoint["config"]
    expected_n_freq = int(checkpoint_config["expected_n_freq"])
    expected_n_modes = int(checkpoint_config["expected_n_modes_out"])

    payload, generation_meta = assemble_unseen_phase1_payload(seed=seed, ng=ng)
    graph = build_phase1_graph_from_payload(payload, expected_n_freq=expected_n_freq, expected_n_modes=expected_n_modes).to(DEVICE)

    with torch.no_grad():
        prediction_t = model(graph, graph.ndata["x"], graph.edata["edge_x"])
    prediction = prediction_t.detach().cpu().numpy()
    target = graph.ndata["y"].detach().cpu().numpy()
    mode_names = tuple(str(x) for x in np.asarray(payload["target_mode_names"]).reshape(-1).tolist())
    coords_xy = np.asarray(payload["coords_xy_mm"], dtype=float)

    metrics = compute_mode_shape_metrics(prediction, target, mode_names)
    save_metrics(output_dir, metrics, mode_names)
    save_raw_outputs(output_dir, payload, prediction, target, metrics)
    plot_mode_shape_comparison(output_dir, coords_xy, prediction, target, mode_names)
    plot_mode_shape_differences(output_dir, coords_xy, prediction, target, mode_names)

    write_json(
        output_dir / "inference_config.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "device": str(DEVICE),
            "seed": int(seed),
            "requested_ng": None if ng is None else int(ng),
            "checkpoint_config": checkpoint_config,
        },
    )
    write_json(output_dir / "sampled_params.json", generation_meta["sampled_params"])
    write_json(output_dir / "generation_metadata.json", generation_meta)

    summary = {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "sample_identifier": "unseen_inference_case",
        "n_nodes": int(prediction.shape[0]),
        "n_modes": int(prediction.shape[1]),
        "mode_names": list(mode_names),
        "metrics": metrics,
        "matched_freq_hz": generation_meta["matched_freq_hz"],
        "matched_mode_indices": generation_meta["matched_mode_indices"],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained Phase-1 GraphSAGE inference on one newly generated unseen structure.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Direct path to a trained checkpoint.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory to infer checkpoint path from. Uses checkpoints/best_model.pt.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INFERENCE_DIR, help="Root output directory for inference cases.")
    parser.add_argument("--case-name", type=str, default=None, help="Optional inference case name.")
    parser.add_argument("--seed", type=int, default=20260313, help="Seed used for new unseen structure generation.")
    parser.add_argument("--ng", type=int, default=None, help="Optional fixed Ng choice for the unseen structure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.run_dir, args.checkpoint_path)
    case_name = args.case_name or f"unseen_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir / case_name

    summary = run_inference_case(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        seed=int(args.seed),
        ng=args.ng,
    )
    print(f"Inference output directory: {output_dir}")
    print(f"Checkpoint used: {checkpoint_path}")
    print(f"Overall MSE: {summary['metrics']['mse']:.6e}")
    print(f"Overall MAE: {summary['metrics']['mae']:.6e}")


if __name__ == "__main__":
    main()

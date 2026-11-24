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
    EDGE_FEATURE_MODES,
    SHAPE_TARGET_MODES,
    FrequencyNormalizationStats,
    build_phase1_graph_from_payload,
    compute_mode_shape_metrics,
    denormalize_frequency_predictions,
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
    validate_sampled_params,
)
from infer_phase1_unseen_structure import (
    choose_ng,
    normalize_json_value,
    resolve_checkpoint_path,
    sample_valid_structure,
    write_csv,
    write_json,
)
from standalone_mode_matching_mmi import match_target_modes


DEFAULT_INFERENCE_DIR = Path("inference_phase2a")


def load_frequency_normalization(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    run_dir: Path | None,
) -> FrequencyNormalizationStats:
    """Load saved frequency normalization metadata for Phase 2A inference.

    Preference order:
    1. explicit `run_dir/frequency_normalization.json`
    2. sibling run directory inferred from checkpoint path
    3. `freq_mean_hz` / `freq_std_hz` embedded in the checkpoint
    """
    candidate_run_dirs: list[Path] = []
    if run_dir is not None:
        candidate_run_dirs.append(Path(run_dir))
    candidate_run_dirs.append(Path(checkpoint_path).resolve().parent.parent)

    for candidate in candidate_run_dirs:
        norm_path = candidate / "frequency_normalization.json"
        if norm_path.exists():
            with norm_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return FrequencyNormalizationStats(
                freq_mean_hz=np.asarray(payload["freq_mean_hz"], dtype=np.float32),
                freq_std_hz=np.asarray(payload["freq_std_hz"], dtype=np.float32),
            )

    if "freq_mean_hz" in checkpoint and "freq_std_hz" in checkpoint:
        return FrequencyNormalizationStats(
            freq_mean_hz=np.asarray(checkpoint["freq_mean_hz"], dtype=np.float32),
            freq_std_hz=np.asarray(checkpoint["freq_std_hz"], dtype=np.float32),
        )

    raise FileNotFoundError(
        "Could not load frequency normalization metadata from frequency_normalization.json or checkpoint."
    )


def assemble_unseen_phase2a_payload(seed: int, ng: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "target_freq_hz": freqs_4,
        "mode_names": list(MODE_NAMES),
        "match_result_json": json.loads(match_res.to_json()),
        "reference_file": str(ref_path),
    }
    return payload, metadata


def save_mode_shape_metrics(output_dir: Path, metrics: dict[str, Any], mode_names: tuple[str, ...]) -> None:
    write_json(output_dir / "mode_shape_metrics.json", metrics)
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
    write_csv(output_dir / "mode_shape_metrics_per_mode.csv", rows, ["mode_name", "mode_shape_mse", "mode_shape_mae", "mode_shape_mac"])


def compute_frequency_comparison(
    pred_freq_hz: np.ndarray,
    target_freq_hz: np.ndarray | None,
    mode_names: tuple[str, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pred_freq_hz": np.asarray(pred_freq_hz, dtype=np.float32).tolist(),
        "mode_names": list(mode_names),
    }
    if target_freq_hz is None:
        summary["target_available"] = False
        return summary

    pred = np.asarray(pred_freq_hz, dtype=float).reshape(-1)
    target = np.asarray(target_freq_hz, dtype=float).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f"pred_freq_hz shape {pred.shape} does not match target_freq_hz shape {target.shape}.")

    abs_error = np.abs(pred - target)
    rel_error = abs_error / np.maximum(np.abs(target), 1e-12) * 100.0
    summary.update(
        {
            "target_available": True,
            "target_freq_hz": target.astype(np.float32).tolist(),
            "abs_error_hz": abs_error.astype(np.float32).tolist(),
            "relative_error_percent": rel_error.astype(np.float32).tolist(),
            "overall_mae_hz": float(np.mean(abs_error)),
        }
    )
    for mode_idx, mode_name in enumerate(mode_names):
        summary[f"pred_freq_{mode_name}_hz"] = float(pred[mode_idx])
        summary[f"target_freq_{mode_name}_hz"] = float(target[mode_idx])
        summary[f"abs_error_{mode_name}_hz"] = float(abs_error[mode_idx])
        summary[f"relative_error_{mode_name}_percent"] = float(rel_error[mode_idx])
    return summary


def save_frequency_outputs(
    output_dir: Path,
    *,
    pred_freq_z: np.ndarray,
    pred_freq_hz: np.ndarray,
    mode_names: tuple[str, ...],
    target_freq_hz: np.ndarray | None,
) -> dict[str, Any]:
    comparison = compute_frequency_comparison(pred_freq_hz, target_freq_hz, mode_names)
    write_json(output_dir / "frequency_summary.json", comparison)

    rows = []
    for mode_idx, mode_name in enumerate(mode_names):
        row = {
            "mode_name": mode_name,
            "pred_freq_z": float(np.asarray(pred_freq_z, dtype=float).reshape(-1)[mode_idx]),
            "pred_freq_hz": float(np.asarray(pred_freq_hz, dtype=float).reshape(-1)[mode_idx]),
        }
        if target_freq_hz is not None:
            row["target_freq_hz"] = float(np.asarray(target_freq_hz, dtype=float).reshape(-1)[mode_idx])
            row["abs_error_hz"] = float(comparison[f"abs_error_{mode_name}_hz"])
            row["relative_error_percent"] = float(comparison[f"relative_error_{mode_name}_percent"])
        rows.append(row)

    fieldnames = ["mode_name", "pred_freq_z", "pred_freq_hz"]
    if target_freq_hz is not None:
        fieldnames.extend(["target_freq_hz", "abs_error_hz", "relative_error_percent"])
    write_csv(output_dir / "frequency_per_mode.csv", rows, fieldnames)
    return comparison


def save_raw_outputs(
    output_dir: Path,
    *,
    payload: dict[str, Any],
    pred_mode_shape_abs: np.ndarray,
    target_mode_shape_abs: np.ndarray,
    pred_freq_z: np.ndarray,
    pred_freq_hz: np.ndarray,
    target_freq_hz: np.ndarray | None,
    mode_shape_metrics: dict[str, Any],
    frequency_summary: dict[str, Any],
) -> None:
    np.savez(
        output_dir / "raw_outputs_phase2a.npz",
        pred_mode_shape_abs=np.asarray(pred_mode_shape_abs, dtype=np.float32),
        target_mode_shape_abs=np.asarray(target_mode_shape_abs, dtype=np.float32),
        pred_freq_z=np.asarray(pred_freq_z, dtype=np.float32),
        pred_freq_hz=np.asarray(pred_freq_hz, dtype=np.float32),
        target_freq_hz=np.asarray(
            np.asarray(target_freq_hz, dtype=np.float32) if target_freq_hz is not None else np.array([], dtype=np.float32),
            dtype=np.float32,
        ),
        x_psd=np.asarray(payload["x_psd"], dtype=np.float32),
        graph_node_ids=np.asarray(payload["graph_node_ids"], dtype=np.int64),
        coords_xy_mm=np.asarray(payload["coords_xy_mm"], dtype=np.float32),
        coords_xyz0_mm=np.asarray(payload["coords_xyz0_mm"], dtype=np.float32),
        matched_mode_indices=np.asarray(payload["matched_mode_indices"], dtype=np.int64),
        matched_freq_hz=np.asarray(payload["matched_freq_hz"], dtype=np.float32),
        target_mode_names=np.asarray(payload["target_mode_names"]),
        mode_shape_metrics_json=np.asarray(json.dumps(normalize_json_value(mode_shape_metrics), sort_keys=True)),
        frequency_summary_json=np.asarray(json.dumps(normalize_json_value(frequency_summary), sort_keys=True)),
    )
    np.savez(output_dir / "assembled_phase2a_sample.npz", **payload)


def _make_mode_shape_cmap():
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        "mode_shape",
        [(0.0, "#3B4CC0"), (0.5, "#FFFFFF"), (1.0, "#B40426")],
    )

_MODE_SHAPE_CMAP = _make_mode_shape_cmap()


def plot_mode_shape_comparison(output_dir: Path, coords_xy: np.ndarray, prediction: np.ndarray, target: np.ndarray, mode_names: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(nrows=len(mode_names), ncols=2, figsize=(9.5, 3.2 * len(mode_names)), squeeze=False)
    for mode_idx, mode_name in enumerate(mode_names):
        pred_col = prediction[:, mode_idx]
        target_col = target[:, mode_idx]

        pred_scale = float(np.max(np.abs(pred_col))) or 1.0
        pred_norm = pred_col / pred_scale
        target_scale = float(np.max(np.abs(target_col))) or 1.0
        target_norm = target_col / target_scale

        for col_idx, (title, values) in enumerate((("FE Target", target_norm), ("GNN Prediction", pred_norm))):
            ax = axes[mode_idx, col_idx]
            sc = ax.scatter(coords_xy[:, 0], coords_xy[:, 1], c=values, cmap=_MODE_SHAPE_CMAP,
                            s=28, vmin=-1.0, vmax=1.0,
                            edgecolors="#333333", linewidths=0.4)
            ax.set_title(f"{mode_name} {title}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Unseen Structure: FE vs GNN Mode Shapes (norm.)", fontsize=13)
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
        sc = ax.scatter(coords_xy[:, 0], coords_xy[:, 1], c=diff_col, cmap=_MODE_SHAPE_CMAP,
                        s=30, vmin=-diff_lim, vmax=diff_lim,
                        edgecolors="#333333", linewidths=0.4)
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


def plot_frequency_comparison(
    output_dir: Path,
    *,
    pred_freq_hz: np.ndarray,
    mode_names: tuple[str, ...],
    target_freq_hz: np.ndarray | None,
) -> None:
    x = np.arange(len(mode_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, pred_freq_hz, width=width, label="Predicted")
    if target_freq_hz is not None:
        ax.bar(x + width / 2, target_freq_hz, width=width, label="Target")
    ax.set_xticks(x, mode_names)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Predicted Graph-Level Natural Frequencies")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "frequency_comparison.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    *,
    mode_names: tuple[str, ...],
    mode_shape_metrics: dict[str, Any],
    frequency_summary: dict[str, Any],
) -> None:
    lines = [
        "# Phase 2A Unseen-Structure Inference",
        "",
        "## Mode-Shape Prediction",
        "",
        f"- Overall MAE: `{mode_shape_metrics['mae']:.6f}`",
        f"- Overall MSE: `{mode_shape_metrics['mse']:.6f}`",
    ]
    for mode_name in mode_names:
        lines.append(
            f"- {mode_name}: MAE `{mode_shape_metrics[f'mae_{mode_name}']:.6f}`, "
            f"MAC `{mode_shape_metrics[f'mac_{mode_name}']:.6f}`"
        )
    lines.extend(
        [
            "",
            "## Frequency Prediction",
            "",
        ]
    )
    if not bool(frequency_summary.get("target_available", False)):
        lines.append("- Target frequencies unavailable; only predicted frequencies were saved.")
    else:
        lines.append(f"- Overall MAE [Hz]: `{frequency_summary['overall_mae_hz']:.6f}`")
    for mode_name in mode_names:
        pred_val = frequency_summary.get(f"pred_freq_{mode_name}_hz")
        if pred_val is None:
            continue
        line = f"- {mode_name}: predicted `{pred_val:.6f}` Hz"
        if bool(frequency_summary.get("target_available", False)):
            line += (
                f", target `{frequency_summary[f'target_freq_{mode_name}_hz']:.6f}` Hz, "
                f"abs error `{frequency_summary[f'abs_error_{mode_name}_hz']:.6f}` Hz, "
                f"relative error `{frequency_summary[f'relative_error_{mode_name}_percent']:.6f}` %"
            )
        lines.append(line)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_inference_from_payload(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    payload: dict[str, Any],
    generation_meta: dict[str, Any],
    run_dir: Path | None = None,
    edge_feature_mode: str = "geom",
    shape_target_mode: str = "abs",
) -> dict[str, Any]:
    if DGL_IMPORT_ERROR is not None:
        raise RuntimeError(f"DGL failed to import in the active environment: {DGL_IMPORT_ERROR}")

    output_dir.mkdir(parents=True, exist_ok=False)

    model, checkpoint = load_trained_model_checkpoint(checkpoint_path, device=DEVICE)
    checkpoint_config = checkpoint["config"]
    expected_n_freq = int(checkpoint_config["expected_n_freq"])
    expected_n_modes = int(checkpoint_config["expected_n_modes_out"])
    freq_stats = load_frequency_normalization(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        run_dir=run_dir,
    )

    graph = build_phase1_graph_from_payload(
        payload,
        expected_n_freq=expected_n_freq,
        expected_n_modes=expected_n_modes,
        edge_feature_mode=edge_feature_mode,
        shape_target_mode=shape_target_mode,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(graph, graph.ndata["x"], graph.edata["edge_x"])

    pred_mode_shape_abs = outputs.pred_mode_shape_abs.detach().cpu().numpy()
    pred_freq_z = outputs.pred_freq.detach().cpu().numpy().reshape(-1)
    pred_freq_hz = (
        denormalize_frequency_predictions(outputs.pred_freq, freq_stats).detach().cpu().numpy().reshape(-1)
    )
    target_mode_shape_abs = graph.ndata["y"].detach().cpu().numpy()
    target_freq_hz = None
    if "y_freq_hz" in payload:
        target_freq_hz = np.asarray(payload["y_freq_hz"], dtype=np.float32).reshape(-1)
    elif "target_freq_hz" in generation_meta:
        target_freq_hz = np.asarray(generation_meta["target_freq_hz"], dtype=np.float32).reshape(-1)

    mode_names = tuple(str(x) for x in np.asarray(payload["target_mode_names"]).reshape(-1).tolist())
    coords_xy = np.asarray(payload["coords_xy_mm"], dtype=float)

    mode_shape_metrics = compute_mode_shape_metrics(pred_mode_shape_abs, target_mode_shape_abs, mode_names)
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
        pred_mode_shape_abs=pred_mode_shape_abs,
        target_mode_shape_abs=target_mode_shape_abs,
        pred_freq_z=pred_freq_z,
        pred_freq_hz=pred_freq_hz,
        target_freq_hz=target_freq_hz,
        mode_shape_metrics=mode_shape_metrics,
        frequency_summary=frequency_summary,
    )
    plot_mode_shape_comparison(output_dir, coords_xy, pred_mode_shape_abs, target_mode_shape_abs, mode_names)
    plot_mode_shape_differences(output_dir, coords_xy, pred_mode_shape_abs, target_mode_shape_abs, mode_names)
    plot_frequency_comparison(output_dir, pred_freq_hz=pred_freq_hz, mode_names=mode_names, target_freq_hz=target_freq_hz)

    write_json(
        output_dir / "inference_config.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "run_dir": None if run_dir is None else str(run_dir),
            "device": str(DEVICE),
            "checkpoint_config": checkpoint_config,
            "edge_feature_mode": edge_feature_mode,
            "shape_target_mode": shape_target_mode,
            "checkpoint_edge_feature_mode": checkpoint_config.get("edge_feature_mode", "geom"),
            "freq_mean_hz": freq_stats.freq_mean_hz,
            "freq_std_hz": freq_stats.freq_std_hz,
        },
    )
    write_json(output_dir / "sampled_params.json", generation_meta.get("sampled_params", {}))
    write_json(output_dir / "generation_metadata.json", generation_meta)
    write_report(
        output_dir,
        mode_names=mode_names,
        mode_shape_metrics=mode_shape_metrics,
        frequency_summary=frequency_summary,
    )

    summary = {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "sample_identifier": str(np.asarray(payload.get("sample_identifier", "unseen_inference_case")).item()),
        "n_nodes": int(pred_mode_shape_abs.shape[0]),
        "n_modes": int(pred_mode_shape_abs.shape[1]),
        "mode_names": list(mode_names),
        "edge_feature_mode": edge_feature_mode,
        "pred_freq_z": pred_freq_z,
        "pred_freq_hz": pred_freq_hz,
        "target_freq_hz": target_freq_hz,
        "mode_shape_metrics": mode_shape_metrics,
        "frequency_summary": frequency_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_inference_case(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int,
    ng: int | None,
    run_dir: Path | None = None,
    edge_feature_mode: str = "geom",
    shape_target_mode: str = "abs",
) -> dict[str, Any]:
    set_global_seed(seed)
    payload, generation_meta = assemble_unseen_phase2a_payload(seed=seed, ng=ng)
    generation_meta = {
        **generation_meta,
        "seed": int(seed),
        "requested_ng": None if ng is None else int(ng),
        "edge_feature_mode": edge_feature_mode,
        "shape_target_mode": shape_target_mode,
    }
    return run_inference_from_payload(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        payload=payload,
        generation_meta=generation_meta,
        run_dir=run_dir,
        edge_feature_mode=edge_feature_mode,
        shape_target_mode=shape_target_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained Phase 2A GraphSAGE inference on one newly generated unseen structure.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Direct path to a trained checkpoint.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory to infer checkpoint path from. Uses checkpoints/best_model.pt.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INFERENCE_DIR, help="Root output directory for inference cases.")
    parser.add_argument("--case-name", type=str, default=None, help="Optional inference case name.")
    parser.add_argument("--seed", type=int, default=20260316, help="Seed used for new unseen structure generation.")
    parser.add_argument("--ng", type=int, default=None, help="Optional fixed Ng choice for the unseen structure.")
    parser.add_argument("--edge-feature-mode", choices=EDGE_FEATURE_MODES, default="geom", help="Edge-feature input mode used during inference.")
    parser.add_argument("--shape-target-mode", choices=SHAPE_TARGET_MODES, default="abs", help="Mode-shape target mode used for saved target comparisons.")
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
        run_dir=args.run_dir,
        edge_feature_mode=args.edge_feature_mode,
        shape_target_mode=args.shape_target_mode,
    )
    print(f"Inference output directory: {output_dir}")
    print(f"Checkpoint used: {checkpoint_path}")
    print(f"Predicted frequencies [Hz]: {summary['pred_freq_hz']}")


if __name__ == "__main__":
    main()

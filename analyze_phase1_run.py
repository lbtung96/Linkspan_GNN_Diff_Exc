from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


MODE_NAMES_DEFAULT = ("B1", "T1", "B2", "T2")


def _make_mode_shape_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "mode_shape",
        [(0.0, "#3B4CC0"), (0.5, "#FFFFFF"), (1.0, "#B40426")],
    )


_MODE_SHAPE_CMAP = _make_mode_shape_cmap()


def mac(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)
    num = np.abs(np.dot(u, v)) ** 2
    den = np.dot(u, u) * np.dot(v, v)
    return float(num / (den + 1e-30))


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing_path(raw_path: str, run_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / raw_path
    if cwd_candidate.exists():
        return cwd_candidate

    run_candidate = run_dir / raw_path
    if run_candidate.exists():
        return run_candidate

    raise FileNotFoundError(f"Could not resolve saved source path: {raw_path}")


def load_prediction_record(pred_path: Path, run_dir: Path) -> dict[str, Any]:
    with np.load(pred_path, allow_pickle=True) as data:
        sample_identifier = str(np.asarray(data["sample_identifier"]).item())
        sample_id = str(np.asarray(data["sample_id"]).item())
        source_path_raw = str(np.asarray(data["source_path"]).item())
        source_path = resolve_existing_path(source_path_raw, run_dir)
        prediction = np.asarray(data["prediction"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.float32)
        graph_node_ids = np.asarray(data["graph_node_ids"], dtype=np.int64).reshape(-1)
        mode_names = tuple(str(x) for x in np.asarray(data["target_mode_names"]).reshape(-1).tolist())

    with np.load(source_path, allow_pickle=True) as sample_npz:
        coords_xy = np.asarray(sample_npz["coords_xy_mm"], dtype=np.float32)
        coords_xyz0 = np.asarray(sample_npz["coords_xyz0_mm"], dtype=np.float32) if "coords_xyz0_mm" in sample_npz else None

    if coords_xy.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"{pred_path}: coords_xy_mm row count {coords_xy.shape[0]} does not match prediction rows {prediction.shape[0]}."
        )

    sample_metrics = {
        "sample_identifier": sample_identifier,
        "sample_id": sample_id,
        "source_path": str(source_path),
        "mode_shape_mse": mse(prediction, target),
        "mode_shape_mae": mae(prediction, target),
    }
    for mode_idx, mode_name in enumerate(mode_names):
        sample_metrics[f"mode_shape_mse_{mode_name}"] = mse(prediction[:, mode_idx], target[:, mode_idx])
        sample_metrics[f"mode_shape_mae_{mode_name}"] = mae(prediction[:, mode_idx], target[:, mode_idx])
        sample_metrics[f"mode_shape_mac_{mode_name}"] = mac(prediction[:, mode_idx], target[:, mode_idx])

    return {
        "prediction_path": pred_path,
        "sample_identifier": sample_identifier,
        "sample_id": sample_id,
        "source_path": source_path,
        "prediction": prediction,
        "target": target,
        "graph_node_ids": graph_node_ids,
        "coords_xy_mm": coords_xy,
        "coords_xyz0_mm": coords_xyz0,
        "mode_names": mode_names,
        "metrics": sample_metrics,
    }


def summarize_by_mode(records: list[dict[str, Any]], mode_names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode_name in mode_names:
        rows.append(
            {
                "mode_name": mode_name,
                "mode_shape_mse_mean": float(np.mean([r["metrics"][f"mode_shape_mse_{mode_name}"] for r in records])),
                "mode_shape_mae_mean": float(np.mean([r["metrics"][f"mode_shape_mae_{mode_name}"] for r in records])),
                "mode_shape_mac_mean": float(np.mean([r["metrics"][f"mode_shape_mac_{mode_name}"] for r in records])),
                "mode_shape_mac_min": float(np.min([r["metrics"][f"mode_shape_mac_{mode_name}"] for r in records])),
                "mode_shape_mac_max": float(np.max([r["metrics"][f"mode_shape_mac_{mode_name}"] for r in records])),
            }
        )
    return rows


def save_mode_shape_summaries(records: list[dict[str, Any]], mode_names: tuple[str, ...], mode_shape_dir: Path) -> None:
    per_sample_rows = [r["metrics"] for r in records]
    per_mode_rows = summarize_by_mode(records, mode_names)
    overall_summary = {
        "n_test_samples": len(records),
        "mode_shape_mse_mean": float(np.mean([r["metrics"]["mode_shape_mse"] for r in records])),
        "mode_shape_mae_mean": float(np.mean([r["metrics"]["mode_shape_mae"] for r in records])),
        "mac_definition": "MAC(u, v) = |u^T v|^2 / ((u^T u)(v^T v)), computed per mode over graph nodes for normalized absolute mode-shape vectors.",
        "mode_names": list(mode_names),
    }
    for mode_name in mode_names:
        overall_summary[f"mode_shape_mse_{mode_name}_mean"] = float(np.mean([r["metrics"][f"mode_shape_mse_{mode_name}"] for r in records]))
        overall_summary[f"mode_shape_mae_{mode_name}_mean"] = float(np.mean([r["metrics"][f"mode_shape_mae_{mode_name}"] for r in records]))
        overall_summary[f"mode_shape_mac_{mode_name}_mean"] = float(np.mean([r["metrics"][f"mode_shape_mac_{mode_name}"] for r in records]))

    with (mode_shape_dir / "mode_shape_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, sort_keys=True)

    per_sample_fields = ["sample_identifier", "sample_id", "source_path", "mode_shape_mse", "mode_shape_mae"]
    for mode_name in mode_names:
        per_sample_fields.extend(
            [
                f"mode_shape_mse_{mode_name}",
                f"mode_shape_mae_{mode_name}",
                f"mode_shape_mac_{mode_name}",
            ]
        )
    write_csv(mode_shape_dir / "mode_shape_metrics_per_sample.csv", per_sample_rows, per_sample_fields)
    write_csv(
        mode_shape_dir / "mode_shape_metrics_by_mode.csv",
        per_mode_rows,
        ["mode_name", "mode_shape_mse_mean", "mode_shape_mae_mean", "mode_shape_mac_mean", "mode_shape_mac_min", "mode_shape_mac_max"],
    )


def plot_total_loss_curves(training_rows: list[dict[str, str]], loss_dir: Path) -> None:
    epochs = np.asarray([int(r["epoch"]) for r in training_rows], dtype=int)
    train_loss = np.asarray([float(r["train_loss"]) for r in training_rows], dtype=float)
    val_loss = np.asarray([float(r["val_loss"]) for r in training_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, label="train_total_loss", linewidth=2.0)
    ax.plot(epochs, val_loss, label="val_total_loss", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Train/Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_dir / "loss_curves_total.png", dpi=180)
    plt.close(fig)


def plot_test_mac_boxplot(records: list[dict[str, Any]], mode_names: tuple[str, ...], mode_shape_dir: Path) -> None:
    data = [[r["metrics"][f"mode_shape_mac_{mode_name}"] for r in records] for mode_name in mode_names]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot(data, tick_labels=mode_names, showmeans=True)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("MAC")
    ax.set_title("Test-Set MAC by Mode")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(mode_shape_dir / "test_mac_boxplot.png", dpi=180)
    plt.close(fig)


def plot_sample_error_scatter(records: list[dict[str, Any]], mode_shape_dir: Path) -> None:
    x = np.arange(len(records))
    y = np.asarray([r["metrics"]["mode_shape_mae"] for r in records], dtype=float)
    labels = [r["sample_identifier"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y, s=45, alpha=0.85)
    for idx, label in enumerate(labels):
        ax.annotate(label, (x[idx], y[idx]), fontsize=7, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Overall MAE")
    ax.set_title("Test Sample Overall MAE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(mode_shape_dir / "test_sample_error_scatter.png", dpi=180)
    plt.close(fig)


def plot_sample_error_ranked(records: list[dict[str, Any]], mode_shape_dir: Path) -> None:
    ranked = sorted(records, key=lambda r: r["metrics"]["mode_shape_mae"])
    y = np.asarray([r["metrics"]["mode_shape_mae"] for r in ranked], dtype=float)
    labels = [r["sample_identifier"] for r in ranked]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(ranked)), y, marker="o", linewidth=1.75)
    if labels:
        ax.annotate(f"best: {labels[0]}", (0, y[0]), fontsize=8, xytext=(5, 8), textcoords="offset points")
        ax.annotate(f"worst: {labels[-1]}", (len(ranked) - 1, y[-1]), fontsize=8, xytext=(-80, 8), textcoords="offset points")
    ax.set_xlabel("Ranked Test Sample")
    ax.set_ylabel("Overall MAE")
    ax.set_title("Ranked Test Sample Error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(mode_shape_dir / "test_sample_error_ranked.png", dpi=180)
    plt.close(fig)


def plot_sample_mode_shapes(record: dict[str, Any], sample_dir: Path) -> None:
    coords = np.asarray(record["coords_xy_mm"], dtype=float)
    pred = np.asarray(record["prediction"], dtype=float)
    target = np.asarray(record["target"], dtype=float)
    diff = pred - target
    mode_names = tuple(record["mode_names"])
    sample_identifier = record["sample_identifier"]

    fig, axes = plt.subplots(nrows=len(mode_names), ncols=3, figsize=(12, 3.2 * len(mode_names)), squeeze=False)
    for mode_idx, mode_name in enumerate(mode_names):
        pred_col = pred[:, mode_idx]
        target_col = target[:, mode_idx]
        diff_col = diff[:, mode_idx]

        target_scale = float(np.max(np.abs(target_col))) or 1.0
        pred_scale = float(np.max(np.abs(pred_col))) or 1.0
        diff_lim = float(max(np.max(np.abs(diff_col)), 1e-8))

        panels = [
            ("Target",     target_col / target_scale, _MODE_SHAPE_CMAP, -1.0, 1.0),
            ("Prediction", pred_col / pred_scale,     _MODE_SHAPE_CMAP, -1.0, 1.0),
            ("Difference", diff_col,                  _MODE_SHAPE_CMAP, -diff_lim, diff_lim),
        ]
        for col_idx, (title, values, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[mode_idx, col_idx]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=28,
                            vmin=vmin, vmax=vmax, edgecolors="#333333", linewidths=0.4)
            ax.set_title(f"{mode_name} {title}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Mode Shape Comparison: {sample_identifier}", fontsize=13)
    fig.tight_layout()
    fig.savefig(sample_dir / f"{sample_identifier}_mode_shapes_compare.png", dpi=180)
    plt.close(fig)

    fig_diff, axes_diff = plt.subplots(nrows=1, ncols=len(mode_names), figsize=(4.0 * len(mode_names), 3.6), squeeze=False)
    for mode_idx, mode_name in enumerate(mode_names):
        ax = axes_diff[0, mode_idx]
        diff_col = diff[:, mode_idx]
        diff_lim = float(max(np.max(np.abs(diff_col)), 1e-8))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=diff_col, cmap=_MODE_SHAPE_CMAP,
                        s=30, vmin=-diff_lim, vmax=diff_lim,
                        edgecolors="#333333", linewidths=0.4)
        ax.set_title(f"{mode_name} Difference")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        fig_diff.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig_diff.suptitle(f"Mode Shape Difference: {sample_identifier}", fontsize=13)
    fig_diff.tight_layout()
    fig_diff.savefig(sample_dir / f"{sample_identifier}_mode_shapes_difference.png", dpi=180)
    plt.close(fig_diff)

    with (sample_dir / f"{sample_identifier}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(record["metrics"], f, indent=2, sort_keys=True)


def choose_inspection_records(records: list[dict[str, Any]], selected_sample: str | None) -> dict[str, dict[str, Any]]:
    by_identifier = {r["sample_identifier"]: r for r in records}
    selected: dict[str, dict[str, Any]] = {}
    ranked = sorted(records, key=lambda r: r["metrics"]["mode_shape_mae"])
    if ranked:
        selected["best"] = ranked[0]
        selected["worst"] = ranked[-1]
        selected["representative"] = ranked[len(ranked) // 2]

    if selected_sample is not None:
        match = by_identifier.get(selected_sample)
        if match is None:
            raise KeyError(f"Selected sample {selected_sample!r} was not found in saved test predictions.")
        selected["selected"] = match
    return selected


def analyze_run(run_dir: Path, selected_sample: str | None = None) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config.json"
    history_path = run_dir / "training_history.csv"
    eval_dir = run_dir / "evaluation"
    predictions_dir = eval_dir / "test_predictions"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing training history file: {history_path}")
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Missing test prediction directory: {predictions_dir}")

    analysis_dir = ensure_dir(run_dir / "analysis")
    loss_dir = ensure_dir(analysis_dir / "losses")
    mode_shape_dir = ensure_dir(analysis_dir / "mode_shape")
    sample_dir = ensure_dir(analysis_dir / "sample_inspection")

    config = load_json(config_path)
    training_rows = load_csv_rows(history_path)
    prediction_paths = sorted(predictions_dir.glob("*_prediction.npz"))
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction NPZ files found in {predictions_dir}")

    records = [load_prediction_record(path, run_dir) for path in prediction_paths]
    mode_names = tuple(records[0]["mode_names"]) if records else MODE_NAMES_DEFAULT

    plot_total_loss_curves(training_rows, loss_dir)
    save_mode_shape_summaries(records, mode_names, mode_shape_dir)
    plot_test_mac_boxplot(records, mode_names, mode_shape_dir)
    plot_sample_error_scatter(records, mode_shape_dir)
    plot_sample_error_ranked(records, mode_shape_dir)

    selected_records = choose_inspection_records(records, selected_sample)
    selected_manifest: dict[str, dict[str, Any]] = {}
    for role, record in selected_records.items():
        role_dir = ensure_dir(sample_dir / role)
        plot_sample_mode_shapes(record, role_dir)
        selected_manifest[role] = {
            "sample_identifier": record["sample_identifier"],
            "sample_id": record["sample_id"],
            "mode_shape_mae": record["metrics"]["mode_shape_mae"],
            "mode_shape_mse": record["metrics"]["mode_shape_mse"],
        }

    with (sample_dir / "selected_samples.json").open("w", encoding="utf-8") as f:
        json.dump(selected_manifest, f, indent=2, sort_keys=True)

    analysis_summary = {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "n_test_predictions": len(records),
        "mode_names": list(mode_names),
        "available_analysis_groups": ["losses", "mode_shape", "sample_inspection"],
        "future_extensibility_note": "Current outputs are mode-shape-specific. Loss and metrics naming stays explicit so future frequency/damping outputs can be added without renaming current artifacts.",
        "config_data_source": config.get("data_source"),
    }
    with (analysis_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2, sort_keys=True)

    return analysis_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-run analysis and visualization for Phase-1 GraphSAGE baseline runs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed run directory, e.g. runs_phase1/task5_debug_check")
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Optional saved test sample identifier for an explicit sample-level comparison, e.g. sample_0003_Ng3_phase1_gnn",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = analyze_run(args.run_dir, selected_sample=args.sample_id)
    print(f"Analysis directory: {analysis_dir}")


if __name__ == "__main__":
    main()

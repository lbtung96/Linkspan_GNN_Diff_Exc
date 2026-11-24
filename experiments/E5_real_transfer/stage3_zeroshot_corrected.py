"""Stage 3 - zero-shot inference on the corrected real features.

Same checkpoint as all previous transfer work:
runs_phase2/EdgeGATConv/800_Samples_Ng2Split/checkpoints/best_model.pt

Reports MAC in both reference frames, runs the mandatory input-dependence check
(all-zero features on the same graph), and writes mode-shape figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import dgl  # noqa: E402

from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint  # noqa: E402
from GraphSAGE_Modal_GNN import denormalize_frequency_predictions  # noqa: E402
from infer_phase2a_unseen_structure import load_frequency_normalization  # noqa: E402
from infer_phase2a_unseen_structure_edgegat import (  # noqa: E402
    apply_input_normalization,
    load_input_normalization,
)
from infer_phase2a_deterministic_edgegat import _MODE_SHAPE_CMAP  # noqa: E402
from structures import LOOCV  # single source of truth

OUT = Path(__file__).resolve().parent
CORRECTED = OUT / "phase3_corrected"
GRAPH_ROOT = REPO / "phase3_real_graphs"
AUDIT_RAW = REPO / "experiments" / "E6_controls" / "task2_raw"
DEFAULT_RUN = REPO / "runs_phase2" / "EdgeGATConv" / "800_Samples_Ng2Split"
STRUCTURES = list(LOOCV)
MODES = ("B1", "T1", "B2", "T2")


def mac(u, v) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.abs(np.dot(u, v)) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def build_graph(ei, x, ea, n):
    g = dgl.graph((ei[0], ei[1]), num_nodes=n)
    g.ndata["x"] = torch.tensor(np.asarray(x, dtype=np.float32))
    g.edata["edge_x"] = torch.tensor(np.asarray(ea, dtype=np.float32))
    return g


def infer(model, freq_stats, norm_stats, ei, x, ea, n, fe_only=False, n_fe=None):
    if fe_only:
        m = (ei[0] < n_fe) & (ei[1] < n_fe)
        ei, ea, x, n = ei[:, m], ea[m], x[:n_fe], n_fe
    g = build_graph(ei, x, ea, n)
    apply_input_normalization(g, norm_stats)
    g = g.to(DEVICE)
    with torch.no_grad():
        o = model(g, g.ndata["x"], g.edata["edge_x"])
        return (o.pred_mode_shape_abs.cpu().numpy(),
                denormalize_frequency_predictions(o.pred_freq, freq_stats)
                .cpu().numpy().reshape(-1))


def plot_structure(s, coords_fe, sensor_xy, fe_t, pred_corr, pred_old, y_ssi, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("FE target", fe_t, coords_fe, 28),
            ("Corrected zero-shot", pred_corr, coords_fe, 28),
            ("Published (broken) zero-shot", pred_old, coords_fe, 28),
            ("SSI (sensors)", y_ssi, sensor_xy, 80)]
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(17.5, 12.6), squeeze=False)
    for m, name in enumerate(MODES):
        for c, (title, arr, xy, dot) in enumerate(cols):
            ax = axes[m, c]
            v = np.asarray(arr[:, m], dtype=float)
            v = v / (float(np.max(np.abs(v))) or 1.0)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=_MODE_SHAPE_CMAP, s=dot,
                            vmin=-1.0, vmax=1.0, edgecolors="#333333", linewidths=0.4)
            ax.set_title(f"{name} - {title}", fontsize=9)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.2)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Stage 3 - {s}: corrected vs published zero-shot "
                 "(each panel max-abs normalised)", fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=STRUCTURES, choices=STRUCTURES)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    ckpt = args.run_dir / "checkpoints" / "best_model.pt"
    model, checkpoint = load_trained_model_checkpoint(ckpt, device=DEVICE)
    model.eval()
    freq_stats = load_frequency_normalization(checkpoint_path=ckpt, checkpoint=checkpoint,
                                              run_dir=args.run_dir)
    norm_stats = load_input_normalization(args.run_dir)
    print(f"checkpoint: {ckpt}")
    print(f"input_normalization.json present: {norm_stats is not None}")

    rows, dep_rows = [], []
    for s in args.structures:
        d = np.load(CORRECTED / s / "phase3_dataset_corrected.npz", allow_pickle=False)
        old = np.load(GRAPH_ROOT / s / "phase3_dataset.npz", allow_pickle=False)
        pkg = np.load(GRAPH_ROOT / s / "graph_package.npz", allow_pickle=False)

        ei = d["edge_index"].astype(np.int64)
        n = int(d["x_psd"].shape[0])
        n_fe = int(d["n_fe_nodes"])
        sensor_idx = d["sensor_node_indices"].astype(int)
        nearest_fe = pkg["nearest_fe_node_idx"].astype(int)
        y_ssi = d["y_mode_shape_ssi"].astype(np.float64)
        ssi_freq = d["y_freq_hz"].astype(np.float64).reshape(-1)
        fe_t_pkg = d["fe_target_mode_shape"].astype(np.float64)
        fe_freq = d["fe_target_freq_hz"].astype(np.float64).reshape(-1)
        coords = np.asarray(pkg["node_coords_xy_mm"], dtype=np.float64)
        # FE target from REAL_STRUCTURE_PARAMS, as used by the audit tables
        raw = np.load(AUDIT_RAW / f"{s}.npz")
        fe_t_det = np.asarray(raw["fe_target"], dtype=np.float64)

        variants = {
            "corrected": d["x_psd"],
            "corrected_orderB": d["x_psd_order_B"],
            "corrected_normfull": d["x_psd_norm_full"],
        }
        preds = {}
        for vname, x in variants.items():
            p, f = infer(model, freq_stats, norm_stats, ei, x,
                         d["edge_attr_cpsd_full"], n)
            preds[vname] = (p, f)

        # published (broken) features on the same graph, and the all-zero control
        p_old, f_old = infer(model, freq_stats, norm_stats, ei, old["x_psd"],
                             old["edge_attr_cpsd_full"], n)
        p_zero, f_zero = infer(model, freq_stats, norm_stats, ei,
                               np.zeros_like(d["x_psd"]),
                               np.zeros_like(d["edge_attr_cpsd_full"]), n)
        preds["published_broken"] = (p_old, f_old)
        preds["zero_input"] = (p_zero, f_zero)

        for vname, (p, f) in preds.items():
            at_sensors = p[sensor_idx]
            for m, name in enumerate(MODES):
                rows.append({
                    "structure": s, "variant": vname, "mode": name,
                    "mac_vs_ssi_sensor_nodes": mac(at_sensors[:, m], y_ssi[:, m]),
                    "mac_vs_fe_pkg_all_fe_nodes": mac(p[:n_fe, m], fe_t_pkg[:, m]),
                    "mac_vs_fe_det_all_fe_nodes": mac(p[:n_fe, m], fe_t_det[:, m]),
                    "pred_freq_hz": float(f[m]),
                    "ssi_freq_hz": float(ssi_freq[m]),
                    "fe_target_freq_hz": float(fe_freq[m]),
                    "pred_spatial_std_over_maxabs": float(
                        np.std(p[:n_fe, m]) / (np.max(np.abs(p[:n_fe, m])) + 1e-30)),
                    "n_fe_nodes": n_fe, "n_sensor_nodes": int(sensor_idx.size),
                })

        p_c, f_c = preds["corrected"]
        for m, name in enumerate(MODES):
            dep_rows.append({
                "structure": s, "mode": name,
                "mac_corrected_vs_zeroinput": mac(p_c[:, m], p_zero[:, m]),
                "mac_published_vs_zeroinput": mac(p_old[:, m], p_zero[:, m]),
                "pred_freq_corrected_hz": float(f_c[m]),
                "pred_freq_zeroinput_hz": float(f_zero[m]),
                "pred_freq_published_hz": float(f_old[m]),
                "std_over_maxabs_corrected": float(
                    np.std(p_c[:n_fe, m]) / (np.max(np.abs(p_c[:n_fe, m])) + 1e-30)),
                "std_over_maxabs_zeroinput": float(
                    np.std(p_zero[:n_fe, m]) / (np.max(np.abs(p_zero[:n_fe, m])) + 1e-30)),
            })

        mm = [r["mac_vs_ssi_sensor_nodes"] for r in rows
              if r["structure"] == s and r["variant"] == "corrected"]
        dep = [r["mac_corrected_vs_zeroinput"] for r in dep_rows if r["structure"] == s]
        print(f"  {s:<12} corrected MAC vs SSI = " +
              " ".join(f"{k}={v:.3f}" for k, v in zip(MODES, mm)) +
              f"  mean={np.mean(mm):.3f} | MAC(vs zero-input) = " +
              " ".join(f"{v:.3f}" for v in dep))

        if not args.no_plots:
            plot_structure(s, coords[:n_fe], coords[sensor_idx], fe_t_pkg,
                           p_c[:n_fe], p_old[:n_fe], y_ssi,
                           OUT / f"stage3_modeshapes_{s}.png")

        np.savez_compressed(OUT / "stage3_raw" / f"{s}.npz",
                            **{f"pred_{k}": v[0] for k, v in preds.items()},
                            coords=coords, n_fe=n_fe, sensor_idx=sensor_idx,
                            fe_target_pkg=fe_t_pkg, y_ssi=y_ssi)

    for path, data in [(OUT / "stage3_zeroshot_corrected.csv", rows),
                       (OUT / "stage3_input_dependence.csv", dep_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
        print(f"Wrote {path}")

    print("\n--- means over structures run ---")
    for v in ["published_broken", "zero_input", "corrected",
              "corrected_orderB", "corrected_normfull"]:
        for frame, col in [("SSI@sensors", "mac_vs_ssi_sensor_nodes"),
                           ("FE@allFE", "mac_vs_fe_pkg_all_fe_nodes")]:
            per = [np.mean([r[col] for r in rows if r["variant"] == v and r["mode"] == m])
                   for m in MODES]
            print(f"{v:<20} {frame:<12} " +
                  " ".join(f"{m}={x:.3f}" for m, x in zip(MODES, per)) +
                  f"  mean={np.mean(per):.3f}")

    dep_mean = np.mean([r["mac_corrected_vs_zeroinput"] for r in dep_rows])
    print(f"\nINPUT-DEPENDENCE CHECK: mean MAC(corrected vs zero-input) = {dep_mean:.4f}")
    print("  " + ("SATURATED - corrected results are meaningless"
                  if dep_mean > 0.9 else "PASS - the network responds to the input"))
    return 0


if __name__ == "__main__":
    (OUT / "stage3_raw").mkdir(exist_ok=True)
    raise SystemExit(main())

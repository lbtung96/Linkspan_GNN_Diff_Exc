"""Stage 4 - control experiments on the corrected conventions.

4.1 graph topology       : FE-only subgraph vs hybrid graph
4.2 training distribution: bandpass checkpoint vs narrowband reference checkpoint
4.3 propagation          : deterministic vs propagation-only vs corrected zero-shot,
                           all built with the corrected Welch/normalisation
                           conventions, same frame, same node set
4.4 decomposition        : arithmetic on 4.3
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
AUDIT = REPO / "experiments" / "E6_controls"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(AUDIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dgl  # noqa: E402

from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint  # noqa: E402
from GraphSAGE_Modal_GNN import denormalize_frequency_predictions  # noqa: E402
from infer_phase2a_unseen_structure import load_frequency_normalization  # noqa: E402
from infer_phase2a_unseen_structure_edgegat import (  # noqa: E402
    apply_input_normalization,
    load_input_normalization,
)
from infer_phase2a_deterministic_edgegat import (  # noqa: E402
    _build_params,
    assemble_deterministic_phase2a_payload,
)
from task2_propagation_only_control import resimulate_deterministic  # noqa: E402
from build_phase3_features_corrected import (  # noqa: E402
    corrected_edge_features,
    corrected_node_features,
    training_freq_grid,
)
from structures import LOOCV  # single source of truth

OUT = Path(__file__).resolve().parent
CORRECTED = OUT / "phase3_corrected"
GRAPH_ROOT = REPO / "phase3_real_graphs"
AUDIT_RAW = AUDIT / "task2_raw"
RUN_NARROW = REPO / "runs_phase2" / "EdgeGATConv" / "800_Samples_Ng2Split"
RUN_BAND = REPO / "runs_phase2" / "EdgeGATConv" / "edgegat_20260507_123555"
STRUCTURES = list(LOOCV)
MODES = ("B1", "T1", "B2", "T2")
MAC_THR = {"Eidsdal": 0.70}


def mac(u, v) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.abs(np.dot(u, v)) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def load_model(run_dir: Path):
    ckpt = run_dir / "checkpoints" / "best_model.pt"
    model, checkpoint = load_trained_model_checkpoint(ckpt, device=DEVICE)
    model.eval()
    fs_ = load_frequency_normalization(checkpoint_path=ckpt, checkpoint=checkpoint,
                                       run_dir=run_dir)
    return model, fs_, load_input_normalization(run_dir)


def infer(bundle, ei, x, ea, n):
    model, freq_stats, norm_stats = bundle
    g = dgl.graph((ei[0], ei[1]), num_nodes=n)
    g.ndata["x"] = torch.tensor(np.asarray(x, dtype=np.float32))
    g.edata["edge_x"] = torch.tensor(np.asarray(ea, dtype=np.float32))
    apply_input_normalization(g, norm_stats)
    g = g.to(DEVICE)
    with torch.no_grad():
        o = model(g, g.ndata["x"], g.edata["edge_x"])
        return (o.pred_mode_shape_abs.cpu().numpy(),
                denormalize_frequency_predictions(o.pred_freq, freq_stats)
                .cpu().numpy().reshape(-1))


def main() -> int:
    grid = training_freq_grid()
    narrow = load_model(RUN_NARROW)
    band = load_model(RUN_BAND)
    rows: list[dict] = []

    for s in STRUCTURES:
        d = np.load(CORRECTED / s / "phase3_dataset_corrected.npz", allow_pickle=False)
        pkg = np.load(GRAPH_ROOT / s / "graph_package.npz", allow_pickle=False)
        ei = d["edge_index"].astype(np.int64)
        x, ea = d["x_psd"], d["edge_attr_cpsd_full"]
        n = int(x.shape[0]); n_fe = int(d["n_fe_nodes"])
        sensor_idx = d["sensor_node_indices"].astype(int)
        nearest_fe = pkg["nearest_fe_node_idx"].astype(int)
        y_ssi = d["y_mode_shape_ssi"].astype(np.float64)
        fe_det = np.asarray(np.load(AUDIT_RAW / f"{s}.npz")["fe_target"], dtype=np.float64)

        # ---- 4.1 topology + 4.2 checkpoint -------------------------------
        results: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        p, f = infer(narrow, ei, x, ea, n)
        results["corrected_hybrid_narrowband"] = (p[:n_fe], p[sensor_idx], "hybrid")
        m_fe = (ei[0] < n_fe) & (ei[1] < n_fe)
        p2, _ = infer(narrow, ei[:, m_fe], x[:n_fe], ea[m_fe], n_fe)
        results["corrected_feonly_narrowband"] = (p2, p2[nearest_fe], "fe_only")
        p3, _ = infer(band, ei, x, ea, n)
        results["corrected_hybrid_bandpass"] = (p3[:n_fe], p3[sensor_idx], "hybrid")
        p4, _ = infer(band, ei[:, m_fe], x[:n_fe], ea[m_fe], n_fe)
        results["corrected_feonly_bandpass"] = (p4, p4[nearest_fe], "fe_only")

        # ---- 4.3 deterministic and propagation-only, corrected conventions
        params = _build_params(s)
        payload, _ = assemble_deterministic_phase2a_payload(
            params, structure_name=s, min_mac_threshold=MAC_THR.get(s))
        det_ei = np.asarray(payload["edge_index"], dtype=np.int64)
        sim = resimulate_deterministic(s, params)
        a3, fs_sim = sim["a3_graph"], sim["fs"]

        # deterministic, rebuilt through the corrected builder (clamp all nodes)
        all_idx = np.arange(n_fe, dtype=int)
        x_det = corrected_node_features(a3, fs_sim, grid, n_nodes=n_fe,
                                        sensor_idx=all_idx, edge_index=det_ei,
                                        order="A", norm_range="grid")
        e_det = corrected_edge_features(a3, fs_sim, grid, det_ei)
        pd_, _ = infer(narrow, det_ei, x_det, e_det, n_fe)
        results["deterministic_corrected"] = (pd_, pd_[nearest_fe], "fe_only")

        # propagation-only: subsample to sensors, bandpass, extend, corrected build
        from phase3_utils import BANDPASS_F_HIGH, BANDPASS_F_LOW, bandpass_filter
        from build_phase3_features_corrected import propagate as prop_fn
        a3_sensor = a3[nearest_fe, :]
        a3_bp = bandpass_filter(a3_sensor, fs_sim, f_low=BANDPASS_F_LOW,
                                f_high=BANDPASS_F_HIGH)
        x_prop = corrected_node_features(a3_bp, fs_sim, grid, n_nodes=n,
                                         sensor_idx=sensor_idx, edge_index=ei,
                                         order="A", norm_range="grid")
        ts = np.zeros((n, a3_bp.shape[1]), dtype=np.float32)
        ts[sensor_idx] = a3_bp.astype(np.float32)
        ts_prop = prop_fn(ts, sensor_idx, ei, n).astype(np.float64)
        e_prop = corrected_edge_features(ts_prop[:n_fe], fs_sim, grid, det_ei)
        pp, _ = infer(narrow, det_ei, x_prop[:n_fe], e_prop, n_fe)
        results["propagation_only_corrected"] = (pp, pp[nearest_fe], "fe_only")

        for name, (p_fe, p_sens, topo) in results.items():
            for m, k in enumerate(MODES):
                rows.append({
                    "structure": s, "variant": name, "topology": topo, "mode": k,
                    "mac_vs_ssi_sensor_nodes": mac(p_sens[:, m], y_ssi[:, m]),
                    "mac_vs_fe_det_all_fe_nodes": mac(p_fe[:, m], fe_det[:, m]),
                })
        print(f"  [{s}] done")

    path = OUT / "stage4_controls.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {path}\n")

    order = ["deterministic_corrected", "propagation_only_corrected",
             "corrected_hybrid_narrowband", "corrected_feonly_narrowband",
             "corrected_hybrid_bandpass", "corrected_feonly_bandpass"]
    for frame, col in [("FE@allFE (vs REAL_STRUCTURE_PARAMS target)", "mac_vs_fe_det_all_fe_nodes"),
                       ("SSI@sensors", "mac_vs_ssi_sensor_nodes")]:
        print(f"--- mean MAC over {len(STRUCTURES)} structures, {frame} ---")
        for v in order:
            per = [np.mean([r[col] for r in rows if r["variant"] == v and r["mode"] == m])
                   for m in MODES]
            print(f"{v:<32} " + " ".join(f"{m}={x:.3f}" for m, x in zip(MODES, per)) +
                  f"  mean={np.mean(per):.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

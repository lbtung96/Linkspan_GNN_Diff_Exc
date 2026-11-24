"""Stage 5 - LOOCV fine-tuning on the corrected Phase-3 features.

Four configurations: {head-only, full} x {narrowband, bandpass} pretrained
checkpoint, 8-fold leave-one-structure-out.

Hyperparameters are the ones the rerun brief specifies, which are the production
full-fine-tuning settings (finetune_phase3_full.py):
    ENCODER_LR 1e-5, HEAD_LR 1e-4, WEIGHT_DECAY 1e-4, GRAD_CLIP_NORM 1.0,
    MAX_EPOCHS 500, PATIENCE 50, COSINE_ETA_MIN 1e-7, FREQ_LOSS_WEIGHT 0.25,
    early stopping on held-out mean sensor MAC.
They are applied to BOTH configurations, so head-only and full differ only by
whether the encoder is frozen.  (The production head-only script instead used
LR 1e-4 / 200 epochs / patience 20 with no cosine schedule - noted in the report.)

The loss, optimizer grouping, freezing rule and MAC definition are copied
verbatim from finetune_phase3_full.py / finetune_phase3_real_sparse.py because
those live at module scope in scripts whose dataset root is hard-coded; the copies
are marked below.
"""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from structures import LOOCV  # single source of truth

OUT = Path(__file__).resolve().parent
CORRECTED = OUT / "phase3_corrected"
RUNS = {
    "narrowband": REPO / "runs_phase2" / "EdgeGATConv" / "800_Samples_Ng2Split",
    "bandpass": REPO / "runs_phase2" / "EdgeGATConv" / "edgegat_20260507_123555",
}
STRUCTURES = list(LOOCV)
MODES = ("B1", "T1", "B2", "T2")

ENCODER_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MAX_EPOCHS = 500
PATIENCE = 50
FREQ_LOSS_WEIGHT = 0.25
COSINE_ETA_MIN = 1e-7
ENCODER_PARTS = ("node_encoder", "edge_encoder", "layers")


# --- copied verbatim from finetune_phase3_full.py --------------------------
def _mac(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.dot(a, a) * np.dot(b, b))
    if denom < 1e-24:
        return 0.0
    return float(np.dot(a, b) ** 2 / denom)


def _mean_sensor_mac(pred_shapes, y_ssi, sensor_idx) -> float:
    pred_at = pred_shapes[sensor_idx, :]
    macs = []
    for m in range(4):
        a = pred_at[:, m]
        b = y_ssi[:, m]
        if np.dot(a, b) < 0:
            a = -a
        macs.append(_mac(a, b))
    return float(np.mean(macs))


def _sparse_loss(pred_shapes, pred_freq_z, y_ssi_shapes, y_freq_hz, sensor_idx,
                 freq_mean, freq_std):
    sensor_t = torch.tensor(sensor_idx, dtype=torch.long, device=pred_shapes.device)
    shape_loss = F.mse_loss(pred_shapes[sensor_t, :], y_ssi_shapes.to(pred_shapes.device))
    y_freq_norm = (y_freq_hz.to(pred_shapes.device) - freq_mean) / freq_std
    freq_loss = F.mse_loss(pred_freq_z.squeeze(0), y_freq_norm)
    return shape_loss + FREQ_LOSS_WEIGHT * freq_loss


def _make_optimizer(model, encoder_lr, head_lr):
    enc, head = [], []
    for name, p in model.named_parameters():
        (enc if any(x in name for x in ENCODER_PARTS) else head).append(p)
    return torch.optim.Adam([
        {"params": enc, "lr": encoder_lr, "weight_decay": WEIGHT_DECAY},
        {"params": head, "lr": head_lr, "weight_decay": WEIGHT_DECAY},
    ])


def _freeze_encoder(model) -> None:
    for name, p in model.named_parameters():
        if any(x in name for x in ENCODER_PARTS):
            p.requires_grad = False
# --- end copied block ------------------------------------------------------


def load_datasets(run_dir: Path) -> dict:
    norm_stats = load_input_normalization(run_dir)
    out = {}
    for s in STRUCTURES:
        raw = {k: v for k, v in
               np.load(CORRECTED / s / "phase3_dataset_corrected.npz",
                       allow_pickle=False).items()}
        ei = raw["edge_index"].astype(np.int64)
        n = int(raw["x_psd"].shape[0])
        g = dgl.graph((ei[0], ei[1]), num_nodes=n)
        g.ndata["x"] = torch.tensor(raw["x_psd"], dtype=torch.float32)
        g.ndata["y"] = torch.zeros(n, 4, dtype=torch.float32)
        g.edata["edge_x"] = torch.tensor(raw["edge_attr_cpsd_full"], dtype=torch.float32)
        apply_input_normalization(g, norm_stats)
        out[s] = {
            "graph": g,
            "sensor_idx": raw["sensor_node_indices"].astype(int),
            "y_ssi": torch.tensor(raw["y_mode_shape_ssi"], dtype=torch.float32),
            "y_freq_hz": torch.tensor(raw["y_freq_hz"].reshape(-1), dtype=torch.float32),
            "n_fe_nodes": int(raw["n_fe_nodes"]),
        }
    return out


def train_fold(held_out: str, run_dir: Path, datasets: dict, freq_stats,
               mode: str) -> dict:
    ckpt = run_dir / "checkpoints" / "best_model.pt"
    model, _ = load_trained_model_checkpoint(ckpt, device=DEVICE)
    if mode == "head_only":
        _freeze_encoder(model)
    opt = _make_optimizer(model, ENCODER_LR, HEAD_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=MAX_EPOCHS, eta_min=COSINE_ETA_MIN)
    freq_mean = torch.tensor(freq_stats.freq_mean_hz, dtype=torch.float32, device=DEVICE)
    freq_std = torch.tensor(freq_stats.freq_std_hz, dtype=torch.float32, device=DEVICE)
    train_s = [s for s in STRUCTURES if s != held_out]

    best_mac, best_state, best_epoch, patience = -1.0, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for name in train_s:
            d = datasets[name]
            g = d["graph"].to(DEVICE)
            opt.zero_grad()
            o = model(g, g.ndata["x"], g.edata["edge_x"])
            loss = _sparse_loss(o.pred_mode_shape_abs, o.pred_freq, d["y_ssi"],
                                d["y_freq_hz"], d["sensor_idx"], freq_mean, freq_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
        sched.step()

        model.eval()
        d = datasets[held_out]
        with torch.no_grad():
            g = d["graph"].to(DEVICE)
            o = model(g, g.ndata["x"], g.edata["edge_x"])
            p = o.pred_mode_shape_abs.cpu().numpy()
        vm = _mean_sensor_mac(p, d["y_ssi"].numpy(), d["sensor_idx"])
        if vm > best_mac:
            best_mac, best_epoch, patience = vm, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    d = datasets[held_out]
    with torch.no_grad():
        g = d["graph"].to(DEVICE)
        o = model(g, g.ndata["x"], g.edata["edge_x"])
        p = o.pred_mode_shape_abs.cpu().numpy()
        f = denormalize_frequency_predictions(o.pred_freq, freq_stats).cpu().numpy().reshape(-1)
    at = p[d["sensor_idx"], :]
    y = d["y_ssi"].numpy()
    macs = []
    for m in range(4):
        a = at[:, m].copy()
        if np.dot(a, y[:, m]) < 0:
            a = -a
        macs.append(_mac(a, y[:, m]))
    return {"mac": macs, "best_epoch": best_epoch, "stopped_epoch": epoch,
            "pred_freq_hz": f.tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["head_only_narrowband", "full_narrowband",
                             "head_only_bandpass", "full_bandpass"])
    args = ap.parse_args(argv)

    rows = []
    for cfg in args.configs:
        mode, ckpt_key = cfg.rsplit("_", 1)
        run_dir = RUNS[ckpt_key]
        ck = run_dir / "checkpoints" / "best_model.pt"
        _, checkpoint = load_trained_model_checkpoint(ck, device=DEVICE)
        freq_stats = load_frequency_normalization(checkpoint_path=ck,
                                                  checkpoint=checkpoint, run_dir=run_dir)
        datasets = load_datasets(run_dir)
        print(f"\n=== {cfg} ({run_dir.name}) ===")
        for s in STRUCTURES:
            torch.manual_seed(123)
            np.random.seed(123)
            r = train_fold(s, run_dir, datasets, freq_stats, mode)
            for m, k in enumerate(MODES):
                rows.append({"config": cfg, "mode_setting": mode, "checkpoint": ckpt_key,
                             "held_out": s, "mode": k, "mac_vs_ssi": r["mac"][m],
                             "best_epoch": r["best_epoch"], "stopped_epoch": r["stopped_epoch"],
                             "pred_freq_hz": r["pred_freq_hz"][m]})
            print(f"  {s:<12} " + " ".join(f"{k}={v:.3f}" for k, v in zip(MODES, r["mac"])) +
                  f"  mean={np.mean(r['mac']):.3f}  best_epoch={r['best_epoch']}"
                  f"  stopped={r['stopped_epoch']}")

    path = OUT / "stage5_finetuning.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {path}\n")

    print("--- LOOCV mean MAC vs SSI (sensor nodes) ---")
    for cfg in args.configs:
        per = [np.mean([r["mac_vs_ssi"] for r in rows
                        if r["config"] == cfg and r["mode"] == m]) for m in MODES]
        eps = [r["best_epoch"] for r in rows if r["config"] == cfg and r["mode"] == "B1"]
        print(f"{cfg:<26} " + " ".join(f"{m}={x:.3f}" for m, x in zip(MODES, per)) +
              f"  mean={np.mean(per):.3f}  best_epoch median={int(np.median(eps))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

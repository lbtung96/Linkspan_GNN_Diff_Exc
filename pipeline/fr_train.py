"""Resumable trainer for every network in Final_results_for_paper.

    python Final_results_for_paper/pipeline/fr_train.py --job edgegat_N600 [--threads 4]

It reproduces `EdgeGATConv_Modal_GNN.train_one_run` and
`GraphSAGE_Modal_GNN.train_one_run` epoch for epoch - the same dataset classes,
loaders, model, optimiser, loss, logging and best-model rule, all imported from
those modules - with three deliberate differences:

 1. **Explicit file lists.**  The train/validation structures come from the
    manifests written by s2_make_splits.py, instead of `sorted(glob)[:max_samples]`
    followed by a random split.  That is what makes the sweep balanced across
    girder counts and nested in N (the first-N-files rule is how the old sweep
    ended up training N <= 2000 on two-girder decks only).
 2. **Resume.**  Every `--resume-every` epochs the model, optimiser, all RNG
    states and the history are written atomically to checkpoints/resume.pt.  A
    run that is interrupted continues from there; on CPU the continuation is
    bit-identical to an uninterrupted run (checked by s3_check_trainer.py).
 3. **No test scoring here.**  Training ends with best_model.pt, last_model.pt,
    training_history.csv and run_summary.json.  Every test number is produced
    later by s4_indomain.py, on CPU, from best_model.pt - one inference path for
    every run, whichever device trained it.

A run is complete when run_summary.json exists.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import fr_common as C


@dataclass(frozen=True)
class Job:
    name: str
    arch: str                     # "edgegat" | "graphsage"
    n: int                        # population size the manifests were cut for
    edge_feature_mode: str        # edgegat: cpsd_full | geom_orient | zero ;
                                  # graphsage: zero | cpsd | cpsd_full | geom_orient | geom
                                  # (every mode but edgegat cpsd_full and graphsage zero/cpsd/geom
                                  # comes from fr_edge_modes)
    shape_target_mode: str = "signed"
    epochs: int = C.EPOCHS
    seed: int = C.SEED
    note: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)


def jobs() -> dict[str, Job]:
    out: dict[str, Job] = {}
    for n in C.SWEEP_N:
        out[C.sweep_run_name(n)] = Job(C.sweep_run_name(n), "edgegat", n, "cpsd_full",
                                       note="population-size sweep; N=600 is the paper's main network")
    # (That note is what the stored run_summary.json files carry: the main network was
    # N = 600 when they were trained.  Since 2026-09-28 it is fr_common.MAIN_N = 2000.)
    # Architecture ablation at N = 600 (fr_common.ABLATION_N).  The EdgeGAT arm IS
    # edgegat_N600: same data, split, seed and epochs.
    out["graphsage_noedge_signed_N600"] = Job("graphsage_noedge_signed_N600", "graphsage", 600, "zero",
                                              note="node PSD only, no edge information")
    out["graphsage_cpsd8_signed_N600"] = Job("graphsage_cpsd8_signed_N600", "graphsage", 600, "cpsd",
                                             note="scalar CPSD edges (8-d), no frequency-resolved phase")
    # (That note is also stored in its run_summary.json.  The 8 features are in fact the
    # complex coherency, Re and Im, at the four FE-matched modal frequencies - phase at the
    # modes, not across the band.)
    # Added 2026-09-28 with the author, so that architecture and edge feature are each
    # varied alone: GraphSAGE on EdgeGAT's own 800-d input, and EdgeGAT on geometry only.
    out["graphsage_cpsdfull_signed_N600"] = Job("graphsage_cpsdfull_signed_N600", "graphsage", 600, "cpsd_full",
                                                note="full-spectrum complex coherency edges (800-d), "
                                                     "the EdgeGAT arm's own input")
    out["edgegat_geomorient_N600"] = Job("edgegat_geomorient_N600", "edgegat", 600, "geom_orient",
                                         note="edge orientation only (cos, sin of the edge angle), no phase")
    # Added 2026-09-29, completing the grid: each layer x {no edge features, orientation,
    # full coherency}.
    out["graphsage_geomorient_signed_N600"] = Job("graphsage_geomorient_signed_N600", "graphsage", 600,
                                                  "geom_orient",
                                                  note="edge orientation only (cos, sin of the edge angle), no phase")
    out["edgegat_noedge_N600"] = Job("edgegat_noedge_N600", "edgegat", 600, "zero",
                                     note="no edge information: constant (zero) edge features")
    return out


# The previous ablation's fourth row (GraphSAGE, geometric edges, absolute targets,
# "context only, not comparable") cannot run on this population, so it is not a job.
UNRUNNABLE = {
    "graphsage_geom_abs_N600": (
        "Fails before training: GraphSAGEEdgeWeighted passes the learned edge weight through "
        "sigmoid and EdgeWeightNorm(norm='both'), which rejects a zero weight. edge_attr_geom is the "
        "raw edge length in mm; the new pool's two-girder decks carry full-length diagonal braces of "
        "19.4-19.9 m, and at initialisation (seed 123) 5 of the 508 N = 600 training decks already "
        "give a pre-sigmoid value near -90, which is exactly 0 in float32. The previous pool had no "
        "two-girder decks (longest edge 9.6 m). Same failure class as the signed geometry arm the "
        "previous ablation reported; not forced to run by changing the architecture or the feature."),
}


def manifest_paths(n: int) -> tuple[Path, Path]:
    return C.SPLITS / f"N{n}_train.txt", C.SPLITS / f"N{n}_val.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description="Resumable trainer (Final_results_for_paper).")
    ap.add_argument("--job", required=True, choices=sorted(jobs()))
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    ap.add_argument("--resume-every", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=None, help="override (tests only)")
    ap.add_argument("--run-root", type=Path, default=C.RUNS, help="override (tests only)")
    ap.add_argument("--train-list", type=Path, default=None, help="override (tests only)")
    ap.add_argument("--val-list", type=Path, default=None, help="override (tests only)")
    ap.add_argument("--pool", type=Path, default=None, help="override the sample folder (tests only)")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="exit after this many epochs in this invocation (resume tests only)")
    args = ap.parse_args()

    if args.threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    import numpy as np
    import torch
    import torch.nn as nn
    from dgl.dataloading import GraphDataLoader

    import GraphSAGE_Modal_GNN as gs
    import EdgeGATConv_Modal_GNN as eg
    import fr_edge_modes

    fr_edge_modes.install()
    if args.threads:
        torch.set_num_threads(int(args.threads))
    C.keep_awake()
    DEVICE = gs.DEVICE

    job = jobs()[args.job]
    epochs = int(args.epochs or job.epochs)
    pool = Path(args.pool) if args.pool else C.POP_BROADBAND
    tr_m, va_m = manifest_paths(job.n)
    tr_m, va_m = (args.train_list or tr_m), (args.val_list or va_m)

    def resolve(m: Path) -> list[Path]:
        names = [s.strip() for s in Path(m).read_text(encoding="utf-8").splitlines() if s.strip()]
        paths = [pool / nm for nm in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest entries missing, e.g. {missing[0]}")
        return paths

    train_paths, val_paths = resolve(tr_m), resolve(va_m)

    run_dir = Path(args.run_root) / job.name
    ckpt_dir = run_dir / "checkpoints"
    ctx = gs.RunContext(run_dir=run_dir, checkpoints_dir=ckpt_dir,
                        eval_dir=run_dir / "evaluation", predictions_dir=run_dir / "evaluation" / "test_predictions")
    if (run_dir / "run_summary.json").exists():
        print(f"[skip] {job.name}: already complete")
        return 0
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "resume.pt"

    # ---- build exactly as the canonical train_one_run does ------------------------------
    if job.arch == "edgegat":
        cfg = eg.TrainingConfig(
            dataset_root=pool, run_root=Path(args.run_root), run_name=job.name, epochs=epochs,
            batch_size=16, hidden_dim=32, edge_hidden_dim=32, num_layers=3, num_heads=4,
            dropout=0.1, attn_dropout=0.1, lr=1e-3, weight_decay=1e-4, seed=job.seed,
            split_seed=C.SPLIT_SEED, max_samples=None, train_frac=C.TRAIN_FRAC,
            val_frac=round(1.0 - C.TRAIN_FRAC, 10), test_frac=0.0, normalize_inputs=False,
            shape_target_mode=job.shape_target_mode, edge_feature_mode=job.edge_feature_mode,
            freq_loss_weight=0.25, disable_freq_multitask=False, enable_freq_multitask=True)
        gs._validate_split_fractions(cfg.train_frac, cfg.val_frac, cfg.test_frac)
        gs.set_global_seed(cfg.seed)
        train_set = eg.Phase2AEdgeDataset(train_paths, edge_feature_mode=cfg.edge_feature_mode,
                                          shape_target_mode=cfg.shape_target_mode,
                                          expected_n_modes=cfg.expected_n_modes_out)
        val_set = eg.Phase2AEdgeDataset(val_paths, edge_feature_mode=cfg.edge_feature_mode,
                                        shape_target_mode=cfg.shape_target_mode,
                                        expected_n_modes=cfg.expected_n_modes_out)
        cfg.node_feat_dim = int(train_set.node_feat_dim or 0)
        cfg.edge_feat_dim = int(train_set.edge_feat_dim or 0)
        freq_stats = gs.compute_frequency_normalization_stats(train_set.labels, eps=cfg.freq_std_eps)
        train_loader = GraphDataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        val_loader = GraphDataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        model = eg.EdgeGATModalGNN(
            node_feat_dim=cfg.node_feat_dim, edge_feat_dim=cfg.edge_feat_dim, hidden_dim=cfg.hidden_dim,
            edge_hidden_dim=cfg.edge_hidden_dim, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dropout=cfg.dropout, attn_dropout=cfg.attn_dropout, n_modes_out=cfg.expected_n_modes_out,
            allow_zero_in_degree=cfg.allow_zero_in_degree).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        move_batch = True
        n_epoch_field = cfg.epochs
        seed_field = cfg.seed
    elif job.arch == "graphsage":
        cfg = gs.TrainingConfig(
            data_source="phase1_npz", phase1_dir=pool, runs_dir=Path(args.run_root), run_name=job.name,
            expected_n_freq=400, expected_n_modes_out=4, batch_size=16, n_epoch=epochs, lr=1e-3,
            train_fraction=C.TRAIN_FRAC, val_fraction=round(1.0 - C.TRAIN_FRAC, 10), test_fraction=0.0,
            seed=job.seed, max_samples=None, enable_freq_multitask=True, freq_loss_weight=0.25,
            show_plot=False, edge_feature_mode=job.edge_feature_mode,
            shape_target_mode=job.shape_target_mode,
            edge_feat_dim=gs._edge_feature_dim_for_mode(job.edge_feature_mode))
        gs._validate_split_fractions(cfg.train_fraction, cfg.val_fraction, cfg.test_fraction)
        gs.set_global_seed(cfg.seed)
        train_set = gs.Phase1NPZDataset(train_paths, cfg.expected_n_freq, cfg.expected_n_modes_out,
                                        edge_feature_mode=cfg.edge_feature_mode,
                                        shape_target_mode=cfg.shape_target_mode)
        val_set = gs.Phase1NPZDataset(val_paths, cfg.expected_n_freq, cfg.expected_n_modes_out,
                                      edge_feature_mode=cfg.edge_feature_mode,
                                      shape_target_mode=cfg.shape_target_mode)
        freq_stats = gs.compute_frequency_normalization_stats(train_set.labels, eps=cfg.freq_std_eps)
        train_loader = GraphDataLoader(train_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle_train,
                                       drop_last=False)
        val_loader = GraphDataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        model = gs.ModalGNN(n_freq=cfg.expected_n_freq, n_modes_out=cfg.expected_n_modes_out,
                            enc_hid=cfg.enc_hid, enc_out=cfg.enc_out, gnn_hid=cfg.gnn_hid,
                            gnn_out=cfg.gnn_out, dec_hid=cfg.dec_hid,
                            edge_feat_dim=cfg.edge_feat_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        move_batch = False                         # Phase1NPZDataset already put every graph on DEVICE
        n_epoch_field = cfg.n_epoch
        seed_field = cfg.seed
    else:
        raise ValueError(job.arch)

    loss_fn = nn.MSELoss()
    gs._save_run_config(ctx, cfg)
    gs._save_split_artifacts(ctx, gs.SplitArtifacts(
        train_ids=[p.name for p in train_paths], val_ids=[p.name for p in val_paths],
        test_ids=[s.strip() for s in (C.SPLITS / "test.txt").read_text(encoding="utf-8").splitlines()
                  if s.strip()] if (C.SPLITS / "test.txt").exists() else []))
    gs._save_frequency_normalization(ctx, freq_stats, cfg)

    history: list[dict] = []
    best_val_loss, best_epoch, start_epoch, elapsed_before = float("inf"), -1, 0, 0.0
    if resume_path.exists():
        # map to CPU: the RNG states must stay CPU ByteTensors; load_state_dict
        # copies weights and optimiser moments onto the parameters' own device.
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        history = list(st["history"])
        best_val_loss, best_epoch = float(st["best_val_loss"]), int(st["best_epoch"])
        start_epoch = int(st["epoch_completed"]) + 1
        elapsed_before = float(st.get("elapsed_s", 0.0))
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        np.random.set_state(st["np_rng"])
        random.setstate(st["py_rng"])
        if int(st.get("threads", torch.get_num_threads())) != torch.get_num_threads():
            print(f"[warn] resumed with {torch.get_num_threads()} threads, saved with {st.get('threads')}"
                  " - CPU results may differ in the last bits")
        if str(st.get("device")) != str(DEVICE):
            raise RuntimeError(f"resume state is from {st.get('device')}, this process is on {DEVICE}")
        print(f"[resume] {job.name}: continuing after epoch {start_epoch - 1} "
              f"(best so far {best_val_loss:.6e} @ {best_epoch})")

    print(f"[train] {job.name}: arch={job.arch} edge={job.edge_feature_mode} target={job.shape_target_mode} "
          f"N={job.n} train={len(train_set)} val={len(val_set)} epochs={epochs} device={DEVICE} "
          f"threads={torch.get_num_threads()}", flush=True)

    def save_resume(epoch_completed: int, elapsed: float) -> None:
        state = {"epoch_completed": epoch_completed, "model": model.state_dict(), "opt": opt.state_dict(),
                 "history": history, "best_val_loss": best_val_loss, "best_epoch": best_epoch,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "np_rng": np.random.get_state(), "py_rng": random.getstate(),
                 "threads": torch.get_num_threads(), "device": str(DEVICE), "elapsed_s": elapsed}
        tmp = resume_path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, resume_path)

    t0 = time.time()
    ran = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        tr_tot = tr_shape = tr_freq = tr_mae = 0.0
        for g_batch, freq_target_hz in train_loader:
            if move_batch:
                g_batch = g_batch.to(DEVICE)
            freq_target_hz = freq_target_hz.to(DEVICE)
            outputs = model(g_batch, g_batch.ndata["x"], g_batch.edata["edge_x"])
            loss_total, loss_shape, loss_freq, pred_freq_hz = gs._compute_multitask_loss(
                outputs, g_batch.ndata["y"], freq_target_hz, loss_fn=loss_fn,
                enable_freq_multitask=True, freq_loss_weight=cfg.freq_loss_weight, freq_stats=freq_stats)
            opt.zero_grad()
            loss_total.backward()
            opt.step()
            tr_tot += float(loss_total.detach().item())
            tr_shape += float(loss_shape.detach().item())
            tr_freq += float(loss_freq.detach().item())
            if pred_freq_hz is not None:
                tr_mae += float(torch.mean(torch.abs(pred_freq_hz - freq_target_hz)).detach().item())
        nb = max(1, len(train_loader))
        tr_tot, tr_shape, tr_freq, tr_mae = tr_tot / nb, tr_shape / nb, tr_freq / nb, tr_mae / nb

        model.eval()
        va_tot = va_shape = va_freq = va_mae = 0.0
        with torch.no_grad():
            for g_batch, freq_target_hz in val_loader:
                if move_batch:
                    g_batch = g_batch.to(DEVICE)
                freq_target_hz = freq_target_hz.to(DEVICE)
                outputs = model(g_batch, g_batch.ndata["x"], g_batch.edata["edge_x"])
                loss_total, loss_shape, loss_freq, pred_freq_hz = gs._compute_multitask_loss(
                    outputs, g_batch.ndata["y"], freq_target_hz, loss_fn=loss_fn,
                    enable_freq_multitask=True, freq_loss_weight=cfg.freq_loss_weight, freq_stats=freq_stats)
                va_tot += float(loss_total.detach().item())
                va_shape += float(loss_shape.detach().item())
                va_freq += float(loss_freq.detach().item())
                if pred_freq_hz is not None:
                    va_mae += float(torch.mean(torch.abs(pred_freq_hz - freq_target_hz)).detach().item())
        nv = max(1, len(val_loader))
        va_tot, va_shape, va_freq, va_mae = va_tot / nv, va_shape / nv, va_freq / nv, va_mae / nv

        history.append({"epoch": epoch, "train_loss_total": tr_tot, "train_loss_shape": tr_shape,
                        "train_loss_freq": tr_freq, "train_freq_mae_hz": tr_mae,
                        "val_loss_total": va_tot, "val_loss_shape": va_shape, "val_loss_freq": va_freq,
                        "val_freq_mae_hz": va_mae, "n_train_samples": len(train_set),
                        "n_val_samples": len(val_set), "batch_size": cfg.batch_size, "seed": seed_field,
                        "learning_rate": cfg.lr, "n_epochs": n_epoch_field})
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d} | train total {tr_tot:.6e} shape {tr_shape:.6e} freq {tr_freq:.6e} "
                  f"freq_mae_hz {tr_mae:.6e} | valid total {va_tot:.6e} shape {va_shape:.6e} "
                  f"freq {va_freq:.6e} freq_mae_hz {va_mae:.6e} | best {best_epoch} "
                  f"| {time.time() - t0 + elapsed_before:.0f}s", flush=True)
        if va_tot < best_val_loss:
            best_val_loss, best_epoch = va_tot, epoch
            gs._save_checkpoint(ckpt_dir / "best_model.pt", model, cfg, epoch, tr_tot, va_tot,
                                freq_stats=freq_stats)
        ran += 1
        done = epoch + 1 == epochs
        if (epoch + 1) % args.resume_every == 0 or done or (args.stop_after and ran >= args.stop_after):
            save_resume(epoch, elapsed_before + time.time() - t0)
        if args.stop_after and ran >= args.stop_after and not done:
            print(f"[stop-after] {job.name}: stopped after epoch {epoch} (resume state saved)")
            return 0

    train_time_s = elapsed_before + time.time() - t0
    gs._save_checkpoint(ckpt_dir / "last_model.pt", model, cfg, epochs - 1,
                        float(history[-1]["train_loss_total"]), float(history[-1]["val_loss_total"]),
                        freq_stats=freq_stats)
    gs._save_training_history_csv(ctx, history)
    summary = {
        "job": job.name, "arch": job.arch, "n_population": job.n, "note": job.note,
        "edge_feature_mode": job.edge_feature_mode, "shape_target_mode": job.shape_target_mode,
        "dataset_root": str(pool), "train_manifest": str(tr_m), "val_manifest": str(va_m),
        "n_train_samples": len(train_set), "n_val_samples": len(val_set),
        "epochs": epochs, "seed": job.seed, "device": str(DEVICE), "threads": torch.get_num_threads(),
        "torch": torch.__version__, "python": sys.executable,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "final_train_loss_total": history[-1]["train_loss_total"],
        "final_val_loss_total": history[-1]["val_loss_total"],
        "train_time_s": train_time_s,
        "freq_mean_hz": freq_stats.freq_mean_hz.tolist(), "freq_std_hz": freq_stats.freq_std_hz.tolist(),
        "checkpoint_rule": "best_model.pt = lowest validation total loss (strict <), as in train_one_run",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {job.name}: best epoch {best_epoch}, best val {best_val_loss:.6e}, {train_time_s:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

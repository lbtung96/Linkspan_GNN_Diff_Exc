"""Gates for the EC T1-T3 trainer wrapper, run before the training queue.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/t_gates.py [--parts a b c ram] [--workers 4]

  a  bit-identity: the paper's edgegat_N600 and graphsage_cpsdfull_signed_N600, trained
     through t_train.py (--paper-job) for 3 epochs into a scratch folder, must give a
     training_history.csv identical, number for number, to the first 3 epochs of the
     paper's own runs (every column but n_epochs, which records the requested length).
     Also ec_sweep_r0_N100, whose manifests equal the paper's N100 split byte for byte,
     against edgegat_N100.
  b  seed: edgegat_N600_s124 (3 epochs) must differ from the paper's seed-123 epoch 0.
  c  smoke: 2 epochs of every new input variant (Re-only and |gamma|-only for both layers,
     record-wise node normalisation) and of one nested-split job (ec_sweep_r1_N100);
     the variant features are checked against their definition on real samples; and the
     trainable parameter count of every T1 arm (total and edge path, as s11 counts them)
     - existing arms must equal s11_ablation_cost.csv, phase arms must equal the full-
     coherency arm of the same layer.
  ram  peak working set of one training process at N = 600, 2000 and 5000 (1-2 epochs
     with t_train.py, 2 threads); fitted a + b*N is what t_jobs.RAM_* should hold.

Scratch runs go to EC_comments/runs/_gates/ (deleted and re-made per gate).  Results:
results/T1_ablation_seeds_phase/t_gates_training.json and t_gates_ram.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import ec_common as E
import fr_common as C

GATES = E.EC_RUNS / "_gates"
HIST_KEYS = ["epoch", "train_loss_total", "train_loss_shape", "train_loss_freq", "train_freq_mae_hz",
             "val_loss_total", "val_loss_shape", "val_loss_freq", "val_freq_mae_hz", "n_train_samples",
             "n_val_samples", "batch_size", "seed", "learning_rate"]
THREADS = 2


def train(tag: str, epochs: int, job: str | None = None, paper_job: str | None = None) -> dict:
    root = GATES / tag
    name = job or paper_job
    shutil.rmtree(root / name, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(E.EC_PIPELINE / "t_train.py"),
           *(["--job", job] if job else ["--paper-job", paper_job]),
           "--threads", str(THREADS), "--epochs", str(epochs), "--run-root", str(root)]
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    log = E.EC / "logs" / f"_gate_{tag}_{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=str(C.REPO), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    mem = json.loads((root / name / "t_mem.json").read_text(encoding="utf-8"))[-1] if (root / name / "t_mem.json").exists() else {}
    print(f"[gate-run] {tag}/{name}: rc={rc} {time.time() - t0:.0f}s mem={mem}", flush=True)
    return {"rc": rc, "dir": root / name, "seconds": time.time() - t0, "mem": mem, "log": str(log)}


def history(path: Path, n: int | None = None) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[:n] if n else rows


def same_history(a: Path, b: Path, n: int) -> dict:
    ha, hb = history(a, n), history(b, n)
    diffs = [(i, k, ra[k], rb[k]) for i, (ra, rb) in enumerate(zip(ha, hb)) for k in HIST_KEYS if ra[k] != rb[k]]
    return {"pass": len(ha) == len(hb) == n and not diffs, "epochs_compared": n, "n_diffs": len(diffs),
            "first_diffs": diffs[:5]}


def params(ck: Path, arch: str) -> tuple[int, int]:
    import torch
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs
    import t_edge_modes

    t_edge_modes.install()
    if arch == "edgegat":
        model, _ = eg.load_trained_model_checkpoint(ck, device=torch.device("cpu"))
    else:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model = gs.build_model_from_config(st["config"])
    named = list(model.named_parameters())
    return (sum(p.numel() for _, p in named if p.requires_grad),
            sum(p.numel() for n, p in named if p.requires_grad and "edge" in n))


def feature_check() -> dict:
    """The variants, built through the patched library on real samples, match their definitions."""
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs
    import t_edge_modes as T

    T.install()
    files = [C.POP_BROADBAND / f for f in (C.SPLITS / "N600_train.txt").read_text().split()]
    pick = files[::25]
    out = {"n_samples": len(pick)}
    worst = {"re_slots": 0.0, "im_zero": 0.0, "abs": 0.0, "gs_eq_eg": 0.0, "recnorm_nodes": 0.0,
             "recnorm_edges_eq_full": 0.0, "full_nodes_unchanged": 0.0}
    for p in pick:
        with np.load(p, allow_pickle=True) as d:
            full = np.asarray(d["edge_attr_cpsd_full"], np.float32)
            x = np.asarray(d["x_psd"], np.float32)
            g_re, _ = eg.build_phase2a_graph_from_payload({k: d[k] for k in d.files}, edge_feature_mode=T.CPSD_RE,
                                                         shape_target_mode="signed")
            g_ab, _ = eg.build_phase2a_graph_from_payload({k: d[k] for k in d.files}, edge_feature_mode=T.CPSD_ABS,
                                                         shape_target_mode="signed")
            g_rn, _ = eg.build_phase2a_graph_from_payload({k: d[k] for k in d.files}, edge_feature_mode=T.RECNORM,
                                                         shape_target_mode="signed")
            g_fu, _ = eg.build_phase2a_graph_from_payload({k: d[k] for k in d.files}, edge_feature_mode="cpsd_full",
                                                         shape_target_mode="signed")
            raw = np.asarray(d["x_psd"], np.float64) * np.asarray(d["x_psd_row_scale"], np.float64)[:, None]
        gs_re = gs._build_graph_from_phase1_npz(p, 400, 4, edge_feature_mode=T.CPSD_RE, shape_target_mode="signed")
        gs_ab = gs._build_graph_from_phase1_npz(p, 400, 4, edge_feature_mode=T.CPSD_ABS, shape_target_mode="signed")
        re = g_re.edata["edge_x"].numpy()
        ab = g_ab.edata["edge_x"].numpy()
        worst["re_slots"] = max(worst["re_slots"], float(np.abs(re[:, 0::2] - full[:, 0::2]).max()))
        worst["im_zero"] = max(worst["im_zero"], float(np.abs(re[:, 1::2]).max()), float(np.abs(ab[:, 1::2]).max()))
        worst["abs"] = max(worst["abs"], float(np.abs(ab[:, 0::2] - np.hypot(full[:, 0::2], full[:, 1::2])).max()))
        worst["gs_eq_eg"] = max(worst["gs_eq_eg"], float(np.abs(gs_re.edata["edge_x"].numpy() - re).max()),
                                float(np.abs(gs_ab.edata["edge_x"].numpy() - ab).max()))
        worst["recnorm_nodes"] = max(worst["recnorm_nodes"],
                                     float(np.abs(g_rn.ndata["x"].numpy() - (raw / raw.max()).astype(np.float32)).max()))
        worst["recnorm_edges_eq_full"] = max(worst["recnorm_edges_eq_full"],
                                             float(np.abs(g_rn.edata["edge_x"].numpy() - full).max()))
        worst["full_nodes_unchanged"] = max(worst["full_nodes_unchanged"], float(np.abs(g_fu.ndata["x"].numpy() - x).max()))
    out["max_abs_dev"] = worst
    out["pass"] = (worst["re_slots"] == 0 and worst["im_zero"] == 0 and worst["abs"] <= 1e-6 and worst["gs_eq_eg"] == 0
                   and worst["recnorm_nodes"] <= 1e-6 and worst["recnorm_edges_eq_full"] == 0
                   and worst["full_nodes_unchanged"] == 0)
    out["recnorm_reconstruction"] = T.recnorm_check(C.POP_BROADBAND / f for f in
                                                    (C.SPLITS / "pool.txt").read_text().split()[::10])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["a", "b", "c", "ram"], choices=["a", "b", "c", "ram"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import t_jobs

    out_json = E.task_dir("T1") / "t_gates_training.json"
    report = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    report["threads"] = THREADS
    report.pop("run_failures", None)
    runs: list[tuple[str, dict]] = []
    if "a" in args.parts:
        runs += [("a", dict(tag="a", epochs=3, paper_job="edgegat_N600")),
                 ("a", dict(tag="a", epochs=3, paper_job="graphsage_cpsdfull_signed_N600")),
                 ("a", dict(tag="a", epochs=3, job="ec_sweep_r0_N100"))]
    if "b" in args.parts:
        runs += [("b", dict(tag="b", epochs=3, job="edgegat_N600_s124"))]
    if "c" in args.parts:
        runs += [("c", dict(tag="c", epochs=2, job=j)) for j in
                 ("edgegat_cpsdre_N600_s123", "edgegat_cpsdabs_N600_s123", "graphsage_cpsdre_signed_N600_s123",
                  "graphsage_cpsdabs_signed_N600_s123", "edgegat_recnorm_N600_s123", "ec_sweep_r1_N100")]
    if "ram" in args.parts:
        runs += [("ram", dict(tag="ram", epochs=2, job="edgegat_N600_s125")),
                 ("ram", dict(tag="ram", epochs=1, job="ec_main_N2000_s124")),
                 ("ram", dict(tag="ram", epochs=1, job="ec_sweep_r1_N5000"))]
    # the N = 5000 RAM run is the longest: start it first
    reg = t_jobs.registry()
    runs.sort(key=lambda r: -(reg[r[1]["job"]].n if r[1].get("job") else C.ABLATION_N))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda r: (r[0], r[1], train(**r[1])), runs))
    by = {(p, kw.get("job") or kw.get("paper_job")): r for p, kw, r in res}
    bad_rc = {f"{p}/{n}": r["rc"] for (p, n), r in by.items() if r["rc"] != 0}
    if bad_rc:
        report["run_failures"] = bad_rc

    if "a" in args.parts:
        ga = {}
        for name, paper in (("edgegat_N600", "edgegat_N600"), ("graphsage_cpsdfull_signed_N600",
                                                               "graphsage_cpsdfull_signed_N600"),
                            ("ec_sweep_r0_N100", "edgegat_N100")):
            r = by[("a", name)]
            ga[f"{name} vs paper {paper}"] = same_history(r["dir"] / "training_history.csv",
                                                         C.RUNS / paper / "training_history.csv", 3) \
                if r["rc"] == 0 else {"pass": False, "rc": r["rc"]}
        ga["pass"] = all(v["pass"] for v in ga.values())
        report["a_bit_identity"] = ga
    if "b" in args.parts:
        r = by[("b", "edgegat_N600_s124")]
        h124 = history(r["dir"] / "training_history.csv", 3)
        h123 = history(C.RUNS / "edgegat_N600" / "training_history.csv", 3)
        report["b_seed_changes_init"] = {
            "seed_field": [x["seed"] for x in h124],
            "epoch0_train_loss_total": {"s123": float(h123[0]["train_loss_total"]), "s124": float(h124[0]["train_loss_total"])},
            "epoch0_val_loss_total": {"s123": float(h123[0]["val_loss_total"]), "s124": float(h124[0]["val_loss_total"])},
            "pass": r["rc"] == 0 and h124[0]["train_loss_total"] != h123[0]["train_loss_total"]
            and all(x["seed"] == "124" for x in h124)}
    if "c" in args.parts:
        gc = {"runs": {}}
        for (p, n), r in by.items():
            if p != "c":
                continue
            h = history(r["dir"] / "training_history.csv") if r["rc"] == 0 else []
            s = json.loads((r["dir"] / "run_summary.json").read_text(encoding="utf-8")) if r["rc"] == 0 else {}
            gc["runs"][n] = {"rc": r["rc"], "epochs": len(h), "edge_feature_mode": s.get("edge_feature_mode"),
                             "n_train": s.get("n_train_samples"), "n_val": s.get("n_val_samples"),
                             "train_manifest": s.get("train_manifest"),
                             "val_loss_total": [float(x["val_loss_total"]) for x in h],
                             "finite": bool(h) and all(np.isfinite(float(x["train_loss_total"])) for x in h),
                             "seconds": round(r["seconds"], 1), "mem": r["mem"]}
        gc["features"] = feature_check()
        # parameter counts of every T1 arm
        s11 = {r["run"]: r for r in csv.DictReader((C.RESULTS / "s11_ablation_cost.csv").open(encoding="utf-8"))}
        pc = {}
        for (layer, feat) in [(lay, f) for lay in C.ABLATION_LAYERS for f in t_jobs.T1_FEATURES]:
            tj = t_jobs.t1_run(layer, feat, 123)
            if tj is None:
                continue
            ck = (tj.run_dir if tj.reuse else GATES / "c" / tj.name) / "checkpoints" / "best_model.pt"
            if not ck.exists():
                pc[f"{layer}|{feat}"] = {"error": f"missing {ck}"}
                continue
            n_all, n_edge = params(ck, tj.job.arch)
            row = {"run_seed123": tj.name, "params_total": n_all, "params_edge_path": n_edge}
            if tj.reuse:
                row["equals_s11"] = (int(s11[tj.name]["params_total"]) == n_all
                                     and int(s11[tj.name]["params_edge_path"]) == n_edge)
            pc[f"{layer}|{feat}"] = row
        for layer in C.ABLATION_LAYERS:
            full = pc.get(f"{layer}|coherency", {})
            for f in ("coherency_re", "coherency_abs", "coherency_recnorm"):
                if f"{layer}|{f}" in pc and "params_total" in pc[f"{layer}|{f}"]:
                    pc[f"{layer}|{f}"]["equals_full_coherency_arm"] = (
                        pc[f"{layer}|{f}"]["params_total"] == full.get("params_total")
                        and pc[f"{layer}|{f}"]["params_edge_path"] == full.get("params_edge_path"))
        gc["param_counts"] = pc
        gc["pass"] = (all(v["rc"] == 0 and v["finite"] for v in gc["runs"].values()) and gc["features"]["pass"]
                      and all(v.get("equals_s11", True) and v.get("equals_full_coherency_arm", True) and "error" not in v
                              for v in pc.values()))
        report["c_smoke"] = gc
    if "ram" in args.parts:
        rows = []
        for (p, n), r in by.items():
            if p == "ram" or (p in ("a", "b", "c") and r["mem"].get("peak_gb")):
                tj = t_jobs.registry().get(n)
                nn = tj.n if tj else C.ABLATION_N
                rows.append({"part": p, "run": n, "N": nn, "epochs": 2 if p == "c" else (3 if p in "ab" else None),
                             "peak_gb": r["mem"].get("peak_gb"), "kind": r["mem"].get("kind"),
                             "seconds": round(r["seconds"], 1)})
        rows.sort(key=lambda x: (x["N"], x["run"]))
        with (E.task_dir("T1") / "t_gates_ram.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        ram = [x for x in rows if x["part"] == "ram" and x["peak_gb"]]
        if len(ram) >= 2:
            b, a = np.polyfit([x["N"] for x in ram], [x["peak_gb"] for x in ram], 1)
            report["ram_fit"] = {"a_gb": float(a), "b_gb_per_N": float(b),
                                 "points": {x["run"]: [x["N"], x["peak_gb"]] for x in ram},
                                 "t_jobs_constants": {"a": t_jobs.RAM_A_GB, "b": t_jobs.RAM_B_GB_PER_N}}
    report["pass"] = all(report[k]["pass"] for k in ("a_bit_identity", "b_seed_changes_init", "c_smoke") if k in report) \
        and "run_failures" not in report
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in report.items()}, indent=1))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

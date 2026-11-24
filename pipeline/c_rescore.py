"""T5 (EC-15) - score any trained network on the paper's fixed test set and/or the fresh ones.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/c_rescore.py \
        [--runs NAME_OR_PATH ...] [--sets paper fresh_test fresh_test_paperexc] [--workers 4]

Runs are found by name in Final_results_for_paper/runs/ first, then in EC_comments/runs/
(any depth: a name may be a relative path such as T1/edgegat_N600_s124), or given as a
directory path.  Default: the 14 paper runs (fr_train.jobs()).  A run is usable when its
folder has run_summary.json (arch, edge_feature_mode, shape_target_mode are read from it)
and checkpoints/<checkpoint>.pt.

The code path is s4_indomain.evaluate's, line for line, with the sample directory and the
run directory as parameters: CPU, torch.set_num_threads(2), the library's datasets
(Phase2AEdgeDataset / Phase1NPZDataset), the library's per-sample metrics, best-validation
checkpoint by default, frequency de-normalised with the checkpoint's own stats.  Edge-feature
modes: t_edge_modes.install() if that module exists (it installs fr_edge_modes first and adds
the T1 modes), else fr_edge_modes.install().

Gate: on --sets paper, every paper run's summary must equal s4_indomain_summary.csv
(best_model rows) exactly, in every column both files carry.  Written to rescore_gate.json.

Writes (results/T5_fresh_test_baselines/):
  rescore_per_sample.csv   one row per test set x run x checkpoint x structure (accumulates:
                           rows of (set, run, checkpoint) re-scored now replace the old ones)
  rescore_summary.csv      s4's summary columns, per test set x run x checkpoint
  rescore_by_ng.csv        the same per girder count
and predictions to EC_comments/data/T5/predictions/<set>/<run-key>__<checkpoint>.npz
(per structure: pred, target, freq_pred, freq_true; plus forward-pass seconds).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import ec_common as E  # noqa: E402

C = E.C
import s4_indomain as S4  # noqa: E402
from fr_train import jobs  # noqa: E402

OUT = E.task_dir("T5")
T5_DATA = E.EC_DATA / "T5"
PRED_ROOT = T5_DATA / "predictions"
NAME_RE = re.compile(r"sample_(\d{4})_Ng(\d)_phase1_gnn\.npz$")
SETS = ("paper", "fresh_test", "fresh_test_paperexc")
GATE_COLS_EXCLUDE = {"run", "checkpoint", "arch", "edge_feature_mode", "shape_target_mode", "device"}


def set_files(test_set: str) -> list[Path]:
    if test_set == "paper":
        names = [s.strip() for s in (C.SPLITS / "test.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
        return [C.POP_BROADBAND / n for n in names]
    d = T5_DATA / test_set / "samples_phase2_bandpass"
    files = sorted(d.glob("sample_*_Ng*_phase1_gnn.npz"))
    if len(files) != 500:
        raise RuntimeError(f"{d}: expected 500 samples, found {len(files)}")
    return files


def resolve_run(spec: str) -> tuple[str, str, Path]:
    """(key, root, run_dir).  key is what the tables show; root is 'paper' or 'ec' or 'path'."""
    p = Path(spec)
    if (C.RUNS / spec / "run_summary.json").exists():
        return spec, "paper", C.RUNS / spec
    if (E.EC_RUNS / spec / "run_summary.json").exists():
        return spec.replace("\\", "/"), "ec", E.EC_RUNS / spec
    if (p / "run_summary.json").exists():
        p = p.resolve()
        try:
            return p.relative_to(E.EC_RUNS.resolve()).as_posix(), "ec", p
        except ValueError:
            return p.name, "path", p
    hits = sorted(E.EC_RUNS.rglob(f"{spec}/run_summary.json")) if E.EC_RUNS.exists() else []
    if len(hits) == 1:
        d = hits[0].parent
        return d.relative_to(E.EC_RUNS).as_posix(), "ec", d
    raise FileNotFoundError(f"run {spec!r}: no run_summary.json in {C.RUNS}, {E.EC_RUNS} or as a path"
                            + (f" (ambiguous: {[str(h.parent) for h in hits]})" if hits else ""))


def install_modes() -> str:
    if (E.EC_PIPELINE / "t_edge_modes.py").exists():
        import t_edge_modes
        t_edge_modes.install()
        return "t_edge_modes"
    import fr_edge_modes
    fr_edge_modes.install()
    return "fr_edge_modes"


def evaluate(task: tuple) -> tuple[list[dict], dict]:
    """s4_indomain.evaluate with the sample files and the run directory as parameters."""
    key, root, run_dir, which, test_set = task
    import torch
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs

    modes = install_modes()
    torch.set_num_threads(2)
    info = json.loads((Path(run_dir) / "run_summary.json").read_text(encoding="utf-8"))
    arch, efm, stm = info["arch"], info["edge_feature_mode"], info.get("shape_target_mode", "signed")
    files = set_files(test_set)
    ck = Path(run_dir) / "checkpoints" / f"{which}.pt"
    dev = torch.device("cpu")
    if arch == "edgegat":
        model, ckpt = eg.load_trained_model_checkpoint(ck, device=dev)
        ds = eg.Phase2AEdgeDataset(files, edge_feature_mode=efm, shape_target_mode=stm, expected_n_modes=4)
        metric = eg._compute_sample_metrics
    else:
        ckpt = torch.load(ck, map_location=dev, weights_only=False)
        model = gs.build_model_from_config(ckpt["config"]).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        ds = gs.Phase1NPZDataset(files, 400, 4, edge_feature_mode=efm, shape_target_mode=stm)
        metric = gs._compute_sample_metrics
    model.eval()
    stats = gs.FrequencyNormalizationStats(freq_mean_hz=np.asarray(ckpt["freq_mean_hz"], np.float32),
                                           freq_std_hz=np.asarray(ckpt["freq_std_hz"], np.float32))
    rows, preds, fwd = [], {}, []
    with torch.no_grad():
        for i in range(len(ds)):
            g, f_true = ds[i]
            g = g.to(dev)
            t0 = time.perf_counter()
            out = model(g, g.ndata["x"], g.edata["edge_x"])
            fwd.append(time.perf_counter() - t0)
            pred = out.pred_mode_shape_abs.cpu().numpy()
            target = g.ndata["y"].cpu().numpy()
            f_pred = gs.denormalize_frequency_predictions(out.pred_freq, stats).cpu().numpy().reshape(-1)
            f_true = f_true.cpu().numpy().reshape(-1)
            m = metric(pred, target, C.MODE_NAMES)
            fm = gs.compute_frequency_metrics(f_pred[None, :], f_true[None, :], C.MODE_NAMES)
            name = files[i].name
            r = {"test_set": test_set, "run": key, "run_root": root, "checkpoint": which, "sample": name,
                 "Ng": int(NAME_RE.search(name).group(2)), "epoch_of_checkpoint": int(ckpt.get("epoch", -1))}
            for k, mode in enumerate(C.MODE_NAMES):
                r[f"mac_{mode}"] = m[f"mac_{mode}"]
                r[f"freq_pred_{mode}_hz"] = float(f_pred[k])
                r[f"freq_true_{mode}_hz"] = float(f_true[k])
                r[f"freq_abs_err_{mode}_hz"] = float(abs(f_pred[k] - f_true[k]))
            r["mac_mean"] = float(np.mean([m[f"mac_{mode}"] for mode in C.MODE_NAMES]))
            r["freq_mae_hz"] = fm["freq_mae_hz"]
            r["forward_s"] = fwd[-1]
            rows.append(r)
            preds[name] = (pred.astype(np.float32), target.astype(np.float32),
                           f_pred.astype(np.float32), f_true.astype(np.float32))
    dst = PRED_ROOT / test_set
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / f"{key.replace('/', '__')}__{which}.npz",
                        forward_s=np.asarray(fwd), samples=np.asarray(list(preds)),
                        **{f"{k.removesuffix('.npz')}__{nm}": v[j] for k, v in preds.items()
                           for j, nm in enumerate(("pred", "target", "freq_pred", "freq_true"))})
    meta = {"run": key, "root": root, "arch": arch, "edge_feature_mode": efm, "shape_target_mode": stm,
            "N": info.get("n_population"), "best_epoch": info.get("best_epoch"), "epochs": info.get("epochs"),
            "seed": info.get("seed"), "device": info.get("device"), "train_time_h": info.get("train_time_s", 0) / 3600,
            "modes_module": modes}
    return rows, meta


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def gate(summ: list[dict]) -> dict:
    ref = {r["run"]: r for r in read_csv(C.RESULTS / "s4_indomain_summary.csv") if r["checkpoint"] == "best_model"}
    out = {"reference": "Final_results_for_paper/pipeline/results/s4_indomain_summary.csv (best_model rows)",
           "rule": "every shared numeric column equal (==)", "runs": {}}
    for s in summ:
        if s["test_set"] != "paper" or s["checkpoint"] != "best_model" or s["run"] not in ref:
            continue
        r = ref[s["run"]]
        cols = [c for c in r if c in s and c not in GATE_COLS_EXCLUDE]
        bad = {c: (float(r[c]), float(s[c])) for c in cols if float(r[c]) != float(s[c])}
        out["runs"][s["run"]] = {"columns_compared": len(cols), "mismatches": bad, "pass": not bad}
    out["n_runs"] = len(out["runs"])
    out["pass"] = bool(out["runs"]) and all(v["pass"] for v in out["runs"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None, help="names or paths (default: the 14 paper runs)")
    ap.add_argument("--sets", nargs="+", default=["paper", "fresh_test"], choices=SETS)
    ap.add_argument("--which", nargs="+", default=["best_model"], choices=["best_model", "last_model"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    specs = args.runs or [r for r in jobs() if (C.RUNS / r / "run_summary.json").exists()]
    runs = [resolve_run(s) for s in specs]
    tasks = [(k, root, str(d), w, ts) for ts in args.sets for (k, root, d) in runs for w in args.which]
    t0 = time.time()
    new, metas = [], {}
    with ProcessPoolExecutor(max_workers=min(args.workers, 4, len(tasks))) as ex:
        for rows, meta in ex.map(evaluate, tasks):
            new += rows
            metas[meta["run"]] = meta
            print(f"[eval] {rows[0]['test_set']:<20} {meta['run']:<34} {rows[0]['checkpoint']:<10} "
                  f"MAC {np.mean([r['mac_mean'] for r in rows]):.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    done = {(r["test_set"], r["run"], r["checkpoint"]) for r in new}
    per_path = OUT / "rescore_per_sample.csv"
    old = [r for r in read_csv(per_path) if (r["test_set"], r["run"], r["checkpoint"]) not in done]
    allrows = old + new
    for r in allrows:
        for k, v in list(r.items()):
            if k not in ("test_set", "run", "run_root", "checkpoint", "sample"):
                r[k] = int(float(v)) if k in ("Ng", "epoch_of_checkpoint") else float(v)
    write_csv(per_path, allrows)

    meta_path = OUT / "rescore_runs.json"
    allmeta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    allmeta.update(metas)
    meta_path.write_text(json.dumps(allmeta, indent=1), encoding="utf-8")

    summ, by_ng = [], []
    groups = sorted({(r["test_set"], r["run"], r["checkpoint"]) for r in allrows},
                    key=lambda t: (SETS.index(t[0]) if t[0] in SETS else 9, t[1], t[2]))
    for ts, run, which in groups:
        rr = [r for r in allrows if (r["test_set"], r["run"], r["checkpoint"]) == (ts, run, which)]
        m = allmeta.get(run, {})
        keys = {"test_set": ts, "run": run, "checkpoint": which, "run_root": rr[0]["run_root"],
                "arch": m.get("arch"), "N": m.get("N"), "edge_feature_mode": m.get("edge_feature_mode"),
                "shape_target_mode": m.get("shape_target_mode"), "seed": m.get("seed"),
                "epoch_of_checkpoint": rr[0]["epoch_of_checkpoint"], "best_epoch": m.get("best_epoch"),
                "epochs": m.get("epochs"), "device": m.get("device"), "train_time_h": m.get("train_time_h")}
        s = S4.summarise(rr, keys)
        s["forward_ms_median"] = float(np.median([r["forward_s"] for r in rr]) * 1e3)
        summ.append(s)
        for g in C.NG_VALUES:
            by_ng.append(S4.summarise([r for r in rr if r["Ng"] == g], {**keys, "Ng": g}))
    write_csv(OUT / "rescore_summary.csv", summ)
    write_csv(OUT / "rescore_by_ng.csv", by_ng)

    rc = 0
    if "paper" in args.sets:
        g = gate([s for s in summ if (s["test_set"], s["run"], s["checkpoint"]) in done])
        (OUT / "rescore_gate.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
        print(f"[gate] paper test set == s4_indomain_summary.csv for {g['n_runs']} paper runs: "
              f"{'PASS' if g['pass'] else 'FAIL'}")
        if g["runs"] and not g["pass"]:
            print(json.dumps({k: v for k, v in g["runs"].items() if not v["pass"]}, indent=1))
            rc = 1
    for s in summ:
        if (s["test_set"], s["run"], s["checkpoint"]) in done:
            print(f"{s['test_set']:<20} {s['run']:<34} MAC {s['mac_mean']:.4f} [B1 {s['mac_B1']:.3f} T1 {s['mac_T1']:.3f} "
                  f"B2 {s['mac_B2']:.3f} T2 {s['mac_T2']:.3f}]  f MAE {s['freq_mae_hz']:.3f} Hz")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""Retrain the archive's networks with THIS repository's trainer and compare them with the archived runs.

    python tools/run_stage.py tools/verify_training.py --prefix [--epochs 50] [--parallel 6] [--only NAME ...]
    python tools/run_stage.py tools/verify_training.py --full-t17 [--parallel 6]
    python tools/run_stage.py tools/verify_training.py --report

The archive trained every EC network through fr_train.main() with in-memory substitutions (fr_train.jobs and
sys.argv replaced, the per-epoch record draw installed by k_train.install_augmentation, the sign-invariant loss
by p_common.install_signinv).  Here they are code (fr_train.train, records.py, Job.shape_loss, library edge
modes).  This stage retrains through the stage wrappers' own entry points (h_train.make_job + fr_train.train,
k_train.run_one, p_train.run_arm, q_train.run), each run in a fresh process with 2 torch threads, into
runs/repro/<mode>/<name>, never into runs/ec/:

  --prefix    every training set-up the archive ran (T11 2 arms, T13 5 arms, T16 6 arms, T20 one seed per
              excitation), first --epochs epochs: training_history.csv must equal the archived run's first
              rows number for number (t_gates.same_history: epoch, every loss, MAE, sizes, seed, lr - every
              column but n_epochs).  No learning-rate schedule and no early stopping, so a short run is the
              long run's prefix.
  --full-t17  the six T17 networks, all 1000 epochs: the whole history, and best_model.pt / last_model.pt
              weights bit-identical (k_train.same_weights), best epoch and best validation loss equal.
              Identical weights imply identical evaluation (q_eval, q_template_compare, u_decomposition).
  --report    write results/reproduction/training_summary.csv and training_gates.json from whatever has run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "pipeline")]
THREADS = 2
ROOT = REPO / "runs" / "repro"
OUT = REPO / "results" / "reproduction"


def specs() -> dict:
    """name -> (task, mode, reference run dir, archived epochs)."""
    import q_common as Q
    import v_common as V
    K, H, P = Q.K, Q.H, Q.P
    out = {}
    for n in H.ARMS:
        out[n] = ("T11", "h", K.EC_RUNS / n)
    for n in K.ARMS:
        out[n] = ("T13", "k", K.EC_RUNS / n)
    for n in P.ARMS:
        out[n] = ("T16", "p", K.EC_RUNS / n)
    for n in Q.RUNS:
        out[n] = ("T17", "q", K.EC_RUNS / n)
    for n, a in V.RUNS.items():
        out[n] = ("T20", "v", K.EC_RUNS / n)
    return out


PREFIX_SET = None  # filled in main(): T11, T13, T16 all arms; T20 seed 123 of each excitation


def run_job(name: str, epochs: int, root: Path) -> int:
    """One retraining in this process (the launcher sets the thread environment first)."""
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
    import q_common as Q
    import v_common as V
    K, H, P = Q.K, Q.H, Q.P
    task, mode, _ref = specs()[name]
    root.mkdir(parents=True, exist_ok=True)
    if mode == "h":
        import fr_train
        import h_train
        return fr_train.train(h_train.make_job(name), threads=THREADS, resume_every=25, epochs=epochs, run_root=root,
                              train_list=H.TRAIN_LIST, val_list=H.VAL_LIST, pool=H.TRAINVAL_DIR)
    if mode == "k":
        import k_train
        a = K.ARMS[name]
        pools = list(a["pools"])
        return k_train.run_one(name, THREADS, epochs, root, pools[0], pools if len(pools) > 1 else None,
                               a["train"], a["val"], a["edge"], a["n"], k_train.arm_note(name))
    if mode == "p":
        import p_train
        return p_train.run_arm(name, THREADS, epochs, run_root=root)
    if mode == "q":
        import q_train
        a = Q.RUNS[name]
        return q_train.run(name, a, a["seed"], 2000, list(Q.POOLS), K.TRAIN2000, K.VAL2000, epochs or Q.EPOCHS, root,
                           f"reproduction of {name}")
    if mode == "v":
        import q_train
        a = V.RUNS[name]
        return q_train.run(name, a, a["seed"], 2000, V.pools(a["exc"]), K.TRAIN2000, K.VAL2000, epochs or V.EPOCHS,
                           root, f"reproduction of {name}")
    raise ValueError(mode)


def launch(names: list[str], epochs: int | None, root: Path, parallel: int) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONIOENCODING": "utf-8"}
    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    pending, running, res = list(names), {}, {}
    while pending or running:
        while pending and len(running) < parallel:
            n = pending.pop(0)
            log = logs / f"repro_{root.name}_{n}.log"
            fh = log.open("a", encoding="utf-8")
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", n, "--root", str(root)]
            if epochs:
                cmd += ["--epochs", str(epochs)]
            running[n] = (subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT), fh,
                          time.time(), log)
            print(f"[launch] {n} -> {log.relative_to(REPO)}", flush=True)
        for n, (p, fh, t0, log) in list(running.items()):
            if p.poll() is not None:
                fh.close()
                res[n] = {"rc": p.returncode, "wall_s": round(time.time() - t0, 1), "log": log.relative_to(REPO).as_posix()}
                print(f"[exit] {n}: rc {p.returncode}, {res[n]['wall_s'] / 60:.1f} min", flush=True)
                del running[n]
        time.sleep(5)
    return res


def compare(name: str, mode: str) -> dict:
    import k_train
    import t_gates
    import torch
    task, _m, ref = specs()[name]
    d = ROOT / mode / name
    h = d / "training_history.csv"
    if not h.exists():
        return {"run": name, "task": task, "mode": mode, "ran": False, "pass": False}
    n = sum(1 for _ in h.open(encoding="utf-8")) - 1
    out = {"run": name, "task": task, "mode": mode, "ran": True, "epochs": n,
           "reference": ref.relative_to(REPO).as_posix()}
    cmp = t_gates.same_history(h, ref / "training_history.csv", n)
    out.update({"history_pass": cmp["pass"], "history_n_diffs": cmp["n_diffs"], "first_diffs": cmp["first_diffs"]})
    ok = cmp["pass"]
    if mode == "full":
        for ck in ("best_model.pt", "last_model.pt"):
            w = k_train.same_weights(d / "checkpoints" / ck, ref / "checkpoints" / ck)
            out[f"{ck}_identical"] = w["pass"]
            out[f"{ck}_n_differing"] = w["n_differing"]
            ok &= w["pass"]
        sa = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
        sb = json.loads((ref / "run_summary.json").read_text(encoding="utf-8"))
        out["best_epoch"], out["best_epoch_archive"] = sa["best_epoch"], sb["best_epoch"]
        out["best_val_loss_equal"] = sa["best_val_loss"] == sb["best_val_loss"]
        ok &= out["best_val_loss_equal"] and sa["best_epoch"] == sb["best_epoch"]
    out["pass"] = bool(ok)
    return out


def report() -> int:
    import k_common as K
    rows = []
    for mode in ("prefix", "full"):
        if not (ROOT / mode).exists():
            continue
        for d in sorted((ROOT / mode).iterdir()):
            if d.is_dir() and d.name in specs():
                r = compare(d.name, mode)
                r["first_diffs"] = json.dumps(r.get("first_diffs", []))
                rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    K.write_csv(OUT / "training_summary.csv", rows)
    gates = {"runs": {f"{r['mode']}/{r['run']}": r["pass"] for r in rows},
             "pass": bool(rows) and all(r["pass"] for r in rows)}
    K.write_json(OUT / "training_gates.json", gates)
    for r in rows:
        extra = (f"  weights best {r.get('best_model.pt_identical')} last {r.get('last_model.pt_identical')}"
                 f"  best epoch {r.get('best_epoch')}/{r.get('best_epoch_archive')}") if r["mode"] == "full" else ""
        print(f"{r['mode']:<6} {r['task']:<4} {r['run']:<44} epochs {r.get('epochs', 0):5}  history "
              f"{'equal' if r.get('history_pass') else 'DIFFERS'}{extra}  {'PASS' if r['pass'] else 'FAIL'}")
    print("TRAINING PASS" if gates["pass"] else "TRAINING FAIL")
    return 0 if gates["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prefix", action="store_true")
    g.add_argument("--full-t17", action="store_true")
    g.add_argument("--report", action="store_true")
    g.add_argument("--job")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    if a.job:
        return run_job(a.job, a.epochs, a.root)
    if a.report:
        return report()
    sp = specs()
    if a.prefix:
        names = [n for n, (t, m, _) in sp.items() if t in ("T11", "T13", "T16") or (t == "T20" and n.endswith("_s123"))]
        names = [n for n in names if (sp[n][2] / "training_history.csv").exists()]
        if a.only:
            names = [n for n in names if n in a.only]
        res = launch(names, a.epochs or 50, ROOT / "prefix", a.parallel)
    else:
        names = [n for n, (t, m, _) in sp.items() if t == "T17"]
        if a.only:
            names = [n for n in names if n in a.only]
        res = launch(names, None, ROOT / "full", a.parallel)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"training_launch_{'prefix' if a.prefix else 'full'}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return report()


if __name__ == "__main__":
    sys.exit(main())

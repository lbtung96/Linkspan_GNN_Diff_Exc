"""T17 step 2 - train the six scale-up runs (2 arms x 3 seeds, N = 2000, 1000 epochs, own records x4).

    python tools/run_stage.py pipeline/q_train.py --check
    python tools/run_stage.py pipeline/q_train.py --launch

As T16's p_train (fr_train unchanged; T13's augmentation wrapper; p_common's sign-invariant loss and
coherency+orientation edges installed in memory), with the training seed as a parameter.
--check: the seed-parameterised wrapper, seed 123, MSE coherency, T13's N600 K4 pools, 2 epochs,
         reproduces T13's ec_t13_coh_K4_N600 history number for number (results/T17_scale_up/train_check.json).
--launch: all six in parallel, 2 threads each; logs logs/T17_train_<run>.log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import q_common as Q

K, P, C = Q.K, Q.P, Q.C
THREADS = 2


def run(name: str, arm: dict, seed: int, n: int, pools: list[Path], train: Path, val: Path, epochs: int,
        run_root: Path, note: str) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
    import k_train

    if Path(run_root).resolve() == C.RUNS.resolve():
        raise SystemExit("refusing to train into the paper's runs/")
    # (archive: P.install_for(arm) + fr_train.jobs / sys.argv replaced + k_train.install_augmentation)
    return k_train.run_one(name, THREADS, epochs, Path(run_root), Path(pools[0]), list(pools), train, val,
                           arm["edge"], n, note, seed=seed, shape_loss=P.shape_loss(arm),
                           node_feature_mode=P.node_feature_mode(arm), job_epochs=epochs)


def run_job(name: str) -> int:
    a = Q.RUNS[name]
    tr, va = K.read_list(K.TRAIN2000), K.read_list(K.VAL2000)
    for k, p in enumerate(Q.POOLS):
        need = tr + (va if k == 0 else [])
        miss = [s for s in need if not (Path(p) / s).exists()]
        if miss:
            raise SystemExit(f"{p}: {len(miss)} samples missing, e.g. {miss[0]}")
    note = (f"T17 scale-up: {a['label']}; N 2000, {Q.EPOCHS} epochs, seed {a['seed']}; 4 own records per training "
            f"structure (T13 bases), validation pool 0 (base 20261125)")
    return run(name, a, a["seed"], 2000, list(Q.POOLS), K.TRAIN2000, K.VAL2000, Q.EPOCHS, K.EC_RUNS, note)


def check() -> int:
    import t_gates
    root = K.EC_RUNS / "ec_t17_gates"
    name = "g_seedparam_coh_K4_mse"
    shutil.rmtree(root / name, ignore_errors=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONIOENCODING": "utf-8"}
    rc = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--gate-job"], cwd=str(C.REPO), env=env).returncode
    out = {"rc": rc, "pass": False}
    if rc == 0:
        cmp = t_gates.same_history(root / name / "training_history.csv",
                                   K.EC_RUNS / "ec_t13_coh_K4_N600" / "training_history.csv", 2)
        out = {"rc": rc, "reference": "runs/ec/ec_t13_coh_K4_N600", **cmp}
    Q.T17_RESULTS.mkdir(parents=True, exist_ok=True)
    K.write_json(Q.T17_RESULTS / "train_check.json", out)
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


def launch(names: list[str]) -> int:
    for f in ("train_check.json", "prepare_summary.json"):
        if not json.loads((Q.T17_RESULTS / f).read_text(encoding="utf-8")).get("pass"):
            raise SystemExit(f"{f} does not pass")
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONIOENCODING": "utf-8"}
    procs = {}
    for n in names:
        log = Q.T17_LOGS / f"T17_train_{n}.log"
        fh = log.open("a", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--job", n], cwd=str(C.REPO),
                             env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs[n] = (p, fh, time.time(), log)
        print(f"[launch] {n} pid {p.pid} -> {log}", flush=True)
    res = {}
    for n, (p, fh, t0, log) in procs.items():
        rc = p.wait()
        fh.close()
        res[n] = {"rc": rc, "wall_s": round(time.time() - t0, 1), "log": str(log.relative_to(C.REPO))}
        print(f"[exit] {n}: rc {rc}, {res[n]['wall_s'] / 60:.1f} min", flush=True)
    K.write_json(Q.T17_RESULTS / "train_launch.json", res)
    return 0 if all(v["rc"] == 0 for v in res.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--gate-job", action="store_true")
    g.add_argument("--launch", action="store_true")
    g.add_argument("--job")
    ap.add_argument("--runs", nargs="+", default=list(Q.RUNS), choices=list(Q.RUNS))
    a = ap.parse_args()
    if a.check:
        return check()
    if a.gate_job:
        return run("g_seedparam_coh_K4_mse", {"edge": "cpsd_full", "loss": "mse", "amp": False}, 123, 600,
                   list(P.POOLS_K4), K.TRAIN600, K.VAL600, 2, K.EC_RUNS / "ec_t17_gates", "T17 trainer check")
    if a.launch:
        return launch(a.runs)
    return run_job(a.job)


if __name__ == "__main__":
    sys.exit(main())

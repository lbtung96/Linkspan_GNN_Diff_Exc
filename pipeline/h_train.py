"""T11 step 2 - train the two varied-excitation arms with the paper's own trainer.

    python tools/run_stage.py pipeline/h_train.py --launch gate
    python tools/run_stage.py pipeline/h_train.py --launch arms

As T1's t_train.py does, fr_train.main() is called unchanged with in-memory substitutions:
fr_train.jobs() returns only the job, and fr_train's own options --pool (the sample folder),
--train-list / --val-list (the paper's N600 manifests) and --run-root (EC_comments/runs) are
set.  fr_train installs fr_edge_modes itself (the geom_orient mode).  Model, hyperparameters,
loss, Adam, batch 16, seed 123, 5000 epochs, best-validation rule, resume: fr_train's.

  ec_varexc_edgegat_N600              edgegat, cpsd_full   (as edgegat_N600)
  ec_varexc_edgegat_geomorient_N600   edgegat, geom_orient (as edgegat_geomorient_N600)

both on data/T11/trainval_varexc/samples (the same 508 + 92 structures, a new excitation
record per structure; h_prepare.py).

--launch gate: the paper's jobs edgegat_N600 and edgegat_geomorient_N600, through this
wrapper, on the paper's own pool, 3 epochs, into runs/ec/ec_varexc_gates/; their
training_history.csv must equal the first 3 epochs of the paper runs' number for number
(T1's t_gates.same_history: every column but n_epochs).  -> results/.../train_gate.json
--launch arms: both arms in parallel, 2 torch threads each (CLAUDE.md), logs in logs.
--job NAME: one training in this process (what --launch runs as a subprocess).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h_common as H

C = H.C
GATE_ROOT = H.EC_RUNS / "ec_varexc_gates"
THREADS = 2
GATE_JOBS = ("edgegat_N600", "edgegat_geomorient_N600")


def make_job(name: str):
    import fr_train
    if name in H.ARMS:
        return fr_train.Job(name, "edgegat", 600, H.ARMS[name]["edge_feature_mode"],
                            note=f"T11 varied-excitation pilot: {H.ARMS[name]['label']}; "
                                 f"same structures and split as {H.ARMS[name]['paper_twin']}, "
                                 f"excitation SeedSequence([{H.TRAIN_EXC_BASE}, sample_index])")
    return fr_train.jobs()[name]


def run_one(name: str, threads: int, epochs: int | None, gate: bool) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    import fr_train

    job = make_job(name)
    if gate:
        if name not in GATE_JOBS:
            raise SystemExit(f"gate jobs: {GATE_JOBS}")
        pool, run_root = C.POP_BROADBAND, GATE_ROOT
    else:
        if name not in H.ARMS:
            raise SystemExit(f"arms: {sorted(H.ARMS)}")
        pool, run_root = H.TRAINVAL_DIR, H.EC_RUNS
        n_tr, n_va = len(H.read_list(H.TRAIN_LIST)), len(H.read_list(H.VAL_LIST))
        have = len(list(pool.glob("sample_*_phase1_gnn.npz")))
        if have < n_tr + n_va:
            raise SystemExit(f"{pool}: {have} samples, need {n_tr + n_va} (run h_prepare.py --part data)")
    if Path(run_root).resolve() == C.RUNS.resolve():
        raise SystemExit("refusing to train into the paper's runs/")
    fr_train.jobs = lambda: {job.name: job}
    sys.argv = ["fr_train.py", "--job", job.name, "--threads", str(threads), "--resume-every", "25",
                "--run-root", str(run_root), "--train-list", str(H.TRAIN_LIST), "--val-list", str(H.VAL_LIST),
                "--pool", str(pool)] + (["--epochs", str(epochs)] if epochs else [])
    return fr_train.main()


def launch(names: list[str], gate: bool, epochs: int | None) -> dict:
    H.T11_LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONIOENCODING": "utf-8"}
    procs = {}
    for n in names:
        log = H.T11_LOGS / f"T11_train_{'gate_' if gate else ''}{n}.log"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", n, "--threads", str(THREADS)]
        if gate:
            cmd += ["--gate"]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        fh = log.open("a", encoding="utf-8")
        procs[n] = (subprocess.Popen(cmd, cwd=str(C.REPO), env=env, stdout=fh, stderr=subprocess.STDOUT), fh,
                    time.time(), log)
        print(f"[launch] {n} pid {procs[n][0].pid} -> {log}", flush=True)
    out = {}
    for n, (p, fh, t0, log) in procs.items():
        rc = p.wait()
        fh.close()
        out[n] = {"rc": rc, "wall_s": round(time.time() - t0, 1), "log": str(log.relative_to(C.REPO))}
        print(f"[exit] {n}: rc {rc}, {out[n]['wall_s'] / 3600:.2f} h", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--job")
    g.add_argument("--launch", choices=["gate", "arms"])
    ap.add_argument("--gate", action="store_true", help="with --job: a paper gate job on the paper pool")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--epochs", type=int, default=None, help="override (gate / smoke only)")
    args = ap.parse_args()
    if args.job:
        return run_one(args.job, args.threads, args.epochs, args.gate)

    if args.launch == "gate":
        import shutil
        import t_gates                                       # T1's history comparison
        for n in GATE_JOBS:
            shutil.rmtree(GATE_ROOT / n, ignore_errors=True)
        res = launch(list(GATE_JOBS), gate=True, epochs=3)
        out = {"rule": "first 3 epochs of training_history.csv equal number for number (t_gates.HIST_KEYS)",
               "threads": THREADS, "runs": {}}
        for n in GATE_JOBS:
            cmp = t_gates.same_history(GATE_ROOT / n / "training_history.csv",
                                       C.RUNS / n / "training_history.csv", 3) if res[n]["rc"] == 0 else {"pass": False}
            out["runs"][n] = {**res[n], **cmp}
            print(f"[gate] {n}: {'PASS' if cmp['pass'] else 'FAIL'} {cmp}", flush=True)
        out["pass"] = all(v["pass"] for v in out["runs"].values())
        H.write_json(H.T11_RESULTS / "train_gate.json", out)
        print("TRAIN GATE PASS" if out["pass"] else "TRAIN GATE FAIL")
        return 0 if out["pass"] else 1

    gate = json.loads((H.T11_RESULTS / "train_gate.json").read_text(encoding="utf-8"))
    if not gate.get("pass"):
        raise SystemExit("train_gate.json does not pass: run --launch gate first")
    summ = H.T11_RESULTS / "prepare_summary.json"
    if not summ.exists() or not json.loads(summ.read_text(encoding="utf-8")).get("pass"):
        raise SystemExit("prepare_summary.json missing or failing: run h_prepare.py --part data first")
    res = launch(list(H.ARMS), gate=False, epochs=args.epochs)
    H.write_json(H.T11_RESULTS / "train_launch.json", res)
    return 0 if all(v["rc"] == 0 for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

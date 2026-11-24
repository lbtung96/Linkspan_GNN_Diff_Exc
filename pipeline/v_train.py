"""T20 step 3 - train T17's best arm on deck forcing + noise: 2 excitations x 3 seeds, N 2000, 1000 epochs.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/v_train.py --launch

coherency + orientation edges (802-d), sign-invariant loss (p_common, in memory), four records per training
structure drawn one per epoch (k_train's augmentation), validation on pool 0, best-validation checkpoint - T17's
set-up exactly, through T17's own wrapper (q_train.run, whose trainer check passed: T17 train_check.json),
with only the record pools changed to v_common.pools(exc).
--launch: all six in parallel, 2 threads each; logs EC_comments/logs/T20_train_<run>.log;
          results/T20_deck_forcing/train_launch.json.  Refuses to start unless prepare_summary.json passes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import v_common as V

K, C = V.K, V.C
THREADS = 2


def run_job(name: str) -> int:
    import q_train as QT

    a = V.RUNS[name]
    pools = V.pools(a["exc"])
    tr, va = K.read_list(K.TRAIN2000), K.read_list(K.VAL2000)
    for k, p in enumerate(pools):
        need = tr + (va if k == 0 else [])
        miss = [s for s in need if not (Path(p) / s).exists()]
        if miss:
            raise SystemExit(f"{p}: {len(miss)} samples missing, e.g. {miss[0]}")
    al = V.alpha()[a["exc"]]
    note = (f"T20: {a['label']} (noise ALPHA {al:.4f}); N 2000, {V.EPOCHS} epochs, seed {a['seed']}; 4 records per "
            f"training structure (bases {V.BASES[a['exc']]}), validation pool 0")
    return QT.run(name, a, a["seed"], 2000, pools, K.TRAIN2000, K.VAL2000, V.EPOCHS, K.EC_RUNS, note)


def launch(names: list[str]) -> int:
    if not json.loads((V.T20_RESULTS / "prepare_summary.json").read_text(encoding="utf-8")).get("pass"):
        raise SystemExit("prepare_summary.json does not pass")
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONIOENCODING": "utf-8"}
    procs = {}
    for n in names:
        log = V.T20_LOGS / f"T20_train_{n}.log"
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
    K.write_json(V.T20_RESULTS / "train_launch.json", res)
    return 0 if all(v["rc"] == 0 for v in res.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--launch", action="store_true")
    g.add_argument("--job")
    ap.add_argument("--runs", nargs="+", default=list(V.RUNS), choices=list(V.RUNS))
    a = ap.parse_args()
    if a.launch:
        return launch(a.runs)
    return run_job(a.job)


if __name__ == "__main__":
    sys.exit(main())

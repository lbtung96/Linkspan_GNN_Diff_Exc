"""T17 step 1 - records 1-3 for the 1700 N2000 training structures.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/q_prepare.py [--workers 14]

Structures that T13 already simulated (its rec1..rec3, the N600 training list) are hard-linked
(npz and check file) - the same seed, so the same file; every other one goes through T13's own
k_prepare.data_job (re-simulation with SeedSequence([base_k, i]), every non-excitation key copied
and checked byte-identical, labels re-solved within 1e-5, features row-normalised).  Writes
results/T17_scale_up/prepare_summary.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import q_common as Q  # noqa: F401  (registers the T17 sets with k_common before k_prepare is used)
import k_prepare as KP

K, C = Q.K, Q.C


def job(j: tuple) -> dict:
    return KP.data_job(j)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    Q.T17_RESULTS.mkdir(parents=True, exist_ok=True)
    names = K.read_list(K.TRAIN2000)
    todo, linked = [], {}
    for k, key in Q.REC_KEYS.items():
        d = K.sdir(key)
        d.mkdir(parents=True, exist_ok=True)
        src = K.T13_DATA / f"rec{k}" / "samples"
        linked[key] = 0
        for n in names:
            s_npz, s_chk = src / n, src / (Path(n).stem + ".check.json")
            t_npz, t_chk = d / n, d / (Path(n).stem + ".check.json")
            if s_npz.exists() and s_chk.exists():
                for s, t in ((s_npz, t_npz), (s_chk, t_chk)):
                    if not t.exists():
                        os.link(s, t)
                linked[key] += 1
            else:
                todo.append((n, key))
    print(f"[prepare] linked {linked}; to simulate {len(todo)}", flush=True)
    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, todo, chunksize=2), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"[prepare] {i}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    bad = [r["sample"] + "/" + r["set"] for r in rows if not r["pass"]]
    complete = {key: sum((K.sdir(key) / n).exists() for n in names) for key in Q.REC_KEYS.values()}
    out = {"linked_from_t13": linked, "simulated": len(rows), "failed": bad[:20], "n_failed": len(bad),
           "complete": complete, "pass": not bad and all(v == len(names) for v in complete.values()),
           "wall_s": round(time.time() - t0, 1)}
    K.write_json(Q.T17_RESULTS / "prepare_summary.json", out)
    print(out, flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

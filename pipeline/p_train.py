"""T16 - train the six sign-test arms (400 epochs) through T13's trainer wrapper.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/p_train.py --check
    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/p_train.py --launch

--check   (results/T16_sign_test/train_checks.json), must pass before --launch:
  1. sign-invariant loss: equals a brute-force min over the 2^(graphs x modes) sign choices on a random
     batch; invariant to flipping any structure's target mode; equals MSE when no flip helps; <= MSE.
  2. coherency + orientation edges: 802 columns = the library's 800-d coherency then fr_edge_modes'
     orientation, on three samples.
  3. amplitude input: deck maximum 1, every support row maximum 1, and at each target-mode bin the
     deck values correlate with |phi| of that mode (median over 50 training structures reported).
  4. trainer unchanged when nothing is switched on: an MSE coherency K4 job through this wrapper
     (with the edge-mode patch installed), 2 epochs, gives T13's ec_t13_coh_K4_N600 history for
     those epochs number for number.
--launch  the six arms in parallel, 2 threads each, logs in EC_comments/logs/T16_train_<arm>.log.
--job     one arm in this process (what --launch runs).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import p_common as P

K, H, C = P.K, P.H, P.C
THREADS = 2


def note(name: str) -> str:
    a = P.ARMS[name]
    return (f"T16 sign test (400 epochs): {a['label']}; edges {a['edge']}, loss {a['loss']}, amplitude nodes {a['amp']}; "
            f"4 own records per training structure (T13 pools), validation pool 0")


def run_arm(name: str, threads: int, epochs: int | None, run_root: Path | None = None, arm: dict | None = None) -> int:
    import k_train
    a = arm or P.ARMS[name]
    P.install_for(a)
    pools = list(P.POOLS_K4)
    return k_train.run_one(name, threads, epochs, run_root or K.EC_RUNS, pools[0], pools, K.TRAIN600, K.VAL600,
                           a["edge"], 600, note(name) if name in P.ARMS else "T16 trainer check")


# ------------------------------------------------------------------ checks
def check_loss() -> dict:
    import torch
    g = torch.Generator().manual_seed(0)
    bnn = torch.tensor([3, 5, 4])
    p = torch.randn(12, 4, generator=g)
    y = torch.randn(12, 4, generator=g)
    seg = np.repeat(np.arange(3), bnn.numpy())
    best = np.inf
    for signs in itertools.product((1.0, -1.0), repeat=12):
        s = np.array(signs).reshape(3, 4)[seg]
        best = min(best, float(((p.numpy() - s * y.numpy()) ** 2).sum()))
    best /= p.numel()
    li = float(P.signinv_shape_loss(p, y, bnn))
    y2 = y.clone()
    y2[3:8, 1] *= -1
    li2 = float(P.signinv_shape_loss(p, y2, bnn))
    same = float(P.signinv_shape_loss(y, y, bnn))
    mse = float(((p - y) ** 2).mean())
    near = y + 0.01 * torch.randn(12, 4, generator=g)
    ok = (abs(li - best) < 1e-6 and abs(li2 - li) < 1e-6 and same == 0.0 and li <= mse + 1e-9
          and abs(float(P.signinv_shape_loss(near, y, bnn)) - float(((near - y) ** 2).mean())) < 1e-9)
    return {"signinv": li, "brute_force": best, "after_flipping_one_structure_mode": li2, "mse": mse, "pass": bool(ok)}


def check_edges() -> dict:
    import EdgeGATConv_Modal_GNN as eg
    import fr_edge_modes
    P.install_cohorient()
    out = []
    for n in K.read_list(K.TRAIN600)[:3]:
        with np.load(P.POOLS_K4[1] / n, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        e = eg._resolve_edge_attr_array(data, P.COH_ORIENT)
        ok = (e.shape[1] == 802 and np.array_equal(e[:, :800], np.asarray(data["edge_attr_cpsd_full"], np.float32))
              and np.array_equal(e[:, 800:], fr_edge_modes.orientation(data)))
        out.append(ok)
    return {"samples": 3, "pass": all(out)}


def check_amp() -> dict:
    corr, ok = [], True
    for n in K.read_list(K.TRAIN600)[:50]:
        with np.load(P.POOLS_K4[0] / n, allow_pickle=True) as d:
            x, s, ei = d["x_psd"], d["x_psd_row_scale"], d["edge_index"]
            phi, f, fb = d["y_mode_shape_signed"], d["y_freq_hz"], d["freq_bins_hz"]
        a = P.amp_features(x, s, ei)
        sup = np.bincount(ei[0], minlength=a.shape[0]) == 1
        ok &= abs(a[~sup].max() - 1) < 1e-6 and np.allclose(a[sup].max(axis=1), 1, atol=1e-6)
        ok &= abs(np.abs(x).max(axis=1) - 1).max() < 1e-6          # x_psd is row-normalised: scale = row max
        b = np.rint(np.asarray(f, float) / (fb[1] - fb[0])).astype(int)
        corr.append([np.corrcoef(a[~sup, b[k]], np.abs(phi[~sup, k]))[0, 1] for k in range(4)])
    med = np.nanmedian(np.array(corr), axis=0)
    return {"structures": 50, "deck_max_and_support_rowmax_1": bool(ok),
            "median_corr_deck_amplitude_vs_abs_phi_at_mode_bin": dict(zip(C.MODE_NAMES, map(float, med))),
            "pass": bool(ok and np.all(med > 0.5))}


def check_trainer() -> dict:
    import t_gates
    root = K.EC_RUNS / "ec_t16_gates"
    name = "g_coh_K4_mse"
    shutil.rmtree(root / name, ignore_errors=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONIOENCODING": "utf-8"}
    rc = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--gate-job", name], cwd=str(C.REPO),
                        env=env).returncode
    if rc:
        return {"rc": rc, "pass": False}
    cmp = t_gates.same_history(root / name / "training_history.csv", K.EC_RUNS / "ec_t13_coh_K4_N600" / "training_history.csv", 2)
    return {"rc": rc, "reference": "EC_comments/runs/ec_t13_coh_K4_N600", **cmp}


def checks() -> int:
    P.T16_RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"loss": check_loss(), "edges": check_edges(), "amp": check_amp(), "trainer": check_trainer()}
    out["pass"] = all(v["pass"] for v in out.values())
    K.write_json(P.T16_RESULTS / "train_checks.json", out)
    print(json.dumps(out, indent=1, default=str))
    print("CHECKS PASS" if out["pass"] else "CHECKS FAIL")
    return 0 if out["pass"] else 1


def launch(names: list[str]) -> int:
    chk = json.loads((P.T16_RESULTS / "train_checks.json").read_text(encoding="utf-8"))
    if not chk.get("pass"):
        raise SystemExit("train_checks.json does not pass")
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONIOENCODING": "utf-8"}
    procs = {}
    for n in names:
        log = P.T16_LOGS / f"T16_train_{n}.log"
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
    K.write_json(P.T16_RESULTS / "train_launch.json", res)
    return 0 if all(v["rc"] == 0 for v in res.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--launch", action="store_true")
    g.add_argument("--job")
    g.add_argument("--gate-job")
    ap.add_argument("--arms", nargs="+", default=list(P.ARMS), choices=list(P.ARMS))
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    if a.check:
        return checks()
    if a.launch:
        return launch(a.arms)
    if a.gate_job:
        return run_arm(a.gate_job, THREADS, 2, K.EC_RUNS / "ec_t16_gates",
                       {"edge": "cpsd_full", "loss": "mse", "amp": False})
    return run_arm(a.job, THREADS, a.epochs)


if __name__ == "__main__":
    sys.exit(main())

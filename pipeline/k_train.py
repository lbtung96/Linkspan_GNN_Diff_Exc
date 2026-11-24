"""T13 step 2 - train the five feasibility arms (400 epochs) with the paper's own trainer.

    python tools/run_stage.py pipeline/k_train.py --launch gate
    python tools/run_stage.py pipeline/k_train.py --launch arms

Every arm runs fr_train.train() (EdgeGAT hyperparameters, seed 123, epochs 400) with its own pool,
train / validation lists and run root, 2 torch threads.  Best-validation rule, resume, logging:
fr_train's.

Augmentation (K > 1): fr_train.train(pools=...) uses records.py (in the archive, two in-memory
substitutions installed before fr_train.main(), with the same code):
  * the training set is a records.VariantSet holding K library datasets, one per record folder
    (pools[k]); the validation set is the library class unchanged.
  * the training loader is records.EpochLoader, whose __iter__ first calls VariantSet.set_epoch().
    fr_train iterates the training loader exactly once per epoch, so this is the per-epoch draw of
    T4's sp_train:
        slot[i] = default_rng([20261130, epoch]).integers(0, K, n_train)
    a numpy Generator of its own - torch's stream (shuffling, dropout) and the global numpy /
    python RNGs are untouched.  VariantSet[i] = the library dataset of record slot[i], item i.
  The K datasets must carry identical graphs (edges), node labels and frequency labels; checked
  at construction.  The draw applied each epoch is logged to <run>/variant_draws.csv.

--launch gate  (results/T13_feasibility/train_gate.json)
  (d) the paper's edgegat_N600 and edgegat_N2000 jobs through this wrapper on the paper pool,
      3 epochs: training_history.csv equals the paper runs' first 3 epochs number for number
      (T1's t_gates.same_history).  T11's train_gate.json (the same wrapper idea, edgegat_N600
      and edgegat_geomorient_N600) is re-read and reported beside it.
  (e) on T11's varied-record pool (N600), 2 epochs: the plain trainer vs the augmentation
      trainer with K = 1, and with K = 4 where all four folders are the same one (so the
      draws switch datasets but the data are equal): history equal number for number, and
      best_model.pt / last_model.pt weights bit-identical.
--launch arms  all five arms in parallel, 2 threads each, logs in logs.
--job NAME     one arm in this process (what --launch runs).
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
from pathlib import Path

import k_common as K

C, H = K.C, K.H
THREADS = 2


def run_one(name: str, threads: int, epochs: int | None, run_root: Path, pool0: Path, pools: list[Path] | None,
            train_list: Path, val_list: Path, edge: str, n: int, note: str, *, seed: int | None = None,
            shape_loss: str = "mse", node_feature_mode: str = "psd", job_epochs: int | None = None) -> int:
    """One EdgeGAT run through fr_train.train.  pools: the K record folders (pools[0] == pool0), or None
    for one record.  (Archive: fr_train.main() with fr_train.jobs and sys.argv replaced, and the record
    draw installed in memory by install_augmentation - now records.py, used by fr_train.train.)"""
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    import fr_train

    if Path(run_root).resolve() == C.RUNS.resolve():
        raise SystemExit("refusing to train into the paper's runs/")
    job = fr_train.Job(name, "edgegat", n, edge, epochs=K.EPOCHS if job_epochs is None else int(job_epochs),
                       seed=C.SEED if seed is None else int(seed), note=note, shape_loss=shape_loss,
                       node_feature_mode=node_feature_mode)
    return fr_train.train(job, threads=threads, resume_every=25, epochs=epochs, run_root=Path(run_root),
                          train_list=train_list, val_list=val_list, pool=Path(pool0),
                          pools=list(pools) if pools is not None else None, epoch_seed=K.EPOCH_SEED)


def arm_note(name: str) -> str:
    a = K.ARMS[name]
    return (f"T13 feasibility (400 epochs): {a['label']}; lever: {a['lever']}; records per training structure "
            f"{len(a['pools'])} (SeedSequence([base_k, sample_index]), bases {list(K.BASES[:len(a['pools'])])}), "
            f"{a['t_total']} s; validation: pool 0")


def run_arm(name: str, threads: int, epochs: int | None) -> int:
    a = K.ARMS[name]
    pools = list(a["pools"])
    tr, va = K.read_list(a["train"]), K.read_list(a["val"])
    for k, p in enumerate(pools):
        need = tr + (va if k == 0 else [])
        miss = [s for s in need if not (Path(p) / s).exists()]
        if miss:
            raise SystemExit(f"{p}: {len(miss)} samples missing, e.g. {miss[0]} (run k_prepare.py --part data)")
    return run_one(name, threads, epochs, K.EC_RUNS, pools[0], pools if len(pools) > 1 else None,
                   a["train"], a["val"], a["edge"], a["n"], arm_note(name))


# gate jobs: name -> (pool0, pools or None, train, val, n, paper run to compare or None)
def gate_jobs() -> dict:
    return {
        "d_edgegat_N600": (C.POP_BROADBAND, None, K.TRAIN600, K.VAL600, 600, "edgegat_N600", 3),
        "d_edgegat_N2000": (C.POP_BROADBAND, None, K.TRAIN2000, K.VAL2000, 2000, "edgegat_N2000", 3),
        "d_aug_K1_edgegat_N600": (C.POP_BROADBAND, [C.POP_BROADBAND], K.TRAIN600, K.VAL600, 600, "edgegat_N600", 3),
        "e_plain": (K.T11_TRAINVAL, None, K.TRAIN600, K.VAL600, 600, None, 2),
        "e_aug_K1": (K.T11_TRAINVAL, [K.T11_TRAINVAL], K.TRAIN600, K.VAL600, 600, None, 2),
        "e_aug_K4_same": (K.T11_TRAINVAL, [K.T11_TRAINVAL] * 4, K.TRAIN600, K.VAL600, 600, None, 2),
    }


def run_gate_job(name: str, threads: int) -> int:
    pool0, pools, tr, va, n, _ref, ep = gate_jobs()[name]
    return run_one(name, threads, ep, K.GATE_ROOT, pool0, pools, tr, va, "cpsd_full", n, "T13 trainer gate")


def launch(names: list[str], gate: bool) -> dict:
    K.T13_LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
           "OPENBLAS_NUM_THREADS": str(THREADS), "PYTHONIOENCODING": "utf-8"}
    procs = {}
    for n in names:
        log = K.T13_LOGS / f"T13_train_{'gate_' if gate else ''}{n}.log"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", n, "--threads", str(THREADS)]
        if gate:
            cmd += ["--gate"]
        fh = log.open("a", encoding="utf-8")
        procs[n] = (subprocess.Popen(cmd, cwd=str(C.REPO), env=env, stdout=fh, stderr=subprocess.STDOUT), fh,
                    time.time(), log)
        print(f"[launch] {n} pid {procs[n][0].pid} -> {log}", flush=True)
    out = {}
    for n, (p, fh, t0, log) in procs.items():
        rc = p.wait()
        fh.close()
        out[n] = {"rc": rc, "wall_s": round(time.time() - t0, 1), "log": str(log.relative_to(C.REPO))}
        print(f"[exit] {n}: rc {rc}, {out[n]['wall_s'] / 60:.1f} min", flush=True)
    return out


def same_weights(a: Path, b: Path) -> dict:
    import torch
    sa = torch.load(a, map_location="cpu", weights_only=False)
    sb = torch.load(b, map_location="cpu", weights_only=False)
    ma, mb = sa["model_state_dict"], sb["model_state_dict"]
    bad = [k for k in ma if k not in mb or not torch.equal(ma[k], mb[k])]
    return {"tensors": len(ma), "differing": bad[:5], "n_differing": len(bad) + len(set(mb) - set(ma)),
            "epoch": [sa.get("epoch"), sb.get("epoch")], "pass": not bad and set(ma) == set(mb)}


def gate() -> int:
    import t_gates
    for n in gate_jobs():
        shutil.rmtree(K.GATE_ROOT / n, ignore_errors=True)
    res = launch(list(gate_jobs()), gate=True)
    out = {"threads": THREADS, "d_paper_history": {}, "e_augmentation_vs_plain": {}}
    for n, (_p0, _pools, _tr, _va, _n, ref, ep) in gate_jobs().items():
        if ref is None:
            continue
        cmp = t_gates.same_history(K.GATE_ROOT / n / "training_history.csv", C.RUNS / ref / "training_history.csv",
                                   ep) if res[n]["rc"] == 0 else {"pass": False}
        out["d_paper_history"][n] = {**res[n], "reference": f"runs/paper/{ref}", **cmp}
    t11 = json.loads((H.T11_RESULTS / "train_gate.json").read_text(encoding="utf-8"))
    out["d_t11_gate_reused"] = {"pass": t11["pass"], "runs": {k: v["pass"] for k, v in t11["runs"].items()},
                                "source": "results/T11_varied_excitation_pilot/train_gate.json"}
    base = K.GATE_ROOT / "e_plain"
    for n in ("e_aug_K1", "e_aug_K4_same"):
        d = K.GATE_ROOT / n
        if res[n]["rc"] or res["e_plain"]["rc"]:
            out["e_augmentation_vs_plain"][n] = {**res[n], "pass": False}
            continue
        h = t_gates.same_history(d / "training_history.csv", base / "training_history.csv", 2)
        wb = same_weights(d / "checkpoints" / "best_model.pt", base / "checkpoints" / "best_model.pt")
        wl = same_weights(d / "checkpoints" / "last_model.pt", base / "checkpoints" / "last_model.pt")
        draws = K.read_csv(d / "variant_draws.csv")
        out["e_augmentation_vs_plain"][n] = {**res[n], "history": h, "best_model": wb, "last_model": wl,
                                            "draws_logged": draws,
                                            "pass": h["pass"] and wb["pass"] and wl["pass"] and len(draws) == 2}
    out["d_pass"] = all(v["pass"] for v in out["d_paper_history"].values())
    out["e_pass"] = all(v["pass"] for v in out["e_augmentation_vs_plain"].values())
    out["pass"] = out["d_pass"] and out["e_pass"]
    K.write_json(K.T13_RESULTS / "train_gate.json", out)
    print(json.dumps(out, indent=1, default=str)[:4000])
    print("TRAIN GATE PASS" if out["pass"] else "TRAIN GATE FAIL")
    return 0 if out["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--job")
    g.add_argument("--launch", choices=["gate", "arms"])
    ap.add_argument("--gate", action="store_true", help="with --job: a gate job")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke only)")
    ap.add_argument("--arms", nargs="+", default=list(K.ARMS), choices=list(K.ARMS))
    args = ap.parse_args()
    if args.job:
        if args.gate:
            return run_gate_job(args.job, args.threads)
        return run_arm(args.job, args.threads, args.epochs)
    if args.launch == "gate":
        return gate()
    tg = json.loads((K.T13_RESULTS / "train_gate.json").read_text(encoding="utf-8"))
    pg = json.loads((K.T13_RESULTS / "prepare_gates.json").read_text(encoding="utf-8"))
    ps = json.loads((K.T13_RESULTS / "prepare_summary.json").read_text(encoding="utf-8"))
    if not (tg.get("pass") and pg.get("pass")):
        raise SystemExit("train_gate.json or prepare_gates.json does not pass")
    bad = [k for k in K.DATASETS if not ps.get(k, {}).get("pass")]
    if bad:
        raise SystemExit(f"prepare_summary.json: sets not passing {bad}")
    res = launch(args.arms, gate=False)
    path = K.T13_RESULTS / "train_launch.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(res)
    K.write_json(path, old)
    return 0 if all(v["rc"] == 0 for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

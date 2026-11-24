"""T16 - score the six sign-test arms and T13's two references.

    python tools/run_stage.py pipeline/p_eval.py [--workers 8]

Synthetic (best-validation checkpoint, CPU, 2 torch threads, one process per (net, set) so an arm's
in-memory input patch never leaks into another's):
  paper_varexc  500 paper test structures, a record of their own each (T11)
  fresh_test    500 new structures, one shared new record 20261124 (T5)
  paper         500 paper test structures, the paper record 12345
Per structure and mode: MAC, signed cosine, frequency error, and the prediction's RMS (to see a
near-zero output).  Gate: the two T13 references reproduce T13's eval_summary.csv (mean MAC per
set, 1e-5; T13 scored them through c_rescore, this script directly).

Real bridges (zero-shot, s5_zeroshot.load_structures / evaluate: Phase P field inputs, hybrid graphs,
SSI labels and FE targets, torch's default threads as s5): every arm except the amplitude arm (the
field inputs are row-normalised with no scale).  Edge inputs per mode: coherency as stored, orientation
from each hybrid graph's edge_attr_geom (as h_real), zeros (2-d), or both side by side.  Gate: the
two references reproduce T13's real_zeroshot.csv exactly.

Writes results/T16_sign_test/: eval_per_sample.csv, eval_summary.csv, eval_contrasts.csv,
real_zeroshot.csv, real_summary.csv, eval_gates.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import p_common as P

K, H, C = P.K, P.H, P.C
MODES = C.MODE_NAMES
SETS = ("paper_varexc", "fresh_test", "paper")
CONTRASTS = [
    ("ec_t16_coh_signinv", "ec_t16_none_signinv", "coherency vs no edges, both sign-invariant"),
    ("ec_t16_coh_signinv", "ec_t13_coh_K4_N600", "coherency: sign-invariant vs MSE"),
    ("ec_t16_cohorient", "ec_t13_orient_K4_N600", "MSE: coherency+orientation vs orientation"),
    ("ec_t16_cohorient_signinv", "ec_t16_orient_signinv", "sign-invariant: coherency+orientation vs orientation"),
    ("ec_t16_orient_signinv", "ec_t13_orient_K4_N600", "orientation: sign-invariant vs MSE"),
    ("ec_t16_cohorient_signinv", "ec_t16_coh_signinv", "sign-invariant: coherency+orientation vs coherency"),
    ("ec_t16_coh_signinv_amp", "ec_t16_coh_signinv", "coherency sign-invariant: amplitude nodes vs row-normalised"),
]


def set_files(ts: str) -> list[Path]:
    names = K.read_list(K.TEST_LIST)
    if ts == "paper_varexc":
        return [K.T11_TEST / n for n in names]
    if ts == "paper":
        return [C.POP_BROADBAND / n for n in names]
    import c_rescore as CR
    return CR.set_files(ts)


def score(task: tuple) -> list[dict]:
    net, ts = task
    import torch
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs

    torch.set_num_threads(2)
    arm = P.ALL[net]
    P.install_for(arm)
    rd = K.EC_RUNS / net
    info = json.loads((rd / "run_summary.json").read_text(encoding="utf-8"))
    if info["edge_feature_mode"] != arm["edge"]:
        raise RuntimeError(f"{net}: run edge mode {info['edge_feature_mode']} != {arm['edge']}")
    model, ckpt = eg.load_trained_model_checkpoint(rd / "checkpoints" / "best_model.pt", device=torch.device("cpu"))
    model.eval()
    stats = gs.FrequencyNormalizationStats(freq_mean_hz=np.asarray(ckpt["freq_mean_hz"], np.float32),
                                           freq_std_hz=np.asarray(ckpt["freq_std_hz"], np.float32))
    files = set_files(ts)
    ds = eg.Phase2AEdgeDataset(files, edge_feature_mode=arm["edge"], shape_target_mode="signed", expected_n_modes=4)
    rows = []
    with torch.no_grad():
        for i, f in enumerate(files):
            g, ft = ds[i]
            o = model(g, g.ndata["x"], g.edata["edge_x"])
            p, y = o.pred_mode_shape_abs.numpy().astype(np.float64), g.ndata["y"].numpy().astype(np.float64)
            fp = gs.denormalize_frequency_predictions(o.pred_freq, stats).numpy().reshape(-1)
            r = {"test_set": ts, "run": net, "sample": f.name, "Ng": int(f.name.split("_Ng")[1][0])}
            for j, m in enumerate(MODES):
                c = float(p[:, j] @ y[:, j] / (np.linalg.norm(p[:, j]) * np.linalg.norm(y[:, j]) + 1e-12))
                r[f"mac_{m}"], r[f"cos_{m}"] = c * c, c
                r[f"rms_{m}"] = float(np.sqrt(np.mean(p[:, j] ** 2)))
                r[f"freq_abs_err_{m}_hz"] = float(abs(fp[j] - float(ft[j])))
            r["mac_mean"] = float(np.mean([r[f"mac_{m}"] for m in MODES]))
            r["freq_mae_hz"] = float(np.mean([r[f"freq_abs_err_{m}_hz"] for m in MODES]))
            rows.append(r)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["test_set"], r["run"]) for r in rows}, key=lambda k: (SETS.index(k[0]), list(P.ALL).index(k[1])))
    for ts, run in keys:
        g = [r for r in rows if r["test_set"] == ts and r["run"] == run]
        s = {"test_set": ts, "run": run, "label": P.ALL[run]["label"], "n": len(g)}
        for m in MODES:
            s[f"mac_{m}"] = float(np.mean([r[f"mac_{m}"] for r in g]))
        v = np.array([r["mac_mean"] for r in g])
        s["mac_mean"] = float(v.mean())
        s["mac_mean_lo"], s["mac_mean_hi"] = K.bootstrap_ci(v)
        s["freq_mae_hz"] = float(np.mean([r["freq_mae_hz"] for r in g]))
        for m in MODES:
            s[f"rms_ratio_{m}_to_B1"] = float(np.mean([r[f"rms_{m}"] for r in g]) / np.mean([r["rms_B1"] for r in g]))
        out.append(s)
    return out


def contrasts(rows: list[dict]) -> list[dict]:
    out = []
    for ts in ("paper_varexc", "fresh_test"):
        for a, b, lab in CONTRASTS:
            ra = {r["sample"]: r for r in rows if r["test_set"] == ts and r["run"] == a}
            rb = {r["sample"]: r for r in rows if r["test_set"] == ts and r["run"] == b}
            common = sorted(set(ra) & set(rb))
            if not common:
                continue
            row = {"test_set": ts, "contrast": lab, "a": a, "b": b, "n": len(common)}
            for col in ("mac_mean", "mac_T1", "mac_T2", "mac_B1", "mac_B2"):
                d = np.array([ra[s][col] - rb[s][col] for s in common])
                lo, hi = K.bootstrap_ci(d)
                row[f"d_{col}"], row[f"d_{col}_lo"], row[f"d_{col}_hi"] = float(d.mean()), lo, hi
            out.append(row)
    return out


def real_zeroshot(nets: list[str]) -> tuple[list[dict], dict]:
    import s5_zeroshot as S5
    import fr_edge_modes

    fr_edge_modes.install()
    data = S5.load_structures()
    ori = {}
    for s, d in data.items():
        pkg = np.load(S5.GRAPH_ROOT / s / "graph_package.npz", allow_pickle=False)
        if not np.array_equal(pkg["edge_index"].astype(np.int64), d["ei"]):
            raise RuntimeError(f"{s}: graph_package edge_index differs from the s5 inputs")
        ori[s] = fr_edge_modes.orientation({"edge_attr_geom": pkg["edge_attr_geom"]})
    rows = []
    for net in nets:
        mode = P.ALL[net]["edge"]
        dd = {}
        for s, d in data.items():
            ea = {"cpsd_full": d["ea"], "geom_orient": ori[s], "zero": np.zeros((d["ea"].shape[0], 2), np.float32),
                  P.COH_ORIENT: np.hstack([d["ea"], ori[s]]).astype(np.float32)}[mode]
            dd[s] = {**d, "ea": ea}
        rr = S5.evaluate(K.EC_RUNS / net, "best_model", dd, net, C.SEED)
        for r in rr:
            r["edge_feature_mode"] = mode
        rows += rr
        print(f"[real] {net:<28} MAC vs SSI {np.mean([r['mac_ssi'] for r in rr]):.3f}  vs FE {np.mean([r['mac_fe_det'] for r in rr]):.3f}",
              flush=True)
    ref = K.read_csv(K.T13_RESULTS / "real_zeroshot.csv")
    gate = {}
    for net in P.REFS:
        a = [r for r in rows if r["arm"] == net]
        b = [r for r in ref if r["arm"] == net]
        same_keys = [(r["structure"], r["mode"]) for r in a] == [(r["structure"], r["mode"]) for r in b]
        md = max(abs(float(x["mac_ssi"]) - float(y["mac_ssi"])) for x, y in zip(a, b)) if same_keys else None
        gate[net] = {"rows": len(a), "max_abs_diff_mac_ssi": md, "pass": bool(same_keys and len(a) == 36 and md == 0.0)}
    return rows, gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args()
    P.T16_RESULTS.mkdir(parents=True, exist_ok=True)
    miss = [n for n in P.ALL if not (K.EC_RUNS / n / "checkpoints" / "best_model.pt").exists()]
    if miss:
        raise SystemExit(f"missing checkpoints: {miss}")
    t0 = time.time()
    tasks = [(n, ts) for ts in SETS for n in P.ALL]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=1) as ex:
        for rr in ex.map(score, tasks):
            rows += rr
            print(f"[eval] {rr[0]['test_set']:<13} {rr[0]['run']:<28} MAC {np.mean([r['mac_mean'] for r in rr]):.4f}  "
                  f"T1 {np.mean([r['mac_T1'] for r in rr]):.3f}  ({time.time() - t0:.0f} s)", flush=True)
    K.write_csv(P.T16_RESULTS / "eval_per_sample.csv", rows)
    summ = summarise(rows)
    K.write_csv(P.T16_RESULTS / "eval_summary.csv", summ)
    K.write_csv(P.T16_RESULTS / "eval_contrasts.csv", contrasts(rows))
    ref = {(r["test_set"], r["run"]): float(r["mac_mean"]) for r in K.read_csv(K.T13_RESULTS / "eval_summary.csv")}
    g13 = {f"{s['test_set']}/{s['run']}": {"mine": s["mac_mean"], "t13": ref[(s["test_set"], s["run"])],
                                         "pass": abs(s["mac_mean"] - ref[(s["test_set"], s["run"])]) < 1e-5}
           for s in summ if s["run"] in P.REFS and (s["test_set"], s["run"]) in ref}
    gates = {"synthetic_refs_reproduce_t13": {"pass": len(g13) == 4 and all(v["pass"] for v in g13.values()), "rows": g13}}
    if not a.skip_real:
        nets = [n for n in P.ALL if not P.ALL[n]["amp"]]
        rr, gr = real_zeroshot(nets)
        K.write_csv(P.T16_RESULTS / "real_zeroshot.csv", rr)
        rs = []
        for n in nets:
            g = [r for r in rr if r["arm"] == n]
            rs.append({"run": n, "label": P.ALL[n]["label"], "mac_ssi_mean": float(np.mean([r["mac_ssi"] for r in g])),
                       "mac_fe_mean": float(np.mean([r["mac_fe_det"] for r in g])),
                       **{f"mac_ssi_{m}": float(np.mean([r["mac_ssi"] for r in g if r["mode"] == m])) for m in MODES}})
        K.write_csv(P.T16_RESULTS / "real_summary.csv", rs)
        gates["real_refs_reproduce_t13"] = {"pass": all(v["pass"] for v in gr.values()), "rows": gr}
    gates["pass"] = all(v["pass"] for v in gates.values() if isinstance(v, dict))
    K.write_json(P.T16_RESULTS / "eval_gates.json", gates)
    print(json.dumps(gates, indent=1))
    for s in summ:
        print(f"{s['test_set']:<13} {s['run']:<28} MAC {s['mac_mean']:.3f} B1 {s['mac_B1']:.3f} T1 {s['mac_T1']:.3f} "
              f"B2 {s['mac_B2']:.3f} T2 {s['mac_T2']:.3f} f {s['freq_mae_hz']:.3f}  T1 rms/B1 {s['rms_ratio_T1_to_B1']:.3f}")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

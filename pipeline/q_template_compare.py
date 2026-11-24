"""T17 step 4 - the scale-up networks against the class template and FDD, paired, on the same deck nodes.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/q_template_compare.py [--workers 8]

T5 (c_fdd.py) scores the class template and FDD on deck nodes (every graph node except the Ng + 2
supports, whose records are the injected excitation); q_eval's network MAC is over every graph node.
This re-runs q_eval's inference (same nets, best-validation checkpoint), scores each mode on the deck
nodes with T5's support masks, and pairs it per structure with T5's rows (fdd_rows.csv, layout `all`):
  template    the class prior's leading shape for the deck's girder count, class given, no record.
              On paper_varexc it is T5's `paper` row: same structures and targets, and it reads no record.
  population  FDD on every deck node, population labeller.  fresh_test only: T5 ran it on the fresh_test
              records, which are the networks' inputs too; T5's `paper` FDD read the shared record, not
              paper_varexc's.
Pairing unit: the structure (mean over the four modes).  Across seeds, the per-structure differences are
averaged over the three seeds first, then bootstrapped over structures (h_common.bootstrap_ci).
Gates: the all-node MAC recomputed here equals q_eval's eval_per_sample.csv (1e-9); every support mask has
Ng + 2 nodes and the graph's node count; T5's row means equal shape_baselines.csv (1e-12).
Writes results/T17_scale_up/: deck_per_sample.csv, baseline_compare.csv, baseline_gates.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import q_common as Q
import p_common as P
import p_eval as PE

K, C, EC = Q.K, Q.C, Q.EC
T16_REFS = ("ec_t16_cohorient_signinv", "ec_t16_orient_signinv")
P.ALL.update({k: {kk: v[kk] for kk in ("edge", "loss", "amp", "label")} for k, v in Q.RUNS.items()})
MODES = list(C.MODE_NAMES)
SETS = ("paper_varexc", "fresh_test")
T5_RES = EC.EC_RESULTS / "T5_fresh_test_baselines"
DECKS = {"paper_varexc": EC.EC_DATA / "T5" / "fdd" / "paper_decks.npz",
         "fresh_test": EC.EC_DATA / "T5" / "fdd" / "fresh_test_decks.npz"}
T5_SPLIT = {"paper_varexc": "paper", "fresh_test": "fresh_test"}
BASELINES = {"template": "class template (no measurement, class given), deck nodes",
             "population": "FDD every deck node, population labeller"}


def mac(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def score_deck(task: tuple) -> list[dict]:
    net, ts = task
    import torch
    import EdgeGATConv_Modal_GNN as eg

    torch.set_num_threads(2)
    arm = P.ALL[net]
    P.install_for(arm)
    rd = K.EC_RUNS / net
    model, _ = eg.load_trained_model_checkpoint(rd / "checkpoints" / "best_model.pt", device=torch.device("cpu"))
    model.eval()
    files = PE.set_files(ts)
    D = np.load(DECKS[ts])
    ds = eg.Phase2AEdgeDataset(files, edge_feature_mode=arm["edge"], shape_target_mode="signed", expected_n_modes=4)
    rows = []
    with torch.no_grad():
        for i, f in enumerate(files):
            g, _ = ds[i]
            o = model(g, g.ndata["x"], g.edata["edge_x"])
            p, y = o.pred_mode_shape_abs.numpy().astype(np.float64), g.ndata["y"].numpy().astype(np.float64)
            sup = D[f"support__{f.name}"].astype(bool)
            ng = int(f.name.split("_Ng")[1][0])
            r = {"test_set": ts, "run": net, "sample": f.name, "Ng": ng, "n_support": int(sup.sum()),
                 "mask_matches_graph": bool(sup.size == p.shape[0])}
            dk = ~sup
            for j, m in enumerate(MODES):
                c = float(p[:, j] @ y[:, j] / (np.linalg.norm(p[:, j]) * np.linalg.norm(y[:, j]) + 1e-12))
                r[f"mac_all_{m}"] = c * c
                r[f"mac_deck_{m}"] = mac(p[dk, j], y[dk, j]) if sup.size == p.shape[0] else float("nan")
            r["mac_deck_mean"] = float(np.mean([r[f"mac_deck_{m}"] for m in MODES]))
            rows.append(r)
    return rows


def paired(diff: pd.DataFrame) -> dict:
    """diff: index sample, columns the four modes (network - baseline).  Structure is the unit."""
    d = diff.mean(axis=1).to_numpy()
    lo, hi = K.bootstrap_ci(d)
    out = {"n": len(d), "d_mean": float(d.mean()), "d_mean_lo": lo, "d_mean_hi": hi,
           "share_net_better": float((d > 0).mean())}
    for m in MODES:
        out[f"d_{m}"] = float(diff[m].mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    nets = list(T16_REFS) + list(Q.RUNS)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=1) as ex:
        for rr in ex.map(score_deck, [(n, ts) for ts in SETS for n in nets]):
            rows += rr
            print(f"[deck] {rr[0]['test_set']:<13} {rr[0]['run']:<40} deck MAC "
                  f"{np.mean([r['mac_deck_mean'] for r in rr]):.4f}", flush=True)
    df = pd.DataFrame(rows)
    K.write_csv(Q.T17_RESULTS / "deck_per_sample.csv", rows)

    # gates
    ev = pd.read_csv(Q.T17_RESULTS / "eval_per_sample.csv")
    ev = ev[ev.test_set.isin(SETS)].set_index(["test_set", "run", "sample"])
    j = df.set_index(["test_set", "run", "sample"])
    g_all = max(float((j[f"mac_all_{m}"] - ev.loc[j.index, f"mac_{m}"]).abs().max()) for m in MODES)
    t5 = pd.read_csv(T5_RES / "fdd_rows.csv")
    t5 = t5[(t5.layout == "all") & t5.method.isin(list(BASELINES))]
    sb = pd.read_csv(T5_RES / "shape_baselines.csv")
    g_t5 = {}
    for ts, split in T5_SPLIT.items():
        for meth, lab in BASELINES.items():
            ref = float(sb[(sb.test_set == split) & (sb.method == lab)]["mean"].iloc[0])
            g_t5[f"{split}/{meth}"] = abs(float(t5[(t5.split == split) & (t5.method == meth)].mac.mean()) - ref)
    gates = {"all_node_mac_equals_q_eval": {"max_abs_diff": g_all, "pass": g_all < 1e-9},
             "support_masks": {"n_support_eq_ng_plus_2": bool((df.n_support == df.Ng + 2).all()),
                               "mask_matches_graph": bool(df.mask_matches_graph.all()),
                               "pass": bool((df.n_support == df.Ng + 2).all() and df.mask_matches_graph.all())},
             "t5_means_reproduce": {"max_abs_diff": max(g_t5.values()), "pass": max(g_t5.values()) < 1e-12}}
    gates["pass"] = all(v["pass"] for v in gates.values())

    # comparisons
    out = []
    for ts in SETS:
        split = T5_SPLIT[ts]
        for meth in BASELINES:
            if meth == "population" and ts != "fresh_test":
                continue
            b = (t5[(t5.split == split) & (t5.method == meth)].pivot(index="sample", columns="mode", values="mac")[MODES])
            per_run = {}
            for net in nets:
                n = df[(df.test_set == ts) & (df.run == net)].set_index("sample")[[f"mac_deck_{m}" for m in MODES]]
                n.columns = MODES
                n = n.loc[b.index]
                per_run[net] = n - b
                out.append({"test_set": ts, "baseline": meth, "what": net, "net_deck_mac": float(n.to_numpy().mean()),
                            "baseline_mac": float(b.to_numpy().mean()), **paired(n - b)})
            for arm in Q.BASE_ARMS:
                runs = [r for r in Q.RUNS if Q.RUNS[r]["base_arm"] == arm]
                dm = sum(per_run[r] for r in runs) / len(runs)
                nm = [float((per_run[r] + b).to_numpy().mean()) for r in runs]
                out.append({"test_set": ts, "baseline": meth, "what": f"{arm} (mean over seeds {list(Q.SEEDS)})",
                            "net_deck_mac": float(np.mean(nm)), "net_deck_mac_sd": float(np.std(nm, ddof=1)),
                            "baseline_mac": float(b.to_numpy().mean()), **paired(dm),
                            "d_mean_seed_min": float(min(per_run[r].mean(axis=1).mean() for r in runs)),
                            "d_mean_seed_max": float(max(per_run[r].mean(axis=1).mean() for r in runs))})
    K.write_csv(Q.T17_RESULTS / "baseline_compare.csv", out)
    K.write_json(Q.T17_RESULTS / "baseline_gates.json", gates)
    print(json.dumps(gates, indent=1))
    for r in out:
        print(f"{r['test_set']:<13} vs {r['baseline']:<10} {r['what']:<52} net {r['net_deck_mac']:.4f} "
              f"base {r['baseline_mac']:.4f}  d {r['d_mean']:+.4f} [{r['d_mean_lo']:+.4f}, {r['d_mean_hi']:+.4f}]  "
              f"better {r['share_net_better']:.2f}  " + " ".join(f"{m} {r[f'd_{m}']:+.3f}" for m in MODES))
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""T5 - why the networks fail on a new excitation realisation: swap inputs between paired sets.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/c_excitation_diag.py

fresh_test and fresh_test_paperexc hold the same 500 structures (same graphs, targets) and
differ only in the excitation seed (20261124 vs the paper's 12345).  For a network, every
structure is scored with inputs assembled from the two:

  new            nodes and edges from the new excitation                 (= c_rescore fresh_test)
  paper          nodes and edges from the paper excitation               (= c_rescore fresh_test_paperexc)
  nodes_new      new node PSDs, paper edges
  edges_new      paper node PSDs, new edges
  supports_new   paper inputs, except the support nodes' PSD rows (the records of the injected
                 excitation itself; support mask from c_fdd's re-simulation) from the new one
  deck_new       new inputs, except the support rows from the paper excitation

Same code path as c_rescore (library datasets, best-validation checkpoint, CPU, 2 threads).
Writes excitation_diag.csv (per run x variant x mode mean MAC, and frequency MAE).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

import ec_common as E  # noqa: E402

C = E.C
import c_rescore as R  # noqa: E402

OUT = E.task_dir("T5")
RUNS = [C.sweep_run_name(C.MAIN_N), C.sweep_run_name(600), C.ABLATION_GRID[("GraphSAGE", "none")],
        C.ABLATION_GRID[("EdgeGATConv", "orientation")]]
VARIANTS = ("new", "paper", "nodes_new", "edges_new", "supports_new", "deck_new")


def main() -> int:
    import torch
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs

    R.install_modes()
    torch.set_num_threads(2)
    fa, fb = R.set_files("fresh_test"), R.set_files("fresh_test_paperexc")
    assert [f.name for f in fa] == [f.name for f in fb]
    sup = np.load(E.EC_DATA / "T5" / "fdd" / "fresh_test_decks.npz")
    rows = []
    for run in RUNS:
        rd = C.RUNS / run
        info = json.loads((rd / "run_summary.json").read_text(encoding="utf-8"))
        ck = rd / "checkpoints" / "best_model.pt"
        dev = torch.device("cpu")
        if info["arch"] == "edgegat":
            model, ckpt = eg.load_trained_model_checkpoint(ck, device=dev)
            mk = lambda fs: eg.Phase2AEdgeDataset(fs, edge_feature_mode=info["edge_feature_mode"],  # noqa: E731
                                                  shape_target_mode="signed", expected_n_modes=4)
            metric = eg._compute_sample_metrics
        else:
            ckpt = torch.load(ck, map_location=dev, weights_only=False)
            model = gs.build_model_from_config(ckpt["config"]).to(dev)
            model.load_state_dict(ckpt["model_state_dict"])
            mk = lambda fs: gs.Phase1NPZDataset(fs, 400, 4, edge_feature_mode=info["edge_feature_mode"],  # noqa: E731
                                                shape_target_mode="signed")
            metric = gs._compute_sample_metrics
        model.eval()
        stats = gs.FrequencyNormalizationStats(freq_mean_hz=np.asarray(ckpt["freq_mean_hz"], np.float32),
                                               freq_std_hz=np.asarray(ckpt["freq_std_hz"], np.float32))
        da, db = mk(fa), mk(fb)
        acc = {v: {"mac": [], "ferr": []} for v in VARIANTS}
        with torch.no_grad():
            for i in range(len(da)):
                ga, ft = da[i]
                gb, _ = db[i]
                assert torch.equal(ga.edges()[0], gb.edges()[0]) and torch.allclose(ga.ndata["y"], gb.ndata["y"], rtol=0, atol=1e-5)  # ARPACK noise
                s = torch.from_numpy(sup[f"support__{fa[i].name}"])
                xa, xb, ea, eb = ga.ndata["x"], gb.ndata["x"], ga.edata["edge_x"], gb.edata["edge_x"]
                xs_new, xd_new = xb.clone(), xa.clone()
                xs_new[s], xd_new[s] = xa[s], xb[s]
                for v, (x, e) in {"new": (xa, ea), "paper": (xb, eb), "nodes_new": (xa, eb), "edges_new": (xb, ea),
                                  "supports_new": (xs_new, eb), "deck_new": (xd_new, ea)}.items():
                    out = model(ga, x, e)
                    m = metric(out.pred_mode_shape_abs.numpy(), ga.ndata["y"].numpy(), C.MODE_NAMES)
                    fp = gs.denormalize_frequency_predictions(out.pred_freq, stats).numpy().reshape(-1)
                    acc[v]["mac"].append([m[f"mac_{k}"] for k in C.MODE_NAMES])
                    acc[v]["ferr"].append(np.abs(fp - ft.numpy().reshape(-1)).mean())
        for v in VARIANTS:
            mm = np.array(acc[v]["mac"])
            rows.append({"run": run, "variant": v, **{f"mac_{k}": float(mm[:, j].mean()) for j, k in enumerate(C.MODE_NAMES)},
                         "mac_mean": float(mm.mean()), "freq_mae_hz": float(np.mean(acc[v]["ferr"]))})
            print(f"{run:<34} {v:<13} MAC {rows[-1]['mac_mean']:.4f}  f MAE {rows[-1]['freq_mae_hz']:.3f}", flush=True)
    with (OUT / "excitation_diag.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # gate: the two unmixed variants equal c_rescore's tables
    summ = {(r["test_set"], r["run"]): r for r in csv.DictReader((OUT / "rescore_summary.csv").open(encoding="utf-8"))
            if r["checkpoint"] == "best_model"}
    worst = max(abs(r["mac_mean"] - float(summ[({"new": "fresh_test", "paper": "fresh_test_paperexc"}[r["variant"]], r["run"])]["mac_mean"]))
                for r in rows if r["variant"] in ("new", "paper"))
    print(f"[check] unmixed variants == c_rescore: max |diff| {worst:.1e} ({'PASS' if worst < 1e-9 else 'FAIL'})")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())

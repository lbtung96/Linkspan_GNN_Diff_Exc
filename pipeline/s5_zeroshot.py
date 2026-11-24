"""Stage 5 - zero-shot mode shapes and frequencies on the nine real bridges, and
transfer against population size.

    python tools/run_stage.py Final_results_for_paper/pipeline/s5_zeroshot.py

Inputs that do not depend on the synthetic population: the corrected real
features in the sensor order of the SSI labels (fr_common.REAL_FEATURES, written
by s5a_field_inputs.py since PROVENANCE Phase P, 2026-11-23), the hybrid graphs
(phase3_real_graphs/), the curated SSI labels inside the features, and the FE
targets of the real geometries (experiments/E6_controls/task2_raw/, label
corrections included).

The scoring code is copied verbatim from
experiments/E2_training_distribution/stage2_zeroshot.py (load_structures, infer,
evaluate).  Before anything new is scored, the copy is proved faithful: it
re-scores E2's broadband seed-123 checkpoint and must reproduce E2's own
stage2_zeroshot.csv rows exactly (inference over stored features is
bit-reproducible).  E2 ran on the legacy raw-order features
(fr_common.LEGACY_REAL_FEATURES), so the proof runs on those; everything new is
scored on fr_common.REAL_FEATURES.

Gates, both required by CLAUDE.md after the 2026-08-21 failure:
  feature convention   corrected real node spectra row-max = 1, edge coherency |v| <= 1
  input dependence     for every structure the mean MAC between the prediction and
                       the network's response to an all-zero input is < 0.9

Outputs
  results/s5_zeroshot.csv          the main network (edgegat_N{MAIN_N}, best_model)
  results/s5_transfer_vs_n.csv     every sweep network, same scoring
  results/s5_gates.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import fr_common as C

import dgl  # noqa: E402

from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint  # noqa: E402
from GraphSAGE_Modal_GNN import denormalize_frequency_predictions  # noqa: E402
from infer_phase2a_unseen_structure import load_frequency_normalization  # noqa: E402
from infer_phase2a_unseen_structure_edgegat import (  # noqa: E402
    apply_input_normalization,
    load_input_normalization,
)

CORRECTED = C.REAL_FEATURES                  # the paper's inputs (sensor rows 6/8 in SSI order)
LEGACY_CORRECTED = C.LEGACY_REAL_FEATURES    # raw sensor order: what E2's published rows ran on
GRAPH_ROOT = C.REPO / "phase3_real_graphs"
AUDIT_RAW = C.REPO / "experiments" / "E6_controls" / "task2_raw"
E2_REF_RUN = C.REPO / "experiments" / "E2_training_distribution" / "runs_ep500" / "bandpass_seed123"
E2_REF_CSV = C.REPO / "experiments" / "E2_training_distribution" / "stage2_zeroshot.csv"
STRUCTURES = list(C.LOOCV)
MODES = C.MODE_NAMES


# ---- copied from experiments/E2_training_distribution/stage2_zeroshot.py -----------------
# (verbatim, except that load_structures no longer opens graph_package.npz, which it never used,
#  and takes the feature root as an argument: legacy for the E2 proof, corrected for the paper)
def mac(u, v) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.abs(np.dot(u, v)) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def build_graph(ei, x, ea, n):
    g = dgl.graph((ei[0], ei[1]), num_nodes=n)
    g.ndata["x"] = torch.tensor(np.asarray(x, dtype=np.float32))
    g.edata["edge_x"] = torch.tensor(np.asarray(ea, dtype=np.float32))
    return g


def infer(model, freq_stats, norm_stats, ei, x, ea, n):
    g = build_graph(ei, x, ea, n)
    apply_input_normalization(g, norm_stats)
    g = g.to(DEVICE)
    with torch.no_grad():
        o = model(g, g.ndata["x"], g.edata["edge_x"])
        return (o.pred_mode_shape_abs.cpu().numpy(),
                denormalize_frequency_predictions(o.pred_freq, freq_stats)
                .cpu().numpy().reshape(-1))


def load_structures(root: Path = CORRECTED) -> dict:
    data = {}
    for s in STRUCTURES:
        d = np.load(root / s / "phase3_dataset_corrected.npz", allow_pickle=False)
        raw = np.load(AUDIT_RAW / f"{s}.npz")
        data[s] = dict(
            ei=d["edge_index"].astype(np.int64), x=d["x_psd"],
            ea=d["edge_attr_cpsd_full"], n=int(d["x_psd"].shape[0]),
            n_fe=int(d["n_fe_nodes"]),
            sensor_idx=d["sensor_node_indices"].astype(int),
            y_ssi=d["y_mode_shape_ssi"].astype(np.float64),
            ssi_freq=d["y_freq_hz"].astype(np.float64).reshape(-1),
            fe_t_pkg=d["fe_target_mode_shape"].astype(np.float64),
            fe_t_det=np.asarray(raw["fe_target"], dtype=np.float64),
            fe_freq=d["fe_target_freq_hz"].astype(np.float64).reshape(-1),
        )
    return data


def evaluate(run_dir: Path, which: str, data: dict, arm: str, seed: int) -> list[dict]:
    ck = run_dir / "checkpoints" / (which + ".pt")
    model, checkpoint = load_trained_model_checkpoint(ck, device=DEVICE)
    model.eval()
    freq_stats = load_frequency_normalization(checkpoint_path=ck, checkpoint=checkpoint,
                                              run_dir=run_dir)
    norm_stats = load_input_normalization(run_dir)
    rows = []
    for s, d in data.items():
        p, f = infer(model, freq_stats, norm_stats, d["ei"], d["x"], d["ea"], d["n"])
        pz, _ = infer(model, freq_stats, norm_stats, d["ei"], np.zeros_like(d["x"]),
                      np.zeros_like(d["ea"]), d["n"])
        at = p[d["sensor_idx"]]
        n_fe = d["n_fe"]
        for m, name in enumerate(MODES):
            col = p[:n_fe, m]
            rows.append({
                "arm": arm, "seed": seed, "checkpoint": which,
                "structure": s, "mode": name,
                "mac_ssi": mac(at[:, m], d["y_ssi"][:, m]),
                "mac_fe_pkg": mac(col, d["fe_t_pkg"][:, m]),
                "mac_fe_det": mac(col, d["fe_t_det"][:, m]),
                "pred_freq_hz": float(f[m]),
                "ssi_freq_hz": float(d["ssi_freq"][m]),
                "fe_target_freq_hz": float(d["fe_freq"][m]),
                "mac_pred_vs_zeroinput": mac(p[:, m], pz[:, m]),
                "pred_std_over_maxabs": float(np.std(col) / (np.max(np.abs(col)) + 1e-30)),
                "best_epoch_in_ckpt": int(checkpoint.get("epoch", -1)),
            })
    return rows
# ---- end of verbatim copy -------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="best_model", choices=["best_model", "last_model"])
    ap.add_argument("--run-dir", type=Path, default=C.RUNS / C.sweep_run_name(C.MAIN_N),
                    help="the network scored as 'main' (default edgegat_N{MAIN_N}; others for dry runs)")
    ap.add_argument("--tag", default="", help="suffix for the output files (dry runs only)")
    ap.add_argument("--skip-sweep", action="store_true", help="do not score the other sweep networks")
    args = ap.parse_args()
    # torch's default thread count, as E2's stage 2 ran: at 2 threads the same
    # inference differs in the last float32 bit (4.8e-07 Hz), which would fail the
    # exact-reproduction gate below for no scientific reason.
    data = load_structures()
    gates: dict = {"torch_threads": torch.get_num_threads()}

    # ---- faithfulness of the copy ------------------------------------------------------
    ref = [r for r in csv.DictReader(E2_REF_CSV.open(encoding="utf-8"))
           if r["arm"] == "bandpass" and r["seed"] == "123" and r["checkpoint"] == "best_model"]
    mine = evaluate(E2_REF_RUN, "best_model", load_structures(LEGACY_CORRECTED), "bandpass", 123)
    num = [k for k in mine[0] if k not in ("arm", "seed", "checkpoint", "structure", "mode")]
    maxdiff = max(abs(float(a[k]) - float(b[k])) for a, b in zip(mine, ref) for k in num) if len(ref) == len(mine) else None
    same_keys = [(a["structure"], a["mode"]) for a in mine] == [(b["structure"], b["mode"]) for b in ref]
    gates["copy_reproduces_E2"] = {"pass": bool(same_keys and maxdiff == 0.0), "rows": len(mine),
                                   "max_abs_diff": maxdiff, "inputs": "legacy raw sensor order (E2's)"}
    gates["inputs"] = {"features": str(CORRECTED.relative_to(C.REPO)).replace("\\", "/"),
                       "field_sensor_swap": list(C.FIELD_SENSOR_SWAP)}

    # ---- feature-convention gate (E5 stage 1) ----------------------------------------------
    feat = {}
    for s, d in data.items():
        rm = np.max(np.abs(d["x"]), axis=1)
        feat[s] = {"node_rowmax_min": float(rm.min()), "node_rowmax_max": float(rm.max()),
                   "edge_absmax": float(np.abs(d["ea"]).max())}
    gates["feature_convention"] = {"pass": all(abs(v["node_rowmax_min"] - 1) <= 1e-5 and abs(v["node_rowmax_max"] - 1) <= 1e-5
                                               and v["edge_absmax"] <= 1 + 1e-5 for v in feat.values()),
                                   "per_structure": feat}
    pool = [C.POP_BROADBAND / f for f in (C.SPLITS / f"N{C.MAIN_N}_train.txt").read_text().split()[:40]]
    rmx = [float(np.max(np.abs(np.load(p)["x_psd"]), axis=1).max()) for p in pool]
    emx = [float(np.abs(np.load(p)["edge_attr_cpsd_full"]).max()) for p in pool]
    gates["feature_convention"]["synthetic_training_reference"] = {
        "node_rowmax_max": max(rmx), "edge_absmax_max": max(emx), "n": len(pool)}

    # ---- the main network --------------------------------------------------------------------
    main_run = args.run_dir
    rows = evaluate(main_run, args.which, data, "broadband", C.SEED)
    main_n = json.loads((main_run / "run_summary.json").read_text(encoding="utf-8"))["n_population"]
    for r in rows:
        r["N"] = main_n
    write_csv(C.RESULTS / f"s5_zeroshot{args.tag}.csv", rows)
    dep = {s: float(np.mean([r["mac_pred_vs_zeroinput"] for r in rows if r["structure"] == s])) for s in STRUCTURES}
    gates["input_dependence"] = {"pass": all(v < 0.9 for v in dep.values()), "gate": "structure mean < 0.9",
                                 "per_structure_mean": dep, "overall_mean": float(np.mean(list(dep.values()))),
                                 "worst_slot": float(max(r["mac_pred_vs_zeroinput"] for r in rows))}

    # ---- transfer against N --------------------------------------------------------------------
    sweep = []
    for n in ([] if args.skip_sweep else C.SWEEP_N):
        rd = C.RUNS / C.sweep_run_name(n)
        if not (rd / "run_summary.json").exists():
            print(f"[wait] {rd.name} not trained yet - transfer-vs-N will be partial")
            continue
        rr = evaluate(rd, args.which, data, "broadband", C.SEED)
        for r in rr:
            r["N"] = n
        sweep += rr
        dn = {s: float(np.mean([r["mac_pred_vs_zeroinput"] for r in rr if r["structure"] == s])) for s in STRUCTURES}
        gates[f"input_dependence_N{n}"] = {"pass": all(v < 0.9 for v in dn.values()),
                                           "overall_mean": float(np.mean(list(dn.values())))}
    if sweep:
        write_csv(C.RESULTS / f"s5_transfer_vs_n{args.tag}.csv", sweep)

    gates["pass"] = all(v.get("pass") is not False for v in gates.values() if isinstance(v, dict))
    (C.RESULTS / f"s5_gates{args.tag}.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(json.dumps({k: v.get("pass") if isinstance(v, dict) else v for k, v in gates.items()}, indent=1))
    per = {m: np.mean([r["mac_ssi"] for r in rows if r["mode"] == m]) for m in MODES}
    print("main network, MAC vs SSI: " + " ".join(f"{m}={v:.3f}" for m, v in per.items())
          + f" mean={np.mean(list(per.values())):.3f}")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

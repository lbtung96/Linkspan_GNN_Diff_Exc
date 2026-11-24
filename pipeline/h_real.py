"""T11 step 4 - the nine real bridges: zero-shot (s5's path) and the FE-simulated ceiling (s6 case a).

    python tools/run_stage.py pipeline/h_real.py --part zeroshot
    python tools/run_stage.py pipeline/h_real.py --part ceiling --workers 5

zeroshot  s5_zeroshot.load_structures() and s5_zeroshot.evaluate(), imported and called unchanged,
          torch's default thread count (as s5).  The inputs are what s5 reads through fr_common:
          the corrected Phase P field features (fr_common.REAL_FEATURES, sensor rows 6/8 in the SSI
          order) on the hybrid graphs, SSI labels and FE targets as s5.
          Orientation arms: s5 has no orientation path (the paper never scored them on the bridges),
          so the 800-d coherency is replaced by fr_edge_modes.orientation() of the hybrid graph's own
          edge_attr_geom (phase3_real_graphs/<S>/graph_package.npz, same edge order - checked); its
          angle convention is checked against the node coordinates here and, for the FE edges,
          against the generator's own graph of the same FE model in --part ceiling.
          Gates: edgegat_N2000 reproduces pipeline/results/s5_zeroshot.csv exactly, edgegat_N600
          reproduces its N = 600 rows of s5_transfer_vs_n.csv exactly; s5's feature-convention gate;
          s5's input-dependence gate reported per network.
ceiling   s6_decomposition's case a (FE-simulated response of each real FE model, every FE node,
          the generator's broadband path) with s6's own functions (_build_params,
          assemble_deterministic_phase2a_payload, simulate_real, infer; 2 threads), for the paper
          record (12345) and three per-bridge new records SeedSequence([20261127, LOOCV position, k]),
          k = 0, 1, 2.  Scored as s6: MAC vs the label-corrected FE target at every FE node, MAC vs
          SSI at the nearest FE nodes.  Gate: edgegat_N2000 on the paper record reproduces
          s6_decomposition.csv's ceiling rows (FE re-solve: <= 1e-5 expected).

Writes results/T11_varied_excitation_pilot/real_zeroshot.csv, real_ceiling.csv, real_gates.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import h_common as H

C = H.C
NETS_REAL = ("ec_varexc_edgegat_N600", "edgegat_N600", "ec_varexc_edgegat_geomorient_N600",
             "edgegat_geomorient_N600")
NETS_CEIL = NETS_REAL + ("edgegat_N2000",)
N_VAR_RECORDS = 3
NUM_EXCLUDE = {"arm", "seed", "checkpoint", "structure", "mode", "N"}


def run_dir(name: str) -> Path:
    for root in (C.RUNS, H.EC_RUNS):
        if (root / name / "run_summary.json").exists():
            return root / name
    raise FileNotFoundError(name)


def edge_mode(name: str) -> str:
    return json.loads((run_dir(name) / "run_summary.json").read_text(encoding="utf-8"))["edge_feature_mode"]


def gates_path() -> Path:
    return H.T11_RESULTS / "real_gates.json"


def load_gates() -> dict:
    return json.loads(gates_path().read_text(encoding="utf-8")) if gates_path().exists() else {}


def exact_compare(mine: list[dict], ref: list[dict]) -> dict:
    keys_ok = [(a["structure"], a["mode"]) for a in mine] == [(b["structure"], b["mode"]) for b in ref]
    num = [k for k in mine[0] if k not in NUM_EXCLUDE and k in ref[0]]
    md = max(abs(float(a[k]) - float(b[k])) for a, b in zip(mine, ref) for k in num) if keys_ok else None
    return {"pass": bool(keys_ok and md == 0.0), "rows": len(mine), "columns": len(num), "max_abs_diff": md}


# ------------------------------------------------------------------------------ zero-shot
def part_zeroshot(nets: tuple, tag: str) -> int:
    import s5_zeroshot as S5
    import fr_edge_modes
    import torch

    fr_edge_modes.install()
    t0 = time.time()
    data = S5.load_structures()
    g = load_gates()
    zg = {"torch_threads": torch.get_num_threads(),
          "inputs": str(S5.CORRECTED.relative_to(C.REPO)).replace("\\", "/")}
    ref = list(csv.DictReader((C.RESULTS / "s5_zeroshot.csv").open(encoding="utf-8")))
    zg["s5_main_edgegat_N2000"] = exact_compare(S5.evaluate(C.RUNS / "edgegat_N2000", "best_model", data,
                                                            "broadband", C.SEED), ref)
    ref600 = [r for r in csv.DictReader((C.RESULTS / "s5_transfer_vs_n.csv").open(encoding="utf-8")) if r["N"] == "600"]
    zg["s5_transfer_edgegat_N600"] = exact_compare(S5.evaluate(C.RUNS / "edgegat_N600", "best_model", data,
                                                               "broadband", C.SEED), ref600)
    feat = {s: {"node_rowmax_min": float(np.abs(d["x"]).max(axis=1).min()),
                "node_rowmax_max": float(np.abs(d["x"]).max(axis=1).max()),
                "edge_absmax": float(np.abs(d["ea"]).max())} for s, d in data.items()}
    zg["feature_convention"] = {"pass": all(abs(v["node_rowmax_min"] - 1) <= 1e-5 and abs(v["node_rowmax_max"] - 1) <= 1e-5
                                            and v["edge_absmax"] <= 1 + 1e-5 for v in feat.values()),
                                "per_structure": feat}
    # orientation inputs from each hybrid graph's own geometry
    data_or, ori = {}, {}
    for s, d in data.items():
        pkg = np.load(S5.GRAPH_ROOT / s / "graph_package.npz", allow_pickle=False)
        ei = pkg["edge_index"].astype(np.int64)
        xy = pkg["node_coords_xy_mm"].astype(np.float64)
        o = fr_edge_modes.orientation({"edge_attr_geom": pkg["edge_attr_geom"]})
        dxy = xy[ei[1]] - xy[ei[0]]
        u = dxy / np.linalg.norm(dxy, axis=1, keepdims=True)
        ori[s] = {"edge_index_equal": bool(np.array_equal(ei, d["ei"])),
                  "orientation_vs_coords_maxdiff": float(np.abs(o - u).max()),
                  "n_edges": int(ei.shape[1])}
        data_or[s] = {**d, "ea": o}
    zg["orientation_inputs"] = {"pass": all(v["edge_index_equal"] and v["orientation_vs_coords_maxdiff"] <= 1e-6
                                            for v in ori.values()),
                                "rule": "cos/sin of graph_package edge_attr_geom theta == unit vector dst - src (1e-6)",
                                "per_structure": ori}
    rows = []
    for net in nets:
        mode = edge_mode(net)
        dd = data_or if mode == "geom_orient" else data
        rr = S5.evaluate(run_dir(net), "best_model", dd, net, C.SEED)
        dep = {s: float(np.mean([r["mac_pred_vs_zeroinput"] for r in rr if r["structure"] == s])) for s in S5.STRUCTURES}
        zg[f"input_dependence[{net}]"] = {"pass": all(v < 0.9 for v in dep.values()), "gate": "structure mean < 0.9",
                                          "overall_mean": float(np.mean(list(dep.values()))),
                                          "worst_structure": max(dep.values())}
        for r in rr:
            r["edge_feature_mode"] = mode
        rows += rr
        print(f"[zeroshot] {net:<36} MAC vs SSI {np.mean([r['mac_ssi'] for r in rr]):.3f}  "
              f"vs FE {np.mean([r['mac_fe_det'] for r in rr]):.3f}", flush=True)
    H.write_csv(H.T11_RESULTS / f"real_zeroshot{tag}.csv", rows)
    zg["pass"] = all(v.get("pass") is not False for v in zg.values() if isinstance(v, dict))
    zg["wall_s"] = round(time.time() - t0, 1)
    g[f"zeroshot{tag}"] = zg
    H.write_json(gates_path(), g)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in zg.items()}, indent=1))
    return 0 if zg["pass"] else 1


# ------------------------------------------------------------------------------ ceiling
def ceiling_job(job: tuple) -> dict:
    s, nets = job
    import torch
    import fr_edge_modes
    import s6_decomposition as S6
    import simulate_uniform_inertial_u3_white_noise as SIM
    from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint
    from infer_phase2a_unseen_structure import load_frequency_normalization

    fr_edge_modes.install()
    torch.set_num_threads(2)
    if S6.simulate_white_noise_from_modal_model.__globals__ is not SIM.__dict__:
        raise RuntimeError("s6's simulate function does not read the simulator module's SEED")
    t0 = time.time()
    models = {}
    for net in nets:
        rd = run_dir(net)
        ck = rd / "checkpoints" / "best_model.pt"
        m, ckpt = load_trained_model_checkpoint(ck, device=DEVICE)
        m.eval()
        models[net] = (m, load_frequency_normalization(checkpoint_path=ck, checkpoint=ckpt, run_dir=rd), edge_mode(net))
    grid = np.load(next(iter(sorted(C.POP_BROADBAND.glob("*.npz")))))["freq_bins_hz"].astype(np.float64)
    pos = list(C.LOOCV).index(s)
    params = S6._build_params(s)
    payload, _ = S6.assemble_deterministic_phase2a_payload(params, structure_name=s,
                                                           min_mac_threshold=S6.MAC_THR.get(s))
    det_ei = np.asarray(payload["edge_index"], dtype=np.int64)
    fe_target = np.asarray(payload["y_mode_shape_signed"], dtype=np.float64)
    fe_freq = np.asarray(payload["y_freq_hz"], dtype=np.float64).reshape(-1)
    n_fe = fe_target.shape[0]
    orient_fe = fr_edge_modes.orientation(payload)
    d = np.load(S6.CORRECTED / s / "phase3_dataset_corrected.npz", allow_pickle=False)
    pkg = S6.load_graph_package(s, graph_root=str(S6.GRAPH_ROOT))
    nearest_fe = pkg["nearest_fe_node_idx"].astype(int)
    y_ssi = d["y_mode_shape_ssi"].astype(np.float64)
    ssi_freq = d["y_freq_hz"].astype(np.float64).reshape(-1)
    # the hybrid graph's FE edges carry the generator's orientation (zero-shot orientation input check)
    hyb_ei = pkg["edge_index"].astype(np.int64)
    hyb_o = fr_edge_modes.orientation({"edge_attr_geom": pkg["edge_attr_geom"]})
    look = {(int(a), int(b)): k for k, (a, b) in enumerate(hyb_ei.T)}
    hit = [look.get((int(a), int(b))) for a, b in det_ei.T]
    check = {"fe_edges_found_in_hybrid": int(sum(h is not None for h in hit)), "fe_edges": int(det_ei.shape[1]),
             "hybrid_vs_generator_orientation_maxdiff": float(max(np.abs(hyb_o[h] - orient_fe[k]).max()
                                                                  for k, h in enumerate(hit) if h is not None)),
             "fe_coords_vs_graph_package_maxdiff_mm": float(np.abs(np.asarray(payload["coords_xy_mm"], float)
                                                                   - pkg["node_coords_xy_mm"][:n_fe].astype(float)).max())}
    # FE edges absent from the hybrid graph: each must be split by an inserted sensor node c (a-c-b)
    nbr = {}
    for a_, b_ in hyb_ei.T:
        nbr.setdefault(int(a_), set()).add(int(b_))
    missing = [(int(a_), int(b_)) for (a_, b_), h in zip(det_ei.T, hit) if h is None]
    check["fe_edges_missing"] = len(missing)
    check["fe_edges_missing_split_by_sensor_node"] = int(sum(
        any(c >= n_fe for c in (nbr.get(a_, set()) & nbr.get(b_, set()))) for a_, b_ in missing))
    records = [("paper", H.PAPER_EXC_SEED)] + [(f"new{k}", np.random.SeedSequence([H.REAL_EXC_BASE, pos, k]))
                                               for k in range(N_VAR_RECORDS)]
    rows = []
    for rec, seed in records:
        SIM.SEED = seed
        try:
            bp, _raw = S6.simulate_real(s, params)
        finally:
            SIM.SEED = H.PAPER_EXC_SEED
        assert np.array_equal(bp["edge_index"], det_ei)
        x = bp["Paa_norm"][:, :S6.PHASE1_FREQ_BINS].astype(np.float32)
        e = S6.build_cpsd_full_edge_payload(edge_index=det_ei, freq_bins_hz=bp["freq_bins"][:S6.PHASE1_FREQ_BINS],
                                            a3_abs=bp["a3"], fs=bp["fs"], nperseg=S6.WELCH_NPERSEG,
                                            noverlap=S6.WELCH_NOVERLAP, nfft=S6.WELCH_NFFT)["edge_attr_cpsd_full"]
        assert np.allclose(bp["freq_bins"][:S6.PHASE1_FREQ_BINS], grid)
        for net, (m, fstats, mode) in models.items():
            p, f = S6.infer(m, fstats, det_ei, x, orient_fe if mode == "geom_orient" else e, n_fe)
            for k, name in enumerate(C.MODE_NAMES):
                rows.append({"structure": s, "record": rec, "exc_seed": H.seed_repr(seed), "net": net, "mode": name,
                             "mac_vs_fe_all_fe_nodes": S6.mac(p[:n_fe, k], fe_target[:, k]),
                             "mac_vs_ssi_sensor_nodes": S6.mac(p[nearest_fe, k], y_ssi[:, k]),
                             "pred_freq_hz": float(f[k]), "fe_target_freq_hz": float(fe_freq[k]),
                             "ssi_freq_hz": float(ssi_freq[k])})
    check["seconds"] = round(time.time() - t0, 1)
    return {"structure": s, "rows": rows, "check": check}


def part_ceiling(workers: int, nets: tuple, tag: str) -> int:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(ceiling_job, [(s, nets) for s in C.LOOCV]))
    rows = [r for x in res for r in x["rows"]]
    H.write_csv(H.T11_RESULTS / f"real_ceiling{tag}.csv", rows)
    ref = [r for r in csv.DictReader((C.RESULTS / "s6_decomposition.csv").open(encoding="utf-8"))
           if r["variant"] == "ceiling"]
    mine = [r for r in rows if r["record"] == "paper" and r["net"] == "edgegat_N2000"]
    key = lambda r: (r["structure"], r["mode"])  # noqa: E731
    rm = {key(r): r for r in ref}
    cols = ("mac_vs_fe_all_fe_nodes", "mac_vs_ssi_sensor_nodes", "pred_freq_hz", "fe_target_freq_hz", "ssi_freq_hz")
    md = {c: max(abs(float(r[c]) - float(rm[key(r)][c])) for r in mine) for c in cols}
    g = load_gates()
    cg = {"s6_ceiling_edgegat_N2000_paper_record": {
              "rows": len(mine), "matched": sum(key(r) in rm for r in mine), "max_abs_diff": md,
              "tol": 1e-5, "pass": len(mine) == len(ref) == 36 and all(v <= 1e-5 for v in md.values())},
          "orientation_fe_edges": {
              "rule": "every FE edge of the generator's graph is in the hybrid graph with the same cos/sin "
                      "(1e-6), or is split by an inserted sensor node (a-c-b)",
              "pass": all(x["check"]["fe_edges_missing_split_by_sensor_node"] == x["check"]["fe_edges_missing"]
                          and x["check"]["hybrid_vs_generator_orientation_maxdiff"] <= 1e-6
                          and x["check"]["fe_coords_vs_graph_package_maxdiff_mm"] <= 1e-3 for x in res),
              "per_structure": {x["structure"]: x["check"] for x in res}},
          "records": {"paper": H.PAPER_EXC_SEED,
                      "new": f"SeedSequence([{H.REAL_EXC_BASE}, LOOCV position, k]), k = 0..{N_VAR_RECORDS - 1}"},
          "wall_s": round(time.time() - t0, 1)}
    cg["pass"] = all(v["pass"] for v in cg.values() if isinstance(v, dict) and "pass" in v)
    g[f"ceiling{tag}"] = cg
    H.write_json(gates_path(), g)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in cg.items()}, indent=1))
    print(json.dumps(md, indent=1))
    for net in nets:
        for rec in ("paper", "new"):
            rr = [r for r in rows if r["net"] == net and r["record"].startswith(rec)]
            print(f"[ceiling] {net:<36} {rec:<6} MAC vs FE {np.mean([r['mac_vs_fe_all_fe_nodes'] for r in rr]):.3f}"
                  f"  vs SSI {np.mean([r['mac_vs_ssi_sensor_nodes'] for r in rr]):.3f}")
    return 0 if cg["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["zeroshot", "ceiling"], required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--nets", nargs="+", default=None, help="subset (pre-training gate runs; output tagged)")
    args = ap.parse_args()
    H.T11_RESULTS.mkdir(parents=True, exist_ok=True)
    full = NETS_REAL if args.part == "zeroshot" else NETS_CEIL
    nets = tuple(args.nets) if args.nets else full
    tag = "" if set(nets) == set(full) else "_pretrain_gate"
    return part_zeroshot(nets, tag) if args.part == "zeroshot" else part_ceiling(args.workers, nets, tag)


if __name__ == "__main__":
    sys.exit(main())

"""T10 (3b) - zero-shot with the field inputs re-ordered: raw sensor rows 6 and 8 (0-based; serials x40131 / x40133)
swapped before the feature builder, as the MATLAB SSI scripts do ("Swap 131 and 133").

    python tools/run_stage.py pipeline/g_t10_swapped_inputs.py

The builder is experiments/E5_real_transfer/build_phase3_features_corrected.py, imported unmodified; its
build_structure body is reproduced here for order A only (the variant s5 scores, x_psd and edge_attr_cpsd_full)
with one extra line, the row swap.  Gate: with swap=False the arrays equal the stored corrected features bit for
bit and the scoring reproduces s5_zeroshot.csv exactly.  Same hybrid graph, labels and FE targets as s5.
Main network edgegat_N2000, best_model, torch threads as s5.

Writes data/T10/<S>_swapped_inputs.npz (features), results/T10_channel_order/t10_swapped_inputs.csv,
t10_swapped_inputs_checks.json.
"""

from __future__ import annotations

import ec_common as E  # noqa: I001

import csv
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(E.REPO / "experiments" / "E5_real_transfer"))
import build_phase3_features_corrected as B  # noqa: E402
import s5_zeroshot as S5  # noqa: E402
from phase3_utils import BANDPASS_F_HIGH, BANDPASS_F_LOW, bandpass_filter, load_graph_package  # noqa: E402

C = E.C
OUT = E.EC_RESULTS / "T10_channel_order"
# T10 is history: it ran on the paper's inputs BEFORE Phase P (sensors 7 and 9 interchanged) and gated against
# the s5 table of that time.  Phase P moved both (s5 now reads data/field_inputs_v2), so the re-run pins them:
# the pre-P features (fr_common.LEGACY_REAL_FEATURES) and the pre-P s5_zeroshot.csv, recovered byte for byte
# from the archive's commit before Phase P (9c1f5da^) into this folder.
PRE_P_FEATURES = C.LEGACY_REAL_FEATURES
PRE_P_S5 = OUT / "s5_zeroshot_preP.csv"
DATA = E.EC_DATA / "T10"
I6, I8 = 6, 8


def swap(a):
    b = np.array(a, copy=True); b[[I6, I8]] = b[[I8, I6]]; return b


def build(structure: str, grid: np.ndarray, do_swap: bool):
    field = np.load(B.FIELD_ROOT / structure / "data" / "field_modal_filtering_outputs.npz", allow_pickle=False)
    u3 = field["raw_u3_timeseries_clean"].astype(np.float64).T
    if do_swap:
        u3 = swap(u3)
    fs = float(2.0 * field["freq_hz"].astype(np.float64)[-1])
    u3_bp = bandpass_filter(u3, fs, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)
    pkg = load_graph_package(structure, graph_root=str(B.GRAPH_ROOT))
    sensor_idx = pkg["sensor_node_indices"].astype(int)
    ei = pkg["edge_index"].astype(np.int64)
    n = int(pkg["node_coords_xy_mm"].shape[0])
    x = B.corrected_node_features(u3_bp, fs, grid, n_nodes=n, sensor_idx=sensor_idx, edge_index=ei,
                                  order="A", norm_range="grid")
    ts = np.zeros((n, u3_bp.shape[1]), dtype=np.float32)
    ts[sensor_idx] = u3_bp.astype(np.float32)
    ts_prop = B.propagate(ts, sensor_idx, ei, n).astype(np.float64)
    ea = B.corrected_edge_features(ts_prop, fs, grid, ei)
    return x, ea


def main() -> int:
    torch.set_num_threads(int(json.loads((C.RESULTS / "s5_gates.json").read_text())["torch_threads"]))
    DATA.mkdir(parents=True, exist_ok=True)
    grid = B.training_freq_grid()
    data = S5.load_structures(PRE_P_FEATURES)
    checks: dict = {"torch_threads": int(torch.get_num_threads())}
    v_un, v_sw = {}, {}
    for s, d in data.items():
        xu, eu = build(s, grid, False)
        xs, es = build(s, grid, True)
        checks[s] = {"unswapped_node_equal": bool(np.array_equal(xu, d["x"])),
                     "unswapped_edge_equal": bool(np.array_equal(eu, d["ea"])),
                     "swapped_node_rel_change": float(np.linalg.norm(xs - d["x"]) / np.linalg.norm(d["x"])),
                     "swapped_edge_rel_change": float(np.linalg.norm(es - d["ea"]) / np.linalg.norm(d["ea"]))}
        np.savez_compressed(DATA / f"{s}_swapped_inputs.npz", x_psd=xs, edge_attr_cpsd_full=es)
        v_un[s] = {**d, "x": xu, "ea": eu}
        v_sw[s] = {**d, "x": xs, "ea": es}
        # labels swapped too (for the "SSI is the mis-ordered side" reading, scored alongside)
        print(s, checks[s], flush=True)
    run = C.RUNS / C.sweep_run_name(C.MAIN_N)
    rows = []
    for v, dd in (("unswapped_rebuild", v_un), ("inputs_swapped", v_sw)):
        for r in S5.evaluate(run, "best_model", dd, "broadband", C.SEED):
            rows.append({"variant": v, **r})
    # SSI with rows swapped, scored against the stored-input prediction is in t10_zeroshot_rescore.csv
    ref = list(csv.DictReader(PRE_P_S5.open(encoding="utf-8")))
    num = [k for k in ref[0] if k not in ("arm", "seed", "checkpoint", "structure", "mode", "N")]
    mine = [r for r in rows if r["variant"] == "unswapped_rebuild"]
    md = max(abs(float(a[k]) - float(b[k])) for a, b in zip(mine, ref) for k in num)
    same = [(a["structure"], a["mode"]) for a in mine] == [(b["structure"], b["mode"]) for b in ref]
    checks["gate_unswapped_reproduces_s5"] = {"pass": bool(same and md == 0.0), "max_abs_diff": md}
    checks["gate_unswapped_features_equal_stored"] = all(checks[s]["unswapped_node_equal"]
                                                         and checks[s]["unswapped_edge_equal"] for s in data)
    with (OUT / "t10_swapped_inputs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for v in ("unswapped_rebuild", "inputs_swapped"):
        sel = [r for r in rows if r["variant"] == v]
        summ[v] = {"mac_ssi_per_mode": {m: float(np.mean([r["mac_ssi"] for r in sel if r["mode"] == m])) for m in S5.MODES},
                   "mac_ssi_mean": float(np.mean([r["mac_ssi"] for r in sel])),
                   "mac_fe_det_mean": float(np.mean([r["mac_fe_det"] for r in sel])),
                   "abs_freq_err_vs_ssi_mean": float(np.mean([abs(r["pred_freq_hz"] - r["ssi_freq_hz"]) for r in sel])),
                   "input_dependence_mean": float(np.mean([r["mac_pred_vs_zeroinput"] for r in sel]))}
    checks["summary"] = summ
    (OUT / "t10_swapped_inputs_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print({k: v for k, v in checks.items() if k.startswith("gate")})
    ok = checks["gate_unswapped_reproduces_s5"]["pass"] and checks["gate_unswapped_features_equal_stored"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

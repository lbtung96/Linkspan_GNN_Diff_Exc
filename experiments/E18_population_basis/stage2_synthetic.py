"""E18 stage 2 - mode shapes from few sensors on the 500 synthetic test decks.

    python tools/run_stage.py experiments/E18_population_basis/stage2_synthetic.py --workers 12

The test set, station layouts (outer girders / every girder, n = 1..12 stations at
x = L(1 - k/n)) and re-simulation are s8's and E16's.  At each layout the four target
mode shapes are filled in over every node, from the sensors only, by

  FP             production feature propagation (D^-1/2 A D^-1/2, 50 iterations)
  PCFI           E16's best diffusion imputer
  PB             population basis, one prior for the four target modes together
  PB-class       population basis, the prior of the mode's class, the class read from the
                 true shape (character and rank; see popbasis.py)
  PB-class-data  the same, the class read from the sensors themselves: bending if the
                 kept sensors on the two outer girders move in phase, ranked by frequency

along two routes:

  shape   the true mode shape at the sensors (E16's ceiling: the value an exact modal
          identification at the sensors would return)
  ods     the deflection read from the re-simulated records at the mode's frequency
          (nearest Welch bin): cross-spectrum with the kept sensor of largest PSD there,
          divided by its amplitude - what operational modal analysis at the sensors gives

MAC against the true shape: over every node on the shape route (E16's convention, so
its FP and PCFI rows are a gate against E16's ceiling.csv), over the deck nodes on the
ods route: the excitation enters at the supports (hinge nodes and bottom-beam end springs),
so their records are the injected noise, tens of times the deck's.
`dense` is the ods route with every node measured: the best that route can do.

The prior (stage1_prior.py) comes from the main network's training split only.

Outputs: synthetic.csv (wide, mac_<mode>), classes.csv (class from the sensors vs from
the true shape), synthetic_checks.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from scipy.signal import get_window  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "Final_results_for_paper" / "pipeline"))
sys.path.insert(0, str(REPO / "experiments" / "E14_corner_sensors"))
sys.path.insert(0, str(REPO / "experiments" / "E16_fp_imputers"))
sys.path.insert(0, str(HERE))

import fr_common as C  # noqa: E402
import imputers as IM  # noqa: E402
import popbasis as PB  # noqa: E402
import stage3_sensor_sweep as S3  # noqa: E402
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402
from simulate_uniform_inertial_u3_white_noise import WELCH_NFFT, WELCH_NOVERLAP, WELCH_NPERSEG  # noqa: E402
import stage1_corner_propagation as E14  # noqa: E402
from stage1_corner_propagation import MODE_NAMES, node_psd_features, propagate  # noqa: E402

LEVELS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
FAMILIES = ("outer", "all")
METHODS = ("FP", "PCFI", "PB", "PB-class", "PB-class-data")
X_TOL_FRAC = 0.02
_P: dict = {}


def resimulate(sampled: dict, selected_idx: np.ndarray) -> dict:
    """E14's resimulate, also returning the graph rows the excitation enters at (the hinge
    nodes and the bottom-beam end springs): their absolute acceleration is the injected
    noise, tens of times the deck's, while their mode shape need not be zero."""
    p = E14.build_solver_params(sampled, n_candidate_modes=E14.N_CANDIDATE_MODES)
    solve_res = E14.run_solver(p, ref_x_fracs=E14.REF_X_FRACS, tol_x=E14.REF_SENSOR_TOL_X)
    graph_payload = E14.build_projected_graph_from_model(solve_res.model, include_sensor_map=False)
    sim_model = dict(solve_res.model)
    sim_model["M_full"] = solve_res.M_full
    sim = E14.simulate_white_noise_from_modal_model(
        sim_model, solve_res.phi_full, solve_res.freq_all_hz, selected_idx,
        zeta=E14.PHASE1_ZETA, plot=False, return_time_series=True,
        use_all_modes=True, bandpass_hz=E14.BANDPASS_HZ)
    ids = np.asarray(graph_payload["graph_node_ids"], dtype=np.int64)
    return {"a3": np.asarray(sim["a3_abs"], dtype=np.float64)[ids - 1, :],
            "fs": float(np.asarray(sim["fs"]).reshape(())), "graph_node_ids": ids,
            "edge_index": np.asarray(graph_payload["edge_index"], dtype=np.int64),
            "support": np.isin(ids, np.asarray(sim["support_nodes"]).ravel())}


def init_worker() -> None:
    d = np.load(HERE / "prior.npz")
    assert int(d["deg_x"]) == PB.DEG_X and int(d["deg_y"]) == PB.DEG_Y and float(d["eta_scale"]) == PB.ETA_SCALE
    _P.update({k: d[k] for k in d.files})


def prior(ng: int, key: str) -> tuple[np.ndarray, float]:
    return _P[f"Sigma_Ng{ng}_{key}"], float(_P[f"tau2_Ng{ng}_{key}"])


def mac_c(u, v) -> float:
    return float(np.abs(np.vdot(u, v)) ** 2 / (np.vdot(u, u).real * np.vdot(v, v).real + 1e-300))


def segment_spectra(a: np.ndarray, fs: float) -> np.ndarray:
    """X (n, nseg, F), scaled so that the Welch CSD is mean_g conj(X_i) X_j."""
    win = get_window("hamming", WELCH_NPERSEG, fftbins=True)
    step = WELCH_NPERSEG - WELCH_NOVERLAP
    nseg = (a.shape[1] - WELCH_NPERSEG) // step + 1
    idx = np.arange(nseg)[:, None] * step + np.arange(WELCH_NPERSEG)[None, :]
    X = np.fft.rfft(a[:, idx] * win, n=WELCH_NFFT, axis=-1)
    w = np.full(X.shape[-1], 2.0)
    w[0] = 1.0
    w[-1] = 1.0
    return X * np.sqrt(w / (fs * (win ** 2).sum()))


def data_classes(vals: np.ndarray, coords_s: np.ndarray, freqs: np.ndarray, ya: float, yb: float,
                 x_tol: float) -> list[str] | None:
    """Classes from the kept sensors: vals (s, 4) real; None if no pair spans the outer girders."""
    on_a = np.abs(coords_s[:, 1] - ya) < PB.TOL_MM
    on_b = np.abs(coords_s[:, 1] - yb) < PB.TOL_MM
    if not (on_a.any() and on_b.any()):
        return None
    return PB.classes(vals, freqs, coords_s, ya, yb, x_tol)


def run_sample(path: str, levels: tuple[int, ...], families: tuple[str, ...]) -> tuple:
    t0 = time.time()
    with np.load(path, allow_pickle=True) as d:
        p = _parse_params_json(d)
        st = {k: d[k] for k in ("graph_node_ids", "edge_index", "coords_xy_mm", "x_psd",
                                "y_mode_shape_signed", "y_freq_hz", "matched_mode_indices")}
    sample = Path(path).stem
    ng, L, dy = int(p["Ng"]), float(p["L_mm"]), float(p["dy_mm"])
    ei = st["edge_index"].astype(np.int64)
    c = st["coords_xy_mm"].astype(np.float64)
    y = st["y_mode_shape_signed"].astype(np.float64)
    f = st["y_freq_hz"].astype(np.float64).reshape(-1)
    n = c.shape[0]
    ya, yb, x_tol = 0.0, (ng - 1) * dy, X_TOL_FRAC * L
    B = PB.design(*PB.norm_coords(c, L, ng, dy))
    cl_true = PB.classes(y, f, c, ya, yb, x_tol)

    sim = resimulate(p, st["matched_mode_indices"].astype(int))
    assert np.array_equal(sim["graph_node_ids"], st["graph_node_ids"]) and np.array_equal(sim["edge_index"], ei)
    deck = ~sim["support"]
    assert sim["support"].sum() >= ng and not np.any(y[np.abs(c[:, 0]) < PB.TOL_MM])
    X = segment_spectra(sim["a3"], sim["fs"])
    freq_axis = np.fft.rfftfreq(WELCH_NFFT, 1.0 / sim["fs"])
    bins = [int(np.argmin(np.abs(freq_axis - fm))) for fm in f]
    Xb = X[:, :, bins]                                # (n, nseg, 4)
    P = np.mean(np.abs(Xb) ** 2, axis=1)              # (n, 4)
    chk = {"sample": sample, "Ng": ng, "n_nodes": n, "n_deck": int(deck.sum()),
           "x_psd_resim_maxdiff": float(np.abs(node_psd_features(sim["a3"], sim["fs"]) - st["x_psd"]).max())}

    rows, crow = [], []

    def row(fam, lv, ns, route, method, macs):
        r = {"sample": sample, "Ng": ng, "family": fam, "level": lv, "n_sensors": ns, "route": route, "method": method}
        r.update({f"mac_{m}": v for m, v in zip(MODE_NAMES, macs)})
        rows.append(r)

    # every node measured, ods route: one reference per mode, the node of largest PSD on the deck
    dense = []
    for m in range(4):
        r0 = int(np.flatnonzero(deck)[np.argmax(P[deck, m])])
        d_all = np.mean(np.conj(Xb[r0, :, m])[None, :] * Xb[:, :, m], axis=1) / np.sqrt(P[r0, m])
        dense.append(mac_c(d_all[deck], y[deck, m]))
    row("-", 0, n, "ods", "dense", dense)

    for fam in families:
        for lv in levels:
            s = S3.station_nodes(c, p, lv, fam)
            ns = int(s.size)
            # ods at every node against the kept reference, per mode
            refs = [int(s[np.argmax(P[s, m])]) for m in range(4)]
            D = np.stack([np.mean(np.conj(Xb[refs[m], :, m])[None, :] * Xb[:, :, m], axis=1) / np.sqrt(P[refs[m], m])
                          for m in range(4)], axis=1)                      # (n, 4) complex
            # operators
            ys = np.zeros_like(y, dtype=np.float32)
            ys[s] = y[s]
            unit = np.zeros((n, ns), dtype=np.float32)
            unit[s, np.arange(ns)] = 1.0
            W_fp = propagate(unit, s, ei, n).astype(np.float64)
            W_pc, xi = IM.operator("PCFI", ei, n, s)
            W_pb = PB.operator(B, s, *prior(ng, "pooled"))
            W_cls = {k: PB.operator(B, s, *prior(ng, k)) for k in PB.CLASSES}
            cl_shape = data_classes(y[s], c[s], f, ya, yb, x_tol)
            cl_ods = data_classes(np.real(D[s]), c[s], f, ya, yb, x_tol)
            for route, cl_data in (("shape", cl_shape), ("ods", cl_ods)):
                crow.append({"sample": sample, "Ng": ng, "family": fam, "level": lv, "route": route,
                             **{f"true_{m}": cl_true[i] for i, m in enumerate(MODE_NAMES)},
                             **{f"data_{m}": (cl_data[i] if cl_data else "") for i, m in enumerate(MODE_NAMES)}})
            # shape route (E16's ceiling: FP and PCFI exactly as E16 computes them)
            fills = {"FP": propagate(ys, s, ei, n).astype(np.float64),
                     "PCFI": IM.apply("PCFI", W_pc, xi, s, ys[s].astype(np.float64), refine=True),
                     "PB": W_pb @ y[s],
                     "PB-class": np.stack([W_cls[cl_true[m]] @ y[s, m] for m in range(4)], axis=1),
                     "PB-class-data": (np.stack([W_cls[cl_shape[m]] @ y[s, m] for m in range(4)], axis=1)
                                       if cl_shape else W_pb @ y[s])}
            for meth in METHODS:
                row(fam, lv, ns, "shape", meth, [mac_c(fills[meth][:, m], y[:, m]) for m in range(4)])
            # ods route (time series get PCFI's diffusion only, as in E16)
            fills = {"FP": W_fp @ D[s], "PCFI": W_pc @ D[s], "PB": W_pb @ D[s],
                     "PB-class": np.stack([W_cls[cl_true[m]] @ D[s, m] for m in range(4)], axis=1),
                     "PB-class-data": (np.stack([W_cls[cl_ods[m]] @ D[s, m] for m in range(4)], axis=1)
                                       if cl_ods else W_pb @ D[s])}
            for meth in METHODS:
                row(fam, lv, ns, "ods", meth, [mac_c(fills[meth][deck, m], y[deck, m]) for m in range(4)])
    chk["seconds"] = round(time.time() - t0, 1)
    return rows, crow, chk


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def gate_e16(rows: list[dict]) -> dict:
    """FP and PCFI on the shape route must be E16's ceiling.csv rows (hinge free)."""
    ref = {}
    with (REPO / "experiments" / "E16_fp_imputers" / "ceiling.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["supports"] == "free" and r["imputer"] in ("FP", "PCFI"):
                ref[(r["sample"], r["family"], int(r["level"]), r["imputer"])] = r
    dmax, matched = 0.0, 0
    for r in rows:
        if r["route"] != "shape" or r["method"] not in ("FP", "PCFI"):
            continue
        q = ref.get((r["sample"], r["family"], r["level"], r["method"]))
        if q is None:
            continue
        matched += 1
        dmax = max(dmax, max(abs(r[f"mac_{m}"] - float(q[f"mac_{m}"])) for m in MODE_NAMES))
    want = sum(1 for r in rows if r["route"] == "shape" and r["method"] in ("FP", "PCFI"))
    return {"rows_matched": matched, "rows_expected": want, "max_abs_diff": dmax, "tol": 1e-6,
            "pass": matched == want and dmax <= 1e-6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-ng", type=int, default=None, help="first K test structures per Ng (pilot)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    test = (C.SPLITS / "test.txt").read_text(encoding="utf-8").split()
    if args.per_ng:
        test = [t for g in C.NG_VALUES for t in [x for x in test if f"_Ng{g}_" in x][: args.per_ng]]
    paths = [str(C.POP_BROADBAND / t) for t in test]
    rows, crows, chks, t0 = [], [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as ex:
        for i, (r, cr, ck) in enumerate(ex.map(run_sample, paths, [LEVELS] * len(paths),
                                               [FAMILIES] * len(paths)), 1):
            rows += r
            crows += cr
            chks.append(ck)
            if i % 50 == 0 or i == len(paths):
                print(f"  [{i}/{len(paths)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    pre = f"{args.tag}_" if args.tag else ""
    write(HERE / f"{pre}synthetic.csv", rows)
    write(HERE / f"{pre}classes.csv", crows)
    g = {"e16_ceiling": gate_e16(rows),
         "resim_vs_stored_x_psd": {"max_abs_diff": max(c["x_psd_resim_maxdiff"] for c in chks), "tol": 1e-6}}
    g["resim_vs_stored_x_psd"]["pass"] = g["resim_vs_stored_x_psd"]["max_abs_diff"] <= 1e-6
    g["pass"] = all(v["pass"] for v in g.values())
    (HERE / f"{pre}synthetic_checks.json").write_text(json.dumps({"gates": g, "per_sample": chks}, indent=1),
                                                       encoding="utf-8")
    print(json.dumps(g, indent=2))
    for route in ("shape", "ods"):
        for fam in FAMILIES:
            print(f"\n[{route}, {fam}] mean MAC over the four modes")
            for meth in METHODS:
                v = [np.mean([np.mean([r[f"mac_{m}"] for m in MODE_NAMES]) for r in rows
                              if r["route"] == route and r["family"] == fam and r["method"] == meth and r["level"] == lv])
                     for lv in LEVELS]
                print(f"  {meth:<14}" + " ".join(f"{x:6.3f}" for x in v))
    dn = [np.mean([r[f"mac_{m}"] for m in MODE_NAMES]) for r in rows if r["method"] == "dense"]
    print(f"\nods, every node measured: {np.mean(dn):.3f}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0 if g["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""E18 stage 3 - the population basis on the nine real bridges (E17's hold-out test).

    python tools/run_stage.py experiments/E18_population_basis/stage3_real.py --workers 9

E17's harness, unchanged: the hybrid graph, the layouts (single, front_pair, front_mid,
alternate, one_line, loso), the sources (real records; FE twin at the real record's
length and at 60 s), the deflection at each mode's frequency (cross-spectrum with the
kept sensor of largest PSD, over its amplitude) and the coherence of filled with
recorded signals.  New are the fill operators:

  FP, PCFI        E17's, as a gate: their rows must reproduce E17's fill.csv
  PB              population basis (stage1_prior.py), one prior for the four modes
  PB-class        the prior of the mode's class, the class read from the label shape
                  (SSI on real, FE target on the twin)
  PB-class-data   the class read from the kept sensors' deflections (bending if the kept
                  sensors on the two main lines move in phase, ranked by frequency); the
                  pooled prior where the kept sensors do not span both lines

The basis is evaluated at the hybrid graph's nodes in the bridge's own normalised
coordinates (L, Ng, dy from build_phase3_graphs.STRUCTURES, the geometry source).

Added score, per layout and mode: `pred_mac_label`, the MAC between the label shape at
every sensor (SSI on real, FE on the twin) and the predicted deflection there - the
kept sensors' own, the hidden ones filled.  It is directly comparable with the
network's MAC against SSI at the sensor nodes (E17's network.csv); `dense` is the same
with every sensor measured.

Outputs: real.csv, real_sensors.csv, real_checks.json.
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

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "E17_real_holdout"))
sys.path.insert(0, str(HERE))

import stage1_holdout as E17  # noqa: E402  (puts the pipeline, E5 and E16 on sys.path)
import popbasis as PB  # noqa: E402
import build_phase3_graphs as BPG  # noqa: E402
from phase3_utils import BANDPASS_F_HIGH, BANDPASS_F_LOW, bandpass_filter, load_graph_package  # noqa: E402

C = E17.C
METHODS = ("FP", "PCFI", "PB", "PB-class", "PB-class-data")
LAYOUTS = E17.FILL_LAYOUTS
PAIR_TOL_MM = 300.0
_P: dict = {}


def init_worker() -> None:
    d = np.load(HERE / "prior.npz")
    assert int(d["deg_x"]) == PB.DEG_X and int(d["deg_y"]) == PB.DEG_Y and float(d["eta_scale"]) == PB.ETA_SCALE
    _P.update({k: d[k] for k in d.files})


def prior(ng: int, key: str) -> tuple[np.ndarray, float]:
    return _P[f"Sigma_Ng{ng}_{key}"], float(_P[f"tau2_Ng{ng}_{key}"])


def fill_source(src, s, x, fs, Y, freqs, sens_nodes, sxy, lay, lines, ei, n, B, ng):
    G, f, nseg = E17.csd_matrix(x, fs)
    ns = len(x)
    P = G[np.arange(ns), np.arange(ns)].real
    bins = [int(np.argmin(np.abs(f - fm))) for fm in freqs]
    ya, yb = lines
    cl_label = PB.classes(Y, freqs, sxy, ya, yb, PAIR_TOL_MM)
    W_cache: dict = {}

    def W_of(name: str, kept: np.ndarray, key: str = "") -> np.ndarray:
        k = (name, key, tuple(kept))
        if k not in W_cache:
            if name in ("FP", "PCFI"):
                W_cache[k] = E17.operator(name, ei, n, sens_nodes[kept])[0]
            else:
                W_cache[k] = PB.operator(B, sens_nodes[kept], *prior(ng, key))
        return W_cache[k]

    def data_cls(kept: np.ndarray, refs: list[int]) -> list[str] | None:
        ks = sxy[kept]
        if not ((np.abs(ks[:, 1] - ya) < PB.TOL_MM).any() and (np.abs(ks[:, 1] - yb) < PB.TOL_MM).any()):
            return None
        vals = np.stack([np.real(G[refs[m], kept, bins[m]]) for m in range(4)], axis=1)
        return PB.classes(vals, freqs, ks, ya, yb, PAIR_TOL_MM)

    def weights(name, kept, m, cl_data):
        if name in ("FP", "PCFI"):
            return W_of(name, kept)
        if name == "PB":
            return W_of(name, kept, "pooled")
        if name == "PB-class":
            return W_of(name, kept, cl_label[m])
        return W_of(name, kept, cl_data[m]) if cl_data else W_of(name, kept, "pooled")

    def per_hidden(W, kept, hidden, m, ref):
        Wh = W[sens_nodes[hidden]]
        Gb = G[:, :, bins[m]]
        sc = np.sqrt(Gb[ref, ref].real)
        d_meas = Gb[ref, hidden] / sc
        d_fill = Wh @ Gb[ref, kept] / sc
        g_fh = np.einsum("hk,kh->h", Wh, Gb[np.ix_(kept, hidden)])
        g_ff = np.einsum("hk,kl,hl->h", Wh, Gb[np.ix_(kept, kept)], Wh).real
        coh = np.abs(g_fh) ** 2 / np.maximum(g_ff * Gb[hidden, hidden].real, 1e-300)
        return d_meas, d_fill, coh, Wh @ Y[kept, m], Gb[ref, :] / sc

    rows, srows = [], []

    def summarise(lname, name, nk, hidden, m, ref, d_meas, d_fill, coh, y_fill, d_all, cl_used):
        a = np.abs(d_meas)
        agree = (np.real(d_fill * np.conj(d_meas)) > 0).astype(float)
        v = d_all.copy()
        v[hidden] = d_fill
        y = Y[:, m].copy()
        y[hidden] = y_fill
        return {"source": src, "structure": s, "layout": lname, "imputer": name, "mode": C.MODE_NAMES[m],
                "class_label": cl_label[m], "class_used": cl_used, "freq_hz": float(freqs[m]), "nseg": nseg,
                "n_kept": nk, "n_hidden": len(hidden), "ref_sensor": int(ref),
                "defl_mac_hidden": E17.mac_c(d_fill, d_meas) if len(hidden) >= 2 else "",
                "sign_agree_w": float((a * agree).sum() / max(a.sum(), 1e-300)),
                "amp_ratio_median": float(np.median(np.abs(d_fill) / np.maximum(a, 1e-300))),
                "coh_mean": float(np.mean(coh)),
                "defl_mac_all": E17.mac_c(v, d_all),
                "label_mac_all": E17.mac_c(y, Y[:, m]),
                "label_mac_hidden": E17.mac_c(y_fill, Y[hidden, m]) if len(hidden) >= 2 else "",
                "pred_mac_label": E17.mac_c(v, Y[:, m])}

    # every sensor measured: the deflection itself against the label
    for m in range(4):
        ref = int(np.argmax(P[:, bins[m]]))
        d_all = G[ref, :, bins[m]] / np.sqrt(G[ref, ref, bins[m]].real)
        rows.append({"source": src, "structure": s, "layout": "full", "imputer": "dense", "mode": C.MODE_NAMES[m],
                     "class_label": cl_label[m], "class_used": "", "freq_hz": float(freqs[m]), "nseg": nseg,
                     "n_kept": ns, "n_hidden": 0, "ref_sensor": ref, "pred_mac_label": E17.mac_c(d_all, Y[:, m])})

    for lname in LAYOUTS[:-1]:
        kept = lay[lname]
        hidden = np.setdiff1d(np.arange(ns), kept)
        refs = [int(kept[np.argmax(P[kept, bins[m]])]) for m in range(4)]
        cl_data = data_cls(kept, refs)
        for name in METHODS:
            for m in range(4):
                W = weights(name, kept, m, cl_data)
                d_meas, d_fill, coh, y_fill, d_all = per_hidden(W, kept, hidden, m, refs[m])
                used = {"FP": "", "PCFI": "", "PB": "pooled", "PB-class": cl_label[m],
                        "PB-class-data": cl_data[m] if cl_data else "pooled"}[name]
                rows.append(summarise(lname, name, len(kept), hidden, m, refs[m], d_meas, d_fill, coh, y_fill,
                                      d_all, used))
                srows += [{"source": src, "structure": s, "layout": lname, "imputer": name, "mode": C.MODE_NAMES[m],
                           "sensor": int(h), "d_meas_re": float(dm.real), "d_meas_im": float(dm.imag),
                           "d_fill_re": float(df.real), "d_fill_im": float(df.imag), "coh": float(ch)}
                          for h, dm, df, ch in zip(hidden, d_meas, d_fill, coh)]
    # leave one out, as E17: one reference per mode, never held out
    for name in METHODS:
        for m in range(4):
            ref = int(np.argmax(P[:, bins[m]]))
            hs = np.array([h for h in range(ns) if h != ref])
            dm, dfl, chs, yf = [], [], [], []
            for h in hs:
                kept = np.setdiff1d(np.arange(ns), [h])
                cl_data = data_cls(kept, [ref] * 4)
                W = weights(name, kept, m, cl_data)
                a, b, c_, y_, d_all = per_hidden(W, kept, np.array([h]), m, ref)
                dm.append(a[0]); dfl.append(b[0]); chs.append(c_[0]); yf.append(y_[0])
            r = summarise("loso", name, ns - 1, hs, m, ref, np.array(dm), np.array(dfl), np.array(chs),
                          np.array(yf), d_all, "")
            r["n_hidden"] = 1
            r["label_mac_all"] = ""
            rows.append(r)
    return rows, srows, {"nseg": nseg, "class_label": cl_label}


def run_structure(s: str) -> dict:
    t0 = time.time()
    pkg = load_graph_package(s, graph_root=str(E17.S6.GRAPH_ROOT))
    ei = pkg["edge_index"].astype(np.int64)
    coords = pkg["node_coords_xy_mm"].astype(np.float64)
    n, n_fe = coords.shape[0], int(pkg["n_fe_nodes"])
    sens = pkg["sensor_node_indices"].astype(int)
    nearest_fe = pkg["nearest_fe_node_idx"].astype(int)
    sxy = coords[sens]
    lay, lines = E17.layouts(sxy, coords[:n_fe, 0].max())
    geo = BPG.STRUCTURES[s]
    L, ng, dy = float(geo["L_mm"]), int(geo["Ng"]), float(geo["dy_mm"])
    assert abs(coords[:n_fe, 0].max() - L) < 1.0 and lines[0] == 0 and abs(lines[1] - (ng - 1) * dy) < 1.0, \
        f"{s}: geometry and sensor lines disagree"
    B = PB.design(*PB.norm_coords(coords, L, ng, dy))
    raw6 = np.load(E17.RUN / "decomposition_raw" / f"{s}.npz")
    fe_target, fe_freq = raw6["fe_target"].astype(np.float64), raw6["fe_freq_hz"].astype(np.float64)
    y_ssi, ssi_freq = raw6["y_ssi"].astype(np.float64), raw6["ssi_freq_hz"].astype(np.float64)

    field = np.load(E17.FIELD_ROOT / s / "data" / "field_modal_filtering_outputs.npz", allow_pickle=False)
    u3 = field["raw_u3_timeseries_clean"].astype(np.float64).T
    fs_r = float(2.0 * field["freq_hz"].astype(np.float64)[-1])
    x_real = bandpass_filter(u3, fs_r, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)
    nseg_real = (x_real.shape[1] - E17.WELCH_NPERSEG) // (E17.WELCH_NPERSEG - E17.WELCH_NOVERLAP) + 1
    t_match = (nseg_real - 1) * (E17.WELCH_NPERSEG - E17.WELCH_NOVERLAP) + E17.WELCH_NPERSEG
    a3_std, a3_long, fs_t = E17.simulate_twin(s, t_match)
    x_twin = bandpass_filter(a3_std[nearest_fe, :], fs_t, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)
    x_match = bandpass_filter(a3_long[nearest_fe, :], fs_t, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)[:, :t_match]
    y_twin = fe_target[nearest_fe]

    rows, srows, info = [], [], {"L_mm": L, "Ng": ng, "dy_mm": dy, "lines_y_mm": list(lines)}
    for src, x, fs, Y, fq in (("real", x_real, fs_r, y_ssi, ssi_freq),
                              ("twin_matched", x_match, fs_t, y_twin, fe_freq),
                              ("twin_full", x_twin, fs_t, y_twin, fe_freq)):
        r, sr, inf = fill_source(src, s, x, fs, Y, fq, sens, sxy, lay, lines, ei, n, B, ng)
        rows += r
        srows += sr
        info[src] = inf
    info["seconds"] = round(time.time() - t0, 1)
    print(f"[{s}] {info['seconds']:.0f}s  classes real {info['real']['class_label']}  "
          f"twin {info['twin_full']['class_label']}", flush=True)
    return {"structure": s, "rows": rows, "sensors": srows, "info": info}


def write(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def gate_e17(rows: list[dict]) -> dict:
    ref = {}
    with (REPO / "experiments" / "E17_real_holdout" / "fill.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["imputer"] in ("FP", "PCFI"):
                ref[(r["source"], r["structure"], r["layout"], r["imputer"], r["mode"])] = r
    cols = ("defl_mac_hidden", "sign_agree_w", "amp_ratio_median", "coh_mean", "defl_mac_all", "label_mac_hidden")
    dmax, matched, want = 0.0, 0, 0
    for r in rows:
        if r["imputer"] not in ("FP", "PCFI"):
            continue
        want += 1
        q = ref.get((r["source"], r["structure"], r["layout"], r["imputer"], r["mode"]))
        if q is None:
            continue
        matched += 1
        for c in cols:
            if r[c] != "" and q[c] != "":
                dmax = max(dmax, abs(float(r[c]) - float(q[c])))
    return {"rows_matched": matched, "rows_expected": want, "max_abs_diff": dmax, "tol": 1e-5,
            "pass": matched == want and dmax <= 1e-5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=list(C.LOOCV), choices=list(C.LOOCV))
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as ex:
        res = list(ex.map(run_structure, args.structures))
    rows = [r for x in res for r in x["rows"]]
    write(HERE / "real.csv", rows)
    write(HERE / "real_sensors.csv", [r for x in res for r in x["sensors"]])
    g = gate_e17(rows)
    (HERE / "real_checks.json").write_text(json.dumps({"gates": {"e17_fill": g},
                                                       "per_structure": {x["structure"]: x["info"] for x in res}},
                                                      indent=1), encoding="utf-8")
    print(json.dumps(g, indent=2))
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[df.imputer != "dense"]
    for col in ("defl_mac_hidden", "sign_agree_w", "pred_mac_label"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for src in ("real", "twin_matched"):
            print(f"\n{col}, {src}")
            print(df[df.source == src].pivot_table(index="imputer", columns="layout", values=col)
                  .reindex(METHODS)[list(LAYOUTS)].round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s")
    return 0 if g["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

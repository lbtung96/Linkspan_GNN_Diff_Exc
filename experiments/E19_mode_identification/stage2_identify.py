"""E19 stage 2 - B1, T1, B2, T2 from two or four sensors, with nothing given.

    python tools/run_stage.py experiments/E19_mode_identification/stage2_identify.py --workers 12

From the kept sensors' cross-spectra alone (stage1_spectra.py's cache on synthetic
decks, the recordings on real ones): pick the peaks, label them, and read each mode's
shape at its peak with E18's class prior (kriging over every node).  Methods:

  given           frequency and class given (E18's PB-class on the deflection route) - the
                  ceiling, and a gate: it must reproduce E18 on both data sets
  template        no measurement at all: the class prior's leading shape (class given) -
                  how much of a shape score the population alone already earns
  rank            ident.assign_rank: phase across the deck, then order
  pop-shape       ident.assign_population with the shape likelihood only (lam = 0)
  population      ident.assign_population with the frequency prior too

Two settings - the peak prominence (all methods) and lam (population) - are chosen per
layout on the 300 validation decks by the share of target modes identified correctly
(mean MAC breaks ties), then used unchanged on the 500 test decks and on the nine real
bridges.  Not by MAC: from two sensors the shape is mostly the class prior's, and a shape
read at the wrong peak scores almost as well.  The whole grid is kept, for the sensitivity.

Per target mode:
  correct    synthetic: the FE mode nearest the picked peak is the target mode;
             real: the picked peak is within 5 % of the SSI frequency
  present    some candidate peak would have been correct (detection, before labelling)
  ferr       |picked - true| frequency (FE target on synthetic, SSI on real)
  mac        the shape read at the picked peak, filled over the deck, against the true
             shape (synthetic: deck nodes; real: SSI at every sensor, the kept sensors'
             own deflections and the hidden ones filled - E18's pred_mac_label); a mode
             with no label scores 0

Layouts: synthetic outer n = 1 (two front corners) and n = 2 (front and mid-span pairs);
real front_pair and front_mid (E17), the same two arrangements on the recorded sensors.

Outputs: grid.csv (every setting, val / test / real means), test.csv and real.csv (per
deck and mode at the chosen settings), checks.json.
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
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
E18 = REPO / "experiments" / "E18_population_basis"
sys.path.insert(0, str(REPO / "experiments" / "E17_real_holdout"))
sys.path.insert(0, str(E18))
sys.path.insert(0, str(HERE))

import stage1_holdout as E17  # noqa: E402  (puts the pipeline, E5 and E16 on sys.path)
import build_phase3_graphs as BPG  # noqa: E402
import ident as ID  # noqa: E402
import popbasis as PB  # noqa: E402
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402
from phase3_utils import BANDPASS_F_HIGH, BANDPASS_F_LOW, bandpass_filter, load_graph_package  # noqa: E402
from stage1_spectra import F_MAX_HZ, X_TOL_FRAC  # noqa: E402

C = E17.C
M = list(C.MODE_NAMES)
LAYOUTS = {"outer1": "front_pair", "outer2": "front_mid"}
THETAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
LAMS = (0.1, 0.3, 1.0, 3.0)
REAL_TOL = 0.05
REAL_PAIR_TOL_MM = 300.0
_S: dict = {}


def init_worker(fprior) -> None:
    d = np.load(E18 / "prior.npz")
    assert int(d["deg_x"]) == PB.DEG_X and int(d["deg_y"]) == PB.DEG_Y
    _S["prior"] = {k: d[k] for k in d.files}
    _S["fprior"] = fprior


def prior(ng: int, key: str) -> tuple[np.ndarray, float]:
    return _S["prior"][f"Sigma_Ng{ng}_{key}"], float(_S["prior"][f"tau2_Ng{ng}_{key}"])


def leading(ng: int, c: str) -> np.ndarray:
    """The class prior's leading coefficient vector: its most typical shape."""
    w, v = np.linalg.eigh(prior(ng, c)[0])
    return v[:, -1]


def freq_prior() -> ID.FreqPrior:
    """From the main network's training decks: class frequencies (E18's prior_fit.csv) and geometry."""
    from stage1_spectra import features
    table: dict = {}
    for r in csv.DictReader((E18 / "prior_fit.csv").open(encoding="utf-8")):
        table.setdefault(r["sample"], {})[r["class"]] = float(r["freq_hz"])
    feats = {}
    for s in table:
        with np.load(C.POP_BROADBAND / s, allow_pickle=True) as d:
            p = _parse_params_json(d)
        feats[s] = (int(p["Ng"]), features(p))
    return ID.FreqPrior(table, feats)


def evaluate(G, f, sxy, kept_nodes, ng, ya, yb, x_tol, B, model, true_f, true_cls, score, given_bins, tscore):
    """Every (theta, method, lam) on one deck and layout.  score(m, cls, bin) -> mac of the shape
    read at `bin` with class `cls`'s prior, against target m; correct(m, f) and present use true_f."""
    Bs = B[kept_nodes]
    K = {c: Bs @ prior(ng, c)[0] @ Bs.T + prior(ng, c)[1] * np.eye(len(kept_nodes)) for c in PB.CLASSES}
    out = [("given", np.nan, np.nan, m, true_f[m], score(m, true_cls[m], given_bins[m]), True) for m in range(4)]
    out += [("template", np.nan, np.nan, m, np.nan, tscore(m, true_cls[m]), False) for m in range(4)]
    for th in THETAS:
        pk = ID.peaks(G, f, th)
        chars = [ID.character(p["u"], sxy, ya, yb, x_tol) for p in pk]
        shape_ll = np.array([[ID.acg_loglik(p["u"], K[c]) for c in PB.CLASSES] for p in pk]).reshape(len(pk), 4)
        assigns = [("rank", np.nan, ID.assign_rank(pk, chars)),
                   ("pop-shape", 0.0, ID.assign_population(pk, shape_ll, model, 0.0))]
        assigns += [("population", lam, ID.assign_population(pk, shape_ll, model, lam)) for lam in LAMS]
        for meth, lam, a in assigns:
            for m in range(4):
                i = a.get(true_cls[m])
                if i is None:
                    out.append((meth, th, lam, m, np.nan, 0.0, False))
                else:
                    out.append((meth, th, lam, m, pk[i]["f"], score(m, true_cls[m], pk[i]["bin"]), True))
    return out


def run_deck(args: tuple[str, str]) -> list[dict]:
    split, path = args
    d = np.load(path)
    ng, L, dy = int(d["Ng"]), float(d["L_mm"]), float(d["dy_mm"])
    coords, y, f = d["coords"], d["y"], d["f_axis"]
    deck = ~d["support"]
    true_f, cls = d["f_true"], [str(c) for c in d["classes"]]
    matched, freq_all = d["matched"], d["freq_all"]
    B = PB.design(*PB.norm_coords(coords, L, ng, dy))
    model = _S["fprior"].fit(ng, d["feat"])
    rows = []
    for lay in LAYOUTS:
        s, G = d[f"{lay}_nodes"], d[f"{lay}_G"]
        W = {c: PB.operator(B, s, *prior(ng, c)) for c in PB.CLASSES}

        def score(m, c, b):
            Gb = G[:, :, b]
            ref = int(np.argmax(np.real(np.diag(Gb))))
            dk = Gb[ref, :] / np.sqrt(Gb[ref, ref].real)
            return E17.mac_c((W[c] @ dk)[deck], y[deck, m])

        given_bins = [int(np.argmin(np.abs(f - fm))) for fm in true_f]
        res = evaluate(G, f, coords[s], s, ng, 0.0, (ng - 1) * dy, X_TOL_FRAC * L, B, model, true_f, cls, score,
                       given_bins, lambda m, c: E17.mac_c((B @ leading(ng, c))[deck], y[deck, m]))
        pk_sets = {th: ID.peaks(G, f, th) for th in THETAS}
        for meth, th, lam, m, fh, mac, labelled in res:
            nearest = int(np.argmin(np.abs(freq_all - fh))) if labelled and np.isfinite(fh) else -1
            present = (any(int(np.argmin(np.abs(freq_all - p["f"]))) == matched[m] for p in pk_sets[th])
                       if th in pk_sets else meth == "given")
            rows.append({"split": split, "sample": str(d["sample"]), "Ng": ng, "layout": lay, "method": meth,
                         "theta": th, "lam": lam, "mode": M[m], "class": cls[m], "f_true": float(true_f[m]),
                         "f_hat": fh, "labelled": labelled, "correct": bool(nearest == matched[m]),
                         "present": bool(present), "ferr": abs(fh - true_f[m]) if labelled else np.nan, "mac": mac})
    return rows


def run_real(fprior) -> list[dict]:
    init_worker(fprior)
    rows = []
    for s in C.LOOCV:
        pkg = load_graph_package(s, graph_root=str(E17.S6.GRAPH_ROOT))
        coords = pkg["node_coords_xy_mm"].astype(np.float64)
        n_fe = int(pkg["n_fe_nodes"])
        sens = pkg["sensor_node_indices"].astype(int)
        sxy = coords[sens]
        lay, (ya, yb) = E17.layouts(sxy, coords[:n_fe, 0].max())
        geo = BPG.STRUCTURES[s]
        L, ng, dy = float(geo["L_mm"]), int(geo["Ng"]), float(geo["dy_mm"])
        B = PB.design(*PB.norm_coords(coords, L, ng, dy))
        raw6 = np.load(E17.RUN / "decomposition_raw" / f"{s}.npz")
        y_ssi, f_ssi = raw6["y_ssi"].astype(np.float64), raw6["ssi_freq_hz"].astype(np.float64)
        cls = PB.classes(y_ssi, f_ssi, sxy, ya, yb, REAL_PAIR_TOL_MM)
        field = np.load(E17.FIELD_ROOT / s / "data" / "field_modal_filtering_outputs.npz", allow_pickle=False)
        u3 = field["raw_u3_timeseries_clean"].astype(np.float64).T
        fs = float(2.0 * field["freq_hz"].astype(np.float64)[-1])
        x = bandpass_filter(u3, fs, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)
        Gall, fax, _ = E17.csd_matrix(x, fs)
        nb = int(np.searchsorted(fax, F_MAX_HZ, side="right"))
        feat = np.array([np.log(L), np.log(dy), float(bool(geo["balcony"]["enable"])), float(geo["bottom_beam"]["x_frac"])])
        model = fprior.fit(ng, feat)
        for key, lname in LAYOUTS.items():
            kept = lay[lname]
            hidden = np.setdiff1d(np.arange(len(sens)), kept)
            G = Gall[np.ix_(kept, kept)][:, :, :nb]
            W = {c: PB.operator(B, sens[kept], *prior(ng, c)) for c in PB.CLASSES}

            def score(m, c, b):
                Gb = Gall[:, :, b]
                ref = int(kept[np.argmax(np.real(np.diag(Gb))[kept])])
                v = Gb[ref, :] / np.sqrt(Gb[ref, ref].real)                 # recorded deflection, every sensor
                v = v.copy()
                v[hidden] = W[c][sens[hidden]] @ v[kept]                     # hidden ones filled from the kept
                return E17.mac_c(v, y_ssi[:, m])

            given_bins = [int(np.argmin(np.abs(fax - fm))) for fm in f_ssi]
            res = evaluate(G, fax[:nb], sxy[kept], sens[kept], ng, ya, yb, REAL_PAIR_TOL_MM, B, model, f_ssi, cls,
                           score, given_bins, lambda m, c: E17.mac_c((B @ leading(ng, c))[sens], y_ssi[:, m]))
            pk_sets = {th: ID.peaks(G, fax[:nb], th) for th in THETAS}
            for meth, th, lam, m, fh, mac, labelled in res:
                ok = labelled and np.isfinite(fh) and abs(fh - f_ssi[m]) <= REAL_TOL * f_ssi[m]
                present = (any(abs(p["f"] - f_ssi[m]) <= REAL_TOL * f_ssi[m] for p in pk_sets[th])
                           if th in pk_sets else meth == "given")
                rows.append({"split": "real", "sample": s, "Ng": ng, "layout": key, "method": meth, "theta": th,
                             "lam": lam, "mode": M[m], "class": cls[m], "f_true": float(f_ssi[m]), "f_hat": fh,
                             "labelled": labelled, "correct": bool(ok), "present": bool(present),
                             "ferr": abs(fh - f_ssi[m]) if labelled else np.nan, "mac": mac})
    return rows


def gates(df: pd.DataFrame) -> dict:
    g = {}
    e18 = pd.read_csv(E18 / "synthetic.csv")
    e18 = e18[(e18.route == "ods") & (e18.method == "PB-class") & (e18.family == "outer")]
    mine = df[(df.split == "test") & (df.method == "given")]
    dmax, n = 0.0, 0
    for (smp, lay), q in mine.groupby(["sample", "layout"]):
        r = e18[(e18["sample"] == smp.replace(".npz", "")) & (e18.level == int(lay[-1]))]
        if len(r) != 1:
            continue
        n += 1
        dmax = max(dmax, max(abs(float(q[q["mode"] == m].mac.iloc[0]) - float(r[f"mac_{m}"].iloc[0])) for m in M))
    g["given_synthetic_reproduces_E18"] = {"pairs": n, "expected": int(mine[["sample", "layout"]].drop_duplicates().shape[0]),
                                           "max_abs_diff": dmax, "tol": 1e-5}
    g["given_synthetic_reproduces_E18"]["pass"] = (n == g["given_synthetic_reproduces_E18"]["expected"] and dmax <= 1e-5)
    e18r = pd.read_csv(E18 / "real.csv")
    e18r = e18r[(e18r.source == "real") & (e18r.imputer == "PB-class") & e18r.layout.isin(LAYOUTS.values())]
    mine = df[(df.split == "real") & (df.method == "given")]
    dmax, n = 0.0, 0
    for _, q in mine.iterrows():
        r = e18r[(e18r.structure == q["sample"]) & (e18r.layout == LAYOUTS[q["layout"]]) & (e18r["mode"] == q["mode"])]
        n += len(r)
        if len(r):
            dmax = max(dmax, abs(float(r.pred_mac_label.iloc[0]) - q["mac"]))
    g["given_real_reproduces_E18"] = {"rows": n, "expected": len(mine), "max_abs_diff": dmax, "tol": 1e-9,
                                      "pass": n == len(mine) and dmax <= 1e-9}
    g["pass"] = all(v["pass"] for v in g.values())
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    fprior = freq_prior()
    print(f"frequency prior from {len(fprior.samples)} training decks, {time.time() - t0:.0f}s", flush=True)
    jobs = [(split, str(p)) for split in ("val", "test") for p in sorted((HERE / "cache" / split).glob("*.npz"))]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(fprior,)) as ex:
        for i, r in enumerate(ex.map(run_deck, jobs, chunksize=4), 1):
            rows += r
            if i % 200 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    rows += run_real(fprior)
    df = pd.DataFrame(rows)
    df["sample"] = df["sample"].astype(str)

    # the grid, and the settings chosen on validation decks only
    keys = ["layout", "method", "theta", "lam"]
    grid = (df[~df.method.isin(["given", "template"])].fillna({"theta": -1, "lam": -1})
            .groupby(["split"] + keys).agg(mac=("mac", "mean"), correct=("correct", "mean"),
                                           present=("present", "mean"), labelled=("labelled", "mean"),
                                           ferr_median=("ferr", "median")).reset_index())
    chosen = {}
    for lay in LAYOUTS:
        for meth in ("rank", "pop-shape", "population"):
            v = grid[(grid.split == "val") & (grid.layout == lay) & (grid.method == meth)]
            b = v.sort_values(["correct", "mac"], ascending=False).iloc[0]
            chosen[f"{lay}/{meth}"] = {"theta": float(b.theta), "lam": float(b.lam),
                                      "val_correct": float(b.correct), "val_mac": float(b.mac)}
    grid.to_csv(HERE / "grid.csv", index=False)

    def pick(sub: pd.DataFrame) -> pd.DataFrame:
        keep = sub.method.isin(["given", "template"])
        for k, v in chosen.items():
            lay, meth = k.split("/")
            lam_ok = (sub.lam.fillna(-1) == v["lam"]) if meth != "rank" else True
            keep |= (sub.layout == lay) & (sub.method == meth) & (sub.theta == v["theta"]) & lam_ok
        return sub[keep]

    pick(df[df.split == "test"]).to_csv(HERE / "test.csv", index=False)
    pick(df[df.split == "real"]).to_csv(HERE / "real.csv", index=False)
    g = gates(df)
    (HERE / "checks.json").write_text(json.dumps({"gates": g, "chosen_on_val": chosen}, indent=1), encoding="utf-8")
    print(json.dumps({"gates": g, "chosen_on_val": chosen}, indent=1))
    for split, f in (("test", HERE / "test.csv"), ("real", HERE / "real.csv")):
        t = pd.read_csv(f)
        print(f"\n{split}: mean over decks x modes")
        print(t.groupby(["layout", "method"]).agg(mac=("mac", "mean"), correct=("correct", "mean"),
                                                 present=("present", "mean"), ferr_med=("ferr", "median"))
              .round(3).to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0 if g["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

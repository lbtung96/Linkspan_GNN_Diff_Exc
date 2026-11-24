"""T5 (EC-25) - conventional FDD and the class template on the same simulated records.

    python tools/run_stage.py pipeline/c_fdd.py \
        --sets val paper fresh_test --workers 4

Per deck (the main network's 300 validation decks - for tuning only -, the paper's 500 test
decks, the 500 fresh test decks):

1. Re-simulate from params_json exactly as s8 / E14 / E19 do (E19 stage1_spectra.resimulate,
   imported), with the excitation seed the set was generated with (paper and validation:
   the simulator's 12345; fresh sets: generation_config.json).  Gate: the re-simulated node
   PSD equals the stored x_psd (max |diff| recorded) and the target frequencies equal the
   stored ones - proves the records are the ones the network was scored on.
2. Layouts:
     all      every deck node measured (every graph node except the supports, whose
              records are the injected support excitation, not the deck's response)
     outer1   s8 outer n = 1, the two front corners             (E19's layouts)
     outer2   s8 outer n = 2, front and mid-span pairs
3. Per layout, E19's whole procedure, imported (ident.py, stage2_identify.evaluate and its
   prior/leading/freq_prior): FDD candidate peaks (largest eigenvalue of the kept sensors'
   Welch cross-spectral matrix, 1.5-20 Hz, prominence theta), labelled bend1/tors1/bend2/
   tors2 by rank / pop-shape / population, each mode's shape read at its peak (the
   deflection G(ref, :)/sqrt(G(ref, ref))) and filled over the deck with E18's class prior
   (for `all` the fill is the identity: every deck node is measured).  Plus E19's two
   references: `given` (true frequency and class) and `template` (no measurement: the class
   prior's leading shape, class given).  The run_deck body below is E19's run_deck with
   the deck passed in memory (a 150-node cross-spectral matrix is too big to cache) and the
   third layout added.
4. Settings: for outer1/outer2, E19's own choice on the validation decks (checks.json); for
   `all`, chosen here the same way (share of modes identified correctly on the 300
   validation decks, MAC breaks ties).  Never on a test set.
5. Runtime per deck, from the records: (a) FDD + population labelling + shape read at one
   fixed setting (theta 0.1, lam 3) on all deck nodes; (b) the network's input features
   (node Welch PSD + 800-d edge coherency, E14's own feature functions) - the network's
   forward pass is timed by c_rescore.py.

Gate: on the paper test set, outer1/outer2 at E19's chosen settings reproduce E19's test.csv
(mac, f_hat, correct) to 1e-5 - the labeller code path is E19's.

Writes data/T5/fdd/<set>_grid.csv.gz (every setting), and in results/T5_fresh_test_baselines/:
fdd_rows.csv (chosen settings, every test deck x layout x method x mode), fdd_timing.csv,
fdd_checks.json; data/T5/fdd/<set>_decks.npz (support masks, classes - for c_report.py).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ec_common as E  # noqa: E402

C = E.C
E19_DIR = C.REPO / "experiments" / "E19_mode_identification"
for _p in (C.REPO / "experiments" / "E14_corner_sensors", C.REPO / "experiments" / "E18_population_basis", E19_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import simulate_uniform_inertial_u3_white_noise as SIM  # noqa: E402
import stage2_identify as E19  # noqa: E402  (E19 stage 2; puts E17/E16/E5 on sys.path)
import stage1_spectra as E19S1  # noqa: E402
import stage1_corner_propagation as E14  # noqa: E402
import stage3_sensor_sweep as S3  # noqa: E402
import ident as ID  # noqa: E402
import popbasis as PB  # noqa: E402
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402
from stage2_synthetic import segment_spectra  # noqa: E402

OUT = E.task_dir("T5")
T5_DATA = E.EC_DATA / "T5"
FDD_DATA = T5_DATA / "fdd"
PAPER_EXC_SEED = 12345
M = list(C.MODE_NAMES)
LAYOUTS = {"all": None, "outer1": ("outer", 1), "outer2": ("outer", 2)}
TIMING_SETTING = (0.1, 3.0)


def set_files(test_set: str) -> tuple[list[Path], int]:
    if test_set == "val":
        return [C.POP_BROADBAND / n for n in (C.SPLITS / f"N{C.MAIN_N}_val.txt").read_text(encoding="utf-8").split()], PAPER_EXC_SEED
    if test_set == "paper":
        return [C.POP_BROADBAND / n for n in (C.SPLITS / "test.txt").read_text(encoding="utf-8").split()], PAPER_EXC_SEED
    root = T5_DATA / test_set
    cfg = json.loads((root / "generation_config.json").read_text(encoding="utf-8"))
    files = sorted((root / "samples_phase2_bandpass").glob("sample_*_Ng*_phase1_gnn.npz"))
    assert len(files) == 500, (test_set, len(files))
    return files, int(cfg["excitation_seed"])


def init_worker(fprior, exc_seed: int) -> None:
    SIM.SEED = int(exc_seed)              # the simulator reads it at call time
    E19.init_worker(fprior)


def run_deck(args: tuple[str, str]) -> dict:
    test_set, path = args
    t_all0 = time.time()
    with np.load(path, allow_pickle=True) as d:
        p = _parse_params_json(d)
        st = {k: d[k] for k in ("graph_node_ids", "edge_index", "coords_xy_mm", "y_mode_shape_signed", "y_freq_hz",
                                "matched_mode_indices", "matched_freq_hz", "x_psd", "freq_bins_hz")}
    sample = Path(path).name
    ng, L, dy = int(p["Ng"]), float(p["L_mm"]), float(p["dy_mm"])
    coords = st["coords_xy_mm"].astype(np.float64)
    y = st["y_mode_shape_signed"].astype(np.float64)
    true_f = st["y_freq_hz"].astype(np.float64).reshape(-1)
    matched = st["matched_mode_indices"].astype(int)
    cls = PB.classes(y, true_f, coords, 0.0, (ng - 1) * dy, E19S1.X_TOL_FRAC * L)
    t0 = time.time()
    sim = E19S1.resimulate(p, matched)
    t_sim = time.time() - t0
    assert np.array_equal(sim["graph_node_ids"], st["graph_node_ids"])
    freq_all = sim["freq_all_hz"]
    fs = sim["fs"]
    # reproduction: these are the records the stored features were computed from
    t0 = time.time()
    x_re = E14.node_psd_features(sim["a3"], fs)
    t_psd = time.time() - t0
    t0 = time.time()
    E14.edge_features(st["edge_index"].astype(np.int64), sim["a3"], fs, st["freq_bins_hz"])
    t_edge = time.time() - t0
    repro = {"x_psd_maxdiff": float(np.abs(x_re - st["x_psd"]).max()),
             "freq_maxdiff_hz": float(np.abs(freq_all[matched] - st["matched_freq_hz"]).max())}

    freq_axis = np.fft.rfftfreq(E14.WELCH_NFFT, 1.0 / fs)
    nb = int(np.searchsorted(freq_axis, E19S1.F_MAX_HZ, side="right"))
    f = freq_axis[:nb]
    deck = ~sim["support"]
    B = PB.design(*PB.norm_coords(coords, L, ng, dy))
    model = E19._S["fprior"].fit(ng, E19S1.features(p))
    rows, timing = [], {}
    for lay, spec in LAYOUTS.items():
        s = np.where(deck)[0] if spec is None else S3.station_nodes(coords, p, spec[1], spec[0])
        t0 = time.time()
        X = segment_spectra(sim["a3"][s], fs)[:, :, :nb]                           # (s, nseg, F)
        G = np.einsum("igf,jgf->ijf", np.conj(X), X) / X.shape[1]
        t_csd = time.time() - t0
        W = {c: PB.operator(B, s, *E19.prior(ng, c)) for c in PB.CLASSES}

        def score(m, c, b, G=G, W=W):
            Gb = G[:, :, b]
            ref = int(np.argmax(np.real(np.diag(Gb))))
            dk = Gb[ref, :] / np.sqrt(Gb[ref, ref].real)
            return E19.E17.mac_c((W[c] @ dk)[deck], y[deck, m])

        if lay == "all":            # one end-to-end pass at a fixed setting, timed
            t0 = time.time()
            th, lam = TIMING_SETTING
            Bs = B[s]
            K = {c: Bs @ E19.prior(ng, c)[0] @ Bs.T + E19.prior(ng, c)[1] * np.eye(len(s)) for c in PB.CLASSES}
            pk = ID.peaks(G, f, th)
            sll = np.array([[ID.acg_loglik(q["u"], K[c]) for c in PB.CLASSES] for q in pk]).reshape(len(pk), 4)
            a = ID.assign_population(pk, sll, model, lam)
            _ = [score(m, cls[m], pk[a[cls[m]]]["bin"]) for m in range(4) if cls[m] in a]
            timing = {"fdd_all_s": t_csd + (time.time() - t0), "n_deck_nodes": int(len(s))}
        given_bins = [int(np.argmin(np.abs(f - fm))) for fm in true_f]
        res = E19.evaluate(G, f, coords[s], s, ng, 0.0, (ng - 1) * dy, E19S1.X_TOL_FRAC * L, B, model, true_f, cls,
                           score, given_bins, lambda m, c: E19.E17.mac_c((B @ E19.leading(ng, c))[deck], y[deck, m]))
        pk_sets = {th: ID.peaks(G, f, th) for th in E19.THETAS}
        for meth, th, lam, m, fh, mac, labelled in res:
            nearest = int(np.argmin(np.abs(freq_all - fh))) if labelled and np.isfinite(fh) else -1
            present = (any(int(np.argmin(np.abs(freq_all - q["f"]))) == matched[m] for q in pk_sets[th])
                       if th in pk_sets else meth == "given")
            rows.append({"split": test_set, "sample": sample, "Ng": ng, "layout": lay, "method": meth,
                         "theta": th, "lam": lam, "mode": M[m], "class": cls[m], "f_true": float(true_f[m]),
                         "f_hat": fh, "labelled": labelled, "correct": bool(nearest == matched[m]),
                         "present": bool(present), "ferr": abs(fh - true_f[m]) if labelled else np.nan, "mac": mac})
    timing.update({"split": test_set, "sample": sample, "Ng": ng, "sim_s": t_sim, "net_features_s": t_psd + t_edge,
                   "net_psd_s": t_psd, "net_edge_s": t_edge, "deck_total_s": time.time() - t_all0, **repro})
    return {"rows": rows, "timing": timing, "support": sim["support"], "classes": cls, "sample": sample}


def chosen_all(grid_val: pd.DataFrame) -> dict:
    """E19's rule, for the `all` layout: best share correct on validation, MAC breaks ties."""
    g = (grid_val[~grid_val.method.isin(["given", "template"])].fillna({"theta": -1, "lam": -1})
         .groupby(["layout", "method", "theta", "lam"]).agg(mac=("mac", "mean"), correct=("correct", "mean"))
         .reset_index())
    out = {}
    for lay in LAYOUTS:
        for meth in ("rank", "pop-shape", "population"):
            v = g[(g.layout == lay) & (g.method == meth)]
            b = v.sort_values(["correct", "mac"], ascending=False).iloc[0]
            out[f"{lay}/{meth}"] = {"theta": float(b.theta), "lam": float(b.lam),
                                   "val_correct": float(b.correct), "val_mac": float(b.mac)}
    return out


def pick(sub: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    keep = sub.method.isin(["given", "template"])
    for k, v in chosen.items():
        lay, meth = k.split("/")
        lam_ok = (sub.lam.fillna(-1) == v["lam"]) if meth != "rank" else True
        keep |= (sub.layout == lay) & (sub.method == meth) & (sub.theta == v["theta"]) & lam_ok
    return sub[keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["val", "paper", "fresh_test"],
                    choices=["val", "paper", "fresh_test", "fresh_test_paperexc"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-ng", type=int, default=None, help="pilot: first K decks per Ng")
    args = ap.parse_args()
    FDD_DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fprior = E19.freq_prior()
    print(f"frequency prior from {len(fprior.samples)} training decks, {time.time() - t0:.0f}s", flush=True)
    timings = []
    for ts in args.sets:
        files, seed = set_files(ts)
        if args.per_ng:
            files = [x for g in C.NG_VALUES for x in [q for q in files if f"_Ng{g}_" in q.name][: args.per_ng]]
        rows, sup, cls = [], {}, {}
        t1 = time.time()
        with ProcessPoolExecutor(max_workers=min(args.workers, 4), initializer=init_worker,
                                 initargs=(fprior, seed)) as ex:
            for i, r in enumerate(ex.map(run_deck, [(ts, str(x)) for x in files], chunksize=2), 1):
                rows += r["rows"]
                timings.append(r["timing"])
                sup[r["sample"]], cls[r["sample"]] = r["support"], r["classes"]
                if i % 50 == 0 or i == len(files):
                    print(f"  [{ts} {i}/{len(files)}] {(time.time() - t1) / 60:.1f} min", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(FDD_DATA / f"{ts}_grid.csv.gz", index=False)
        np.savez(FDD_DATA / f"{ts}_decks.npz", samples=np.array(list(sup)), excitation_seed=seed,
                 **{f"support__{k}": v for k, v in sup.items()}, **{f"classes__{k}": np.array(v) for k, v in cls.items()})
    tpath = OUT / "fdd_timing.csv"          # accumulates: rows of the sets run now replace old ones
    old_t = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame(columns=["split"])
    tm = pd.concat([old_t[~old_t.split.isin(args.sets)], pd.DataFrame(timings)], ignore_index=True)
    tm.to_csv(tpath, index=False)

    # settings: outer layouts = E19's own choice; `all` = same rule on our validation grid
    e19 = json.loads((E19_DIR / "checks.json").read_text(encoding="utf-8"))["chosen_on_val"]
    val = pd.read_csv(FDD_DATA / "val_grid.csv.gz")
    mine = chosen_all(val)
    chosen = {k: v for k, v in mine.items() if k.startswith("all/")}
    chosen.update(e19)
    checks = {"chosen": chosen,
              "outer_choice_recomputed_equals_E19": {k: (mine[k]["theta"], mine[k]["lam"]) == (e19[k]["theta"], e19[k]["lam"])
                                                     for k in e19}}
    out_rows = []
    test_sets = [t for t in ("paper", "fresh_test", "fresh_test_paperexc") if (FDD_DATA / f"{t}_grid.csv.gz").exists()]
    for ts in test_sets:          # every test set on disk, not only those run now
        out_rows.append(pick(pd.read_csv(FDD_DATA / f"{ts}_grid.csv.gz"), chosen))
    allrows = pd.concat(out_rows)
    allrows.to_csv(OUT / "fdd_rows.csv", index=False)

    checks["reproduction"] = {ts: {"x_psd_maxdiff": float(q.x_psd_maxdiff.max()), "freq_maxdiff_hz": float(q.freq_maxdiff_hz.max()),
                                   "n": int(len(q))} for ts, q in tm.groupby("split")}
    checks["reproduction_pass"] = all(v["x_psd_maxdiff"] <= 1e-5 and v["freq_maxdiff_hz"] <= 1e-5
                                      for v in checks["reproduction"].values())
    if "paper" in test_sets:
        ref = pd.read_csv(E19_DIR / "test.csv")
        mine_t = allrows[(allrows.split == "paper") & allrows.layout.isin(["outer1", "outer2"])].copy()
        mine_t["sample"] = mine_t["sample"].str.replace(".npz", "", regex=False)
        key = ["sample", "layout", "method", "mode"]
        j = ref.merge(mine_t, on=key, suffixes=("_e19", "_t5"))
        dm = float(np.nanmax(np.abs(j.mac_e19 - j.mac_t5)))
        df_ = float(np.nanmax(np.abs(j.f_hat_e19.fillna(-1) - j.f_hat_t5.fillna(-1))))
        dc = int((j.correct_e19 != j.correct_t5).sum())
        checks["paper_outer_reproduces_E19_test_csv"] = {"rows_matched": int(len(j)), "rows_E19": int(len(ref)),
                                                         "max_abs_mac_diff": dm, "max_abs_fhat_diff": df_,
                                                         "correct_disagreements": dc,
                                                         "pass": len(j) == len(ref) and dm <= 1e-5 and df_ <= 1e-5 and dc == 0}
        tpl = allrows[(allrows.split == "paper") & (allrows.method == "template") & (allrows.layout == "all")]
        checks["template_paper_mean_mac"] = float(tpl.mac.mean())
        checks["template_paper_equals_E19_0946"] = bool(abs(tpl.mac.mean() - ref[ref.method == "template"].mac.mean()) <= 1e-9)
    (OUT / "fdd_checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    print(allrows.groupby(["split", "layout", "method"]).agg(mac=("mac", "mean"), correct=("correct", "mean"),
                                                             ferr=("ferr", "mean")).round(3).to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    ok = checks["reproduction_pass"] and checks.get("paper_outer_reproduces_E19_test_csv", {}).get("pass", True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""T14 - input anatomy: how do the network inputs differ between the paper's one-record population and
a population with a record per structure?  No training, no network (see m_occlusion.py for that).

    python tools/run_stage.py pipeline/m_input_anatomy.py [--workers 6]

Data (all already on disk, nothing simulated here):
  the 508 N600 training structures under FIVE records -
    paper   the paper's record (seed 12345, shared by every structure)   data/pop_broadband
    r0      SeedSequence([20261125, i])  (T11 trainval_varexc)
    r1..r3  SeedSequence([20261131..33, i])  (T13 rec1..rec3)
  the 500 paper test structures under the paper record and under T11's paper_varexc record.

Inputs: x_psd (row-normalised Welch PSD, 400 bins; analysed as log10) and edge_attr_cpsd_full
(complex coherency, Re/Im interleaved, 400 bins; one direction per undirected edge).  Only the
1.5-20 Hz band (the bandpass) is analysed.

(a) structure vs record.  For one structure under two records, the correlation ACROSS NODES (or
    edges) of the input at one frequency bin: 1 = the spatial pattern at that bin is set by the
    structure; 0 = set by the record.  Profiled against the bin's distance from each target mode.
(b) shared fingerprint.  Structure-level spectra (mean over support nodes / deck nodes / edges of the
    detrended input) correlated across bins, OFF resonance, between: two different bridges under the
    same paper record; two different bridges under their own records; one bridge under two records.
(c) the physical relative-sign cue.  At each target mode's frequency, how often sign(Re coherency)
    on an edge equals sign(phi_i * phi_j) of the target shape, and the mean Re coherency on anti-phase
    edges, under each record.
(d) a left/right label.  For mirror-symmetric decks, the left-minus-right asymmetry spectrum of the
    inputs; a fixed template (mean over the training structures of each Ng) scored on the test
    structures: the fraction whose asymmetry points the same way as the template.

Writes to results/T14_input_anatomy/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import k_common as K  # noqa: E402

H, EC, C = K.H, K.EC, K.C
OUT = EC.EC_RESULTS / "T14_input_anatomy"
MODES = C.MODE_NAMES
VERSIONS = ("paper", "r0", "r1", "r2", "r3")
SRC = {"paper": C.POP_BROADBAND, "r0": H.TRAINVAL_DIR, "r1": K.sdir("rec1"), "r2": K.sdir("rec2"), "r3": K.sdir("rec3")}
TEST_SRC = {"paper": C.POP_BROADBAND, "varexc": H.TEST_DIR}
OFFS = np.arange(-25, 26)
RES_W = 3            # "resonance" = within +-3 bins (+-0.29 Hz) of a target mode
OFF_W = 6            # "off resonance" = more than 6 bins (0.59 Hz) from every target mode
PHI_MIN = 0.3        # edges where both |phi| >= 0.3 (max-abs normalised target) carry a usable sign
FEATS = ("node log PSD", "edge Re coherency", "edge Im coherency")
SUMMARIES = ("support nodes, log PSD", "deck nodes, log PSD", "edges, Re coherency", "edges, |coherency|")


def _load(p: Path) -> dict:
    with np.load(p, allow_pickle=True) as d:
        return {k: d[k] for k in ("x_psd", "edge_attr_cpsd_full", "edge_index", "coords_xy_mm", "y_mode_shape_signed",
                                  "y_freq_hz", "freq_bins_hz", "Ng")}


def band(fb: np.ndarray) -> np.ndarray:
    return (fb >= 1.5) & (fb <= 20.0)


def colcorr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation per column (across rows)."""
    xc, yc = X - X.mean(0), Y - Y.mean(0)
    den = np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-12, (xc * yc).sum(0) / den, np.nan)


def vcorr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else np.nan


def feats(s: dict) -> dict:
    ei = np.asarray(s["edge_index"])
    keep = ei[0] < ei[1]
    e = np.asarray(s["edge_attr_cpsd_full"], float)[keep]
    return {"L": np.log10(np.asarray(s["x_psd"], float) + 1e-10), "Re": e[:, 0::2], "Im": e[:, 1::2],
            "ei": ei[:, keep], "ei_all": ei, "e_all": np.asarray(s["edge_attr_cpsd_full"], float)}


def detrend(x: np.ndarray) -> np.ndarray:
    from scipy.ndimage import median_filter
    return x - median_filter(x, size=(1, 13) if x.ndim == 2 else 13, mode="nearest")


def mirror_maps(coords: np.ndarray, ei_all: np.ndarray):
    y = coords[:, 1]
    yc = 0.5 * (y.min() + y.max())
    tol = 1e-6 * max(1.0, float(np.abs(coords).max()))
    perm = np.empty(coords.shape[0], dtype=np.int64)
    for i in range(coords.shape[0]):
        d = np.abs(coords[:, 0] - coords[i, 0]) + np.abs(coords[:, 1] - (2 * yc - coords[i, 1]))
        j = int(np.argmin(d))
        if d[j] > tol:
            return None
        perm[i] = j
    lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(ei_all.T)}
    left = y > yc + tol
    return perm, lookup, left, (y >= yc - tol)


def asymmetry(F: dict, mm) -> dict | None:
    if mm is None:
        return None
    perm, lookup, left, left_or_c = mm
    L = F["L"]
    a_node = (L[left] - L[perm[left]]).mean(0)
    re_all = F["e_all"][:, 0::2]
    rows = []
    for k, (a, b) in enumerate(F["ei_all"].T):
        if a < b and left_or_c[a] and left_or_c[b] and (left[a] or left[b]):
            m = lookup.get((int(perm[a]), int(perm[b])))
            if m is not None and m != k:
                rows.append(re_all[k] - re_all[m])
    a_re = np.mean(rows, axis=0) if rows else np.full(L.shape[1], np.nan)
    return {"node": a_node, "re": a_re}


def support_mask(ei_all: np.ndarray, n: int) -> np.ndarray:
    deg = np.bincount(ei_all[0], minlength=n)
    return deg == 1


# ------------------------------------------------------------------ per-structure worker
def train_job(name: str) -> dict:
    S = {v: _load(SRC[v] / name) for v in VERSIONS}
    s0 = S["paper"]
    for v in VERSIONS[1:]:
        if not np.array_equal(S[v]["edge_index"], s0["edge_index"]):
            raise RuntimeError(f"{name}: graph differs in version {v}")
    fb = np.asarray(s0["freq_bins_hz"], float)
    inband = band(fb)
    df = fb[1] - fb[0]
    F = {v: feats(S[v]) for v in VERSIONS}
    tb = np.rint(np.asarray(s0["y_freq_hz"], float) / df).astype(int)
    dist = np.min(np.abs(np.arange(fb.size)[:, None] - tb[None, :]), axis=1)
    res_bins, off_bins = inband & (dist <= RES_W), inband & (dist > OFF_W)

    # (a) cross-record spatial correlation per bin - deck nodes and deck-deck edges only: the support
    # nodes are 40-540x louder in every record and would make any across-node correlation look high
    n = F["paper"]["L"].shape[0]
    sup = support_mask(np.asarray(s0["edge_index"]), n)
    ei_u = F["paper"]["ei"]
    deck_e = ~sup[ei_u[0]] & ~sup[ei_u[1]]

    def cc(va, vb):
        return np.stack([colcorr(F[va]["L"][~sup], F[vb]["L"][~sup]), colcorr(F[va]["Re"][deck_e], F[vb]["Re"][deck_e]),
                         colcorr(F[va]["Im"][deck_e], F[vb]["Im"][deck_e])])     # (3, nbins)
    vv = np.nanmean(np.stack([cc(a, b) for i, a in enumerate(VERSIONS[1:]) for b in VERSIONS[2 + i:]]), axis=0)
    pv = np.nanmean(np.stack([cc("paper", b) for b in VERSIONS[1:]]), axis=0)
    prof = {}
    for key, M in (("vv", vv), ("pv", pv)):
        P = np.full((3, 4, OFFS.size), np.nan)
        for k, b in enumerate(tb):
            idx = b + OFFS
            ok = (idx >= 0) & (idx < fb.size)
            ok[ok] &= inband[idx[ok]]
            P[:, k, ok] = M[:, idx[ok]]
        prof[key] = P
    cls = {key: {"res": np.nanmean(M[:, res_bins], axis=1), "off": np.nanmean(M[:, off_bins], axis=1)}
           for key, M in (("vv", vv), ("pv", pv))}

    # (b) structure-level detrended spectra
    summ = {}
    for v in VERSIONS:
        dL = detrend(F[v]["L"])
        summ[v] = np.stack([dL[sup].mean(0), dL[~sup].mean(0), detrend(F[v]["Re"]).mean(0),
                            detrend(np.hypot(F[v]["Re"], F[v]["Im"])).mean(0)])   # (4, nbins)

    # (c) relative-sign cue at the target modes
    phi = np.asarray(s0["y_mode_shape_signed"], float)
    ei = F["paper"]["ei"]
    cue = {}
    for v in VERSIONS:
        rows = []
        for k, b in enumerate(tb):
            pi, pj = phi[ei[0], k], phi[ei[1], k]
            use = (np.abs(pi) >= PHI_MIN) & (np.abs(pj) >= PHI_MIN)
            sgn = np.sign(pi * pj)[use]
            re = F[v]["Re"][use, b]
            anti = sgn < 0
            rows.append([float(np.mean(np.sign(re) == sgn)) if use.any() else np.nan,
                         float(re[anti].mean()) if anti.any() else np.nan,
                         float(re[~anti].mean()) if (~anti).any() else np.nan,
                         int(use.sum()), int(anti.sum())])
        cue[v] = np.array(rows, dtype=float)                                     # (4 modes, 5)

    # (d) left/right asymmetry
    mm = mirror_maps(np.asarray(s0["coords_xy_mm"], float), np.asarray(s0["edge_index"]))
    asym = {v: asymmetry(F[v], mm) for v in ("paper", "r0")}
    return {"name": name, "Ng": int(s0["Ng"]), "tb": tb, "res_bins": res_bins, "off_bins": off_bins, "inband": inband,
            "prof": prof, "cls": cls, "summ": summ, "cue": cue, "asym": asym, "symmetric": mm is not None}


def test_job(name: str) -> dict:
    out = {"name": name, "Ng": None, "asym": {}, "cue": {}}
    for v, root in TEST_SRC.items():
        s = _load(root / name)
        out["Ng"] = int(s["Ng"])
        F = feats(s)
        mm = mirror_maps(np.asarray(s["coords_xy_mm"], float), np.asarray(s["edge_index"]))
        out["asym"][v] = asymmetry(F, mm)
        out["symmetric"] = mm is not None
    return out


# ------------------------------------------------------------------ analysis
def analyse(tr: list[dict], te: list[dict]) -> dict:
    rows_a, rows_b, rows_c, rows_d = [], [], [], []
    # (a)
    for key, lab in (("vv", "two varied records"), ("pv", "paper record vs a varied record")):
        for c in ("res", "off"):
            M = np.nanmean(np.stack([t["cls"][key][c] for t in tr]), axis=0)
            rows_a.append({"comparison": lab, "bins": "resonance (+-3 bins of a target mode)" if c == "res"
                           else "off resonance (> 6 bins from every target mode)",
                           **{f: float(M[i]) for i, f in enumerate(FEATS)}})
    prof = {key: np.nanmean(np.stack([t["prof"][key] for t in tr]), axis=0) for key in ("vv", "pv")}

    # (b) fingerprint correlations, off resonance, same-Ng pairs
    rng = np.random.default_rng(20261123)
    for ng in C.NG_VALUES:
        g = [t for t in tr if t["Ng"] == ng]
        pairs = [(i, j) for i in range(len(g)) for j in range(i + 1, len(g))]
        for si, slab in enumerate(SUMMARIES):
            acc = {"different bridges, same (paper) record": [], "different bridges, own records": [],
                   "same bridge, paper vs own record": [], "same bridge, two own records": []}
            for i, j in pairs:
                m = g[i]["off_bins"] & g[j]["off_bins"]
                acc["different bridges, same (paper) record"].append(vcorr(g[i]["summ"]["paper"][si, m], g[j]["summ"]["paper"][si, m]))
                acc["different bridges, own records"].append(vcorr(g[i]["summ"]["r0"][si, m], g[j]["summ"]["r0"][si, m]))
            for t in g:
                m = t["off_bins"]
                acc["same bridge, paper vs own record"].append(vcorr(t["summ"]["paper"][si, m], t["summ"]["r0"][si, m]))
                acc["same bridge, two own records"].append(vcorr(t["summ"]["r0"][si, m], t["summ"]["r1"][si, m]))
            for comp, v in acc.items():
                v = np.asarray(v, float)
                v = v[np.isfinite(v)]
                rows_b.append({"Ng": ng, "summary": slab, "comparison": comp, "n": v.size, "median": float(np.median(v)),
                               "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))})
    # (c)
    for v in VERSIONS:
        A = np.stack([t["cue"][v] for t in tr])                                  # (S, 4, 5)
        for k, m in enumerate(MODES):
            rows_c.append({"record": v, "mode": m, "sign_agreement": float(np.nanmean(A[:, k, 0])),
                           "mean_Re_anti_phase_edges": float(np.nanmean(A[:, k, 1])),
                           "mean_Re_in_phase_edges": float(np.nanmean(A[:, k, 2])),
                           "edges_used_per_structure": float(np.mean(A[:, k, 3])),
                           "anti_phase_edges_per_structure": float(np.mean(A[:, k, 4]))})
    # (d) side detector: template from training structures (per Ng, record kind), scored on test
    det_scores = {}
    for feat in ("node", "re"):
        for tkind, tests in (("paper", ("paper", "varexc")), ("r0", ("varexc", "paper"))):
            for ng in C.NG_VALUES:
                tt = [t for t in tr if t["Ng"] == ng and t["asym"][tkind] is not None]
                if not tt:
                    continue
                m = tt[0]["inband"]
                T = np.nanmean(np.stack([t["asym"][tkind][feat][m] for t in tt]), axis=0)
                for tv in tests:
                    z = [vcorr(e["asym"][tv][feat][m], T) for e in te if e["Ng"] == ng and e["asym"].get(tv) is not None]
                    z = np.asarray(z, float)
                    z = z[np.isfinite(z)]
                    det_scores[(feat, tkind, tv, ng)] = z
                    rows_d.append({"input": "node log PSD" if feat == "node" else "edge Re coherency",
                                   "template_from": "paper-record training structures" if tkind == "paper"
                                   else "own-record training structures (r0)",
                                   "scored_on": "test, paper record" if tv == "paper" else "test, own records (paper_varexc)",
                                   "Ng": ng, "n_train": len(tt), "n_test": int(z.size),
                                   "frac_same_side": float(np.mean(z > 0)) if z.size else np.nan,
                                   "median_corr_with_template": float(np.median(z)) if z.size else np.nan})
    return {"a": rows_a, "prof": prof, "b": rows_b, "c": rows_c, "d": rows_d, "det": det_scores}


def figures(tr: list[dict], R: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = 200.0 / 2048

    # F1 mode-centred profile
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.5), sharex=True, sharey="row")
    for r, f in enumerate(FEATS):
        for k, m in enumerate(MODES):
            ax = axes[r, k]
            ax.plot(OFFS * df, R["prof"]["vv"][r, k], color="#4c72b0", label="same bridge, two own records")
            ax.plot(OFFS * df, R["prof"]["pv"][r, k], color="#c44e52", ls="--", label="same bridge, paper vs own record")
            ax.axhline(0, color="k", lw=0.5)
            ax.axvspan(-RES_W * df, RES_W * df, color="#eee", zorder=0)
            if r == 0:
                ax.set_title(f"around the {m} frequency")
            if k == 0:
                ax.set_ylabel(f"{f}\ncorrelation across the graph")
            if r == 2:
                ax.set_xlabel("frequency - modal frequency (Hz)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Is the input at a frequency set by the structure (1) or by the record (0)?  508 bridges, 4 own records each", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_a_structure_vs_record.png", dpi=150)
    plt.close(fig)

    # F2 fingerprint
    import pandas as pd
    b = pd.DataFrame(R["b"])
    comps = list(dict.fromkeys(b.comparison))
    cols = ["#c44e52", "#4c72b0", "#dd8452", "#55a868"]
    fig, axes = plt.subplots(1, len(SUMMARIES), figsize=(14, 3.8), sharey=True)
    for ax, s in zip(axes, SUMMARIES):
        for k, c in enumerate(comps):
            g = b[(b.summary == s) & (b.comparison == c)]
            x = np.array(C.NG_VALUES) + (k - 1.5) * 0.18
            ax.errorbar(x, g["median"], yerr=[g["median"] - g["q25"], g["q75"] - g["median"]], fmt="o", ms=4,
                        color=cols[k], label=c)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(s, fontsize=9)
        ax.set_xticks(C.NG_VALUES, [f"Ng {n}" for n in C.NG_VALUES])
    axes[0].set_ylabel("correlation of the detrended spectra,\noff resonance (median, IQR)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Off resonance: do two bridges shaken by the same record look more alike than one bridge under two records?", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_b_shared_fingerprint.png", dpi=150)
    plt.close(fig)

    # F2b example
    g = [t for t in tr if t["Ng"] == 3][:2]
    if len(g) == 2:
        fb = np.arange(400) * df
        m = (fb >= 2) & (fb <= 8)
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(fb[m], g[0]["summ"]["paper"][1, m], color="#c44e52", lw=1.2, label=f"bridge A, paper record")
        ax.plot(fb[m], g[1]["summ"]["paper"][1, m] - 0.6, color="#8172b2", lw=1.2, label=f"bridge B, paper record (offset -0.6)")
        ax.plot(fb[m], g[0]["summ"]["r0"][1, m] - 1.2, color="#4c72b0", lw=1.2, label=f"bridge A, own record (offset -1.2)")
        for t, c in ((g[0], "#c44e52"), (g[1], "#8172b2")):
            for b_ in t["tb"]:
                if m[b_]:
                    ax.axvline(fb[b_], color=c, lw=0.6, ls=":")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("deck nodes, detrended log PSD")
        ax.legend(fontsize=7, ncol=3)
        ax.set_title(f"{g[0]['name']} (A) and {g[1]['name']} (B); dotted = their target modes", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig_b_example_two_bridges.png", dpi=150)
        plt.close(fig)

    # F3 cue
    c = pd.DataFrame(R["c"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    w = 0.16
    for k, v in enumerate(VERSIONS):
        g = c[c.record == v]
        axes[0].bar(np.arange(4) + (k - 2) * w, g["sign_agreement"], w, label=v, color="#c44e52" if v == "paper" else None)
        axes[1].bar(np.arange(4) + (k - 2) * w, g["mean_Re_anti_phase_edges"], w, label=v, color="#c44e52" if v == "paper" else None)
    for ax, t in zip(axes, ("sign(Re coherency) = sign(phi_i phi_j), at the modal frequency", "mean Re coherency on anti-phase edges")):
        ax.set_xticks(range(4), MODES)
        ax.set_title(t, fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=7, ncol=5)
    fig.suptitle("The physical relative-sign cue at each mode: paper record vs four own records", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c_relative_sign_cue.png", dpi=150)
    plt.close(fig)

    # F4 side detector
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, feat, lab in ((axes[0], "node", "node log PSD"), (axes[1], "re", "edge Re coherency")):
        for tv, col, l in (("paper", "#c44e52", "test, paper record"), ("varexc", "#4c72b0", "test, own records")):
            z = np.concatenate([R["det"].get((feat, "paper", tv, ng), np.zeros(0)) for ng in C.NG_VALUES])
            ax.hist(z, bins=np.linspace(-1, 1, 41), alpha=0.6, color=col, label=f"{l}: {np.mean(z > 0):.0%} same side")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"left-minus-right asymmetry, {lab}", fontsize=9)
        ax.set_xlabel("correlation with the paper-record template")
        ax.legend(fontsize=7)
    fig.suptitle("Does one fixed template tell the deck's left half from its right?", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_d_side_label.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train = H.read_list(K.TRAIN600)
    test = H.read_list(K.TEST_LIST)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        tr = list(pool.imap(train_job, train, chunksize=4))
        print(f"[load] train {len(tr)} in {time.time() - t0:.0f} s", flush=True)
        te = list(pool.imap(test_job, test, chunksize=4))
        print(f"[load] test {len(te)} in {time.time() - t0:.0f} s", flush=True)
    R = analyse(tr, te)
    H.write_csv(OUT / "a_structure_vs_record.csv", R["a"])
    H.write_csv(OUT / "b_shared_fingerprint.csv", R["b"])
    H.write_csv(OUT / "c_relative_sign_cue.csv", R["c"])
    H.write_csv(OUT / "d_side_label.csv", R["d"])
    prof_rows = []
    for key in ("vv", "pv"):
        for r, f in enumerate(FEATS):
            for k, m in enumerate(MODES):
                for o, val in zip(OFFS, R["prof"][key][r, k]):
                    prof_rows.append({"comparison": key, "feature": f, "mode": m, "offset_bins": int(o), "corr": float(val)})
    H.write_csv(OUT / "a_profile.csv", prof_rows)
    H.write_json(OUT / "counts.json", {"train_structures": len(tr), "test_structures": len(te),
                                       "train_mirror_symmetric": int(sum(t["symmetric"] for t in tr)),
                                       "test_mirror_symmetric": int(sum(t["symmetric"] for t in te)),
                                       "res_w_bins": RES_W, "off_w_bins": OFF_W, "phi_min": PHI_MIN})
    figures(tr, R)
    import pandas as pd
    pd.set_option("display.width", 220)
    for k in ("a", "c", "d"):
        print(pd.DataFrame(R[k]).round(3).to_string(index=False))
    b = pd.DataFrame(R["b"])
    print(b.groupby(["summary", "comparison"])["median"].mean().round(3).to_string())
    print(f"[done] {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

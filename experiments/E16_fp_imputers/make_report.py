"""Assemble REPORT.md, gates.json and the figures from stage1_imputer_sweep.py's CSVs.

    python tools/run_stage.py experiments/E16_fp_imputers/make_report.py

Generated, never hand-edited: every number in REPORT.md is computed here.
Unit of replication: the test structure (one network, one seed), so every
paired comparison is structure by structure against production FP on the same
structure, layout and level.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
S8 = REPO / "results" / "paper"
sys.path.insert(0, str(REPO / "pipeline"))
import fr_common as C  # noqa: E402

M = list(C.MODE_NAMES)
MAC = [f"mac_{m}" for m in M]
FERR = [f"ferr_{m}" for m in M]
IMPUTERS = ["Zero", "Mean", "NeighborMean", "FP", "FP-rw", "Harmonic", "Tikhonov", "PCFI"]
DIFFUSION = ["FP-rw", "Harmonic", "Tikhonov", "PCFI"]
FAMILIES = {"outer": "outer girders only (2n sensors)", "all": "every girder (Ng x n sensors)"}
GATE_ATOL = 1e-5            # the FE re-solve (eigsh, free start vector) moves MAC by < 1e-6

# s10's style: the validated reference palette, categorical slots in fixed order
S = {"FP": "#0b0b0b", "FP-rw": "#2a78d6", "Harmonic": "#eb6834", "Tikhonov": "#1baf7a",
     "PCFI": "#eda100", "NeighborMean": "#e87ba4"}
MARK = {"FP": "o", "FP-rw": "s", "Harmonic": "^", "Tikhonov": "D", "PCFI": "v", "NeighborMean": "P"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "lines.linewidth": 1.6, "lines.markersize": 5.5, "lines.solid_capstyle": "round",
    "axes.axisbelow": True,
})


def load():
    ps = pd.read_csv(HERE / "per_sample.csv")
    ce = pd.read_csv(HERE / "ceiling.csv")
    dg = pd.read_csv(HERE / "diagnostics.csv")
    for df in (ps, ce):
        df["mac"] = df[MAC].mean(axis=1)
    ps["ferr"] = ps[FERR].mean(axis=1)
    return ps, ce, dg


def gates(ps: pd.DataFrame, dg: pd.DataFrame) -> dict:
    """FP, full and zeros rows against s8, structure by structure."""
    out = {}
    for fam in FAMILIES:
        s8 = pd.read_csv(S8 / f"s8_sensor_sweep_{fam}.csv")
        mine = pd.concat([ps[(ps.family == fam) & (ps.imputer == "FP")],
                          ps[ps.variant.isin(["full", "zeros"])]])
        j = mine.merge(s8, on=["sample", "variant"], suffixes=("", "_s8"), validate="one_to_one")
        diff = max(float(np.abs(j[c] - j[f"{c}_s8"]).max()) for c in MAC + FERR)
        out[fam] = {"rows_matched": int(len(j)), "rows_s8": int(len(s8)),
                    "max_abs_diff": diff,
                    "n_sensors_equal": bool((j.n_sensors == j.n_sensors_s8).all()),
                    "pass": bool(len(j) == len(s8) and diff <= GATE_ATOL
                                 and (j.n_sensors == j.n_sensors_s8).all())}
    out["resim_x_psd_maxdiff"] = float(dg.x_psd_resim_maxdiff.max())
    out["structures"] = int(dg["sample"].nunique())
    out["atol"] = GATE_ATOL
    out["pass"] = all(out[f]["pass"] for f in FAMILIES)
    return out


def levels(df: pd.DataFrame) -> list[int]:
    return sorted(int(v) for v in df.loc[df.level > 0, "level"].unique())


def paired(ps: pd.DataFrame, fam: str, imp: str, lv: int, col: str) -> dict:
    a = ps[(ps.family == fam) & (ps.level == lv) & (ps.imputer == imp)].set_index("sample")[col]
    b = ps[(ps.family == fam) & (ps.level == lv) & (ps.imputer == "FP")].set_index("sample")[col]
    d = (a - b.loc[a.index]).to_numpy()
    half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"d": d.mean(), "half": half, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "n": d.size, "p": p}


def fmt_p(p: float) -> str:
    return "<1e-4" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}"


def mean_table(df: pd.DataFrame, fam: str, col: str, imps: list[str], best=max, extra=None) -> list[str]:
    lv = levels(df)
    sub = df[df.family == fam]
    sens = [sub[sub.level == v].n_sensors.mean() for v in lv]
    piv = sub.groupby(["imputer", "level"])[col].mean().unstack()
    out = ["| imputer | " + " | ".join(f"n{v} ({s:.1f})" for v, s in zip(lv, sens)) + " |",
           "|---|" + "---:|" * len(lv)]
    for imp in imps:
        cells = []
        for v in lv:
            x = piv.loc[imp, v]
            top = best(piv.loc[imps, v])
            cells.append(f"**{x:.3f}**" if np.isclose(x, top) else f"{x:.3f}")
        out.append(f"| {imp} | " + " | ".join(cells) + " |")
    for name, vals in (extra or []):
        out.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in vals) + " |")
    return out


def first_reach(ps: pd.DataFrame, fam: str, imp: str, target: float) -> str:
    sub = ps[(ps.family == fam) & (ps.imputer == imp)]
    for v in levels(sub):
        s = sub[sub.level == v]
        if s.mac.mean() >= target:
            return f"n{v} ({s.n_sensors.mean():.1f})"
    return "not reached"


# ----------------------------------------------------------------------------- figures
def fig_delta(ps: pd.DataFrame) -> str:
    """Paired MAC difference against production FP, per family."""
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True, sharey=True)
    for ax, fam in zip(axs, FAMILIES):
        lv = levels(ps)
        x = [ps[(ps.family == fam) & (ps.level == v)].n_sensors.mean() for v in lv]
        ax.axhline(0, color=INK, linewidth=1.0)
        ax.annotate("production FP", (x[-1], 0), xytext=(0, -10), textcoords="offset points",
                    ha="right", fontsize=7.5, color=INK2)
        for imp in DIFFUSION:
            r = [paired(ps, fam, imp, v, "mac") for v in lv]
            d, h = np.array([q["d"] for q in r]), np.array([q["half"] for q in r])
            ax.fill_between(x, d - h, d + h, color=S[imp], alpha=0.12, linewidth=0)
            ax.plot(x, d, color=S[imp], marker=MARK[imp], markersize=4, label=imp)
        ax.set_title(FAMILIES[fam].split(" (")[0].capitalize(), loc="left", color=INK)
        ax.set_xlabel("measured nodes (mean over test structures)")
        ax.set_xlim(0, None)
    axs[0].set_ylabel("network MAC minus production FP")
    axs[1].legend(loc="lower right", fontsize=8)
    p = HERE / "fig1_network_mac_vs_fp.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p.name


def fig_absolute(ps: pd.DataFrame, ce: pd.DataFrame, best: dict) -> str:
    """The paper's fig3 (FP), the best alternative per family, and the imputation ceiling."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    full = ps[ps.variant == "full"]
    zero = ps[ps.variant == "zeros"]
    for fam, c, mk in (("outer", "#2a78d6", "o"), ("all", "#eb6834", "s")):
        lv = levels(ps)
        sub = ps[ps.family == fam]
        x = [sub[sub.level == v].n_sensors.mean() for v in lv]
        lab = "outer" if fam == "outer" else "all"
        for imp, ls in (("FP", "-"), (best[fam], (0, (4, 2)))):
            ym = [sub[(sub.level == v) & (sub.imputer == imp)].mac.mean() for v in lv]
            yf = [sub[(sub.level == v) & (sub.imputer == imp)].ferr.mean() for v in lv]
            a.plot(x, ym, color=c, marker=mk, markersize=4, linestyle=ls, label=f"{lab}, {imp}")
            b.plot(x, yf, color=c, marker=mk, markersize=4, linestyle=ls, label=f"{lab}, {imp}")
        cs = ce[(ce.family == fam) & (ce.imputer == "FP") & (ce.supports == "free")]
        yc = [cs[cs.level == v].mac.mean() for v in lv]
        a.plot(x, yc, color=c, linewidth=1.0, linestyle=(0, (1, 1.5)),
               label=f"{lab}, FP on the true mode shape (no network)")
    for ax, yv, txt, dy in ((a, full.mac.mean(), "every node measured", -10),
                            (a, zero.mac.mean(), "nothing measured", 3),
                            (b, full.ferr.mean(), "every node measured", 3)):
        ax.axhline(yv, color=MUTED, linewidth=0.9)
        ax.annotate(txt, (1.0, yv), xycoords=("axes fraction", "data"), xytext=(-2, dy),
                    textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    a.set_ylim(0, 1)
    a.set_xlabel("measured nodes (mean over test structures)")
    a.set_ylabel("mean MAC")
    a.set_title("Mode shapes", loc="left", color=INK)
    b.set_ylim(bottom=0)
    b.set_xlabel("measured nodes (mean over test structures)")
    b.set_ylabel("mean |frequency error| (Hz)")
    b.set_title("Natural frequencies", loc="left", color=INK)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=7.5)
    p = HERE / "fig2_sensor_sweep_best_imputer.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p.name


def fig_ceiling(ce: pd.DataFrame, ps: pd.DataFrame) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True, sharey=True)
    for ax, fam in zip(axs, FAMILIES):
        lv = levels(ce)
        x = [ce[(ce.family == fam) & (ce.level == v)].n_sensors.mean() for v in lv]
        for imp in ["FP"] + DIFFUSION:
            for sup, ls in (("free", "-"), ("clamped", (0, (4, 2)))):
                s = ce[(ce.family == fam) & (ce.imputer == imp) & (ce.supports == sup)]
                y = [s[s.level == v].mac.mean() for v in lv]
                ax.plot(x, y, color=S[imp], linestyle=ls, marker=MARK[imp], markersize=3.5,
                        linewidth=1.2, label=f"{imp}" if sup == "free" else None)
        net = ps[(ps.family == fam) & (ps.imputer == "FP")]
        ax.plot(x, [net[net.level == v].mac.mean() for v in lv], color=MUTED, linewidth=1.0,
                linestyle=(0, (1, 1.5)), label="network on FP inputs")
        ax.set_title(FAMILIES[fam].split(" (")[0].capitalize(), loc="left", color=INK)
        ax.set_xlabel("measured nodes (mean over test structures)")
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("MAC of the imputed true mode shape")
    axs[1].legend(loc="lower right", fontsize=7.5, title="solid: hinge free\ndashed: hinge at rest",
                  title_fontsize=7.5)
    p = HERE / "fig3_imputation_ceiling.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p.name


# ----------------------------------------------------------------------------- prose
def answer(ps: pd.DataFrame, ce: pd.DataFrame) -> list[str]:
    """The findings, every number computed."""
    lv = levels(ps)
    fams = list(FAMILIES)
    P = {(f, i, v): paired(ps, f, i, v, "mac") for f in fams for i in DIFFUSION for v in lv}
    sens = {(f, v): ps[(ps.family == f) & (ps.level == v)].n_sensors.mean() for f in fams for v in lv}
    net = {(f, i, v): ps[(ps.family == f) & (ps.imputer == i) & (ps.level == v)].mac.mean()
           for f in fams for i in IMPUTERS for v in lv}
    cei = {(f, i, s, v): ce[(ce.family == f) & (ce.imputer == i) & (ce.supports == s) & (ce.level == v)].mac.mean()
           for f in fams for i in ["FP"] + DIFFUSION for s in ("free", "clamped") for v in lv}
    full = ps[ps.variant == "full"].mac.mean()

    def at(f, v):
        return f"{FAMILIES[f].split(' (')[0]}, n{v} ({sens[(f, v)]:.0f} nodes)"

    rw = max(abs(P[(f, "FP-rw", v)]["d"]) for f in fams for v in lv)
    h_worst = min(((f, v) for f in fams for v in lv), key=lambda k: P[(k[0], "Harmonic", k[1])]["d"])
    t = [P[(f, "Tikhonov", v)]["d"] for f in fams for v in lv]
    pc = {f: [P[(f, "PCFI", v)] for v in lv] for f in fams}
    pc_best = max(((f, v) for f in fams for v in lv), key=lambda k: P[(k[0], "PCFI", k[1])]["d"])
    pc_neg = [(f, v) for f in fams for v in lv if P[(f, "PCFI", v)]["d"] < 0 and P[(f, "PCFI", v)]["p"] < 0.05]
    reach = [(f, tg, first_reach(ps, f, "FP", tg), first_reach(ps, f, "PCFI", tg))
             for f in fams for tg in (0.6, 0.7, 0.8)]
    reach_diff = [r for r in reach if r[2] != r[3]]
    fe = {f: [ps[(ps.family == f) & (ps.imputer == "FP") & (ps.level == v)].ferr.mean()
              - ps[(ps.family == f) & (ps.imputer == "PCFI") & (ps.level == v)].ferr.mean() for v in lv]
          for f in fams}
    # from which level on Harmonic stays within 0.005 of FP; from which NeighborMean has the lowest |df|
    h_level = {f: next(v for v in lv if all(abs(P[(f, "Harmonic", w)]["d"]) <= 0.005 for w in lv if w >= v))
               for f in fams}
    fe_tab = {f: ps[ps.family == f].groupby(["level", "imputer"]).ferr.mean().unstack() for f in fams}
    nm_level = {f: next((v for v in lv if all(fe_tab[f].loc[w].idxmin() == "NeighborMean" for w in lv if w >= v)),
                        None) for f in fams}
    # the hinge rows' share of the stored full node PSD
    share, frac = [], []
    for s in ps[ps.variant == "full"]["sample"]:
        with np.load(C.POP_BROADBAND / f"{s}.npz", allow_pickle=True) as d:
            x2 = (d["x_psd"].astype(np.float64) ** 2).sum(axis=1)
            h = np.abs(d["coords_xy_mm"][:, 0]) < 1.0
        share.append(x2[h].sum() / x2.sum())
        frac.append(h.mean())
    def spread(vs):
        return max(max(cei[(f, i, "free", v)] for i in ["FP"] + DIFFUSION)
                   - min(cei[(f, i, "free", v)] for i in ["FP"] + DIFFUSION) for f in fams for v in vs)
    top_n1 = {f: max(["FP"] + DIFFUSION, key=lambda i: cei[(f, i, "free", lv[0])]) for f in fams}
    clamp = [cei[(f, "FP", "clamped", v)] - cei[(f, "FP", "free", v)] for f in fams for v in lv]
    beat_full = {f: next((v for v in lv if cei[(f, "FP", "free", v)] >= full), None) for f in fams}

    A = ["## The answer\n"]
    A.append(
        "**No feature-propagation algorithm changes the sensor curve.** Swapped into the same pipeline, "
        "the diffusion imputers land within a few hundredths of production FP, and only one is better:\n")
    A.append(f"- **FP-rw** (random-walk normalisation) is FP: |difference| ≤ {rw:.3f} MAC at every level.")
    hw = P[(h_worst[0], "Harmonic", h_worst[1])]
    A.append(f"- **Harmonic** - FP-rw run to convergence, the repository's overall winner - is worse with few "
             f"sensors (down to {hw['d']:+.3f} at {at(*h_worst)}) and within 0.005 of FP from "
             + " and ".join(at(f, v) for f, v in h_level.items()) + " up.")
    A.append(f"- **Tikhonov** is worse everywhere ({max(t):+.3f} to {min(t):+.3f}). It does not pin the "
             "measured values, so it smooths the sensors' own records too.")
    po, pa = pc["outer"], pc["all"]
    pb = P[(pc_best[0], "PCFI", pc_best[1])]
    A.append(
        f"- **PCFI** is the only improvement: {min(q['d'] for q in po):+.3f} to {max(q['d'] for q in po):+.3f} "
        f"on the outer girders and {min(q['d'] for q in pa):+.3f} to {max(q['d'] for q in pa):+.3f} on every "
        f"girder. The largest gain is {pb['d']:+.3f} at {at(*pc_best)}: "
        f"{net[(pc_best[0], 'FP', pc_best[1])]:.3f} -> {net[(pc_best[0], 'PCFI', pc_best[1])]:.3f}, "
        f"{pb['wins']} of {pb['n']} structures better."
        + (" It is significantly *worse* at " + ", ".join(
            f"{at(f, v)} ({P[(f, 'PCFI', v)]['d']:+.3f})" for f, v in pc_neg) + "." if pc_neg else ""))
    A.append("- The naive baselines (Zero, Mean, NeighborMean) are far worse in MAC.\n")
    if reach_diff:
        A.append(f"Sensors needed for a target mean MAC (0.6, 0.7, 0.8) change in {len(reach_diff)} of "
                 f"{len(reach)} cases between FP and PCFI: "
                 + "; ".join(f"{FAMILIES[f].split(' (')[0]} {tg}: FP {a}, PCFI {b} "
                             f"(where FP scores {net[(f, 'FP', int(b.split()[0][1:]))]:.3f} and PCFI "
                             f"{net[(f, 'PCFI', int(b.split()[0][1:]))]:.3f})" for f, tg, a, b in reach_diff)
                 + ". A threshold crossing by a few thousandths, not a material gain.\n")
    A.append(f"Frequency error moves the same way and as little: PCFI lowers it by at most "
             f"{max(max(fe['outer']), max(fe['all'])):.3f} Hz. (NeighborMean has the lowest frequency error "
             "of all imputers from "
             + " and ".join(at(f, v) for f, v in nm_level.items() if v is not None)
             + " up, despite its poor MAC. Noticed, not pursued.)\n")
    A.append(
        "**Why so little moves.** The network's inputs are scale-free by construction: node PSDs are "
        "row-normalised by their maximum, and the coherency divides by both autospectra. An imputer therefore "
        "reaches the network only through the proportions in which it mixes the sensors' records at each "
        "unmeasured node, never through the amplitude it assigns there. The ordering is consistent with "
        "locality being what helps, though this run does not isolate it. PCFI weights each node towards its "
        "nearest sensors and does best. FP and FP-rw stop after 50 iterations, which also favours near "
        "sensors. Harmonic, their converged limit, mixes distant sensors more evenly and loses with few "
        "sensors. On the inputs themselves the diffusion imputers sit close together (see *Inputs*), and "
        "Tikhonov has the largest coherency error of the four. Every imputer leaves the node PSD far from "
        f"the full one. Part of that is the hinge rows: {100 * np.mean(frac):.1f} % of the nodes, carrying "
        "broadband base-excitation spectra that no deck sensor reproduces, with "
        f"{100 * np.mean(share):.0f} % of the squared norm on average.\n")
    A.append(
        "**Where the loss actually is.** Run on the true mode shapes from their exact values at the same "
        "sensors, the same graph interpolation does far better than the network on the interpolated "
        f"spectra. On the outer girders it scores {cei[('outer', 'FP', 'free', 4)]:.3f} against "
        f"{net[('outer', 'FP', 4)]:.3f} at n4 ({sens[('outer', 4)]:.0f} nodes) and "
        f"{cei[('outer', 'FP', 'free', 12)]:.3f} against {net[('outer', 'FP', 12)]:.3f} at n12. On every "
        f"girder it scores {cei[('all', 'FP', 'free', 4)]:.3f} against {net[('all', 'FP', 4)]:.3f} at n4 "
        f"and {cei[('all', 'FP', 'free', 12)]:.3f} against {net[('all', 'FP', 12)]:.3f} at n12. It passes "
        f"the network's own every-node score ({full:.3f}) at "
        + " and ".join(f"{at(f, v)}" for f, v in beat_full.items() if v is not None)
        + f". The choice of imputer matters only at n{lv[0]}: the diffusion imputers differ there by up to "
        f"{spread(lv[:1]):.3f} (highest: " + ", ".join(f"{top_n1[f]} on {FAMILIES[f].split(' (')[0]}"
                                                       for f in fams)
        + f"), and by at most {spread(lv[1:]):.3f} from n{lv[1]} up. Clamping the hinge line at rest adds "
        f"{min(clamp):+.3f} to {max(clamp):+.3f} for FP. "
        "So interpolation is not the bottleneck. The loss sits between interpolated response spectra and a "
        "network trained only on fully measured decks, which is E14's conclusion seen from the imputer side. "
        "The comparison is an upper bound, not a pipeline: it is given the mode shape at the sensors, which "
        "would have to come from modal identification there. Whether operational modal analysis at the "
        "sensors followed by FP of the mode shape would hold up is not tested here.\n")
    return A


# ----------------------------------------------------------------------------- report
def main() -> int:
    ps, ce, dg = load()
    g = gates(ps, dg)
    (HERE / "gates.json").write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
    lv = levels(ps)
    full, zero = ps[ps.variant == "full"], ps[ps.variant == "zeros"]
    # the alternative with the largest mean paired MAC gain over FP, per family
    best = {}
    for fam in FAMILIES:
        gain = {imp: np.mean([paired(ps, fam, imp, v, "mac")["d"] for v in lv]) for imp in DIFFUSION}
        best[fam] = max(gain, key=gain.get)
    figs = [fig_delta(ps), fig_absolute(ps, ce, best), fig_ceiling(ce, ps)]

    A = []
    A.append("# E16 - would another feature-propagation algorithm need fewer sensors?\n")
    A.append("**Generated by `make_report.py` from `stage1_imputer_sweep.py`'s CSVs. Do not hand-edit.**\n")
    A.append(
        f"The main network (`{C.sweep_run_name(C.MAIN_N)}`, best-validation checkpoint) on the fixed "
        f"test set ({g['structures']} structures, 125 per girder count), exactly as "
        "`pipeline/s8_sensor_sweep.py` runs it, except for the operator that fills "
        "the unmeasured nodes. s8 uses the production Feature Propagation (Rossi et al. 2022; "
        "D^-1/2 A D^-1/2, 50 iterations, measured rows reset). Here every model-free imputer benchmarked "
        "in [lbtung96/Feature-propagation](https://github.com/lbtung96/Feature-propagation) takes its "
        "place in turn (`imputers.py`, adapted from its `fpbench/imputers.py` at commit 9b7582d):\n")
    A.append("| imputer | what it is |\n|---|---|")
    for k, v in (("Zero, Mean, NeighborMean", "naive baselines (unmeasured node = 0 / sensor mean / mean of measured neighbours)"),
                 ("**FP**", "production operator, D^-1/2 A D^-1/2, 50 iterations from zero - the s8 reference"),
                 ("FP-rw", "random-walk normalisation D^-1 A, 50 iterations"),
                 ("Harmonic", "exact harmonic extension, the fixed point FP-rw converges to (direct solve)"),
                 ("Tikhonov", "(M + L_sym) x = M y: Laplacian-regularised fit, measured values not pinned"),
                 ("PCFI", "pseudo-confidence FP (Um et al. 2023): diffusion weighted by alpha^(hops to a sensor), "
                          "then a cross-channel refinement (PSD bins; time series get the diffusion only)")):
        A.append(f"| {k} | {v} |")
    A.append(
        "\nEverything else is s8's: the FE solve and broadband simulation rebuilt from `params_json`, "
        "the station layouts (n stations per girder line at x = L(1 - k/n), front beam always measured, "
        "hinge line never), the feature order (Welch PSD at the sensors -> impute -> row-normalise; "
        "time series imputed -> complex coherency on every edge), one CPU thread per worker. "
        "The learned imputers in that repository (SAT, GINN) and its end-to-end models (PaGNN, GCNmf) "
        "are not run: they would replace or retrain the network. Neither is its `ModalExpansion`: every "
        "structure here is generated from a known FE model of its own geometry, so a nominal-model "
        "expansion would be an oracle.\n")
    A.append("Comparisons are paired by structure against FP on the same structure, layout and level; "
             "the unit of replication is the structure (one network, one seed). Columns are n stations "
             "per girder, with the mean number of measured nodes in brackets.\n")
    A += answer(ps, ce)

    A.append("## Gate: the FP rows are s8\n")
    for fam in FAMILIES:
        q = g[fam]
        A.append(f"- **{fam}**: {q['rows_matched']} of {q['rows_s8']} s8 rows matched (FP, full, zeros); "
                 f"max |difference| in MAC or frequency error {q['max_abs_diff']:.1e} "
                 f"(tolerance {GATE_ATOL:.0e}, the FE re-solve's ARPACK noise) - "
                 f"**{'PASS' if q['pass'] else 'FAIL'}**")
    A.append(f"- re-simulation reproduces the stored node features to {g['resim_x_psd_maxdiff']:.1e}\n")

    A.append("## Network: mean MAC over B1, T1, B2, T2\n")
    A.append(f"Every node measured: **{full.mac.mean():.3f}**; nothing measured (zero input): "
             f"**{zero.mac.mean():.3f}**. Best per column in bold.\n")
    for fam, title in FAMILIES.items():
        A.append(f"*{title}:*\n")
        A += mean_table(ps, fam, "mac", IMPUTERS)
        A.append("")

    A.append("## Network: difference from production FP (paired, per structure)\n")
    A.append("Mean MAC difference ± 95 % CI half-width; then structures improved / worsened, and the "
             "Wilcoxon signed-rank p. Positive = better than FP.\n")
    for fam, title in FAMILIES.items():
        A.append(f"*{title}:*\n")
        A.append("| imputer | " + " | ".join(f"n{v}" for v in lv) + " |")
        A.append("|---|" + "---:|" * len(lv))
        for imp in ["NeighborMean"] + DIFFUSION:
            cells = []
            for v in lv:
                q = paired(ps, fam, imp, v, "mac")
                cells.append(f"{q['d']:+.3f} ±{q['half']:.3f}<br>{q['wins']}/{q['losses']}, p {fmt_p(q['p'])}")
            A.append(f"| {imp} | " + " | ".join(cells) + " |")
        A.append("")

    A.append("## Network: mean |frequency error| (Hz)\n")
    A.append(f"Every node measured: **{full.ferr.mean():.3f}** Hz; nothing measured: "
             f"**{zero.ferr.mean():.3f}** Hz. Lowest per column in bold.\n")
    for fam, title in FAMILIES.items():
        A.append(f"*{title}:*\n")
        A += mean_table(ps, fam, "ferr", IMPUTERS, best=min)
        A.append("")

    A.append("## Network: MAC per mode\n")
    for fam, title in FAMILIES.items():
        A.append(f"*{title}:*\n")
        A.append("| level | imputer | B1 | T1 | B2 | T2 | mean |")
        A.append("|---|---|---:|---:|---:|---:|---:|")
        for v in (1, 2, 4, 8):
            for imp in ["FP"] + DIFFUSION:
                s = ps[(ps.family == fam) & (ps.level == v) & (ps.imputer == imp)]
                A.append(f"| n{v} | {imp} | " + " | ".join(f"{s[c].mean():.3f}" for c in MAC)
                         + f" | {s.mac.mean():.3f} |")
        A.append("")

    A.append("## Network: mean MAC by girder count\n")
    for fam, title in FAMILIES.items():
        A.append(f"*{title}, n4:*\n")
        A.append("| imputer | Ng = 2 | Ng = 3 | Ng = 4 | Ng = 5 |")
        A.append("|---|---:|---:|---:|---:|")
        for imp in ["FP"] + DIFFUSION + ["NeighborMean"]:
            s = ps[(ps.family == fam) & (ps.level == 4) & (ps.imputer == imp)]
            A.append(f"| {imp} | " + " | ".join(f"{s[s.Ng == k].mac.mean():.3f}" for k in C.NG_VALUES) + " |")
        A.append("")

    A.append("## Network: first level whose mean MAC reaches a target\n")
    A.append("| imputer | " + " | ".join(f"{fam}, {t}" for fam in FAMILIES for t in (0.6, 0.7, 0.8)) + " |")
    A.append("|---|" + "---|" * 6)
    for imp in ["FP"] + DIFFUSION + ["NeighborMean"]:
        A.append(f"| {imp} | " + " | ".join(first_reach(ps, fam, imp, t)
                                            for fam in FAMILIES for t in (0.6, 0.7, 0.8)) + " |")
    A.append("")

    A.append("## Inputs: how far the imputed features sit from the full ones\n")
    A.append("Relative Frobenius error against the stored full-field inputs: node PSD over the "
             "unmeasured nodes (after row normalisation, as the network sees it), complex coherency "
             "over every edge.\n")
    for fam, title in FAMILIES.items():
        for col, what in (("x_err", "node PSD"), ("ea_err", "edge coherency")):
            A.append(f"*{title}, {what}:*\n")
            A += mean_table(ps, fam, col, IMPUTERS, best=min)
            A.append("")

    A.append("## No network: the imputed true mode shape\n")
    A.append("MAC of each imputer applied to the true mode shapes themselves, from their exact values at "
             "the same sensors - the FP repository's modal benchmark on these decks. It needs the mode "
             "shape at the sensors, which the network does not get; read it as what the graph "
             "interpolation alone could do. *Hinge free*: the hinge line is unknown, as for the network's "
             "inputs. *Hinge at rest*: the hinge nodes are clamped to zero (the repository's 'supports "
             "known at rest' - true for mode shapes, not for the inputs, which are absolute accelerations "
             "and carry the base excitation at the hinge). Last row: the network on FP inputs, for scale.\n")
    for fam, title in FAMILIES.items():
        net = ps[(ps.family == fam) & (ps.imputer == "FP")]
        for sup, stitle in (("free", "hinge free"), ("clamped", "hinge at rest")):
            A.append(f"*{title}, {stitle}:*\n")
            A += mean_table(ce[ce.supports == sup], fam, "mac", IMPUTERS,
                            extra=[("network, FP inputs", [net[net.level == v].mac.mean() for v in lv])])
            A.append("")

    A.append("## Figures\n")
    for f, cap in zip(figs, ("Network MAC minus production FP, paired by structure, 95 % CI bands.",
                             "s8's figure (solid: FP) with the best alternative per family (dashed) and "
                             "FP applied to the true mode shape with no network (dotted).",
                             "Imputing the true mode shape directly: every diffusion imputer, hinge free "
                             "(solid) and at rest (dashed), against the network on FP inputs.")):
        A.append(f"![{f}]({f})\n\n*{cap}*\n")
    (HERE / "REPORT.md").write_text("\n".join(A) + "\n", encoding="utf-8")
    print(f"wrote REPORT.md, gates.json, {', '.join(figs)}; gate {'PASS' if g['pass'] else 'FAIL'}")
    return 0 if g["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

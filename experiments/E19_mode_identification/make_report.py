"""Assemble REPORT.md and the figures from E19's CSVs.

    python tools/run_stage.py experiments/E19_mode_identification/make_report.py

Generated, never hand-edited: every number in REPORT.md is computed here.  Units of
replication: the test structure (500) on synthetic decks, the bridge (9) on real ones.
The network's numbers are read from E16 (per_sample.csv, = s8) and E17 (network.csv).
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import stage2_identify as S2  # noqa: E402  (E17, E18 and the pipeline on sys.path)

C, ID, PB, BPG = S2.C, S2.ID, S2.PB, S2.BPG
M = list(C.MODE_NAMES)
LAY = {"outer1": "two front corners", "outer2": "front and mid-span pairs (4)"}
METH = ["rank", "pop-shape", "population"]
INK2, MUTED, GRID, BASE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"rank": "#0b0b0b", "pop-shape": "#2a78d6", "population": "#1baf7a"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "axes.axisbelow": True,
})


def f3(v) -> str:
    if isinstance(v, str):
        return v
    return "" if pd.isna(v) else f"{v:.3f}"


def table(df: pd.DataFrame, index_name: str) -> list[str]:
    cols = list(df.columns)
    out = [f"| {index_name} | " + " | ".join(str(c) for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for k, r in df.iterrows():
        k = " / ".join(str(x) for x in k) if isinstance(k, tuple) else k
        out.append(f"| {k} | " + " | ".join(f3(r[c]) for c in cols) + " |")
    return out


def summary(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["correct"] = d["correct"].astype(float)
    d["present"] = d["present"].astype(float)
    d.loc[d.method == "template", ["correct", "present"]] = np.nan     # nothing is picked
    return d.groupby(["layout", "method"]).agg(
        correct=("correct", "mean"), present=("present", "mean"), mac=("mac", "mean"),
        ferr_median=("ferr", "median"), ferr_mean=("ferr", "mean"))


def main() -> int:
    test = pd.read_csv(HERE / "test.csv")
    real = pd.read_csv(HERE / "real.csv")
    grid = pd.read_csv(HERE / "grid.csv")
    chk = json.loads((HERE / "checks.json").read_text(encoding="utf-8"))
    s1 = json.loads((HERE / "stage1_checks.json").read_text(encoding="utf-8"))
    g, chosen = chk["gates"], chk["chosen_on_val"]
    e16 = pd.read_csv(REPO / "experiments" / "E16_fp_imputers" / "per_sample.csv")
    e16["mean"] = e16[[f"mac_{m}" for m in M]].mean(axis=1)
    e16["ferr"] = e16[[f"ferr_{m}" for m in M]].mean(axis=1)
    e17 = pd.read_csv(REPO / "experiments" / "E17_real_holdout" / "network.csv")
    e17 = e17[e17.source == "real"].copy()
    e17["ferr"] = (e17.pred_freq_hz - e17.ssi_freq_hz).abs()
    full = e16[e16.variant == "full"]
    net_full_mac, net_full_ferr = float(full["mean"].mean()), float(full["ferr"].mean())
    net_syn = {lay: e16[(e16.imputer == "FP") & (e16.family == "outer") & (e16.level == int(lay[-1]))] for lay in LAY}
    net_real = {lay: e17[e17.layout == S2.LAYOUTS[lay]] for lay in LAY}
    st, sr = summary(test), summary(real)
    v = lambda df, lay, meth, col: float(df.loc[(lay, meth), col])  # noqa: E731
    best_real = {lay: max(METH, key=lambda mth: v(sr, lay, mth, "correct")) for lay in LAY}
    tmpl_syn = float(test[test.method == "template"].mac.mean())
    tmpl_real = float(real[real.method == "template"].mac.mean())

    # template vs network, deck by deck (synthetic; network with every node measured)
    tp = test[test.method == "template"].groupby("sample").mac.mean()
    nf = full.assign(sample=full["sample"].str.replace(".npz", "", regex=False)).set_index("sample")["mean"]
    both = pd.concat([tp.rename("template"), nf.rename("network")], axis=1).dropna()
    tmpl_wins = int((both.template > both.network).sum())
    tmpl_ng = test[test.method == "template"].groupby("Ng").mac.mean()
    net_ng = full.groupby("Ng")["mean"].mean()

    # where the real decks sit in the population's frequency prior
    fpr = S2.freq_prior()
    zrows = []
    for s in C.LOOCV:
        geo = BPG.STRUCTURES[s]
        L, ng, dy = float(geo["L_mm"]), int(geo["Ng"]), float(geo["dy_mm"])
        feat = np.array([np.log(L), np.log(dy), float(bool(geo["balcony"]["enable"])), float(geo["bottom_beam"]["x_frac"])])
        mdl = fpr.fit(ng, feat)
        q = real[(real["sample"] == s) & (real.method == "given") & (real.layout == "outer1")].set_index("class").f_true
        z = {"bend1 (log f)": (np.log(q["bend1"]) - mdl["m1"]) / mdl["s1"]}
        for k, c in enumerate(ID.CLASSES[1:]):
            z[f"{c} / bend1"] = (np.log(q[c] / q["bend1"]) - mdl["mr"][k]) / mdl["sr"][k]
        zrows.append(pd.Series(z, name=s))
    zt = pd.DataFrame(zrows)
    zt_abs_mean = float(zt.abs().values.mean())
    zt_pos = float((zt.values > 0).mean())
    zt_max = float(zt.values.max())

    # ---- figure ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, (name, sm) in zip(axes, (("Synthetic test decks (500)", st), ("Real bridges (9)", sr))):
        x = np.arange(len(LAY))
        for j, mth in enumerate(METH):
            ax.bar(x + (j - 1) * 0.26, [v(sm, lay, mth, "correct") for lay in LAY], 0.26, color=COL[mth],
                   label={"rank": "phase + order", "pop-shape": "population shape", "population": "population shape + frequency"}[mth])
        ax.plot(x, [v(sm, lay, "population", "present") for lay in LAY], "_", ms=40, mew=1.6, color=MUTED)
        ax.set_xticks(x, [LAY[k] for k in LAY])
        ax.set_ylim(0, 1.05)
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("share of target modes identified correctly")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=MUTED, lw=1.6))
    lab.append("target peak among the candidates at all")
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig1 = "fig1_identification.png"
    fig.savefig(HERE / fig1, dpi=200)
    plt.close(fig)

    # ---- report ---------------------------------------------------------------------------------
    A = ["# E19 - B1, T1, B2, T2 from two or four sensors, with nothing given", "",
         "**Generated by `make_report.py` from the stage CSVs. Do not hand-edit.**", "",
         "E18 read each mode's shape at its known frequency. Here nothing about the modes is given: only the kept "
         "sensors' records and the deck's geometry (what drawings provide: L, girder count and spacing, balcony, "
         "bottom-beam position). From the sensors' cross-spectra (frequency domain decomposition) every spectral peak "
         "in 1.5-20 Hz is a candidate; a labeller decides which peaks are bend1, tors1, bend2 and tors2 (E18's classes); "
         "each mode's shape is read at its peak and filled over the deck with E18's class prior.", "",
         "Labellers (`ident.py`):", "",
         "- **rank**: bending if the kept sensors on the two outer girders move in phase at the peak, torsion "
         "otherwise; the lowest two of each kind. Uses nothing but the sensors.",
         "- **pop-shape**: every assignment of four peaks to the four classes, scored by how likely each peak's "
         "sensor vector is under the class's shape prior at those sensors (angular central Gaussian of E18's prior).",
         "- **population**: pop-shape plus a frequency prior. Bend1's frequency and the other three's ratios to it are "
         "modelled as Gaussians, fitted to the 100 training decks nearest in geometry.", "",
         "Two settings, the peak prominence and the weight of the frequency prior, were chosen per layout on the main "
         "network's 300 **validation** decks, by the share of modes identified correctly. They were then used "
         "unchanged on the 500 test decks and the nine real bridges. Chosen: " +
         "; ".join(f"{k}: prominence {v_['theta']:g}" + (f", weight {v_['lam']:g}" if k.endswith('population') else "")
                   for k, v_ in chosen.items()) + ".", "",
         "A mode counts as **correct** on synthetic decks if the FE mode nearest the picked peak is the target mode, "
         f"on real ones if the peak is within {S2.REAL_TOL:.0%} of the SSI frequency. **Present**: some candidate "
         "peak would have been correct (detection, before labelling). **MAC**: the shape read at the picked peak "
         "against the true shape (synthetic: deck nodes; real: SSI at every sensor, kept sensors measured and hidden "
         "ones filled); a mode left without a peak scores 0. Two references: **given** (the true frequency and class: "
         "E18) and **template**. The template involves no measurement at all: it is the class prior's leading shape, "
         "for the right girder count and class.", "",
         "## Gates", "",
         f"- *given* reproduces E18 on synthetic decks ({g['given_synthetic_reproduces_E18']['pairs']} deck-layouts, max "
         f"difference {g['given_synthetic_reproduces_E18']['max_abs_diff']:.1e}) and on real ones "
         f"({g['given_real_reproduces_E18']['rows']} rows, {g['given_real_reproduces_E18']['max_abs_diff']:.1e}) - "
         f"**{'PASS' if g['pass'] else 'FAIL'}**",
         f"- re-simulated target frequencies = stored, max {s1['matched_freq_maxdiff_hz']:.1e} Hz; all four classes "
         f"present on {s1['class_sets_complete']} of {sum(s1['n'].values())} decks", "",
         "## What came out", "",
         f"**1. On synthetic decks, the population labeller finds the modes.** It identifies "
         f"{v(st, 'outer1', 'population', 'correct'):.0%} of the target modes from the two front corners and "
         f"{v(st, 'outer2', 'population', 'correct'):.0%} from four sensors. Median frequency error: "
         f"{v(st, 'outer1', 'population', 'ferr_median'):.3f} and {v(st, 'outer2', 'population', 'ferr_median'):.3f} Hz. "
         f"Mean frequency error, including the wrong picks: {v(st, 'outer1', 'population', 'ferr_mean'):.2f} and "
         f"{v(st, 'outer2', 'population', 'ferr_mean'):.2f} Hz. The network on the same sensors (s8, FP-filled) errs by "
         f"{net_syn['outer1'].ferr.mean():.2f} and {net_syn['outer2'].ferr.mean():.2f} Hz, and by "
         f"{net_full_ferr:.2f} Hz with every node measured. The frequency prior does the most work: "
         f"phase and order alone reach {v(st, 'outer1', 'rank', 'correct'):.0%} and "
         f"{v(st, 'outer2', 'rank', 'correct'):.0%}. The end-to-end shape scores "
         f"{v(st, 'outer1', 'population', 'mac'):.3f} and {v(st, 'outer2', 'population', 'mac'):.3f}, against "
         f"{v(st, 'outer1', 'given', 'mac'):.3f} and {v(st, 'outer2', 'given', 'mac'):.3f} with the frequencies given.", "",
         f"**2. On the real bridges, labelling is the bottleneck.** The candidate peaks contain the target mode for "
         f"{v(sr, 'outer1', 'population', 'present'):.0%} (two corners) and {v(sr, 'outer2', 'population', 'present'):.0%} "
         f"(four sensors) of the modes. The best labeller (rank, pop-shape or population, picked per layout by real "
         f"correctness, so optimistic) gets "
         f"{v(sr, 'outer1', best_real['outer1'], 'correct'):.0%} and {v(sr, 'outer2', best_real['outer2'], 'correct'):.0%} "
         f"right ({best_real['outer1']}, {best_real['outer2']}). The full population labeller, tuned on synthetic "
         f"decks, gets only {v(sr, 'outer1', 'population', 'correct'):.0%} and "
         f"{v(sr, 'outer2', 'population', 'correct'):.0%}. Its frequency prior misleads it. The real frequencies sit "
         f"above where the population's nearest decks put them in {zt_pos:.0%} of the cells of the table below, "
         f"by up to {zt_max:.1f} standard deviations (mean |z| {zt_abs_mean:.1f}). The real decks are stiffer than "
         "their population neighbours, more so in the higher modes, in line with SSI sitting ~2 Hz above FE at B2/T2. "
         f"When a mode is found, its frequency is within ~{v(sr, 'outer1', best_real['outer1'], 'ferr_median'):.2f} "
         f"Hz (median). The wrong picks are off by Hz, so the mean error ({v(sr, 'outer1', best_real['outer1'], 'ferr_mean'):.2f} "
         f"and {v(sr, 'outer2', best_real['outer2'], 'ferr_mean'):.2f} Hz) is no better than the network's "
         f"({net_real['outer1'].ferr.mean():.2f} Hz).", "",
         f"**3. With two sensors the shape is the population's, not the sensors'.** With no measurement at all, the "
         f"class prior's leading shape scores mean MAC **{tmpl_syn:.3f}** on the synthetic test decks and "
         f"**{tmpl_real:.3f}** against SSI on the real bridges. From the two corners, with the frequency given, the "
         f"filled shape scores less: {v(st, 'outer1', 'given', 'mac'):.3f} and {v(sr, 'outer1', 'given', 'mac'):.3f}. "
         f"From four sensors it scores more: {v(st, 'outer2', 'given', 'mac'):.3f} and {v(sr, 'outer2', 'given', 'mac'):.3f}. "
         "Two corners contribute the frequencies and the bending/torsion class, not the shape. That is also why the "
         "settings were tuned on correctness: a shape read at the wrong peak scores almost as well. So E18's two-corner "
         "result (0.84 synthetic, 0.81 real) is below this template.", "",
         f"**4. The same template is a baseline the main network has not been measured against.** On the synthetic "
         f"test decks it scores {tmpl_syn:.3f}, against {net_full_mac:.3f} for the network with every node measured. "
         f"The template is higher on {tmpl_wins} of {len(both)} decks. By girder count: " +
         ", ".join(f"Ng {k} {tmpl_ng[k]:.3f} vs {net_ng[k]:.3f}" for k in tmpl_ng.index) + ". It is given each target's "
         "class from the true shape (the network infers which output slot is which mode). Its MAC is over the deck "
         "nodes (the network's, from E16, over every node). On the real bridges the gap is wider: "
         f"{tmpl_real:.3f} against {float(e17[e17.layout == 'full'].mac_ssi_sensor_nodes.mean()):.3f} for the "
         "network with every sensor.", "",
         f"![{fig1}]({fig1})", ""]

    A += ["## Synthetic test decks (500)", ""]
    A += table(st.reindex(["outer1", "outer2"], level=0), "layout / method") + [""]
    q = test[~test.method.isin(["template"])].pivot_table(index=["layout", "method"], columns="mode", values="correct")[M]
    A += ["*Identified correctly, by mode:*", ""] + table(q, "layout / method") + [""]
    q = test[test.method == "population"].pivot_table(index="Ng", columns="layout", values="correct")
    A += ["*Population labeller, identified correctly, by girder count:*", ""] + table(q, "Ng") + [""]
    q = test[test.method == "template"].pivot_table(index="Ng", columns="mode", values="mac")[M]
    q["mean"] = q.mean(axis=1)
    q["network, every node"] = net_ng
    A += ["*Template (no measurement) by girder count and mode, against the network with every node measured:*", ""] + \
        table(q, "Ng") + [""]
    A += [f"*Network on the same sensors (s8, FP-filled records): mean MAC {net_syn['outer1']['mean'].mean():.3f} / "
          f"{net_syn['outer2']['mean'].mean():.3f}, mean frequency error {net_syn['outer1'].ferr.mean():.2f} / "
          f"{net_syn['outer2'].ferr.mean():.2f} Hz (two corners / four sensors).*", ""]

    A += ["## Real bridges (9)", ""]
    A += table(sr.reindex(["outer1", "outer2"], level=0), "layout / method") + [""]
    q = real[~real.method.isin(["template"])].pivot_table(index=["layout", "method"], columns="mode", values="correct")[M]
    A += ["*Identified correctly, by mode:*", ""] + table(q, "layout / method") + [""]
    q = real[real.method.isin(METH)].pivot_table(index="sample", columns=["layout", "method"], values="correct")
    q.columns = [f"{a} {b}" for a, b in q.columns]
    A += ["*Identified correctly, by bridge (share of the four modes):*", ""] + table(q, "bridge") + [""]
    A += ["*Real frequencies in the population's frequency prior (z-scores; bend1 as log frequency, the others as log "
          "ratio to bend1):*", ""] + table(zt, "bridge") + [""]
    A += [f"*Network (E17): mean MAC vs SSI {net_real['outer1'].mac_ssi_sensor_nodes.mean():.3f} / "
          f"{net_real['outer2'].mac_ssi_sensor_nodes.mean():.3f}, mean frequency error {net_real['outer1'].ferr.mean():.2f} / "
          f"{net_real['outer2'].ferr.mean():.2f} Hz (two corners / four sensors).*", ""]

    A += ["## Sensitivity to the two settings", "",
          "Share identified correctly at every setting (rows chosen on val are marked *).", ""]
    for lay in LAY:
        q = grid[(grid.layout == lay) & grid.method.isin(METH)].pivot_table(
            index=["method", "theta", "lam"], columns="split", values="correct")[["val", "test", "real"]]
        q.index = [f"{a} / {b:g} / {'-' if c < 0 else f'{c:g}'}" + (" *" if (chosen[f'{lay}/{a}']['theta'] == b and
                   (a == 'rank' or chosen[f'{lay}/{a}']['lam'] == c)) else "") for a, b, c in q.index]
        A += [f"*{LAY[lay]} (method / prominence / weight):*", ""] + table(q, "setting") + [""]

    (HERE / "REPORT.md").write_text("\n".join(A) + "\n", encoding="utf-8")
    print("wrote REPORT.md,", fig1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

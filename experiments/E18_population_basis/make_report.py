"""Assemble REPORT.md and the figures from E18's CSVs.

    python tools/run_stage.py experiments/E18_population_basis/make_report.py

Generated, never hand-edited: every number in REPORT.md is computed here.  Units of
replication: the test structure (500) on synthetic decks, the bridge (9) on real ones.
The network's numbers for comparison are read from E16 (per_sample.csv, = s8) and E17
(network.csv, = s5 with every sensor).
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
sys.path.insert(0, str(REPO / "Final_results_for_paper" / "pipeline"))
sys.path.insert(0, str(HERE))
import fr_common as C  # noqa: E402
import popbasis as PB  # noqa: E402

M = list(C.MODE_NAMES)
MAC = [f"mac_{m}" for m in M]
LEVELS = [1, 2, 3, 4, 5, 6, 8, 10, 12]
METHODS = ["FP", "PCFI", "PB", "PB-class", "PB-class-data"]
REAL_LAYOUTS = ["single", "front_pair", "front_mid", "alternate", "one_line", "loso"]
FAM = {"outer": "outer girders only (2n sensors)", "all": "every girder (Ng x n sensors)"}

INK2, MUTED, GRID, BASE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"FP": "#0b0b0b", "PCFI": "#eda100", "PB": "#9ec5f0", "PB-class": "#2a78d6", "PB-class-data": "#1baf7a",
       "network": "#e87ba4"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "lines.linewidth": 1.6, "lines.markersize": 5.0, "axes.axisbelow": True,
})


def f3(v) -> str:
    return "" if pd.isna(v) else f"{v:.3f}"


def table(df: pd.DataFrame, index_name: str) -> list[str]:
    cols = list(df.columns)
    out = [f"| {index_name} | " + " | ".join(str(c) for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for k, r in df.iterrows():
        out.append(f"| {k} | " + " | ".join(f3(r[c]) for c in cols) + " |")
    return out


def main() -> int:
    syn = pd.read_csv(HERE / "synthetic.csv")
    syn["mean"] = syn[MAC].mean(axis=1)
    cls = pd.read_csv(HERE / "classes.csv")
    fit = pd.read_csv(HERE / "prior_fit.csv")
    real = pd.read_csv(HERE / "real.csv")
    for c in ("defl_mac_hidden", "sign_agree_w", "pred_mac_label", "label_mac_hidden"):
        real[c] = pd.to_numeric(real[c], errors="coerce")
    g2 = json.loads((HERE / "synthetic_checks.json").read_text(encoding="utf-8"))["gates"]
    g3 = json.loads((HERE / "real_checks.json").read_text(encoding="utf-8"))["gates"]
    e16 = pd.read_csv(REPO / "experiments" / "E16_fp_imputers" / "per_sample.csv")
    e16["mean"] = e16[MAC].mean(axis=1)
    e17 = pd.read_csv(REPO / "experiments" / "E17_real_holdout" / "network.csv")

    net_full = float(e16[e16.variant == "full"]["mean"].mean())
    net = {fam: e16[(e16.family == fam) & (e16.imputer == "FP")].groupby("level")["mean"].mean() for fam in FAM}
    sm = lambda route, fam, meth, lv: float(syn[(syn.route == route) & (syn.family == fam) & (syn.method == meth)  # noqa: E731
                                                 & (syn.level == lv)]["mean"].mean())
    dense = float(syn[syn.method == "dense"]["mean"].mean())
    rr = real[(real.source == "real") & (real.imputer != "dense")]
    rm = lambda lay, meth, col="pred_mac_label", src="real": float(  # noqa: E731
        real[(real.source == src) & (real.layout == lay) & (real.imputer == meth)][col].mean())
    rdense = {src: float(real[(real.source == src) & (real.imputer == "dense")].pred_mac_label.mean())
              for src in ("real", "twin_matched")}
    rnet = lambda lay: float(e17[(e17.source == "real") & (e17.layout == lay)].mac_ssi_sensor_nodes.mean())  # noqa: E731
    # class read from the sensors vs from the true shape
    cq = cls[cls.route == "ods"]
    agree = {(fam, lv): float(np.mean([(cq[(cq.family == fam) & (cq.level == lv)][f"data_{m}"]
                                        == cq[(cq.family == fam) & (cq.level == lv)][f"true_{m}"]).mean() for m in M]))
             for fam in FAM for lv in LEVELS}
    rc = rr[(rr.layout == "front_pair") & (rr.imputer == "PB-class-data")]
    real_cls_ok = int((rc.class_used == rc.class_label).sum())

    # ---- figures -------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, fam in zip(axes, FAM):
        for meth in ("FP", "PB", "PB-class-data"):
            ax.plot(LEVELS, [sm("ods", fam, meth, lv) for lv in LEVELS], marker="o", color=COL[meth],
                    label={"FP": "FP fill of the sensor deflections", "PB": "population basis, pooled",
                           "PB-class-data": "population basis, class from the sensors"}[meth])
        ax.plot(LEVELS, [net[fam][lv] for lv in LEVELS], marker="s", ls="--", color=COL["network"],
                label="network on FP-filled records (s8)")
        ax.axhline(net_full, color=COL["network"], lw=0.9, ls=":")
        ax.axhline(dense, color=MUTED, lw=0.9, ls=":")
        ax.text(12, net_full - 0.012, "network, every node", ha="right", va="top", fontsize=7.5, color=COL["network"])
        ax.text(12, dense + 0.008, "deflection, every node", ha="right", va="bottom", fontsize=7.5, color=MUTED)
        ax.set_xticks(LEVELS)
        ax.set_xlabel("stations per girder line (n)")
        ax.set_title(FAM[fam], loc="left")
        ax.set_ylim(0.2, 1.02)
    axes[0].set_ylabel("mean MAC vs true shape (500 test decks)")
    axes[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig1 = "fig1_synthetic.png"
    fig.savefig(HERE / fig1, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    lays = ["single", "front_pair", "front_mid", "alternate", "one_line"]
    series = [("FP", "FP"), ("PB", "PB"), ("PB-class-data", "PB-class-data"), ("PB-class", "PB-class"), ("network", "network")]
    w = 0.16
    for j, (key, meth) in enumerate(series):
        vals = [rnet(l) if key == "network" else rm(l, meth) for l in lays]
        ax.bar(np.arange(len(lays)) + (j - 2) * w, vals, w, color=COL[key],
               label={"FP": "FP", "PB": "population basis, pooled", "PB-class-data": "population basis, class from sensors",
                      "PB-class": "population basis, class given", "network": "network (E17)"}[key])
    ax.axhline(rdense["real"], color=MUTED, lw=0.9, ls=":")
    ax.text(len(lays) - 0.5, rdense["real"] + 0.01, "deflection, every sensor", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(np.arange(len(lays)), lays)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean MAC vs SSI at the sensors")
    ax.set_title("Nine real bridges: mode shape from the kept sensors", loc="left")
    ax.legend(loc="upper left", fontsize=7.2, ncol=2)
    fig.tight_layout()
    fig2 = "fig2_real.png"
    fig.savefig(HERE / fig2, dpi=200)
    plt.close(fig)

    # ---- report --------------------------------------------------------------------------------
    pm = fit.projection_mac
    A = ["# E18 - a population basis: what the linkspan family knows about mode shapes", "",
         "**Generated by `make_report.py` from the stage CSVs. Do not hand-edit.**", "",
         "Every fill-in tested so far (E14, E16, E17) is model-free: it knows the graph and nothing about "
         "linkspans. Its weights are non-negative and the same at every frequency, so it cannot place a nodal "
         "line between sensors that move together. Here the fill knows the family instead.", "",
         f"**The prior** (`stage1_prior.py`, `popbasis.py`). Each training deck of the main network "
         f"({int(fit['sample'].nunique())} structures, N = {C.MAIN_N} training split; never validation or test) is put in "
         "normalised coordinates: xi = x / L, and eta = -1 and +1 at the outer girders. Its four target mode shapes "
         f"are expanded in a tensor Legendre basis ({PB.DEG_X + 1} x {PB.DEG_Y + 1} = "
         f"{(PB.DEG_X + 1) * (PB.DEG_Y + 1)} functions). The basis represents them with projection MAC median "
         f"{pm.median():.5f} (1st percentile {pm.quantile(0.01):.4f}). The coefficients give a zero-mean "
         "Gaussian prior per girder count: pooled over the four modes, and per mode class. A class is the shape's "
         "character (bending if the outer girders move in phase, else torsion) and its rank by frequency: bend1, "
         "tors1, bend2, tors2. Classes are read from the shape, not from the matcher's labels, which swap B2 and "
         "T2 on two-girder balcony decks.", "",
         "**The fill** is kriging with that prior. It is still a linear map from the sensors to every node, but "
         "its weights can be negative, and with a class prior they differ from mode to mode. Variants:", "",
         "- **PB**: one prior for all four modes; needs nothing but the sensors.",
         "- **PB-class**: the prior of the mode's class, with the class given (from the true or label shape).",
         "- **PB-class-data**: the class read from the kept sensors themselves (bending if those on the two outer "
         "girders move in phase, ranked by frequency); pooled where the kept sensors do not span both outer girders.",
         "",
         "What every variant here is **given**: the frequency of each of the four target modes. The shape is read "
         "at the nearest Welch bin (the FE target's frequency on synthetic decks, SSI's on real ones), so which "
         "peaks are the four targets is not tested. What is tested is the shape between the sensors.", "",
         "## Gates", "",
         f"- Synthetic, shape route: FP and PCFI rows = E16's `ceiling.csv` (hinge free), "
         f"{g2['e16_ceiling']['rows_matched']} of {g2['e16_ceiling']['rows_expected']} rows, max difference "
         f"{g2['e16_ceiling']['max_abs_diff']:.1e} - **{'PASS' if g2['e16_ceiling']['pass'] else 'FAIL'}**",
         f"- Synthetic: re-simulation reproduces the stored node features to "
         f"{g2['resim_vs_stored_x_psd']['max_abs_diff']:.1e} - **{'PASS' if g2['resim_vs_stored_x_psd']['pass'] else 'FAIL'}**",
         f"- Real: FP and PCFI rows = E17's `fill.csv`, {g3['e17_fill']['rows_matched']} of "
         f"{g3['e17_fill']['rows_expected']} rows, max difference {g3['e17_fill']['max_abs_diff']:.1e} - "
         f"**{'PASS' if g3['e17_fill']['pass'] else 'FAIL'}**", ""]

    A += ["## What came out", "",
          f"**1. On the 500 synthetic test decks, four sensors and the population prior recover the mode shapes "
          f"almost as well as measuring every node.** The deflections are read from the re-simulated records at the "
          f"sensors, as operational modal analysis there would give them. From two stations on the outer girders (4 "
          f"sensors) the pooled prior reaches mean MAC {sm('ods', 'outer', 'PB', 2):.3f}. That compares with "
          f"{sm('ods', 'outer', 'FP', 2):.3f} for FP on the same deflections, {dense:.3f} for the deflection measured "
          f"at every node, and {net['outer'][2]:.3f} for the network on FP-filled records (s8). The network scores "
          f"{net_full:.3f} with every node measured. From the two front corners alone (n = 1): "
          f"{sm('ods', 'outer', 'PB-class-data', 1):.3f} with the class read from the two sensors, "
          f"{sm('ods', 'outer', 'PB-class', 1):.3f} with it given, {sm('ods', 'outer', 'PB', 1):.3f} pooled, "
          f"{sm('ods', 'outer', 'FP', 1):.3f} for FP and {net['outer'][1]:.3f} for the network.", "",
          f"**2. On the nine real bridges it holds.** The predicted shape is scored at every sensor (the kept sensors' "
          f"own deflections, the hidden ones filled) against SSI, the same frame as the network's zero-shot score.", "",
          f"| kept sensors | network | FP | PB | PB-class-data | PB-class |",
          "|---|---:|---:|---:|---:|---:|"]
    for lay in ["single", "front_pair", "front_mid", "alternate", "one_line"]:
        kept = float(rr[rr.layout == lay].n_kept.mean())
        A.append(f"| {lay} ({kept:.1f}) | {rnet(lay):.3f} | {rm(lay, 'FP'):.3f} | {rm(lay, 'PB'):.3f} | "
                 f"{rm(lay, 'PB-class-data'):.3f} | {rm(lay, 'PB-class'):.3f} |")
    A += [f"| every sensor ({float(real[(real.source == 'real') & (real.imputer == 'dense')].n_kept.mean()):.1f}) | "
          f"{rnet('full'):.3f} | | | | |", "",
          f"With every sensor measured and nothing filled, the deflection itself scores {rdense['real']:.3f} against "
          f"SSI. From the two front corners the class "
          f"read from the sensors scores {rm('front_pair', 'PB-class-data'):.3f}, against {rnet('front_pair'):.3f} for "
          f"the network on the same two sensors and {rnet('full'):.3f} for the network with every sensor. With four "
          f"sensors (front and mid-span pairs) the population basis scores {rm('front_mid', 'PB'):.3f} "
          f"(pooled), against {rdense['real']:.3f} for the deflection measured at every sensor. Read from the two "
          f"corners, the class was right in {real_cls_ok} of {len(rc)} mode slots.", "",
          f"**3. Where it cannot help: the sensors must span the deck.** From one girder line the class cannot be "
          f"read (no phase across the deck), and the pooled prior scores {rm('one_line', 'PB'):.3f} on real decks. "
          f"With the class given it reaches {rm('one_line', 'PB-class'):.3f}. From a single sensor, the class given "
          f"gives {rm('single', 'PB-class'):.3f} and pooled {rm('single', 'PB'):.3f}. One sensor fixes the scale; the "
          "shape is then the population's.", "",
          f"**4. The real decks are still harder than their twins**, as E17 found, but far less so. Hidden-sensor "
          f"deflection MAC from the front pair, real vs FE twin: PB-class-data "
          f"{rm('front_pair', 'PB-class-data', 'defl_mac_hidden'):.3f} vs "
          f"{rm('front_pair', 'PB-class-data', 'defl_mac_hidden', 'twin_matched'):.3f}, FP "
          f"{rm('front_pair', 'FP', 'defl_mac_hidden'):.3f} vs {rm('front_pair', 'FP', 'defl_mac_hidden', 'twin_matched'):.3f}. "
          f"Leaving one sensor out: PB-class-data {rm('loso', 'PB-class-data', 'defl_mac_hidden'):.3f} vs FP "
          f"{rm('loso', 'FP', 'defl_mac_hidden'):.3f} on real decks.", "",
          "**What this does and does not show.** The population basis is a better *interpolator* of linkspan mode "
          "shapes than any graph diffusion, by a wide margin and on real bridges. It does not replace the rest of "
          "modal identification: the four target frequencies, and which peaks they are, are given here. On real "
          "decks the label (SSI) comes from the same recordings as the deflections, and the kept sensors' own "
          "values count in the score at every sensor (2 of 14-16 entries for the front pair). On synthetic decks "
          "the prior is in-distribution: same generator, different structures.", "",
          f"![{fig1}]({fig1})", "",
          "*Synthetic test set, deflection route, mean over B1, T1, B2, T2. Dotted: the deflection measured at every "
          "node, and the network with every node measured.*", "",
          f"![{fig2}]({fig2})", ""]

    A += ["## Synthetic decks (500 test structures)", "",
          "Mean MAC over the four modes against the true shape. *shape*: the true shape at the sensors (E16's "
          "ceiling). *ods*: the deflection read from the re-simulated records (deck nodes; the supports carry the "
          f"injected excitation). Every node measured, ods route: **{dense:.3f}**. Network: every node "
          f"**{net_full:.3f}**; row *network* = s8 (FP-filled records).", ""]
    for route in ("ods", "shape"):
        for fam in FAM:
            t = syn[(syn.route == route) & (syn.family == fam) & (syn.method != "dense")].pivot_table(
                index="method", columns="level", values="mean").reindex(METHODS)
            t.columns = [f"n{c}" for c in t.columns]
            if route == "ods":
                t.loc["network"] = [net[fam][lv] for lv in LEVELS]
            A += [f"*{route}, {FAM[fam]}:*", ""] + table(t, "method") + [""]
    for fam in FAM:
        t = syn[(syn.route == "ods") & (syn.family == fam) & (syn.level.isin([1, 2]))].pivot_table(
            index=["level", "method"], values=MAC).reindex(columns=MAC)
        t.index = [f"n{a}, {b}" for a, b in t.index]
        t.columns = M
        A += [f"*ods, {FAM[fam]}, per mode:*", ""] + table(t, "level, method") + [""]
    t = pd.DataFrame({fam: [agree[(fam, lv)] for lv in LEVELS] for fam in FAM}, index=[f"n{lv}" for lv in LEVELS]).T
    A += ["*Class read from the sensors' deflections = class of the true shape (share of mode slots):*", ""] + \
        table(t, "family") + [""]
    t = syn[(syn.route == "ods") & (syn.family == "outer") & (syn.level == 2)].pivot_table(
        index="Ng", columns="method", values="mean")[["FP", "PB", "PB-class-data"]]
    A += ["*ods, outer girders, n2 (4 sensors), by girder count:*", ""] + table(t, "Ng") + [""]

    A += ["## Real bridges (E17's hold-out)", "",
          "*pred vs label*: MAC at every sensor between the label shape (SSI on real, FE target on the twin) and the "
          "predicted deflection (kept measured, hidden filled). *hidden deflection*: filled vs recorded deflection "
          "at the hidden sensors. *sign*: share of the hidden motion with the right sign (amplitude-weighted). Mean "
          "over 9 bridges x 4 modes.", ""]
    for col, lab in (("pred_mac_label", "pred vs label"), ("defl_mac_hidden", "hidden deflection"),
                     ("sign_agree_w", "sign")):
        for src in ("real", "twin_matched"):
            t = real[(real.source == src) & (real.imputer != "dense")].pivot_table(
                index="imputer", columns="layout", values=col).reindex(METHODS)[REAL_LAYOUTS]
            if col == "pred_mac_label" and src == "real":
                t.loc["network (E17)"] = [rnet(l) if l != "loso" else np.nan for l in REAL_LAYOUTS]
            A += [f"*{lab}, {src}:*", ""] + table(t, "method") + [""]
    for lay in ("front_pair", "front_mid"):
        t = rr[rr.layout == lay].pivot_table(index="imputer", columns="mode", values="pred_mac_label").reindex(METHODS)[M]
        t.loc["network (E17)"] = [float(e17[(e17.source == "real") & (e17.layout == lay) & (e17["mode"] == m)]
                                        .mac_ssi_sensor_nodes.mean()) for m in M]
        A += [f"*pred vs SSI by mode, real, {lay}:*", ""] + table(t, "method") + [""]
    t = rr[rr.layout.isin(["front_pair", "front_mid"]) & rr.imputer.isin(["FP", "PB", "PB-class-data"])].pivot_table(
        index="structure", columns=["layout", "imputer"], values="pred_mac_label")
    t.columns = [f"{a} {b}" for a, b in t.columns]
    A += ["*pred vs SSI by bridge, real:*", ""] + table(t, "bridge") + [""]
    t = real[real.imputer == "dense"].pivot_table(index="source", columns="mode", values="pred_mac_label")[M]
    A += ["*Every sensor measured: deflection vs label, by mode:*", ""] + table(t, "source") + [""]

    (HERE / "REPORT.md").write_text("\n".join(A) + "\n", encoding="utf-8")
    print("wrote REPORT.md,", fig1, fig2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

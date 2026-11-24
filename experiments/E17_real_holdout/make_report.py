"""Assemble REPORT.md and the figures from stage1_holdout.py's CSVs.

    python tools/run_stage.py experiments/E17_real_holdout/make_report.py

Generated, never hand-edited: every number in REPORT.md is computed here.
Unit of replication: the bridge (nine; one network, one seed).  Real and twin are
compared bridge by bridge, imputers against production FP on the same bridge and
layout.
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
import fr_common as C  # noqa: E402

M = list(C.MODE_NAMES)
LAYOUTS = ["single", "front_pair", "front_mid", "alternate", "one_line", "loso"]
POOLED = ["front_mid", "alternate", "one_line", "loso"]                     # >= 4 sensors kept
NET_LAYOUTS = ["full", "single", "front_pair", "front_mid", "alternate", "one_line"]
IMPUTERS = ["FP", "FP-rw", "Harmonic", "Tikhonov", "PCFI", "NeighborMean", "Mean"]
LAY_TXT = {
    "single": "the front sensor of line a",
    "front_pair": "the front sensor of each line (~ s8 outer n = 1)",
    "front_mid": "front pair + each line's sensor nearest mid-span (~ s8 outer n = 2)",
    "alternate": "every other sensor along each line, front kept",
    "one_line": "every sensor on line a; line b and the rest hidden",
    "loso": "every sensor but one, each in turn",
    "full": "every sensor (s5 / s6)",
}
REL_MOVE = 0.3            # a hidden sensor "moves" at a mode if |d| >= 30 % of the largest |d| there
COH_BINS = [0, 0.5, 0.8, 0.9, 0.95, 1.0001]

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"real": "#0b0b0b", "twin_matched": "#2a78d6", "twin_full": "#9ec5f0"}
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


def f3(v) -> str:
    return "" if pd.isna(v) else f"{v:.3f}"


def table(df: pd.DataFrame, index_name: str) -> list[str]:
    cols = list(df.columns)
    out = [f"| {index_name} | " + " | ".join(str(c) for c in cols) + " |",
           "|---|" + "---:|" * len(cols)]
    for k, r in df.iterrows():
        out.append(f"| {k} | " + " | ".join(f3(r[c]) if not isinstance(r[c], str) else r[c] for c in cols) + " |")
    return out


def coherence_bins(s: pd.DataFrame, src: str) -> pd.DataFrame:
    q = s[(s.imputer == "FP") & (s.source == src) & (s.layout != "loso")].copy()
    dm = q.d_meas_re + 1j * q.d_meas_im
    df = q.d_fill_re + 1j * q.d_fill_im
    q["wrong"] = np.real(df * np.conj(dm)) < 0
    q["amp"] = np.abs(df) / np.abs(dm)
    q["a"] = np.abs(dm)
    q["rel"] = q.a / q.groupby(["structure", "layout", "mode"]).a.transform("max")
    q = q[q.rel >= REL_MOVE]
    q["coherence"] = pd.cut(q.coh, COH_BINS, labels=["< 0.5", "0.5-0.8", "0.8-0.9", "0.9-0.95", "> 0.95"])
    return q.groupby("coherence", observed=True).agg(cases=("wrong", "size"), wrong_sign=("wrong", "mean"),
                                                     amplitude_ratio_median=("amp", "median"))


def fig_real_vs_twin(f: pd.DataFrame) -> str:
    fp = f[f.imputer == "FP"]
    per = fp.groupby(["source", "layout", "structure"]).defl_mac_hidden.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(LAYOUTS))
    for j, (src, lab) in enumerate((("real", "real recordings"), ("twin_matched", "FE twin, same record length"))):
        for i, lay in enumerate(LAYOUTS):
            v = per[(per.source == src) & (per.layout == lay)].defl_mac_hidden.values
            xi = x[i] + (j - 0.5) * 0.3
            ax.scatter(np.full(v.size, xi) + np.linspace(-0.06, 0.06, v.size), v, s=12, color=COL[src], alpha=0.45, lw=0)
            ax.plot([xi - 0.11, xi + 0.11], [v.mean()] * 2, color=COL[src], lw=2.4, label=lab if i == 0 else None)
    ax.set_xticks(x, LAYOUTS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("MAC, filled vs recorded deflection\nat the hidden sensors")
    ax.legend(loc="lower right")
    ax.set_title("Same graph, sensors and operator (production FP): real decks vs their FE twins", loc="left")
    fig.tight_layout()
    p = HERE / "fig1_real_vs_twin.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p.name


def fig_coherence(cb: dict) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for src, lab in (("real", "real recordings"), ("twin_matched", "FE twin")):
        t = cb[src]
        ax.plot(range(len(t)), t.wrong_sign.values, marker="o", color=COL[src], label=lab)
    ax.set_xticks(range(len(cb["real"])), cb["real"].index.astype(str))
    ax.set_ylim(0, 0.7)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("coherence, filled vs recorded signal, at the mode")
    ax.set_ylabel("share filled with the wrong sign")
    ax.legend(loc="upper right")
    ax.set_title("High coherence does not mean the right sign", loc="left")
    fig.tight_layout()
    p = HERE / "fig2_coherence_vs_sign.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p.name


def main() -> int:
    f = pd.read_csv(HERE / "fill.csv")
    s = pd.read_csv(HERE / "fill_sensors.csv")
    p = pd.read_csv(HERE / "psd.csv")
    n = pd.read_csv(HERE / "network.csv")
    chk = json.loads((HERE / "checks.json").read_text(encoding="utf-8"))
    g, per = chk["gates"], chk["per_structure"]
    fp = f[f.imputer == "FP"]

    def lm(src, col, lay=None, mode=None, imp="FP"):
        q = f[(f.imputer == imp) & (f.source == src)]
        if lay is not None:
            q = q[q.layout == lay]
        if mode is not None:
            q = q[q["mode"] == mode]
        return float(q[col].mean())

    # real vs twin, bridge by bridge
    bridge = {}
    for col in ("defl_mac_hidden", "label_mac_hidden"):
        t = fp[fp.layout.isin(POOLED)].pivot_table(
            index="structure", columns="source", values=col)
        t["gap"] = t.twin_matched - t.real
        bridge[col] = t
    gap_lay = {lay: lm("twin_matched", "defl_mac_hidden", lay) - lm("real", "defl_mac_hidden", lay) for lay in LAYOUTS}
    worse = [lay for lay in LAYOUTS if gap_lay[lay] > 0]
    better = [lay for lay in LAYOUTS if gap_lay[lay] <= 0]
    length_effect = max(abs(lm("twin_full", c, lay) - lm("twin_matched", c, lay))
                        for lay in LAYOUTS for c in ("defl_mac_hidden", "label_mac_hidden"))
    cb = {src: coherence_bins(s, src) for src in ("real", "twin_matched")}

    # imputers against FP, paired by bridge and layout
    imp = {}
    for src in ("real", "twin_matched"):
        t = f[f.source == src].pivot_table(index=["structure", "layout"], columns="imputer",
                                                                    values="defl_mac_hidden")
        d = t.sub(t.FP, axis=0)
        imp[src] = pd.DataFrame({"mean difference": d.mean(), "share better": (d > 0).mean()}).reindex(IMPUTERS)
    nm = lambda src, lay, col="mac_ssi_sensor_nodes": float(n[(n.source == src) & (n.layout == lay)][col].mean())  # noqa: E731

    fig1, fig2 = fig_real_vs_twin(f), fig_coherence(cb)
    nsens = {sname: len(v["layouts"]["full"]) for sname, v in per.items()}
    kept = {lay: np.mean([len(v["layouts"][lay]) for v in per.values()]) for lay in NET_LAYOUTS}
    nseg = {sname: v["real"]["nseg"] for sname, v in per.items()}

    A = ["# E17 - fill in real sensors that were really measured", "",
         "**Generated by `make_report.py` from `stage1_holdout.py`'s CSVs. Do not hand-edit.**", "",
         "E14 and E16 judged every fill-in against simulated ground truth. The nine real bridges carry their own: "
         f"each has {min(nsens.values())}-{max(nsens.values())} recorded sensors. Here some are hidden, filled in from the "
         "rest over the hybrid graph the network reads (FE nodes plus sensor nodes, `phase3_real_graphs/`), and compared "
         "with what they recorded. The same test runs on each bridge's FE twin, the broadband simulation of its real FE "
         "model that s6 uses, taken at the same sensors. Real and twin therefore share the graph, the sensors and the "
         "operator; only the signals differ.", "",
         "Signals: the real records (`raw_u3_timeseries_clean`, 128 Hz, "
         f"{min(nseg.values())}-{max(nseg.values())} Welch segments) and the twin at the generator's 200 Hz, either "
         "simulated to the same number of Welch segments as the real record (`twin_matched`) or at the generator's 60 s "
         "(`twin_full`, s6's propagation-only input). All are band-passed 1.5-20 Hz and use the pipeline's Welch settings.", "",
         "Each mode is read at its own frequency, the nearest Welch bin: SSI on real data, the label-corrected FE target on "
         "the twin. A sensor's **deflection** is its cross-spectrum with a reference sensor, divided by the reference's "
         "amplitude. The reference is the kept sensor with the largest PSD there. A filled sensor's deflection is the same "
         "weighted sum of the kept sensors' deflections, since every fill here is a weighted sum of their records. "
         "**Coherence** compares the filled signal with the recorded one, as in the explorer's step 3. All of it follows "
         "exactly from the sensors' cross-spectral matrix.", "",
         "Layouts (kept sensors), from the two girder lines with the most sensors (line a at y = 0); "
         "mean number of sensors kept in brackets:", "",
         "| layout | kept |", "|---|---|"]
    A += [f"| {lay} ({kept[lay]:.1f}) | {LAY_TXT[lay]} |" for lay in LAYOUTS[:-1]]
    A += ["| loso | every sensor but one, each in turn; one reference per mode, never held out |", ""]

    A += ["## Gates", ""]
    names = {"real_full_reproduces_s5": "every sensor kept, real: network MAC and frequency = `s5_zeroshot.csv`",
             "twin_full_reproduces_s6_prop_hybrid": "every sensor kept, twin: network MAC and frequency = `s6_decomposition.csv` prop_hybrid",
             "real_full_features_vs_stored": "every sensor kept, real: node and edge features = E5's stored corrected features",
             "csd_matrix_vs_scipy": "cross-spectral matrix = scipy `welch` / `csd`",
             "fp_operator_linear": "production FP's weight matrix applied to data = FP run on the data"}
    for k, txt in names.items():
        v = g[k]
        d = v.get("max_abs_diff", v.get("max_rel_diff"))
        A.append(f"- {txt}: max difference {d:.1e} (tolerance {v['tol']:.0e}) - **{'PASS' if v['pass'] else 'FAIL'}**")
    A.append("")

    # ---- the answer ----------------------------------------------------------------------------
    bd, bl = bridge["defl_mac_hidden"], bridge["label_mac_hidden"]
    A += ["## What came out", "",
          f"**1. The real decks fill worse than their FE twins.** With the same graph, sensors and operator (production FP), "
          f"the filled deflections at the hidden sensors match the recorded ones less well on the real decks. The MAC "
          f"is lower in {len(worse)} of {len(LAYOUTS)} layouts, by {min(gap_lay[l] for l in worse):.3f} to "
          f"{max(gap_lay[l] for l in worse):.3f}"
          + (f"; the exception is {', '.join(better)}, where every fill is a scaled copy of one record (real "
             f"{lm('real', 'defl_mac_hidden', better[0]):.3f}, twin {lm('twin_matched', 'defl_mac_hidden', better[0]):.3f})" if better else "")
          + f". Bridge by bridge it is lower on {int((bd.gap > 0).sum())} of 9 ({', '.join(POOLED)} pooled). "
          f"Applying the fill to the mode shapes themselves (SSI on real, FE target on the twin) shows the same: lower "
          f"on {int((bl.gap > 0).sum())} of 9. Record length is not the cause: the twin at the real record's length and "
          f"at 60 s differ by at most {length_effect:.3f}. The real shapes vary more from one sensor to the next than the "
          "FE shapes do. This test cannot tell how much of that is the structure and how much is sensor error "
          "(gain, orientation, noise), because both real estimates come from the same recordings.", "",
          f"**2. The sign fails where non-negative weights predict.** Every imputer here mixes the kept sensors with "
          f"non-negative weights, so it cannot place a nodal line between sensors that move together. "
          f"From the front pair, B2's filled sign matches the recorded one for "
          f"{lm('real', 'sign_agree_w', 'front_pair', 'B2'):.2f} of the hidden motion on real decks "
          f"({lm('twin_matched', 'sign_agree_w', 'front_pair', 'B2'):.2f} on the twin; amplitude-weighted). "
          f"From one girder line, the torsion modes are filled with the wrong sign across the deck: T1 "
          f"{lm('real', 'sign_agree_w', 'one_line', 'T1'):.2f} and T2 {lm('real', 'sign_agree_w', 'one_line', 'T2'):.2f} "
          f"real ({lm('twin_matched', 'sign_agree_w', 'one_line', 'T1'):.2f} and "
          f"{lm('twin_matched', 'sign_agree_w', 'one_line', 'T2'):.2f} twin). Even with every other sensor kept "
          f"(loso), T1's sign agreement is {lm('real', 'sign_agree_w', 'loso', 'T1'):.2f} on real decks and "
          f"{lm('twin_matched', 'sign_agree_w', 'loso', 'T1'):.2f} on the twin. That is "
          "`docs/TORSION_TRANSFER_FINDING.md`'s mechanism, here with recorded ground truth: no edge joins the two "
          "sides of the deck, and propagation averages the two signs.", "",
          f"**3. Coherence is not a usable criterion.** Take every FP-filled hidden sensor that moves at a mode "
          f"(|deflection| >= {REL_MOVE:.0%} of the largest there; fixed layouts). Where coherence with the recording "
          f"exceeds 0.95, {cb['real'].wrong_sign.iloc[-1]:.0%} are filled with the wrong sign on real decks "
          f"({cb['twin_matched'].wrong_sign.iloc[-1]:.0%} on the twin), and the filled amplitude is a median "
          f"{cb['real'].amplitude_ratio_median.iloc[-1]:.2f} of the recorded one. Lower coherence does mean worse "
          f"({cb['real'].wrong_sign.iloc[0]:.0%} wrong below 0.5), so it is a warning sign at best, never a pass mark. "
          "Coherence ignores the scale, and the mode shape is the scale.", "",
          f"**4. E16's ranking of the imputers holds on real decks.** Paired by bridge and layout, against FP: "
          f"PCFI {imp['real'].loc['PCFI', 'mean difference']:+.3f} "
          f"(better in {imp['real'].loc['PCFI', 'share better']:.0%} of cells), "
          f"FP-rw {imp['real'].loc['FP-rw', 'mean difference']:+.3f}, "
          f"Harmonic {imp['real'].loc['Harmonic', 'mean difference']:+.3f}, "
          f"Tikhonov {imp['real'].loc['Tikhonov', 'mean difference']:+.3f}, "
          f"Mean {imp['real'].loc['Mean', 'mean difference']:+.3f}. The twin gives the same order "
          f"(PCFI {imp['twin_matched'].loc['PCFI', 'mean difference']:+.3f}, "
          f"Tikhonov {imp['twin_matched'].loc['Tikhonov', 'mean difference']:+.3f}). NeighborMean equals Mean here: "
          "a hidden sensor node's graph neighbours are FE nodes, never measured.", "",
          f"**5. The network's real-bridge score hardly moves with fewer sensors, because it was low to begin with.** Main network, FP, "
          f"mean MAC against SSI at the sensors: every sensor {nm('real', 'full'):.3f} (s5), front_mid "
          f"{nm('real', 'front_mid'):.3f}, front pair {nm('real', 'front_pair'):.3f}, one sensor "
          f"{nm('real', 'single'):.3f}. With one sensor its prediction has MAC "
          f"{nm('real', 'single', 'mac_pred_vs_zeroinput'):.2f} with its response to an all-zero input "
          f"({nm('real', 'full', 'mac_pred_vs_zeroinput'):.2f} with every sensor). B1's score with one sensor "
          f"({float(n[(n.source == 'real') & (n.layout == 'single') & (n['mode'] == 'B1')].mac_ssi_sensor_nodes.mean()):.3f}) "
          f"is the network's default shape (MAC {float(n[(n.source == 'real') & (n.layout == 'single') & (n['mode'] == 'B1')].mac_pred_vs_zeroinput.mean()):.2f} "
          "with its zero-input response), not something read from the record. On the twin the same layouts score "
          f"{nm('twin_full', 'single', 'mac_fe_fe_nodes'):.3f} (one sensor) to {nm('twin_full', 'full', 'mac_fe_fe_nodes'):.3f} "
          "(every sensor) against the FE target.", ""]

    # ---- tables --------------------------------------------------------------------------------
    A += ["## Fill-in at the hidden sensors, production FP", "",
          "Mean over 9 bridges x 4 modes. *Deflection MAC*: filled vs recorded deflection over the hidden sensors "
          "(>= 2 hidden). *Sign*: share of the hidden motion filled with the right sign (amplitude-weighted). "
          "*Amplitude*: median filled / recorded |deflection|. *Coherence*: filled vs recorded signal, at the mode. "
          "*Shape MAC*: the same fill applied to the label shape (SSI on real, FE target on twin), over the hidden "
          "sensors - E16's ceiling, on real decks.", ""]
    for col, lab in (("defl_mac_hidden", "Deflection MAC"), ("sign_agree_w", "Sign"), ("amp_ratio_median", "Amplitude"),
                     ("coh_mean", "Coherence"), ("label_mac_hidden", "Shape MAC")):
        t = fp.pivot_table(index="layout", columns="source", values=col).reindex(LAYOUTS)[["real", "twin_matched", "twin_full"]]
        A += [f"*{lab}:*", ""] + table(t, "layout") + [""]
    for col, lab in (("defl_mac_hidden", "Deflection MAC"), ("sign_agree_w", "Sign")):
        for src in ("real", "twin_matched"):
            t = fp[fp.source == src].pivot_table(index="layout", columns="mode", values=col).reindex(LAYOUTS)[M]
            A += [f"*{lab} by mode, {src}:*", ""] + table(t, "layout") + [""]
    A += [f"*Bridge by bridge* (FP; the layouts with at least four sensors kept, {', '.join(POOLED)}, pooled; "
          "gap = twin_matched - real):", ""]
    t = pd.concat([bd[["real", "twin_matched", "gap"]].add_prefix("deflection "),
                   bl[["real", "twin_matched", "gap"]].add_prefix("shape ")], axis=1)
    A += table(t, "bridge") + ["", f"![{fig1}]({fig1})", "",
                               "*Each dot is one bridge (mean over modes); bars are the means.*", ""]

    A += ["## Coherence as a pass mark", "",
          f"Every FP-filled hidden sensor x mode that moves (|deflection| >= {REL_MOVE:.0%} of the largest in its bridge, "
          "layout and mode), fixed layouts, binned by its coherence with the recording:", ""]
    for src in ("real", "twin_matched"):
        t = cb[src].copy()
        t["cases"] = t["cases"].astype(int).astype(str)
        A += [f"*{src}:*", ""] + table(t, "coherence") + [""]
    A += [f"![{fig2}]({fig2})", ""]

    A += ["## Imputers against production FP", "",
          f"Deflection MAC at the hidden sensors, paired by bridge and layout ({len(LAYOUTS)} layouts x 9 bridges = "
          f"{9 * len(LAYOUTS)} cells), mean over modes. Positive = better than FP.", ""]
    t = pd.concat([imp["real"].add_prefix("real "), imp["twin_matched"].add_prefix("twin ")], axis=1)
    A += table(t, "imputer") + [""]
    t = f.pivot_table(index="imputer", columns=["source", "layout"], values="defl_mac_hidden")
    for src in ("real", "twin_matched"):
        A += [f"*Deflection MAC by layout, {src}:*", ""] + table(t[src][LAYOUTS].reindex(IMPUTERS), "imputer") + [""]

    A += ["## Node PSD at the hidden sensors, as the network sees it", "",
          "Relative error of the filled row-normalised PSD on the training grid (PSD route: propagate the sensors' PSD, "
          "as E5's builder does), over the hidden sensors.", ""]
    t = p[p.imputer == "FP"].pivot_table(index="layout", columns="source", values="psd_rel_err_hidden").reindex(LAYOUTS)
    A += ["*FP:*", ""] + table(t[["real", "twin_matched", "twin_full"]], "layout") + [""]
    t = p.pivot_table(index="imputer", columns="source", values="psd_rel_err_hidden").reindex(IMPUTERS)
    A += ["*Every imputer, mean over layouts:*", ""] + table(t[["real", "twin_matched", "twin_full"]], "imputer") + [""]

    A += ["## The network on fewer real sensors", "",
          f"`edgegat_N{C.MAIN_N}`, best-validation, FP, features from E5's corrected builder with only the kept "
          "sensors; scored as s5 (SSI at the sensor nodes) and s6 (FE target at the FE nodes). Mean over 9 bridges.", ""]
    rows = {}
    for lay in NET_LAYOUTS:
        rows[lay] = {"kept": kept[lay], "real vs SSI": nm("real", lay), "real vs FE": nm("real", lay, "mac_fe_fe_nodes"),
                     "twin vs FE": nm("twin_full", lay, "mac_fe_fe_nodes"),
                     "real, MAC with zero-input response": nm("real", lay, "mac_pred_vs_zeroinput")}
    A += table(pd.DataFrame(rows).T, "layout") + [""]
    for src, col in (("real", "mac_ssi_sensor_nodes"), ("twin_full", "mac_fe_fe_nodes")):
        t = n[n.source == src].pivot_table(index="layout", columns="mode", values=col).reindex(NET_LAYOUTS)[M]
        A += [f"*By mode, {src} ({'vs SSI' if src == 'real' else 'vs FE target'}):*", ""] + table(t, "layout") + [""]

    (HERE / "REPORT.md").write_text("\n".join(A) + "\n", encoding="utf-8")
    print("wrote REPORT.md,", fig1, fig2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

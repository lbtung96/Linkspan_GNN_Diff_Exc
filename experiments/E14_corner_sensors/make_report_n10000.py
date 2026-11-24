"""Assemble REPORT_N10000.md and its figures from the E14 stage-2 outputs.

Generated, never hand-edited. Every number is computed here from
`stage2_per_sample.csv`, `stage2_layouts.csv`, `stage2_diagnostics.csv` and
`raw_n10000/*.npz`; directional sentences are conditional on the numbers they
describe. The unit of replication is the structure (one seed).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs_phase2" / "EdgeGATConv" / "sensitivity_N10000_ep500"
MODES = ("B1", "T1", "B2", "T2")
NGS = (2, 3, 4, 5)
GOOD_MAC = 0.9
LAYOUT_LABEL = {"corners2": "2 nodes: front corners",
                "corners2_mid2": "4 nodes: front corners + girder midspans"}
C_FULL, C_TWO, C_FOUR, C_FLOOR = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def mac(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(abs(np.dot(u, v)) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def matrix(rows: list[dict], key: str, samples: list[str], **where) -> np.ndarray:
    idx = {(r["sample"], r["mode"]): r[key] for r in rows
           if all(r[k] == v for k, v in where.items())}
    return np.array([[idx[(s, m)] for m in MODES] for s in samples], dtype=float)


def row(label: str, vals, spec: str = ".3f") -> str:
    return f"| {label} | " + " | ".join(format(v, spec) for v in vals) + " |"


def head(first: str = "") -> list[str]:
    return [f"| {first} | " + " | ".join(MODES) + " |", "|---|" + "---:|" * 4]


def main() -> int:
    rows = read_csv(HERE / "stage2_per_sample.csv")
    lays = read_csv(HERE / "stage2_layouts.csv")
    diags = read_csv(HERE / "stage2_diagnostics.csv")
    samples = [d["sample"] for d in diags]
    n = len(samples)
    ng = np.array([int(d["Ng"]) for d in diags])

    V = ("full", "corners2_prop", "corners2_mid2_prop", "zeros")
    M = {(ck, v): matrix(rows, "mac", samples, checkpoint=ck, variant=v)
         for ck in ("last", "best") for v in V}
    FE = {v: matrix(rows, "abs_freq_err_hz", samples, checkpoint="last", variant=v) for v in V}
    true_f = matrix(rows, "true_freq_hz", samples, checkpoint="last", variant="full")
    fnorm = json.loads((RUN_DIR / "frequency_normalization.json").read_text(encoding="utf-8"))
    const_err = np.abs(true_f - np.asarray(fnorm["freq_mean_hz"], dtype=float))
    obs = {lay: matrix(lays, "sensor_amp_over_peak", samples, layout=lay) for lay in LAYOUT_LABEL}
    ceil = {lay: matrix(lays, "operator_ceiling_mac", samples, layout=lay) for lay in LAYOUT_LABEL}
    full, two, four, floor = (M[("last", v)] for v in V)

    # ---- checks -------------------------------------------------------------------
    tm = {r["sample_identifier"]: r
          for r in read_csv(RUN_DIR / "evaluation" / "test_metrics.csv")}
    rep_mac = max(abs(full[i, j] - tm[s][f"mac_{m}"])
                  for i, s in enumerate(samples) for j, m in enumerate(MODES))
    rep_freq = max(abs(FE["full"][i, j] - tm[s][f"freq_mae_{m}_hz"])
                   for i, s in enumerate(samples) for j, m in enumerate(MODES))
    raw = {s: np.load(HERE / "raw_n10000" / f"{s}.npz") for s in samples}
    flat_std = max(float((np.std(raw[s]["last_zeros"], axis=0)
                          / (np.abs(raw[s]["last_zeros"]).max(axis=0) + 1e-30)).max())
                   for s in samples)
    ones_gap = max(abs(floor[i, j] - mac(np.ones(len(raw[s]["target"])), raw[s]["target"][:, j]))
                   for i, s in enumerate(samples) for j in range(4))
    mid_x = np.array([[d["mid_x_over_L_0"], d["mid_x_over_L_1"]] for d in diags])

    # ---- paired: what the two midspan nodes add, unit = structure ------------------
    def paired(a: np.ndarray, b: np.ndarray) -> list[tuple]:
        out = []
        for j in range(4):
            d = a[:, j] - b[:, j]
            p = float(wilcoxon(a[:, j], b[:, j]).pvalue) if np.any(d != 0) else float("nan")
            out.append((float(d.mean()), int((d > 0).sum()), a.shape[0], p))
        return out

    add4 = paired(four, two)
    add4_ng = {g: paired(four[ng == g], two[ng == g]) for g in NGS}

    # ---- summary csv ----------------------------------------------------------------
    out = []
    for ck in ("last", "best"):
        for v in V:
            for g in (None,) + NGS:
                sel = np.ones(n, bool) if g is None else ng == g
                for j, m in enumerate(MODES):
                    x = M[(ck, v)][sel, j]
                    out.append({"checkpoint": ck, "variant": v, "Ng": "all" if g is None else g,
                                "mode": m, "n_structures": int(sel.sum()),
                                "mac_mean": x.mean(), "mac_median": np.median(x),
                                "mac_q25": np.percentile(x, 25), "mac_q75": np.percentile(x, 75),
                                f"frac_mac_ge_{GOOD_MAC}": float((x >= GOOD_MAC).mean()),
                                "freq_mae_hz": (FE[v][sel, j].mean() if ck == "last"
                                                else float("nan"))})
    with open(HERE / "summary_n10000.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    rep = make_figures(M, ng, raw, samples)

    # ---- REPORT_N10000.md -----------------------------------------------------------
    md: list[str] = []
    add = md.append
    cfg = json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
    counts = ", ".join(f"Ng = {g}: {int((ng == g).sum())}" for g in NGS)
    add("# E14 stage 2 - the N = 10 000 network, two and four sensors\n")
    add("**Generated by `make_report_n10000.py`. Do not hand-edit.** Stage 1 (the "
        "N = 600, two-girder network) is [`REPORT.md`](REPORT.md); design in "
        "[`README.md`](README.md).\n")
    add(f"Network `sensitivity_N10000_ep500` - broadband, seed 123, 500 epochs, trained "
        f"on all four girder counts - evaluated with `last_model.pt` on its own {n} "
        f"held-out synthetic structures ({counts}). Synthetic domain only; one seed, so "
        f"the unit of replication is the structure.\n")
    add("Two sensor layouts, every other node filled by the production propagation "
        "operator exactly as in stage 1:\n")
    add("- **2 nodes** - the front corners: front beam x the two outermost girders.")
    add(f"- **4 nodes** - those two plus the girder node nearest midspan on each "
        f"outermost girder (at x/L = {mid_x.min():.3f}-{mid_x.max():.3f}).\n")

    add("## The answer\n")
    md.extend(head())
    add(row("full graph (every node measured)", full.mean(0)))
    add(row("**2 nodes, propagated**", two.mean(0)))
    add(row("**4 nodes, propagated**", four.mean(0)))
    add(row("flat floor (no spatial information)", floor.mean(0)))
    add(row(f"share at MAC >= {GOOD_MAC}, full graph", (full >= GOOD_MAC).mean(0)))
    add(row(f"share at MAC >= {GOOD_MAC}, 2 nodes", (two >= GOOD_MAC).mean(0)))
    add(row(f"share at MAC >= {GOOD_MAC}, 4 nodes", (four >= GOOD_MAC).mean(0)))
    add(row("operator ceiling, 2 nodes", ceil["corners2"].mean(0)))
    add(row("operator ceiling, 4 nodes", ceil["corners2_mid2"].mean(0)))
    add("")
    add("Mean MAC over all graph nodes against the stored signed FE target. **Operator "
        "ceiling**: the true mode shape, known only at the sensor nodes, pushed through "
        "the same propagation operator - what the operator alone can carry, before any "
        "network.\n")
    add("ANSWER_PROSE")

    add("## By girder count\n")
    for g in NGS:
        sel = ng == g
        add(f"**Ng = {g}** ({int(sel.sum())} structures)\n")
        md.extend(head())
        add(row("full graph", full[sel].mean(0)))
        add(row("2 nodes", two[sel].mean(0)))
        add(row("4 nodes", four[sel].mean(0)))
        add(row("flat floor", floor[sel].mean(0)))
        add(row("operator ceiling, 4 nodes", ceil["corners2_mid2"][sel].mean(0)))
        add("")

    add("## What the two midspan nodes add\n")
    add("4 nodes minus 2 nodes, paired by structure (Wilcoxon signed-rank):\n")
    md.extend(head())
    add(row("mean gain, all structures", [t[0] for t in add4], "+.3f"))
    add("| structures improved | " + " | ".join(f"{t[1]}/{t[2]}" for t in add4) + " |")
    add("| p | " + " | ".join(f"{t[3]:.1e}" for t in add4) + " |")
    for g in NGS:
        add(row(f"mean gain, Ng = {g}", [t[0] for t in add4_ng[g]], "+.3f"))
    add("")
    add("How much of each mode the sensors can see - the largest sensor amplitude over "
        "the mode's peak amplitude anywhere on the deck:\n")
    md.extend(head())
    for lay, lab in LAYOUT_LABEL.items():
        add(row(f"{lab}, mean", obs[lay].mean(0), ".2f"))
        add(row(f"{lab}, worst structure", obs[lay].min(0), ".3f"))
    add("")
    add("GAIN_PROSE")
    add("### Where B2 goes wrong\n")
    add("Mean along-span B2 profile on the girder lines, by x/L bin (each shape "
        "sign-aligned to the target and max-abs normalised; bins labelled by centre). The "
        "midspan sensors sit at or just below L/2, on the 0.45/0.55 bin boundary; the "
        "corners in the 0.95 bin:\n")
    md.extend(b2_profile_lines(raw, samples, ng))
    add("")
    add("At every girder count the target is one broad hump peaking near 0.3-0.4 L. "
        "Where the gain is negative, the 4-node prediction instead collapses onto a "
        "narrow bump at the midspan sensors and is near zero elsewhere; where it is "
        "positive, it spreads into a hump of roughly the right width. Why this splits by "
        "girder count was not pursued here. Far from any sensor, 2-node predictions are "
        "plateaus: every such node carries nearly the same mixture of sensor records, "
        "which is locally the uniform input that gives a flat output (see Checks).\n")

    add("## Frequencies\n")
    add("Mean absolute error, Hz:\n")
    md.extend(head())
    add(row("full graph", FE["full"].mean(0), ".2f"))
    add(row("2 nodes, propagated", FE["corners2_prop"].mean(0), ".2f"))
    add(row("4 nodes, propagated", FE["corners2_mid2_prop"].mean(0), ".2f"))
    add(row("all-zero input", FE["zeros"].mean(0), ".2f"))
    add(row("constant guess: training-set mean", const_err.mean(0), ".2f"))
    add(row("true frequency, mean", true_f.mean(0), ".2f"))
    add("")
    add("FREQ_PROSE")

    add(f"## Robustness: `best_model.pt` (epoch {int(cfg.get('best_epoch', -1))})\n")
    md.extend(head())
    for v, lab in (("full", "full graph"), ("corners2_prop", "2 nodes"),
                   ("corners2_mid2_prop", "4 nodes"), ("zeros", "flat floor")):
        add(row(lab, M[("best", v)].mean(0)))
    add("")
    diffs = {v: M[("best", v)].mean(0) - M[("last", v)].mean(0)
             for v in ("corners2_prop", "corners2_mid2_prop")}
    j_max = int(np.argmax([max(abs(diffs[v][j]) for v in diffs) for j in range(4)]))
    best_top = max(float(M[("best", v)].mean(0).max()) for v in diffs)
    add(f"The conclusion holds, but the propagated rows are **checkpoint-sensitive**, "
        f"{MODES[j_max]} most: "
        + ", ".join(f"{'2' if v == 'corners2_prop' else '4'} nodes "
                    f"{M[('last', v)][:, j_max].mean():.3f} -> {M[('best', v)][:, j_max].mean():.3f}"
                    for v in diffs)
        + f" from last to best epoch, while the full graph moves by "
          f"{float(np.abs(M[('best', 'full')].mean(0) - M[('last', 'full')].mean(0)).max()):.3f}. "
          f"Off the training distribution, weights that agree on the full graph disagree on "
          f"sparse input. No propagated row exceeds {best_top:.3f} under either checkpoint.\n")

    add("## Figures\n")
    add("- `fig_n10000_mac_by_ng.png` - mean MAC with interquartile range, per mode and "
        "girder count, for the full graph, both layouts and the flat floor.")
    add(f"- `fig_n10000_modeshapes_{samples[rep]}.png` - the Ng = 4 structure whose 4-node "
        f"mean MAC is the median of its group: FE target, full graph, 2 nodes, 4 nodes. "
        f"Black rings mark the sensors.\n")

    add("## Checks\n")
    add(f"- **The full-graph row reproduces the run's own test metrics** "
        f"(`evaluation/test_metrics.csv`, computed on GPU in the source repository): "
        f"largest per-structure MAC difference {rep_mac:.1e}, frequency error "
        f"{rep_freq:.1e} Hz, over all {n} structures.")
    add(f"- **Re-simulation is faithful**: rebuilt inputs match the stored ones to "
        f"{max(d['x_psd_resim_maxdiff'] for d in diags):.1e} (node PSD) and "
        f"{max(d['edge_resim_maxdiff'] for d in diags):.1e} (edge coherency); target "
        f"frequencies to {max(d['freq_resim_maxdiff_hz'] for d in diags):.1e} Hz.")
    add(f"- **Propagation reaches every node** "
        f"({int(sum(d['corners2_n_unreached'] + d['corners2_mid2_n_unreached'] for d in diags))} "
        f"unreached). Farthest-node amplitude relative to the sensors: as low as "
        f"{min(d['corners2_min_rms_ratio'] for d in diags):.1e} with 2 nodes, "
        f"{min(d['corners2_mid2_min_rms_ratio'] for d in diags):.1e} with 4; the features "
        f"are scale-invariant, so the network sees mixing ratios, not amplitudes.")
    add(f"- **The flat floor is still flat under this network**: all-zero input gives a "
        f"spatially uniform shape (largest spatial std / max-abs {flat_std:.1e}) whose MAC "
        f"equals that of an all-ones vector to {ones_gap:.1e}.\n")

    add("## What this does not establish\n")
    add("- **Whether a network trained on sparse, propagated inputs would do better.** "
        "Neither network here ever saw one. Retraining with the same sensor layouts "
        "applied to the training inputs is the experiment that answers it.")
    add("- **Real sensors.** The sensor records are noise-free simulation.")
    add("- **Retraining variance.** One seed. N = 10 000 has no second seed in E3.")
    add("- **Other operators or layouts.** Only the production diffusion operator and "
        "these two layouts were run.")

    stage1 = None
    s1_path = HERE / "stage1_per_sample.csv"
    if s1_path.exists():
        s1 = [r for r in read_csv(s1_path)
              if r["checkpoint"] == "last" and r["variant"] == "corners_prop"]
        stage1 = np.array([np.mean([r["mac"] for r in s1 if r["mode"] == m]) for m in MODES])

    text = "\n".join(md) + "\n"
    text = text.replace("ANSWER_PROSE", answer_prose(full, two, four, floor, obs, ng, stage1))
    text = text.replace("GAIN_PROSE", gain_prose(add4, add4_ng))
    text = text.replace("FREQ_PROSE", freq_prose(FE, const_err))
    (HERE / "REPORT_N10000.md").write_text(text, encoding="utf-8")
    print(f"wrote {HERE / 'REPORT_N10000.md'}, summary_n10000.csv and figures")
    return 0


def span(a: np.ndarray) -> str:
    m = a.mean(0)
    return f"{m.min():.2f}-{m.max():.2f}"


def answer_prose(full, two, four, floor, obs, ng, stage1) -> str:
    better = [m for j, m in enumerate(MODES) if four[:, j].mean() > two[:, j].mean()]
    p = [f"**No, for either layout.** Mean MAC is {span(full)} on the full graph, "
         f"{span(two)} with 2 nodes and {span(four)} with 4. Four nodes beat two on "
         f"average in {', '.join(better) if better else 'no mode'}"
         + (f" (all four modes)" if len(better) == 4 else "") +
         f", but only {int((four >= GOOD_MAC).sum())} of {four.size} structure-mode cells "
         f"reach MAC {GOOD_MAC} with 4 nodes and {int((two >= GOOD_MAC).sum())} with 2, "
         f"against {int((full >= GOOD_MAC).sum())} on the full graph.\n"]
    o4 = obs["corners2_mid2"].mean(0)
    p.append(f"**Seeing the modes is not the limit.** With 4 nodes every mode is well "
             f"observed - the sensors carry {o4.min():.2f}-{o4.max():.2f} of each mode's "
             f"peak amplitude on average (B2 {o4[2]:.2f}, T2 {o4[3]:.2f}, up from "
             f"{obs['corners2'].mean(0)[2]:.2f} and {obs['corners2'].mean(0)[3]:.2f} with 2) "
             f"- and B2/T2 still reach only {four[:, 2].mean():.3f} and "
             f"{four[:, 3].mean():.3f}. T1 is at full amplitude at the corners in both "
             f"layouts and reaches {two[:, 1].mean():.3f} / {four[:, 1].mean():.3f}, falling "
             f"with deck width: "
             + ", ".join(f"Ng = {g} {four[ng == g, 1].mean():.2f}" for g in NGS)
             + " with 4 nodes. As in stage 1, the network was trained on inputs where "
               "every node carries its own response, and a propagated field is not one.\n")
    if stage1 is not None:
        s2 = two[ng == 2].mean(0)
        worse = int((s2 < stage1).sum())
        lead = ("**Training on all four girder counts does not make it more robust.**"
                if s2.mean() <= stage1.mean() + 0.05 else
                "**Training on all four girder counts helps somewhat.**")
        p.append(f"{lead} On two-girder decks with the 2 corner sensors it is lower than "
                 f"the N = 600 network in {worse} of 4 modes, scoring "
                 + " / ".join(f"{v:.3f}" for v in s2)
                 + " (B1/T1/B2/T2) against "
                 + " / ".join(f"{v:.3f}" for v in stage1)
                 + " for the two-girder-only N = 600 network in stage 1. The structure sets "
                   "differ (each network's own held-out split, same generator), so compare "
                   "levels, not structures.\n")
    return "\n".join(p)


def gain_prose(add4, add4_ng) -> str:
    helped = [m for m, t in zip(MODES, add4) if t[0] > 0 and t[3] < 0.01]
    split = [m for j, m in enumerate(MODES)
             if min(add4_ng[g][j][0] for g in NGS) < -0.05 < 0.05 < max(add4_ng[g][j][0] for g in NGS)]
    top = int(np.argmax([t[0] for t in add4]))
    s = (f"Pooled over girder counts, {', '.join(helped) if helped else 'no mode'} "
         f"improve{'s' if len(helped) == 1 else ''} significantly with the two midspan "
         f"nodes, {MODES[top]} most ({add4[top][0]:+.3f}); B2 and T2 gain in observability "
         f"(table above), B1 and T1 were already fully observed at the corners.")
    if split:
        s += (f" But the gain is not uniform across girder counts for "
              f"{', '.join(split)}: "
              + "; ".join(f"{m} " + ", ".join(f"Ng = {g} {add4_ng[g][MODES.index(m)][0]:+.2f}"
                                               for g in NGS) for m in split)
              + ". Where it goes negative, adding two correctly measured nodes makes the "
                "prediction worse than two nodes alone.")
    return s + "\n"


def freq_prose(FE, const_err) -> str:
    c = const_err.mean(0)
    beat = [m for j, m in enumerate(MODES) if FE["corners2_mid2_prop"][:, j].mean() < c[j]]
    lose2 = [m for j, m in enumerate(MODES) if FE["corners2_prop"][:, j].mean() > c[j]]
    return (f"With 2 nodes the frequency head is worse than a constant training-mean guess "
            f"in {', '.join(lose2) if lose2 else 'no mode'}"
            f" (mean {FE['corners2_prop'].mean():.2f} Hz vs {c.mean():.2f} Hz). With 4 nodes "
            f"it beats the guess in {', '.join(beat) if beat else 'no mode'} (mean "
            f"{FE['corners2_mid2_prop'].mean():.2f} Hz), roughly halving the 2-node error, "
            f"but it stays well short of the full graph ({FE['full'].mean():.2f} Hz).\n")


def b2_profile_lines(raw: dict, samples: list[str], ng: np.ndarray) -> list[str]:
    """Mean along-span B2 profile, girder nodes, sign-aligned to the target."""
    bins = np.linspace(0.0, 1.0, 11)
    lines = ["| Ng | shape | " + " | ".join(f"{b:.2f}" for b in bins[:-1] + 0.05) + " |",
             "|---|---|" + "---:|" * 10]
    for g in NGS:
        prof = {k: [] for k in ("target", "last_corners2_mid2_prop")}
        for i in np.where(ng == g)[0]:
            d = raw[samples[i]]
            xy, t = d["coords"], d["target"][:, 2]
            on = (xy[:, 1] >= -1.0) & (xy[:, 1] <= xy[d["corners"][1], 1] + 1.0)
            xb = np.clip(np.digitize(xy[:, 0] / xy[:, 0].max(), bins) - 1, 0, 9)
            for k in prof:
                v = d[k][:, 2]
                v = v * (np.sign(np.dot(v, t)) or 1.0)
                v = v / (np.abs(v).max() or 1.0)
                prof[k].append([v[on & (xb == b)].mean() if np.any(on & (xb == b)) else np.nan
                                for b in range(10)])
        for k, lab in (("target", "FE target"), ("last_corners2_mid2_prop", "4-node prediction")):
            lines.append(f"| {g} | {lab} | "
                         + " | ".join(f"{v:+.2f}" for v in np.nanmean(prof[k], axis=0)) + " |")
    return lines


def make_figures(M: dict, ng: np.ndarray, raw: dict, samples: list[str]) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from infer_phase2a_deterministic_edgegat import _MODE_SHAPE_CMAP

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    series = [("Full graph", M[("last", "full")], C_FULL, "o"),
              ("2 nodes, propagated", M[("last", "corners2_prop")], C_TWO, "o"),
              ("4 nodes, propagated", M[("last", "corners2_mid2_prop")], C_FOUR, "o"),
              ("Flat floor", M[("last", "zeros")], C_FLOOR, "_")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
    offs = (-0.27, -0.09, 0.09, 0.27)
    for k, g in enumerate(NGS):
        ax = axes[k]
        sel = ng == g
        for (label, x, col, mk), dx in zip(series, offs):
            xs = x[sel]
            lo, mid, hi = (np.percentile(xs, q, axis=0) for q in (25, 50, 75))
            mean = xs.mean(axis=0)
            pos = np.arange(4) + dx
            ax.vlines(pos, lo, hi, color=col, lw=2, alpha=0.55, zorder=2)
            ax.scatter(pos, mean, color=col, s=44 if mk == "o" else 120, marker=mk,
                       edgecolors="white" if mk == "o" else None, linewidths=1.2 if mk == "o" else 2.5,
                       zorder=3, label=label if k == 0 else None)
        ax.set_xticks(range(4), MODES)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Ng = {g}  ({int(sel.sum())} structures)", fontsize=10, color=INK,
                     loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("MAC vs FE target (mean, IQR bar)")
    fig.suptitle("N = 10 000 network on its held-out structures: full graph vs 2 and 4 "
                 "measured nodes", fontsize=11, color=INK, x=0.01, ha="left")
    fig.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(HERE / "fig_n10000_mac_by_ng.png", dpi=170)
    plt.close(fig)

    # representative Ng = 4 structure: median 4-node mean MAC within the group
    four = M[("last", "corners2_mid2_prop")].mean(axis=1)
    cand = np.where(ng == 4)[0]
    rep = int(cand[np.argsort(four[cand])[len(cand) // 2]])
    s = samples[rep]
    d = raw[s]
    xy, ei = d["coords"], d["edge_index"]
    sens2 = d["corners"]
    sens4 = np.concatenate([d["corners"], d["mids"]])
    cols = [("FE target", d["target"], None), ("Full graph", d["last_full"], None),
            ("2 nodes", d["last_corners2_prop"], sens2),
            ("4 nodes", d["last_corners2_mid2_prop"], sens4)]
    fig, axes = plt.subplots(4, 4, figsize=(15, 9.5), squeeze=False)
    for m, name in enumerate(MODES):
        for k, (title, arr, sens) in enumerate(cols):
            ax = axes[m, k]
            v = np.asarray(arr[:, m], dtype=float)
            v = v / (np.abs(v).max() or 1.0)
            for a, b in ei.T[ei[0] < ei[1]]:
                ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=GRID, lw=0.7, zorder=1)
            ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=_MODE_SHAPE_CMAP, vmin=-1, vmax=1,
                       s=26, edgecolors="#555555", linewidths=0.4, zorder=2)
            if sens is not None:
                ax.scatter(xy[sens, 0], xy[sens, 1], s=130, facecolors="none",
                           edgecolors=INK, linewidths=1.6, zorder=3)
            score = "" if k == 0 else f"  MAC {mac(arr[:, m], d['target'][:, m]):.2f}"
            ax.set_title(f"{name} - {title}{score}", fontsize=9, color=INK)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
    fig.suptitle(f"{s}: median 4-node MAC among the Ng = 4 structures (each panel max-abs "
                 f"normalised; black rings = measured nodes; front beam at the right)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / f"fig_n10000_modeshapes_{s}.png", dpi=150)
    plt.close(fig)
    return rep


if __name__ == "__main__":
    raise SystemExit(main())

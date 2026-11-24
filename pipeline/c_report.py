"""T5 - analysis tables and README.md, generated from the stage CSVs (never hand-edited).

    python tools/run_stage.py pipeline/c_report.py

Reads rescore_*.csv, ec23_*.csv, freq_baselines.csv, fdd_rows.csv, fdd_timing.csv,
fdd_checks.json, fresh_checks.json, rescore_gate.json and the stored predictions; writes
fresh_vs_fixed.csv, ablation_fresh_vs_fixed.csv, sweep_fresh_vs_fixed.csv,
shape_baselines.csv, fdd_summary.csv and README.md in results/T5_fresh_test_baselines/.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import ec_common as E

C = E.C
from s4_indomain import bootstrap_ci  # noqa: E402

OUT = E.task_dir("T5")
T5_DATA = E.EC_DATA / "T5"
M = list(C.MODE_NAMES)
MAIN = C.sweep_run_name(C.MAIN_N)
GRID = C.ABLATION_GRID
N_BOOT = 2000
BOOT_SEED = 20261123
SET_LABEL = {"paper": "fixed (paper)", "fresh_test": "fresh", "fresh_test_paperexc": "fresh, paper excitation"}


def f3(x, n=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def pm(d, lo, hi, n=3):
    return f"{d:+.{n}f} [{lo:+.{n}f}, {hi:+.{n}f}]"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def boot_unpaired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """mean(b) - mean(a), each set resampled within girder count (125 per Ng), independently."""
    rng = np.random.default_rng(BOOT_SEED)
    ga = [a[a.Ng == g][col].to_numpy() for g in C.NG_VALUES]
    gb = [b[b.Ng == g][col].to_numpy() for g in C.NG_VALUES]
    d = np.empty(N_BOOT)
    for k in range(N_BOOT):
        ma = np.concatenate([x[rng.integers(0, x.size, x.size)] for x in ga]).mean()
        mb = np.concatenate([x[rng.integers(0, x.size, x.size)] for x in gb]).mean()
        d[k] = mb - ma
    return float(b[col].mean() - a[col].mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    per = pd.read_csv(OUT / "rescore_per_sample.csv")
    per = per[per.checkpoint == "best_model"]
    summ = pd.read_csv(OUT / "rescore_summary.csv")
    summ = summ[summ.checkpoint == "best_model"]
    gate = json.loads((OUT / "rescore_gate.json").read_text(encoding="utf-8"))
    fchk = json.loads((OUT / "fresh_checks.json").read_text(encoding="utf-8"))
    sets_avail = [s for s in ("paper", "fresh_test", "fresh_test_paperexc") if s in set(summ.test_set)]
    paper_runs = [r for r in summ[summ.test_set == "paper"].run if summ[(summ.run == r) & (summ.test_set == "paper")].run_root.iloc[0] == "paper"]

    # ---------------- 1. fresh vs fixed, every paper run ------------------------------------
    rows = []
    for run in paper_runs:
        a = per[(per.test_set == "paper") & (per.run == run)]
        b = per[(per.test_set == "fresh_test") & (per.run == run)]
        if b.empty:
            continue
        r = {"run": run}
        for col in [f"mac_{m}" for m in M] + ["mac_mean", "freq_mae_hz"]:
            d, lo, hi = boot_unpaired(a, b, col)
            r[f"{col}_fixed"], r[f"{col}_fresh"] = a[col].mean(), b[col].mean()
            r[f"{col}_diff"], r[f"{col}_ci95_lo"], r[f"{col}_ci95_hi"] = d, lo, hi
        c = per[(per.test_set == "fresh_test_paperexc") & (per.run == run)]
        if not c.empty:            # paired: same structures, only the excitation differs
            j = b.merge(c, on="sample", suffixes=("_new", "_old"))
            for col in ("mac_mean", "mac_T1", "freq_mae_hz"):
                dd = (j[f"{col}_new"] - j[f"{col}_old"]).to_numpy()
                lo, hi = bootstrap_ci(dd)
                r[f"{col}_fresh_paperexc"] = c[col].mean()
                r[f"{col}_excitation_effect"], r[f"{col}_excitation_effect_ci95_lo"], r[f"{col}_excitation_effect_ci95_hi"] = dd.mean(), lo, hi
        rows.append(r)
    fvf = pd.DataFrame(rows)
    fvf.to_csv(OUT / "fresh_vs_fixed.csv", index=False)

    # ---------------- 2. sweep: smallest N with every mode >= 0.9 -----------------------------
    sw = []
    for n in C.SWEEP_N:
        run = C.sweep_run_name(n)
        for ts in sets_avail:
            s = summ[(summ.run == run) & (summ.test_set == ts)]
            if s.empty:
                continue
            s = s.iloc[0]
            sw.append({"N": n, "test_set": ts, **{f"mac_{m}": s[f"mac_{m}"] for m in M}, "mac_mean": s.mac_mean,
                       "min_mode_mac": min(s[f"mac_{m}"] for m in M), "freq_mae_hz": s.freq_mae_hz,
                       "mac_T1_ci95_lo": s.mac_T1_ci95_lo, "mac_T1_ci95_hi": s.mac_T1_ci95_hi})
    sw = pd.DataFrame(sw)
    sw["all_modes_ge_0.9"] = sw.min_mode_mac >= 0.9
    sw.to_csv(OUT / "sweep_fresh_vs_fixed.csv", index=False)
    smallest = {ts: (int(sw[(sw.test_set == ts) & sw["all_modes_ge_0.9"]].N.min())
                     if sw[(sw.test_set == ts) & sw["all_modes_ge_0.9"]].size else None) for ts in sets_avail}

    # ---------------- 3. ablation: ranking and contrasts, on each set -------------------------
    arms = {f"{l} / {f}": GRID[(l, f)] for l in C.ABLATION_LAYERS for f in C.ABLATION_FEATURES}
    arms["GraphSAGE / scalar CPSD 8-d"] = C.ABLATION_EXTRA[0]
    ab = []
    for ts in sets_avail:
        for lab, run in arms.items():
            s = summ[(summ.run == run) & (summ.test_set == ts)]
            if not s.empty:
                ab.append({"test_set": ts, "arm": lab, "run": run, "mac_mean": s.mac_mean.iloc[0],
                           "mac_T1": s.mac_T1.iloc[0], "freq_mae_hz": s.freq_mae_hz.iloc[0]})
    ab = pd.DataFrame(ab)
    ab["rank"] = ab.groupby("test_set").mac_mean.rank(ascending=False).astype(int)
    con = []

    def diff(ts, x, y, col):
        a = per[(per.test_set == ts) & (per.run == x)].set_index("sample")[col]
        b = per[(per.test_set == ts) & (per.run == y)].set_index("sample")[col]
        return (a - b.loc[a.index]).to_numpy()

    contrasts = [(f"layer, {f}: EdgeGATConv - GraphSAGE", lambda ts, col, f=f: diff(ts, GRID[("EdgeGATConv", f)], GRID[("GraphSAGE", f)], col))
                 for f in C.ABLATION_FEATURES]
    for l in C.ABLATION_LAYERS:
        for f0, f1 in zip(C.ABLATION_FEATURES, C.ABLATION_FEATURES[1:]):
            contrasts.append((f"feature, {l}: {f1} - {f0}", lambda ts, col, l=l, f0=f0, f1=f1: diff(ts, GRID[(l, f1)], GRID[(l, f0)], col)))
    contrasts.append(("interaction: (coherency - orientation) in EdgeGATConv minus in GraphSAGE",
                      lambda ts, col: diff(ts, GRID[("EdgeGATConv", "coherency")], GRID[("EdgeGATConv", "orientation")], col)
                      - diff(ts, GRID[("GraphSAGE", "coherency")], GRID[("GraphSAGE", "orientation")], col)))
    s11 = pd.read_csv(C.RESULTS / "s11_ablation_contrasts.csv").set_index("contrast")
    s11_ok = True
    for name, fn in contrasts:
        for ts in sets_avail:
            r = {"contrast": name, "test_set": ts}
            for col in ("mac_mean", "mac_T1"):
                d = fn(ts, col)
                lo, hi = bootstrap_ci(d)
                r[f"{col}_diff"], r[f"{col}_ci95_lo"], r[f"{col}_ci95_hi"] = float(d.mean()), lo, hi
                if ts == "paper":
                    s11_ok &= abs(r[f"{col}_diff"] - s11.loc[name, f"{col}_diff"]) < 1e-12 and \
                        abs(lo - s11.loc[name, f"{col}_ci95_lo"]) < 1e-12 and abs(hi - s11.loc[name, f"{col}_ci95_hi"]) < 1e-12
            con.append(r)
    con = pd.DataFrame(con)
    abl = ab.merge(con, how="outer", on=["test_set"], suffixes=("", "")) if False else None
    ab.to_csv(OUT / "ablation_ranking.csv", index=False)
    con.to_csv(OUT / "ablation_contrasts_fresh_vs_fixed.csv", index=False)
    tau = None
    if {"paper", "fresh_test"} <= set(ab.test_set):
        x = ab[ab.test_set == "paper"].set_index("arm").mac_mean
        y = ab[ab.test_set == "fresh_test"].set_index("arm").mac_mean.loc[x.index]
        tau = kendalltau(x, y).statistic

    # ---------------- 4. EC-23 ---------------------------------------------------------------
    e23 = pd.read_csv(OUT / "ec23_summary.csv")
    cross = pd.read_csv(OUT / "ec23_crossmac.csv")

    # ---------------- 5. EC-25 baselines -------------------------------------------------------
    fb = pd.read_csv(OUT / "freq_baselines.csv")
    fdd = pd.read_csv(OUT / "fdd_rows.csv")
    fdd["fail"] = (~fdd.correct.astype(bool)) | (fdd.mac < 0.5)
    fchk_fdd = json.loads((OUT / "fdd_checks.json").read_text(encoding="utf-8"))
    tim = pd.read_csv(OUT / "fdd_timing.csv")
    fs = (fdd.groupby(["split", "layout", "method"])
          .agg(mac=("mac", "mean"), identified=("correct", "mean"), present=("present", "mean"),
               fail_rate=("fail", "mean"), ferr_mean=("ferr", "mean"), ferr_median=("ferr", "median"))
          .reset_index())
    fs.to_csv(OUT / "fdd_summary.csv", index=False)
    fsm = (fdd.groupby(["split", "layout", "method", "mode"])
           .agg(mac=("mac", "mean"), identified=("correct", "mean"), fail_rate=("fail", "mean"),
                ferr_mean=("ferr", "mean")).reset_index())
    fsm.to_csv(OUT / "fdd_summary_by_mode.csv", index=False)
    # network on deck nodes only (the template's and FDD's scoring nodes), from the stored predictions
    netdeck = []
    for ts in ("paper", "fresh_test"):
        pf = T5_DATA / "predictions" / ts / f"{MAIN}__best_model.npz"
        df = T5_DATA / "fdd" / f"{ts}_decks.npz"
        if not (pf.exists() and df.exists()):
            continue
        P, D = np.load(pf), np.load(df)
        for smp in [str(s) for s in P["samples"]]:
            k = smp.removesuffix(".npz")
            p, y, dk = P[f"{k}__pred"].astype(np.float64), P[f"{k}__target"].astype(np.float64), ~D[f"support__{smp}"]
            for i, m in enumerate(M):
                u, v = p[dk, i], y[dk, i]
                netdeck.append({"split": ts, "sample": smp, "mode": m,
                                "mac_deck": float(np.dot(u, v) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))})
    netdeck = pd.DataFrame(netdeck)
    shp = []
    for ts in ("paper", "fresh_test"):
        q = per[(per.test_set == ts) & (per.run == MAIN)]
        if q.empty:
            continue
        row = {"test_set": ts, "method": "network N2000 (every node measured), all graph nodes",
               **{m: q[f"mac_{m}"].mean() for m in M}, "mean": q.mac_mean.mean(),
               "fail_rate (MAC<0.5)": float(np.mean([(q[f"mac_{m}"] < 0.5).mean() for m in M]))}
        shp.append(row)
        nd = netdeck[netdeck.split == ts] if len(netdeck) else netdeck
        if len(nd):
            shp.append({"test_set": ts, "method": "network N2000 (every node measured), deck nodes",
                        **{m: nd[nd["mode"] == m].mac_deck.mean() for m in M}, "mean": nd.mac_deck.mean(),
                        "fail_rate (MAC<0.5)": float((nd.mac_deck < 0.5).mean())})
        for lay, meth, lab in (("all", "template", "class template (no measurement, class given), deck nodes"),
                               ("all", "given", "FDD every deck node, frequency+class given"),
                               ("all", "population", "FDD every deck node, population labeller"),
                               ("all", "rank", "FDD every deck node, rank labeller (phase+order only)"),
                               ("outer1", "population", "FDD 2 front corners + E18 fill, population labeller"),
                               ("outer2", "population", "FDD 4 sensors + E18 fill, population labeller")):
            z = fdd[(fdd.split == ts) & (fdd.layout == lay) & (fdd.method == meth)]
            if z.empty:
                continue
            shp.append({"test_set": ts, "method": lab, **{m: z[z["mode"] == m].mac.mean() for m in M},
                        "mean": z.mac.mean(), "fail_rate (MAC<0.5)": float((z.mac < 0.5).mean()),
                        "identified": float(z.correct.mean()) if meth not in ("template", "given") else np.nan,
                        "fail_rate (not identified or MAC<0.5)": float(z.fail.mean()) if meth not in ("template", "given") else np.nan,
                        "ferr_mean_hz": float(z.ferr.mean()) if meth != "template" else np.nan})
    shp = pd.DataFrame(shp)
    shp.to_csv(OUT / "shape_baselines.csv", index=False)

    # ---------------- README -------------------------------------------------------------------
    L = []
    w = L.append
    w("# T5 - a fresh untouched synthetic test set, shape metrics beyond MAC, and baselines\n")
    w("**Generated by `c_report.py` from the CSVs in this folder. Do not hand-edit.**  "
      "These results answer review comments; they are not part of `PAPER_RESULTS.md` (see `../../../README.md`). "
      "Every network is scored from its **best-validation checkpoint**, one training seed (123), as in the paper.\n")
    w("## The comments\n")
    w("- **EC-15** - \"the fixed 500-structure test set is used to select both the architecture/edge input (Section 3.2) "
      "and population size (Section 3.4)... Use validation results for these choices and evaluate once on a new untouched "
      "synthetic test set, or label the current results exploratory/model-selection results.\"")
    w("- **EC-23** - \"MAC measures alignment, not amplitude calibration... Add a sign-aligned shape error, full cross-MAC "
      "matrices and per-mode success rates. If SSI shapes are complex, explain the real-mode projection/phase alignment or "
      "use the Hermitian MAC.\"")
    w("- **EC-25** - \"compare with conventional FDD/SSI on the same simulated records, a mean-frequency/mean-template or "
      "geometry-only predictor, and the previous PSD-based model. Report failure rates and runtime at matched sensor layouts "
      "and noise levels. This separates response-driven identification from interpolation of a strongly constrained "
      "population prior.\"\n")

    # --- headline
    hm = fvf[fvf.run == MAIN].iloc[0] if len(fvf[fvf.run == MAIN]) else None
    if hm is not None and "mac_mean_fresh_paperexc" in hm:
        w("## Headline\n")
        w(f"- **On a new excitation realisation every network collapses.** Main network (N = {C.MAIN_N}): mean MAC "
          f"{hm.mac_mean_fixed:.3f} on the fixed test set, {hm.mac_mean_fresh_paperexc:.3f} on 500 new structures excited "
          f"by the paper's noise realisation, **{hm.mac_mean_fresh:.3f}** on the same 500 structures with a new one "
          f"(frequency MAE {hm.freq_mae_hz_fixed:.3f} / {hm.freq_mae_hz_fresh_paperexc:.3f} / {hm.freq_mae_hz_fresh:.3f} Hz). "
          "Every paper structure - training, validation and test - was excited by the same white-noise realisation "
          "(simulator seed 12345, re-seeded per structure), so the fixed test set never tested a new one. New structures "
          "alone change nothing; the drop is the excitation.")
        w("- For the coherency-edge networks the dependence enters through the edge coherency, not the node PSDs and not the "
          "support records (section 2, input swap). Node-only and orientation-only networks lose less, through the node PSDs.")
        w("- FDD on the very same new records is unaffected (every deck node, population labeller: mean MAC "
          + f3(shp[(shp.test_set == 'fresh_test') & shp.method.str.startswith('FDD every deck node, population')]['mean'].iloc[0])
          + "), so the records are sound; the class template (no record) scores "
          + f3(shp[(shp.test_set == 'fresh_test') & shp.method.str.startswith('class template')]['mean'].iloc[0]) + ".")
        w("- With the paper's excitation, the fresh set reproduces the fixed-set conclusions: smallest N with every mode >= 0.9 "
          f"is {smallest.get('fresh_test_paperexc')}, and the ablation ranking is unchanged. With the new excitation no N "
          "reaches it and the ranking changes (orientation arms first).\n")
    # --- fresh set
    gen = fchk["generation"]
    nov = fchk["novelty"]
    w("## 1. The fresh test set (EC-15)\n")
    w("**Which seed drives what.** The generator draws structure parameters from `generate_population_dataset_modes4.SEED` "
      "(paper: 20260507), split by `SeedSequence.spawn` into one stream per girder count. The excitation comes from a "
      "different seed: `simulate_uniform_inertial_u3_white_noise.SEED = 12345`, and `simulate_white_noise_from_modal_model` "
      "calls `np.random.default_rng(SEED)` afresh for **every** structure. So all 5500 paper structures (training, validation "
      "and test) were excited by the **same white-noise realisation** (per support DOF row; the row count varies with the "
      "number of supports). A test set drawn the paper's way therefore never tests a new excitation realisation.\n")
    w("Continuing the paper's streams would mean re-running all 5500 draws with their FE solves (the accept/reject decision "
      "needs the solve), so a new parameter seed was used. Two sets were generated with the paper's own in-memory patches "
      "(`s0_generate_population.install_patches`: Ng >= 3 spacing to 3000 mm, selected-mode MAC gate 0.85; broadband, all "
      "15 FE modes, 1.5-20 Hz band-pass), 125 structures per Ng:\n")
    w(md(pd.DataFrame([{"set": s, "parameter seed": gen[s]["param_seed"], "excitation seed": gen[s]["excitation_seed"],
                        "generation time (min)": f3(gen[s]["seconds"] / 60, 1),
                        "s / structure": f3(gen[s]["seconds"] / 500, 2)}
                       for s in ("fresh_test", "fresh_test_paperexc") if s in gen])))
    w(f"\n`fresh_test` is the EC-15 test set (new structures, new excitation). `fresh_test_paperexc` holds the **same 500 "
      f"structures** with the paper's excitation (12345): it separates the effect of new structures from that of a new noise "
      f"realisation. Both ran side by side with other jobs on the machine (2 BLAS threads each); the paper's 5500 took "
      f"{f3(gen['paper_broadband_5500']['seconds'] / 3600, 1)} h ({f3(gen['paper_broadband_5500']['per_structure_s'], 2)} s per structure).\n")
    w("**Checks** (`c_fresh_checks.py`, `fresh_checks.json`):\n")
    w(f"- counts: " + "; ".join(f"{s}: {v['n']} ({', '.join(f'Ng {g}: {n}' for g, n in sorted(v['per_Ng'].items()))})"
                                 for s, v in fchk["counts"].items()))
    if "fresh_sets_paired" in fchk:
        fp = fchk["fresh_sets_paired"]
        w(f"- the two fresh sets: identical params_json for all 500: {fp['same_params_json_all']}; target shapes differ by at most "
          f"{fp['max_abs_diff_targets']:.1e} and frequencies by {fp['max_abs_diff_freq_hz']:.1e} Hz (ARPACK start-vector noise); "
          f"node PSDs differ (median max |diff| {fp['node_psd_max_abs_diff_median']:.2f} on the row-normalised scale) - "
          f"only the excitation differs")
    for s in ("fresh_test", "fresh_test_paperexc"):
        if s in nov:
            w(f"- novelty ({s}): {nov[s]['identical_params_json_in_paper_population']} of 500 params_json identical to any of "
              f"the 5500 paper structures; nearest paper structure (same Ng, standardised parameters) at distance >= "
              f"{nov[s]['nn_dist_to_any_paper_structure_min']:.2f}; median distance to the training pool "
              f"{nov[s]['nn_dist_to_paper_pool_median']:.2f}, against {nov['paper_test_to_paper_pool']['nn_dist_median']:.2f} "
              f"for the paper's own test set - as far from the training data as the paper test set is")
    ks = fchk.get("ks_summary", {})
    if ks:
        w(f"- distribution against the paper's fixed test set: {ks['n_tests']} two-sample KS tests (14 parameters + 4 target "
          f"frequencies + node count, pooled and per Ng); {ks['n_p_below_0.05']} with p < 0.05 (~{0.05 * ks['n_tests']:.0f} "
          f"expected by chance); smallest p = {ks['min_p']:.3f} ({ks['min_p_quantity']}); Bonferroni at 0.05: "
          f"{'no rejection' if ks['bonferroni_0.05_pass'] else 'REJECTED'}. Table: `fresh_param_distribution.csv`.")
    rj = pd.read_csv(OUT / "fresh_rejections.csv")
    tab = rj.pivot_table(index="Ng", columns="set", values="per_100_accepted", aggfunc="sum").round(1).reset_index()
    w("- rejections logged by the generator (no feasible mode candidate, or the selected-mode MAC gate), per 100 accepted "
      "structures (`fresh_rejections.csv` by reason; parameter-validation rejections are not logged, as in the paper):\n")
    w(md(tab.fillna(0)))
    w("")

    # --- re-scoring
    w("## 2. Every paper network on the fresh set (EC-15)\n")
    w(f"**Gate.** `c_rescore.py` scores with `s4_indomain.evaluate`'s code path (CPU, 2 threads, library datasets and metrics, "
      f"best-validation checkpoint). On the paper's fixed test set it reproduces `s4_indomain_summary.csv` **exactly** "
      f"(every shared column, `==`) for {gate['n_runs']} of 14 paper runs: **{'PASS' if gate['pass'] else 'FAIL'}** "
      f"(`rescore_gate.json`). The contrasts below reproduce `s11_ablation_contrasts.csv` exactly on the fixed set: "
      f"**{'PASS' if s11_ok else 'FAIL'}**.\n")
    t = []
    for _, r in fvf.iterrows():
        t.append({"run": r.run, "MAC fixed": f3(r.mac_mean_fixed), "MAC fresh": f3(r.mac_mean_fresh),
                  "fresh - fixed [95 % CI]": pm(r.mac_mean_diff, r.mac_mean_ci95_lo, r.mac_mean_ci95_hi),
                  "T1 fresh - fixed": pm(r.mac_T1_diff, r.mac_T1_ci95_lo, r.mac_T1_ci95_hi),
                  "freq MAE fixed / fresh (Hz)": f"{r.freq_mae_hz_fixed:.3f} / {r.freq_mae_hz_fresh:.3f}",
                  **({"excitation effect on MAC (paired)": pm(r.mac_mean_excitation_effect, r.mac_mean_excitation_effect_ci95_lo,
                                                               r.mac_mean_excitation_effect_ci95_hi)}
                     if "mac_mean_excitation_effect" in r and not pd.isna(r.get("mac_mean_excitation_effect")) else {})})
    w("Mean MAC over the four modes. *fresh - fixed*: the two test sets are independent, so each is bootstrapped within "
      f"girder count ({N_BOOT} resamples) and the interval is that of the difference of means. *Excitation effect*: "
      "`fresh_test` minus `fresh_test_paperexc` - the same structures, new vs the paper's noise realisation - paired by "
      "structure.\n")
    w(md(pd.DataFrame(t)))
    w("")
    mm = fvf[fvf.run == MAIN]
    if len(mm):
        r = mm.iloc[0]
        w(f"**Main network (N = {C.MAIN_N}), per mode, fresh - fixed:** " + "; ".join(
            f"{m} {r[f'mac_{m}_fixed']:.3f} -> {r[f'mac_{m}_fresh']:.3f} ({pm(r[f'mac_{m}_diff'], r[f'mac_{m}_ci95_lo'], r[f'mac_{m}_ci95_hi'])})"
            for m in M) + f"; frequency MAE {r.freq_mae_hz_fixed:.3f} -> {r.freq_mae_hz_fresh:.3f} Hz "
          f"({pm(r.freq_mae_hz_diff, r.freq_mae_hz_ci95_lo, r.freq_mae_hz_ci95_hi)}). Full table: `fresh_vs_fixed.csv`.\n")
    if (OUT / "excitation_diag.csv").exists():
        dg = pd.read_csv(OUT / "excitation_diag.csv")
        w("### Why: the networks depend on the one excitation realisation they were trained on\n")
        w("`c_excitation_diag.py` scores the same 500 fresh structures with inputs assembled from the two paired sets "
          "(*new*: excitation 20261124, *paper*: 12345). *support rows*: the support nodes' PSD rows - the records of the "
          "injected excitation itself, identical in every paper structure - taken from the other set. Mean MAC "
          "(frequency MAE in Hz):\n")
        lab = {"new": "all inputs new", "paper": "all inputs paper exc.", "nodes_new": "new node PSDs, paper edges",
               "edges_new": "paper node PSDs, new edges", "supports_new": "paper inputs, new support rows",
               "deck_new": "new inputs, paper support rows"}
        tb = []
        for v, l in lab.items():
            r = {"inputs": l}
            for run in dg.run.unique():
                q = dg[(dg.run == run) & (dg.variant == v)]
                if len(q):
                    r[run] = f"{q.mac_mean.iloc[0]:.3f} ({q.freq_mae_hz.iloc[0]:.2f})"
            tb.append(r)
        w(md(pd.DataFrame(tb)))
        w("")
    w("### Population size: the smallest N with every mode >= 0.9\n")
    st = sw.pivot_table(index="N", columns="test_set", values="min_mode_mac").reset_index()
    st2 = sw.pivot_table(index="N", columns="test_set", values="mac_T1").reset_index()
    tt = pd.DataFrame({"N": st.N})
    for ts in sets_avail:
        if ts in st:
            tt[f"lowest mode MAC, {SET_LABEL[ts]}"] = st[ts].map(f3)
            tt[f"T1, {SET_LABEL[ts]}"] = st2[ts].map(f3)
    w(md(tt))
    w("\nSmallest N with every mode >= 0.9: " + "; ".join((f"{SET_LABEL[ts]}: **N = {smallest[ts]}**" if smallest[ts] else f"{SET_LABEL[ts]}: **none in the sweep**") for ts in sets_avail)
      + ". (On every set the lowest mode is T1.) The paper chose N = 2000 by this rule on the fixed set.\n")
    w("### Ablation grid (N = 600): ranking and contrasts\n")
    rk = ab.pivot_table(index="arm", columns="test_set", values="mac_mean").reset_index()
    rr = ab.pivot_table(index="arm", columns="test_set", values="rank").reset_index()
    tt = pd.DataFrame({"arm": rk.arm})
    for ts in sets_avail:
        tt[f"MAC {SET_LABEL[ts]}"] = rk[ts].map(f3)
        tt[f"rank {SET_LABEL[ts]}"] = rr[ts].astype(int)
    tt = tt.sort_values(f"rank {SET_LABEL['paper']}")
    w(md(tt))
    if tau is not None:
        w(f"\nKendall tau between the fixed-set and fresh-set orderings of the 7 arms: {tau:.3f}.\n")
    w("Paired contrasts (per structure, bootstrap 95 %, as `s11_ablation_cost.py`), mean MAC and T1:\n")
    ct = []
    for name in con.contrast.unique():
        r = {"contrast": name}
        for ts in ("paper", "fresh_test"):
            z = con[(con.contrast == name) & (con.test_set == ts)]
            if len(z):
                z = z.iloc[0]
                r[f"mean MAC, {SET_LABEL[ts]}"] = pm(z.mac_mean_diff, z.mac_mean_ci95_lo, z.mac_mean_ci95_hi)
                r[f"T1, {SET_LABEL[ts]}"] = pm(z.mac_T1_diff, z.mac_T1_ci95_lo, z.mac_T1_ci95_hi)
        ct.append(r)
    w(md(pd.DataFrame(ct)))
    w("")

    # --- EC-23
    w("## 3. Shape error, success rates and cross-MAC (EC-23)\n")
    w("Over every graph node (where the paper's MAC is computed). *RMSE*: the prediction scaled by its max |value| (the "
      "targets' own normalisation: max |y| = 1 per mode), its global sign set by its dot product with the target; "
      "0 is a perfect shape, sqrt(2) x RMS(target) an orthogonal one. *success*: share of structures with MAC above the "
      "threshold. *swap*: the predicted slot matches another mode's target better than its own.\n")
    names = {MAIN: f"main (EdgeGAT, N = {C.MAIN_N})"}
    names.update({v: f"{l} / {f} (N = 600)" for (l, f), v in GRID.items()})
    for ts in ("paper", "fresh_test"):
        z = e23[(e23.test_set == ts) & (e23.Ng == "all")]
        if z.empty:
            continue
        w(f"**{SET_LABEL[ts]} test set**\n")
        tb = []
        for run in names:
            q = z[z.run == run].set_index("mode")
            if q.empty:
                continue
            tb.append({"network": names[run],
                       **{f"RMSE {m}": f3(q.loc[m, "rmse_signed_mean"]) for m in M},
                       **{f">0.9 {m}": f3(q.loc[m, "success_mac_gt_0.9"], 2) for m in M},
                       **{f">0.8 {m}": f3(q.loc[m, "success_mac_gt_0.8"], 2) for m in M},
                       "swap B1/T1/B2/T2": "/".join(f3(q.loc[m, "swap_rate"], 3) for m in M)})
        w(md(pd.DataFrame(tb)))
        w("")
    zz = e23[(e23.run == MAIN) & (e23.Ng != "all")]
    if len(zz):
        w(f"Main network by girder count (RMSE / success MAC > 0.9):\n")
        tb = []
        for ts in ("paper", "fresh_test"):
            for g in C.NG_VALUES:
                q = zz[(zz.test_set == ts) & (zz.Ng.astype(str) == str(g))].set_index("mode")
                if len(q):
                    tb.append({"set": SET_LABEL[ts], "Ng": g, **{m: f"{q.loc[m, 'rmse_signed_mean']:.3f} / {q.loc[m, 'success_mac_gt_0.9']:.2f}" for m in M}})
        w(md(pd.DataFrame(tb)))
        w("")
    for ts in ("paper", "fresh_test"):
        q = cross[(cross.test_set == ts) & (cross.run == MAIN)]
        if len(q):
            w(f"Mean cross-MAC, main network, {SET_LABEL[ts]} set (rows: predicted slot, columns: target mode):\n")
            w(md(pd.DataFrame([{"slot": r.pred_slot, **{m: f3(r[f"target_{m}"]) for m in M}} for _, r in q.iterrows()])))
            w("")
    w("All seven networks' matrices: `ec23_crossmac.csv`; per structure: `ec23_per_sample.csv`.\n")
    w("**How SSI shapes enter the real-bridge scoring (s5, s7).** They are real-valued throughout. The SSI files "
      "(`field_measurements_raw/*_Data_picked.mat` and the curated `SSI_modal_infor/*_Data_picked.mat`) store `phi_sorted` "
      "as float64 - no imaginary part reaches this repository; whatever made the identified modes real (projection or "
      "taking the real part) happened upstream in the MATLAB SSI code (`COMAC_PBSHM/COMACPBSHM_Rev01.m` and the "
      "\"For GNN\" export), not here. `_load_ssi_data` keeps the u3 rows; `build_phase3_real_dataset.py` stores them as "
      "`y_mode_shape_ssi` (float32), sign-aligned to the FE target. s5 scores with the real MAC "
      "`(u.v)^2/((u.u)(v.v))` at the sensor nodes (sign-invariant); s7 flips the prediction's sign to the SSI shape before "
      "its validation MAC and fine-tunes on an MSE against these real shapes. No Hermitian MAC is used, because there is "
      "nothing complex left to apply it to. (E19's FDD baseline below reads complex operating deflection shapes and scores "
      "them with the Hermitian MAC, `E17.mac_c`.) The phase handling of the original SSI shapes should be stated from the "
      "MATLAB code, which is outside this repository.\n")

    # --- EC-25
    w("## 4. Baselines (EC-25)\n")
    w("### (a) Frequency without any record\n")
    w(f"Fitted on the main network's training split (1700 decks), never on validation or test. *geom_knn*: mean of the 10 "
      f"nearest training decks of the same girder count in standardised sampled parameters (geometry, springs, masses, "
      f"E and rho); *noEr*: without E and rho. MAE in Hz:\n")
    tb = []
    for ts in ("paper", "fresh_test"):
        for meth in ("mean", "ng_mean", "geom_knn", "geom_knn_noEr", "network_N2000"):
            q = fb[(fb.test_set == ts) & (fb.method == meth) & (fb.Ng.astype(str) == "all")]
            if len(q):
                q = q.iloc[0]
                tb.append({"set": SET_LABEL[ts], "method": meth, **{m: f3(q[f"mae_{m}_hz"]) for m in M},
                           "mean [95 % CI]": f"{q.mae_hz:.3f} [{q.mae_hz_ci95_lo:.3f}, {q.mae_hz_ci95_hi:.3f}]",
                           "mean rel. error %": f3(q.rel_err_pct_mean, 1)})
    w(md(pd.DataFrame(tb)))
    w("\nPer girder count: `freq_baselines.csv`; per structure: `freq_baselines_per_sample.csv`.\n")
    w("### (b, c) Shapes: template, FDD on the same records, and the network\n")
    w("FDD runs on the **same simulated records** the network's features come from: every deck is re-simulated from its "
      "params_json with the set's own excitation seed, and the re-simulated node PSDs equal the stored network inputs "
      f"(max |diff| " + ", ".join(f"{k} {v['x_psd_maxdiff']:.1e}" for k, v in fchk_fdd["reproduction"].items())
      + "). The records are noise-free apart from the excitation (the generator adds no measurement noise), so the noise "
      "level is matched by construction. FDD and labelling are E19's (`ident.py`, imported): peaks of the first "
      "singular value of the kept nodes' cross-spectral matrix, labelled bend1/tors1/bend2/tors2; *population* uses E18's "
      "class shape prior and a geometry-conditioned frequency prior from the training split (rank: phase and order only, "
      "no population information). Each shape is the deflection at the picked peak, scored with the Hermitian MAC on the "
      "deck nodes (supports excluded: their records are the injected excitation). Settings were chosen on the 300 "
      "validation decks (outer layouts: E19's own choice; every-node layout: "
      + ", ".join(f"{k.split('/')[1]} theta {v['theta']}" + (f", lam {v['lam']}" if v['lam'] > 0 else "")
                  for k, v in fchk_fdd["chosen"].items() if k.startswith("all/"))
      + "). The class template uses no record: the leading shape of E18's class prior for the deck's girder count, "
        "with the class read from the true shape.\n")
    g19 = fchk_fdd.get("paper_outer_reproduces_E19_test_csv", {})
    if g19:
        w(f"**Gates.** On the paper's test set the two- and four-sensor rows reproduce E19's `test.csv` "
          f"({g19['rows_matched']} of {g19['rows_E19']} rows, max |MAC diff| {g19['max_abs_mac_diff']:.1e}, max |f diff| "
          f"{g19['max_abs_fhat_diff']:.1e} Hz, {g19['correct_disagreements']} disagreements on correctness): "
          f"**{'PASS' if g19['pass'] else 'FAIL'}**. Template mean MAC on the paper test set "
          f"{fchk_fdd['template_paper_mean_mac']:.4f} = E19's 0.946: **{'PASS' if fchk_fdd['template_paper_equals_E19_0946'] else 'FAIL'}**.\n")
    for ts in ("paper", "fresh_test"):
        q = shp[shp.test_set == ts]
        if q.empty:
            continue
        w(f"**{SET_LABEL[ts]} test set** - mean MAC per mode; *fail*: share of mode slots with MAC < 0.5, or (FDD) not "
          f"identified or MAC < 0.5; *f err*: mean |f - f_true| (Hz) including wrong picks:\n")
        rows_ = []
        for _, r in q.iterrows():
            rows_.append({"method": r.method, **{m: f3(r[m]) for m in M}, "mean": f3(r["mean"]),
                          "identified": f3(r.get("identified"), 3), "fail": f3(r.get("fail_rate (not identified or MAC<0.5)", np.nan), 3)
                          if not pd.isna(r.get("fail_rate (not identified or MAC<0.5)", np.nan)) else f3(r["fail_rate (MAC<0.5)"], 3),
                          "f err (Hz)": f3(r.get("ferr_mean_hz"), 3)})
        net = fb[(fb.test_set == ts) & (fb.method == "network_N2000") & (fb.Ng.astype(str) == "all")]
        if len(net):
            rows_[0]["f err (Hz)"] = f3(net.iloc[0].mae_hz, 3)
        w(md(pd.DataFrame(rows_)))
        w("")
    w("Per mode (identification rate, failure rate, frequency error): `fdd_summary_by_mode.csv`; every setting of every "
      "deck: `../../data/T5/fdd/<set>_grid.csv.gz` (not tracked).\n")
    w("### Runtime per structure\n")
    tg = tim[tim.split.isin(["paper", "fresh_test"])]
    fwd = summ[(summ.run == MAIN)].set_index("test_set").forward_ms_median
    w(md(pd.DataFrame([{"step": "network input features from the records (node Welch PSD + 800-d edge coherency; E14's functions)",
                        "median s": f3(tg.net_features_s.median(), 3), "90th pct s": f3(tg.net_features_s.quantile(0.9), 3)},
                       {"step": f"network forward pass (main network, CPU, 2 threads; c_rescore)",
                        "median s": f3(fwd.median() / 1e3, 4), "90th pct s": ""},
                       {"step": "FDD on every deck node + population labelling + shape read (one setting)",
                        "median s": f3(tg.fdd_all_s.median(), 3), "90th pct s": f3(tg.fdd_all_s.quantile(0.9), 3)},
                       {"step": "(re-simulation: FE solve + response, needed by both, not counted)",
                        "median s": f3(tg.sim_s.median(), 2), "90th pct s": f3(tg.sim_s.quantile(0.9), 2)}])))
    w("\nWall-clock on a shared machine (up to four of these jobs and other agents' runs side by side); compare orders of "
      "magnitude, not digits. `fdd_timing.csv` has every deck.\n")
    w("### (d) \"The previous PSD-based model\"\n")
    w("Not retrained. The ablation arm **GraphSAGE / no edge features** (`graphsage_noedge_signed_N600`: node Welch PSDs "
      "only, the edge path fed zeros) is that model on the paper's population and protocol; its rows are in sections 2 and 3 "
      "on both test sets.\n")

    # --- caveats
    w("## Caveats\n")
    w("- One training seed (123) per network, as in the paper; intervals are over test structures, not over training runs.")
    w("- Only one new excitation seed (20261124) was tested. The paired design (same structures, only the seed changed) and "
      "the input swap make a bug unlikely, but a second new seed would confirm the size of the drop.")
    w("- One of 95 distribution tests rejects after Bonferroni: balcony length on Ng = 3 decks (64 balconies per set). The "
      "generator code is identical; against the paper's training pool the fresh set is shorter (p = 0.0006) and the paper's "
      "own test set longer (p = 0.017), so this is sampling, in both directions.")
    w("- The fresh set is drawn from the paper's generator with the paper's patches, so it tests generalisation within the "
      "population distribution, not to other bridges.")
    w("- `fresh_test` changes two things at once: new structures and a new excitation realisation. `fresh_test_paperexc` "
      "isolates the second (paired, same structures).")
    w("- The template and FDD *given* rows are told each target's class from its true shape; the network infers which output "
      "slot is which mode. The template and FDD are scored on deck nodes, the network's paper MAC on every graph node; the "
      "table shows the network both ways.")
    w("- FDD's *population* labeller uses population priors (E18's class shapes, a frequency prior from the training split); "
      "the *rank* labeller uses no population information. The \"identified\" criterion: the FE mode nearest the picked peak "
      "is the target mode (E19's).")
    w("- No measurement noise is added anywhere (as in the paper's population); FDD on noise-free records is an optimistic "
      "conventional baseline. SSI was not run on the synthetic records.")
    w("- The Ng = 2 balcony decks carry the matcher's B2/T2 label swap (568 of the paper's 5500); FDD and the template "
      "score each slot against its own target shape, with the class read from that shape, so they are unaffected; the "
      "network was trained with the swapped labels.\n")

    # --- how to run
    w("## What ran\n")
    w("```bash")
    w("python tools/run_stage.py pipeline/c_gen_fresh_test.py --set fresh_test")
    w("python tools/run_stage.py pipeline/c_gen_fresh_test.py --set fresh_test_paperexc")
    w("python tools/run_stage.py pipeline/c_fresh_checks.py")
    w("python tools/run_stage.py pipeline/c_rescore.py --sets paper fresh_test fresh_test_paperexc --workers 3")
    w("python tools/run_stage.py pipeline/c_ec23_metrics.py")
    w("python tools/run_stage.py pipeline/c_freq_baselines.py")
    w("python tools/run_stage.py pipeline/c_fdd.py --sets val paper --workers 2")
    w("python tools/run_stage.py pipeline/c_fdd.py --sets fresh_test --workers 3")
    w("python tools/run_stage.py pipeline/c_excitation_diag.py")
    w("python tools/run_stage.py pipeline/c_report.py")
    w("```\n")
    w("**Re-scoring networks trained later** (e.g. into `runs/ec/`): any folder with `run_summary.json` and "
      "`checkpoints/best_model.pt` is accepted, by name or path; new edge modes come from `t_edge_modes.py`:\n")
    w("```bash")
    w("python tools/run_stage.py pipeline/c_rescore.py --runs <name-or-path> ... --sets paper fresh_test")
    w("```\n")
    w("Rows accumulate in `rescore_per_sample.csv` / `rescore_summary.csv` (a re-scored (set, run, checkpoint) replaces "
      "its old rows); add `--runs` to `c_ec23_metrics.py` for the EC-23 metrics of a new run, then re-run `c_report.py`.\n")
    led = {}
    for line in (C.REPO / "results" / "RUN_LEDGER.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        st = str(e.get("stage", "")).replace("\\", "/")
        if "/pipeline/c_" in st and e.get("exit_code") == 0 and "--per-ng" not in e.get("args", []):
            led[(st.rsplit("/", 1)[-1], " ".join(e.get("args", [])))] = e["seconds"]
    if led:
        w("**Compute time** (wall clock of the last successful run of each stage and arguments, from "
          "`results/RUN_LEDGER.jsonl`; shared machine):\n")
        w(md(pd.DataFrame([{"stage": k[0], "args": k[1], "minutes": f3(v / 60, 1)} for k, v in led.items()])))
        w("")
    (OUT / "README.md").write_text("\n".join(L), encoding="utf-8")
    print((OUT / "README.md").read_text(encoding="utf-8")[:3000])
    print(f"[check] contrasts on the fixed set == s11_ablation_contrasts.csv: {'PASS' if s11_ok else 'FAIL'}")
    return 0 if (gate["pass"] and s11_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

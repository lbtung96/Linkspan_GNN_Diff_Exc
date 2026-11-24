"""Re-run every stage behind the results carried over from the archive, and report what reproduced.

    python tools/reproduce.py --list                     # the plan
    python tools/reproduce.py [--tasks T5 T10 ...] [--from N] [--dry-run]
    python tools/reproduce.py --report                   # results/reproduction/RESULTS_REPRODUCED.md

Each step is run through tools/run_stage.py (so it is logged and in results/RUN_LEDGER.jsonl), in the order the
archive ran it (its ledger), with the same arguments, and writes into results/<task>/ - the folders carried over
from the archive and committed unchanged in migration batch 1.  So after a run:

    rewritten and unchanged in git   reproduced (bit-identical file)
    rewritten and changed in git     a difference: read `git diff results/<task>` (the report lists them)
    not rewritten                    not produced by any step here (listed: archive-only, or produced by a
                                     data / training stage that is verified by tools/verify_*.py instead)

Data-writing stages (h_/k_/q_/v_prepare --part data / --run, c_gen_fresh_test) are not re-run here: their
outputs are hard-linked inputs.  tools/verify_records.py re-simulates every own-record sample in memory, and
tools/verify_fresh_test.py regenerates a prefix of T5's fresh test sets.  Training is verified by
tools/verify_training.py.  The trainer *gates* below (h_train/k_train --launch gate, p_train/q_train --check)
retrain a few epochs through this repository's trainer and compare with the archive's histories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

# (task, script, args) - the archive ledger's order (2026-11-23 .. 24); smoke runs and failed attempts dropped
PLAN: list[tuple[str, str, list[str]]] = [
    ("T5", "c_fresh_checks.py", []),
    ("T5", "c_rescore.py", ["--sets", "paper", "--workers", "3"]),
    ("T5", "c_ec23_metrics.py", ["--sets", "paper"]),
    ("T5", "c_fdd.py", ["--sets", "val", "paper", "--workers", "2"]),
    ("T10", "g_t10_channel_order.py", []),
    ("T10", "g_t10_swapped_inputs.py", []),
    ("T5", "c_rescore.py", ["--sets", "fresh_test", "fresh_test_paperexc", "--workers", "3"]),
    ("T5", "c_ec23_metrics.py", ["--sets", "paper", "fresh_test", "fresh_test_paperexc"]),
    ("T5", "c_freq_baselines.py", []),
    ("T5", "c_fdd.py", ["--sets", "fresh_test", "--workers", "3"]),
    ("T5", "c_excitation_diag.py", []),
    ("T5", "c_report.py", []),
    ("T11", "h_prepare.py", ["--part", "gate", "--workers", "10"]),
    ("T11", "h_train.py", ["--launch", "gate"]),
    ("T11", "h_eval.py", ["--runs", "edgegat_N600", "edgegat_geomorient_N600", "edgegat_N2000",
                          "--sets", "paper", "fresh_test", "fresh_test_paperexc", "--workers", "4"]),
    ("T11", "h_real.py", ["--part", "zeroshot", "--nets", "edgegat_N600", "edgegat_geomorient_N600"]),
    ("T11", "h_real.py", ["--part", "ceiling", "--workers", "9", "--nets", "edgegat_N600", "edgegat_geomorient_N600",
                          "edgegat_N2000"]),
    ("T11", "h_eval.py", ["--runs", "edgegat_N600", "edgegat_geomorient_N600", "edgegat_N2000",
                          "--sets", "paper_varexc", "--workers", "3"]),
    ("T11", "h_eval.py", ["--runs", "ec_varexc_edgegat_N600", "ec_varexc_edgegat_geomorient_N600", "--workers", "4"]),
    ("T11", "h_real.py", ["--part", "zeroshot"]),
    ("T11", "h_real.py", ["--part", "ceiling", "--workers", "9"]),
    ("T11", "h_report.py", []),
    ("T12", "x_plot_excitation.py", []),
    ("T13", "k_prepare.py", ["--part", "gate", "--workers", "10"]),
    ("T13", "k_prepare.py", ["--part", "crosscheck", "--workers", "8"]),
    ("T13", "k_train.py", ["--launch", "gate"]),
    ("T13", "k_eval.py", ["--runs", "ec_varexc_edgegat_N600", "ec_varexc_edgegat_geomorient_N600", "--workers", "3"]),
    ("T13", "k_eval.py", ["--workers", "7"]),
    ("T13", "k_real.py", []),
    ("T13", "k_diag.py", []),
    ("T13", "k_report.py", []),
    ("T15", "n_mirror_test.py", ["--per-ng", "25", "--workers", "6"]),
    ("T14", "m_input_anatomy.py", ["--workers", "6"]),
    ("T14", "m_occlusion.py", []),
    ("T16", "p_train.py", ["--check"]),
    ("T16", "p_eval.py", ["--workers", "8"]),
    ("T17", "q_train.py", ["--check"]),
    ("T18", "r_excitation_realism.py", ["--workers", "9"]),
    ("T17", "q_eval.py", ["--workers", "8"]),
    ("T17", "q_template_compare.py", ["--workers", "8"]),
    ("T19", "u_decomposition.py", ["--workers", "4"]),
    ("T20", "v_fit_noise.py", []),
    ("T20", "v_prepare.py", ["--check"]),
    ("T20", "v_eval.py", ["--workers", "8"]),
]
TASK_DIR = {"T5": "T5_fresh_test_baselines", "T10": "T10_channel_order", "T11": "T11_varied_excitation_pilot",
            "T12": "T12_excitation_plots", "T13": "T13_feasibility", "T14": "T14_input_anatomy",
            "T15": "T15_mirror_test", "T16": "T16_sign_test", "T17": "T17_scale_up",
            "T18": "T18_excitation_realism", "T19": "T19_decomposition", "T20": "T20_deck_forcing"}
STATE = REPO / "results" / "reproduction" / "reproduce_state.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8").stdout


def report() -> int:
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"steps": []}
    t_start = min((s["t0"] for s in st["steps"]), default=time.time())
    changed = {l[3:].strip().strip('"') for l in git("status", "--porcelain", "--", "results").splitlines()}
    lines = ["# Reproduction of the carried-over results", "",
             "Generated by `tools/reproduce.py --report`.  A file counts as *rewritten* when its modification time",
             "is later than the first step's start; *identical* when git reports no change against the archive",
             "copy committed in migration batch 1.", "",
             "| step | task | stage | exit | minutes |", "|---|---|---|---|---|"]
    for i, s in enumerate(st["steps"]):
        lines.append(f"| {i} | {s['task']} | `{s['script']} {' '.join(s['args'])}` | {s['rc']} | {s['minutes']:.1f} |")
    lines += ["", "| task | files | rewritten, identical | rewritten, DIFFERENT | not rewritten |", "|---|---|---|---|---|"]
    detail = []
    for t, d in TASK_DIR.items():
        files = [p for p in (REPO / "results" / d).rglob("*") if p.is_file()]
        rel = {p: p.relative_to(REPO).as_posix() for p in files}
        rew = [p for p in files if p.stat().st_mtime >= t_start]
        diff = [p for p in rew if rel[p] in changed]
        same = [p for p in rew if rel[p] not in changed]
        notr = [p for p in files if p not in rew]
        lines.append(f"| {t} | {len(files)} | {len(same)} | {len(diff)} | {len(notr)} |")
        if diff or notr:
            detail.append(f"### {t}")
            detail += [f"- DIFFERENT: `{rel[p]}`" for p in diff]
            detail += [f"- not rewritten: `{rel[p]}`" for p in notr]
            detail.append("")
    out = REPO / "results" / "reproduction" / "RESULTS_REPRODUCED.md"
    out.write_text("\n".join(lines + ["", "## Files that differ or were not rewritten", ""] + detail) + "\n",
                   encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--from", dest="start", type=int, default=0)
    a = ap.parse_args()
    steps = [(i, t, s, args) for i, (t, s, args) in enumerate(PLAN) if i >= a.start and (not a.tasks or t in a.tasks)]
    if a.list or a.dry_run:
        for i, t, s, args in steps:
            print(f"{i:3d}  {t:<4} pipeline/{s} {' '.join(args)}")
        return 0
    if a.report:
        return report()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"steps": []}
    for i, t, s, args in steps:
        t0 = time.time()
        print(f"\n=== step {i}: {t} pipeline/{s} {' '.join(args)}", flush=True)
        rc = subprocess.run([PY, str(REPO / "tools" / "run_stage.py"), f"pipeline/{s}", *args], cwd=str(REPO)).returncode
        st["steps"].append({"i": i, "task": t, "script": s, "args": args, "rc": rc, "t0": t0,
                            "minutes": (time.time() - t0) / 60})
        STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
        if rc:
            print(f"step {i} failed (exit {rc}); stopping.  Resume with --from {i}", flush=True)
            return rc
    return report()


if __name__ == "__main__":
    sys.exit(main())

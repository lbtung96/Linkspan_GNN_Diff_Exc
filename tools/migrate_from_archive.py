"""One-time migration: build this repository from the frozen archive `../Linkspan_GNN_OMA` (2026-11-24).

    python tools/migrate_from_archive.py --dry-run      # list what would be copied / linked, with sizes
    python tools/migrate_from_archive.py                # do it (refuses to overwrite anything that exists)

Code, small data, networks and the reference results are COPIED.  Large inputs that no stage ever rewrites
(the 5500-structure population, the own-record sets, field inputs) are HARD-LINKED: same NTFS volume, zero
extra disk.  Every linked file is then marked read-only.  The read-only attribute belongs to the file, not
to the link, so the archive's copy is protected too: a stage that tried to rewrite a linked file in place
would fail loudly instead of silently changing the archive.

Archive snapshot: commit f4e1ad5 of ../Linkspan_GNN_OMA ("Snapshot before freezing").  What is kept and why
is RESTRUCTURE_MANIFEST.md; what each path became is written to docs/migration_log.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path

NEW = Path(__file__).resolve().parent.parent
OLD = NEW.parent / "Linkspan_GNN_OMA"
FR = OLD / "Final_results_for_paper"
EC = FR / "EC_comments"

# ---- code -----------------------------------------------------------------------------------------------
LIBRARY = [  # repository root, flat (CLAUDE.md rule 3 of the archive)
    "grid_modal_Kinematic_MPC.py", "standalone_mode_matching_mmi.py", "export_projected_graph.py",
    "simulate_uniform_inertial_u3_white_noise.py", "generate_population_dataset_modes4.py",
    "GraphSAGE_Modal_GNN.py", "EdgeGATConv_Modal_GNN.py", "phase3_utils.py",
    "infer_phase1_unseen_structure.py", "infer_phase2a_unseen_structure.py",
    "infer_phase2a_unseen_structure_edgegat.py", "infer_phase2a_deterministic_edgegat.py",
    "build_phase3_graphs.py", "analyze_phase1_run.py",
    "structures.py", "process_field_modal_filtering.py",
]
PAPER_STAGES = ["fr_common.py", "fr_train.py", "fr_edge_modes.py", "s0_generate_population.py",
                "s4_indomain.py", "s5_zeroshot.py", "s5a_field_inputs.py", "s6_decomposition.py"]
EC_PREFIXES = ("c_", "g_t10_", "h_", "k_", "m_", "n_", "p_", "q_", "r_", "u_", "v_", "x_")
EC_EXTRA = ["ec_common.py", "t_gates.py"]
EXPERIMENTS = ["E5_real_transfer", "E14_corner_sensors", "E16_fp_imputers", "E17_real_holdout",
               "E18_population_basis", "E19_mode_identification"]
EXP_SUFFIXES = (".py", ".csv", ".json", ".npz")          # top level only: modules + the small tables they read

# ---- results kept as the reference the re-runs are compared with ---------------------------------------
EC_RESULTS = ["T5_fresh_test_baselines", "T10_channel_order", "T11_varied_excitation_pilot", "T12_excitation_plots",
              "T13_feasibility", "T14_input_anatomy", "T15_mirror_test", "T16_sign_test", "T17_scale_up",
              "T18_excitation_realism", "T19_decomposition", "T20_deck_forcing"]

# ---- data -------------------------------------------------------------------------------------------------
LINK_DIRS = {  # new path -> old path; hard-linked, then read-only
    "data/population/samples_phase2_bandpass": FR / "data/pop_broadband/samples_phase2_bandpass",
    "data/field_inputs_v2": FR / "data/field_inputs_v2",
    "data/legacy_preP/field_modal_filtering_outputs": OLD / "field_modal_filtering_outputs",
    "data/legacy_preP/E5_phase3_corrected": OLD / "experiments/E5_real_transfer/phase3_corrected",
    "experiments/E19_mode_identification/cache": OLD / "experiments/E19_mode_identification/cache",
    **{f"data/T5/{s}": EC / f"data/T5/{s}" for s in ("fresh_test", "fresh_test_paperexc")},
    **{f"data/T11/{s}": EC / f"data/T11/{s}" for s in ("paper_varexc", "trainval_varexc")},
    **{f"data/T13/{s}": EC / f"data/T13/{s}" for s in ("trainval_N2000", "rec1", "rec2", "rec3", "trainval_T120",
                                                      "paper_varexc_T120")},
    **{f"data/T17/rec{k}_N2000": EC / f"data/T17/rec{k}_N2000" for k in (1, 2, 3)},
    **{f"data/T20/{e}_{p}": EC / f"data/T20/{e}_{p}" for e in ("deck", "walk")
       for p in ("p0", "p1", "p2", "p3", "test")},
    "data/T20/checks": EC / "data/T20/checks",
}
COPY_DIRS = {  # new path -> old path; small, copied
    "data/splits": FR / "data/splits",
    "phase3_real_graphs": OLD / "phase3_real_graphs",
    "SSI_modal_infor": OLD / "SSI_modal_infor",
    "field_measurements_raw": OLD / "field_measurements_raw",
    "experiments/E6_controls/task2_raw": OLD / "experiments/E6_controls/task2_raw",
    "results/paper": FR / "pipeline/results",
}
POP_ROOT_FILES = ["generation_config.json", "summary.csv", "filtered_summary.csv", "outlier_report.json",
                  "failures.log"]
EC_RUNS_KEEP_PREFIXES = ("ec_t13_coh", "ec_t13_orient", "ec_t16_", "ec_t17_", "ec_varexc_edgegat")
LATE_RUNS_PREFIX = "ec_t20_"  # still training at migration time: copied by a second batch (--late)
SKIP_NAMES = {"__pycache__"}


def files_under(d: Path):
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x not in SKIP_NAMES]
        for f in files:
            yield Path(root) / f


class Plan:
    def __init__(self) -> None:
        self.ops: list[tuple[str, Path, Path]] = []  # (copy|link, src, dst)

    def copy(self, src: Path, dst: str | Path) -> None:
        self.ops.append(("copy", src, NEW / dst))

    def copy_tree(self, src: Path, dst: str) -> None:
        for f in files_under(src):
            self.copy(f, Path(dst) / f.relative_to(src))

    def link_tree(self, src: Path, dst: str) -> None:
        for f in files_under(src):
            self.ops.append(("link", f, NEW / dst / f.relative_to(src)))


def build_plan(late: bool = False) -> Plan:
    P = Plan()
    if late:
        for run in sorted((EC / "runs").iterdir()):
            if run.is_dir() and run.name.startswith(LATE_RUNS_PREFIX):
                P.copy_tree(run, f"runs/ec/{run.name}")
        return P
    for f in LIBRARY:
        P.copy(OLD / f, f)
    for f in sorted(OLD.glob("ref_modes_*.npz")):
        P.copy(f, f.name)
    for f in PAPER_STAGES:
        P.copy(FR / "pipeline" / f, Path("pipeline") / f)
    for f in sorted((EC / "pipeline").glob("*.py")):
        if f.name.startswith(EC_PREFIXES) or f.name in EC_EXTRA:
            P.copy(f, Path("pipeline") / f.name)
    for e in EXPERIMENTS:
        for f in sorted((OLD / "experiments" / e).iterdir()):
            if f.is_file() and f.suffix in EXP_SUFFIXES:
                P.copy(f, Path("experiments") / e / f.name)
    P.copy(OLD / "tools" / "run_stage.py", "tools/run_stage.py")
    if (OLD / "tools" / "INTERPRETER").exists():
        P.copy(OLD / "tools" / "INTERPRETER", "tools/INTERPRETER")
    for r in EC_RESULTS:
        src = EC / "pipeline" / "results" / r
        if src.exists():
            P.copy_tree(src, f"results/{r}")
    P.copy(OLD / "results" / "RUN_LEDGER.jsonl", "results/RUN_LEDGER.jsonl")
    P.copy(EC / "README.md", "docs/history/EC_comments_README.md")
    P.copy(OLD / "CLAUDE.md", "docs/history/CLAUDE_archive.md")
    for d, src in COPY_DIRS.items():
        P.copy_tree(src, d)
    for f in POP_ROOT_FILES:
        if (FR / "data/pop_broadband" / f).exists():
            P.copy(FR / "data/pop_broadband" / f, Path("data/population") / f)
    for run in sorted((FR / "runs").iterdir()):
        if run.is_dir():
            P.copy_tree(run, f"runs/paper/{run.name}")
    for run in sorted((EC / "runs").iterdir()):
        if run.is_dir() and run.name.startswith(EC_RUNS_KEEP_PREFIXES) and not run.name.endswith("_gates"):
            P.copy_tree(run, f"runs/ec/{run.name}")
    for d, src in LINK_DIRS.items():
        P.link_tree(src, d)
    return P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="restrict to destination prefixes (e.g. runs/paper/)")
    ap.add_argument("--late", action="store_true", help="second batch: the T20 networks, once trained")
    a = ap.parse_args()
    P = build_plan(late=a.late)
    ops = [o for o in P.ops if not a.only or any(o[2].relative_to(NEW).as_posix().startswith(x) for x in a.only)]
    missing = [o for o in ops if not o[1].exists()]
    if missing:
        for o in missing[:20]:
            print("MISSING", o[1])
        return 1
    tally: dict[str, dict[str, float]] = {}
    for kind, src, dst in ops:
        top = "/".join(dst.relative_to(NEW).parts[:2])
        t = tally.setdefault(f"{kind:4s} {top}", {"files": 0, "MB": 0.0})
        t["files"] += 1
        t["MB"] += src.stat().st_size / 1e6
    for k in sorted(tally):
        print(f"{k:70s} {tally[k]['files']:7d} files {tally[k]['MB']:10.1f} MB")
    print(f"total: {len(ops)} files; copied {sum(v['MB'] for k, v in tally.items() if k.startswith('copy')):.0f} MB,"
          f" linked {sum(v['MB'] for k, v in tally.items() if k.startswith('link')):.0f} MB")
    if a.dry_run:
        return 0
    clash = [o for o in ops if o[2].exists()]
    if clash:
        print(f"refusing: {len(clash)} destination(s) exist, e.g. {clash[0][2]}")
        return 1
    for kind, src, dst in ops:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if kind == "copy":
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)
            os.chmod(dst, stat.S_IREAD)
    log = NEW / "docs" / "migration_log.json"
    prev = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {"batches": []}
    prev["batches"].append({"archive": OLD.name, "archive_commit": "f4e1ad5", "only": a.only, "late": a.late,
                            "tally": tally, "n_ops": len(ops)})
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

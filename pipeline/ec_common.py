"""Shared paths for the EC-comment runs (co-author review of the 2026-11-16 draft).

Every script under EC_comments/pipeline/ imports this first.  Run everything from the
repository root (CLAUDE.md rule 1), through tools/run_stage.py.

    Final_results_for_paper/EC_comments/
        pipeline/            scripts (owner prefix per task) + ec_common.py
        pipeline/results/    one subfolder per task, T1_... to T9_...  (tracked tables)
        data/                generated data (gitignored)
        runs/                checkpoints (gitignored)

These results answer review comments. They are NOT part of
Final_results_for_paper/pipeline/results/PAPER_RESULTS.md and must not be quoted beside
it until the author decides how to use them.  Nothing here may write into
Final_results_for_paper/pipeline/, data/ or runs/, or edit a library module.
"""

from __future__ import annotations

import sys
from pathlib import Path

EC_PIPELINE = Path(__file__).resolve().parent          # .../EC_comments/pipeline
EC = EC_PIPELINE.parent                                # .../EC_comments
FINAL = EC.parent                                      # .../Final_results_for_paper
REPO = FINAL.parent                                    # repository root (three parents up)
FR_PIPELINE = FINAL / "pipeline"                       # the paper pipeline (read-only from here)

for _p in (REPO, FR_PIPELINE, EC_PIPELINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fr_common as C  # noqa: E402  (the paper pipeline's constants, re-exported)

EC_DATA = EC / "data"
EC_RUNS = EC / "runs"
EC_RESULTS = EC_PIPELINE / "results"

TASK_DIRS = {
    "T1": "T1_ablation_seeds_phase",
    "T2": "T2_sweep_nested",
    "T3": "T3_main_seeds",
    "T4": "T4_sparse_training",
    "T5": "T5_fresh_test_baselines",
    "T6": "T6_sensitivity",
    "T7": "T7_stored_outputs",
    "T8": "T8_finetune_variants",
    "T9": "T9_facts",
}

# Seeds: 123 is the paper's single seed; 124-127 are the added ones (5 in all).
PAPER_SEED = 123
EXTRA_SEEDS = (124, 125, 126, 127)
ALL_SEEDS = (PAPER_SEED,) + EXTRA_SEEDS


def task_dir(task: str) -> Path:
    d = EC_RESULTS / TASK_DIRS[task]
    d.mkdir(parents=True, exist_ok=True)
    return d

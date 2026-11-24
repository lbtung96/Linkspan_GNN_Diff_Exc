"""Shared paths for the pipeline stages (the archive's EC_comments chain, T5 and T10-T20).

Every stage script under pipeline/ imports this first (directly or through its task's common
module).  Run everything from the repository root, through tools/run_stage.py.

    pipeline/            stage scripts (prefix per task, see README.md) + the paper modules they import
    results/<task>/      one folder per task (tracked tables); results/paper/ = the archived paper's
                         tables, read as gate references only
    data/                inputs and generated data (gitignored; DATA.md)
    runs/ec/             networks trained with a record per structure (gitignored)
    runs/paper/          the archived paper's networks, shared excitation record: history only
    logs/                stage logs (gitignored)

In the archive these were Final_results_for_paper/EC_comments/{pipeline, pipeline/results, data,
runs, logs}.  Task-folder names (T5_..., T10_..., T20_...) are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

EC_PIPELINE = Path(__file__).resolve().parent          # <repo>/pipeline
REPO = EC_PIPELINE.parent                              # repository root (one parent up; three in the archive)
EC = REPO                                              # logs/ lives at the root (EC.EC / "logs")
FINAL = REPO
FR_PIPELINE = EC_PIPELINE                              # the paper modules share pipeline/ with the EC chain

for _p in (REPO, EC_PIPELINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fr_common as C  # noqa: E402  (the paper pipeline's constants, re-exported)

# archive: Final_results_for_paper/EC_comments/{data, runs, pipeline/results}
EC_DATA = REPO / "data"
EC_RUNS = REPO / "runs" / "ec"
EC_RESULTS = REPO / "results"

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

"""Stage 6 - the §7.3-7.5 decomposition on the E2 checkpoints, five seeds.

    python tools/run_stage.py experiments/E5_real_transfer/stage6_e2_decomposition.py

Until 2026-09-14 the paper's decomposition sat on
runs_phase2/EdgeGATConv/800_Samples_Ng2Split, a single 100-epoch checkpoint whose
training set no longer exists. With the switch to 500 epochs it has no
counterpart, so the paper quotes the same chain on the five E2 seeds instead,
with a spread:

  E6/task2   deterministic ceiling and propagation-only control, narrowband
             checkpoint only - its features simulate the four target modes with
             no bandpass, the narrowband training distribution, so for a
             broadband checkpoint they would not be a ceiling
  E5/stage3  corrected zero-shot and the input-dependence check, both arms
  E5/stage4  graph-topology and checkpoint controls, the pair

Each is the production stage, imported and pointed at an output folder under
stage6/seed<S>/. This is the chain experiments/E11_epoch500/ ran and checked.
The per-seed outputs are then concatenated into the four CSVs the paper reads:

  stage6_task2.csv                seed + task2_propagation_only_control.csv
  stage6_zeroshot_corrected.csv   seed, arm + stage3_zeroshot_corrected.csv
  stage6_input_dependence.csv     seed, arm + stage3_input_dependence.csv
  stage6_controls.csv             seed + stage4_controls.csv

stage3/stage4/task2 on the legacy checkpoint are left as they were, as history.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from structures import LOOCV  # noqa: E402  single source of truth

E2_RUNS = REPO / "experiments" / "E2_training_distribution" / "runs_ep500"
E6 = REPO / "experiments" / "E6_controls"
OUT = HERE / "stage6"
SEEDS = [123, 124, 125, 126, 127]
ARMS = ("narrowband", "bandpass")


def _load(path: Path):
    """Import a stage by file under its own name, so sibling imports reuse it."""
    name = path.stem
    if name in sys.modules:
        return sys.modules[name]
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_seed(seed: int, structures: list[str]) -> int:
    base = OUT / f"seed{seed}"
    raw = base / "narrowband" / "task2_raw"
    t2 = _load(E6 / "task2_propagation_only_control.py")
    z3 = _load(HERE / "stage3_zeroshot_corrected.py")
    for arm in ARMS:
        (base / arm / "stage3_raw").mkdir(parents=True, exist_ok=True)

    t2.OUT_DIR = base / "narrowband"
    sys.argv = ["task2_propagation_only_control.py",
                "--run-dir", str(E2_RUNS / f"narrowband_seed{seed}"),
                "--structures", *structures]
    rc = t2.main()
    if rc:
        return rc
    for arm in ARMS:
        # stage3 reads fe_target from the task2_raw just written - the ordering
        # run_all.py enforces for the legacy chain. fe_target does not depend on
        # the checkpoint, so both arms share it.
        z3.OUT, z3.AUDIT_RAW = base / arm, raw
        rc = z3.main(["--run-dir", str(E2_RUNS / f"{arm}_seed{seed}"), "--no-plots",
                      "--structures", *structures])
        if rc:
            return rc
    c4 = _load(HERE / "stage4_controls.py")
    c4.OUT, c4.AUDIT_RAW = base, raw
    c4.RUN_NARROW = E2_RUNS / f"narrowband_seed{seed}"
    c4.RUN_BAND = E2_RUNS / f"bandpass_seed{seed}"
    c4.STRUCTURES = list(structures)
    return c4.main()


def _concat(parts: list[tuple[dict, Path]], out: Path) -> int:
    data: list[dict] = []
    for extra, p in parts:
        with p.open(newline="", encoding="utf-8") as fh:
            data += [{**extra, **r} for r in csv.DictReader(fh)]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
    print(f"wrote {out.relative_to(REPO)} ({len(data)} rows)")
    return len(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--structures", nargs="+", default=list(LOOCV), choices=list(LOOCV))
    a = ap.parse_args(argv)

    missing = [f"{arm}_seed{s}" for s in a.seeds for arm in ARMS
               if not (E2_RUNS / f"{arm}_seed{s}" / "checkpoints" / "best_model.pt").is_file()]
    if missing:
        print(f"[FAIL] missing E2 checkpoints in {E2_RUNS.relative_to(REPO)}: {missing}")
        return 1

    for s in a.seeds:
        print(f"\n===== seed {s} =====", flush=True)
        rc = run_seed(s, a.structures)
        if rc:
            return rc

    per = [(s, OUT / f"seed{s}") for s in a.seeds]
    _concat([({"seed": s}, d / "narrowband" / "task2_propagation_only_control.csv")
             for s, d in per], HERE / "stage6_task2.csv")
    _concat([({"seed": s, "arm": arm}, d / arm / "stage3_zeroshot_corrected.csv")
             for s, d in per for arm in ARMS], HERE / "stage6_zeroshot_corrected.csv")
    _concat([({"seed": s, "arm": arm}, d / arm / "stage3_input_dependence.csv")
             for s, d in per for arm in ARMS], HERE / "stage6_input_dependence.csv")
    _concat([({"seed": s}, d / "stage4_controls.csv") for s, d in per],
            HERE / "stage6_controls.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

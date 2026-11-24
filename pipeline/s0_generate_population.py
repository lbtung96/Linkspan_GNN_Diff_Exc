"""Stage 0 - generate the paper population, one response arm per invocation.

    python tools/run_stage.py Final_results_for_paper/pipeline/s0_generate_population.py --arm broadband
    python tools/run_stage.py Final_results_for_paper/pipeline/s0_generate_population.py --arm narrowband

Both arms draw the same 5500 structures (generator seed 20260507, per-Ng random
streams) and differ only in the synthesised response:

  broadband   all 15 solved FE modes, 1.5-20 Hz zero-phase Butterworth bandpass
              - the arm every network in this folder trains on
  narrowband  the 4 matched target modes only, no filter
              - generated for later use; nothing here trains on it

The production generator is never edited.  It is imported, and five of its
module-level names are replaced before main() runs; main() looks every one of
them up as a global at call time:

  OUT_DIR                  -> Final_results_for_paper/data/pop_<arm>
  N_STRUCTURES_TOTAL       -> 5500  (1375 per Ng)
  sample_params_for_ng     -> girder spacing for Ng >= 3 drawn up to 3000 mm
                              (was 2500 mm), so the five real bridges at 3000 mm
                              lie inside the population
  validate_sampled_params  -> accepts that spacing
  match_target_modes       -> additionally rejects a structure whose selected
                              modes match their reference template with
                              MAC < 0.85 (the matcher's own threshold).  In the
                              previous generation 4.8 % of two-girder decks
                              carried such a label, some near MAC 0.

The two sampling functions are rebuilt from the generator's own source with a
single exact-string substitution each, which hard-fails if the source changes,
so every other line of the sampling stays byte-identical to production.
"""

from __future__ import annotations

import __future__ as _future
import argparse
import inspect
import io
import json
import platform
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# main() finishes with plot_probability_histograms_before_after(), which calls
# plt.show().  With an interactive backend that opens windows and blocks until
# someone closes them - the smoke test sat there for ten minutes after all the
# work was done.  Agg must be selected before the generator imports pyplot.
matplotlib.use("Agg")

import numpy as np  # noqa: E402

import fr_common as C  # noqa: E402

import generate_population_dataset_modes4 as gen  # noqa: E402  (fr_common puts REPO on sys.path)

DY_OLD = "dy_high = min(2500.0, 12000.0 / (ng - 1))"
DY_NEW = f"dy_high = min({C.DY_MAX_NG3PLUS_MM:.1f}, 12000.0 / (ng - 1))"
VAL_OLD = "if not (2000.0 <= dy <= 2500.0):"
VAL_NEW = f"if not (2000.0 <= dy <= {C.DY_MAX_NG3PLUS_MM:.1f}):"


def _rebuild(func, old: str, new: str):
    """Recompile `func` from the generator's source with one exact substitution."""
    src = textwrap.dedent(inspect.getsource(func))
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"expected exactly one {old!r} in {func.__name__}, found {n} - "
                           "the generator source changed; re-derive the patch")
    code = compile(src.replace(old, new), f"<Final_results_for_paper patch of {func.__name__}>",
                   "exec", flags=_future.annotations.compiler_flag, dont_inherit=True)
    ns: dict = {}
    exec(code, gen.__dict__, ns)                       # globals: the generator module itself
    return ns[func.__name__]


_ORIGINAL_MATCH = gen.match_target_modes


def match_with_selected_mac_gate(*args, **kwargs):
    res = _ORIGINAL_MATCH(*args, **kwargs)
    mac = np.asarray(res.selected_mac, dtype=float).reshape(-1)
    if np.any(mac < C.SELECTED_MAC_GATE):
        # main() catches this, counts the attempt as rejected, logs it to
        # failures.log and draws the next structure from the same Ng stream.
        raise ValueError(f"selected-mode MAC below {C.SELECTED_MAC_GATE}: "
                         f"{np.round(mac, 4).tolist()}")
    return res


def install_patches(out_root: Path, n_total: int) -> dict:
    sample_new = _rebuild(gen.sample_params_for_ng, DY_OLD, DY_NEW)
    validate_new = _rebuild(gen.validate_sampled_params, VAL_OLD, VAL_NEW)
    gen.OUT_DIR = Path(out_root)
    gen.N_STRUCTURES_TOTAL = int(n_total)
    gen.sample_params_for_ng = sample_new
    gen.validate_sampled_params = validate_new
    gen.match_target_modes = match_with_selected_mac_gate
    return {"sample_params_for_ng": [DY_OLD, DY_NEW],
            "validate_sampled_params": [VAL_OLD, VAL_NEW],
            "match_target_modes": f"reject if any selected MAC < {C.SELECTED_MAC_GATE}"}


def self_test() -> None:
    """The patched sampler and validator behave as intended; Ng = 2 is untouched."""
    install_patches(Path("unused"), 8)
    rng = np.random.default_rng(0)
    for ng, lo, hi in [(3, 2250.0, 3000.0), (4, 2000.0, 3000.0), (5, 2000.0, 3000.0), (2, 4000.0, 4500.0)]:
        dys = np.array([gen.sample_params_for_ng(ng, rng)["dy_mm"] for _ in range(3000)])
        assert dys.min() >= lo - 1e-9 and dys.max() <= hi + 1e-9, (ng, dys.min(), dys.max())
        assert dys.max() > hi - 20.0 and dys.min() < lo + 20.0, (ng, dys.min(), dys.max())
    base = gen.sample_params_for_ng(4, np.random.default_rng(1))
    for dy, ok in [(2999.0, True), (3001.0, False), (1999.0, False)]:
        assert gen.validate_sampled_params({**base, "dy_mm": dy}) is ok, dy
    b5 = gen.sample_params_for_ng(5, np.random.default_rng(2))
    assert gen.validate_sampled_params({**b5, "dy_mm": 3000.0}) is True       # width 12 000 mm, allowed
    b2 = gen.sample_params_for_ng(2, np.random.default_rng(3))
    assert gen.validate_sampled_params({**b2, "dy_mm": 3000.0}) is False      # Ng = 2 rule unchanged
    # draw order unchanged: for Ng = 2 the patched sampler reproduces production exactly
    a = _ORIGINAL_SAMPLE(2, np.random.default_rng(7))
    b = gen.sample_params_for_ng(2, np.random.default_rng(7))
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    print("[self-test] patched sampler/validator OK "
          "(Ng>=3 spacing up to 3000 mm, Ng=2 unchanged and bit-identical to production)")


_ORIGINAL_SAMPLE = gen.sample_params_for_ng


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(C.REPO), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=["broadband", "narrowband"], required=False)
    ap.add_argument("--n-total", type=int, default=C.N_TOTAL)
    ap.add_argument("--out-root", type=Path, default=None,
                    help="override the output root (smoke tests only)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.arm is None:
        ap.error("--arm is required")

    out_root = args.out_root or (C.POP_BROADBAND_ROOT if args.arm == "broadband" else C.POP_NARROWBAND_ROOT)
    out_root = Path(out_root).resolve()
    if "population_dataset_modes4" in str(out_root):
        raise RuntimeError("refusing to write into population_dataset_modes4/")
    if args.out_root is None and C.DATA.resolve() not in out_root.parents:
        raise RuntimeError(f"output root {out_root} is not under {C.DATA}")
    if gen.SEED != C.GENERATOR_SEED:
        raise RuntimeError(f"generator SEED is {gen.SEED}, expected {C.GENERATOR_SEED}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[keep-awake] {'requested' if C.keep_awake() else 'unavailable'}")

    patches = install_patches(out_root, args.n_total)
    meta = {
        "stage": "s0_generate_population",
        "arm": args.arm,
        "bandpass": args.arm == "broadband",
        "n_structures_total": args.n_total,
        "per_ng_target": gen.compute_ng_targets(args.n_total, gen.NG_CHOICES),
        "generator_seed": gen.SEED,
        "n_candidate_modes": gen.N_CANDIDATE_MODES,
        "patches": patches,
        "git_head": git_head(),
        "python": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_root / "generation_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    # main() ends with an interactive "inspect a sample?" prompt, guarded by
    # sys.stdin.isatty().  Under this shell and under tools/run_stage.py the child
    # can inherit a console stdin, so the guard passes and input() blocks for ever
    # after all the work is done (it did, in the smoke test).  An empty, non-TTY
    # stdin makes the guard skip the prompt.
    sys.stdin = io.StringIO("")
    t0 = time.time()
    gen.main(bandpass=(args.arm == "broadband"))
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["seconds"] = round(time.time() - t0, 1)
    (out_root / "generation_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {args.arm}: {meta['seconds']:.0f} s -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

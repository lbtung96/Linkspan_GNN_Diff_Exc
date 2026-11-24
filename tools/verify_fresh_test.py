"""Regenerate the first structures of T5's two fresh test sets with this repository's code and compare.

    python tools/run_stage.py tools/verify_fresh_test.py [--per-ng 10]

T5 generated 500 new structures (parameter seed 20261123) twice: `fresh_test` (one shared new record, 20261124)
and `fresh_test_paperexc` (the paper's record, 12345), through pipeline/c_gen_fresh_test.py (the generator with
s0's patches; 52 + 56 min).  Both sets are hard-linked inputs here.  Regenerating all 1000 structures would
need 1.5 GB of scratch disk and two hours; this stage regenerates the first --per-ng accepted structures of each
girder count, which is a true prefix: each Ng draws from its own stream (SeedSequence(seed).spawn(4)) and every
rejection happens before the excitation, so the k-th accepted structure of an Ng does not depend on how many are
requested.  Only the global numbering differs (sample_index / sample_id / file name).

Each set is generated in a child process (c_gen_fresh_test.main with --n-total 4 x per-ng) into
data/T5/_verify_<set>/, compared key by key with the stored structure of the same Ng and rank, then deleted.
Arrays: 1e-5 absolute (the FE model is re-solved; eigsh has no fixed start vector).  JSON strings: parsed and
compared numerically at the same tolerance (NaN equals NaN, equal infinities are equal).  The four keys taken
from the raw eigenvectors before sign alignment (SIGN_KEYS) must agree up to one sign per mode: eigsh's raw sign
is arbitrary from solve to solve, the aligned labels are not, and they are compared exactly like every other
array.  The number of modes whose raw sign came out differently is reported.
Writes results/reproduction/fresh_test_prefix.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "pipeline")]
import numpy as np  # noqa: E402

T5 = REPO / "data" / "T5"
OUT = REPO / "results" / "reproduction"
TOL = 1e-5
RENUMBERED = {"sample_index", "sample_id"}
SETS = ("fresh_test", "fresh_test_paperexc")


def by_ng(folder: Path) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {}
    for p in sorted(folder.glob("sample_*_Ng*_phase1_gnn.npz"), key=lambda p: int(p.name.split("_")[1])):
        out.setdefault(int(p.name.split("_Ng")[1][0]), []).append(p)
    return out


def json_diffs(a, b, path: str = "") -> list[str]:
    """Leaf paths at which two parsed JSON values differ (numbers at TOL, relative above 1)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k} (only in one)" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += json_diffs(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in json_diffs(x, y, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None:
        return [] if a == b else [path]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        x, y = float(a), float(b)
        if x == y or (math.isnan(x) and math.isnan(y)):  # also equal infinities; NaN only equals NaN
            return []
        return [] if abs(x - y) <= TOL * max(1.0, abs(y)) else [path]
    return [] if a == b else [path]


# Taken from the raw eigenvectors BEFORE sign alignment.  eigsh has no fixed start vector, so each mode's raw sign
# is arbitrary from one solve to the next; the aligned labels (y_mode_shape_signed) are not.  These keys must agree
# up to one sign per mode, consistent across all four.
SIGN_KEYS = ("phi_u3_sensor_4", "sign_reference_dot_before", "sign_flip_factors", "sign_fallback_reason")
REASON_UNDER_FLIP = {"dot_positive_kept": "dot_negative_flipped", "dot_negative_flipped": "dot_positive_kept"}


def sign_check(a, b) -> tuple[bool, int]:
    """(agree up to a per-mode raw sign, number of modes whose raw sign differs)."""
    pa, pb = a["phi_u3_sensor_4"].astype(np.float64), b["phi_u3_sensor_4"].astype(np.float64)
    if pa.shape != pb.shape:
        return False, -1
    s = np.sign((pa * pb).sum(axis=0))
    s[s == 0] = 1.0
    ok = float(np.abs(pa - pb * s).max()) <= TOL
    ok &= float(np.abs(a["sign_reference_dot_before"] - s * b["sign_reference_dot_before"]).max()) <= TOL
    ok &= bool(np.array_equal(a["sign_flip_factors"], s * b["sign_flip_factors"]))
    ok &= all(str(ra) == (REASON_UNDER_FLIP.get(str(rb), str(rb)) if sm < 0 else str(rb))
              for ra, rb, sm in zip(a["sign_fallback_reason"], b["sign_fallback_reason"], s))
    return bool(ok), int((s < 0).sum())


def compare(new: Path, old: Path) -> dict:
    a, b = np.load(new, allow_pickle=False), np.load(old, allow_pickle=False)
    bad, maxd, jd, kd = [], 0.0, {}, {}
    if set(a.files) != set(b.files):
        bad.append(f"keys {sorted(set(a.files) ^ set(b.files))}")
    sign_ok, n_raw_flipped = sign_check(a, b)
    if not sign_ok:
        bad.append("sign keys (not equal up to a per-mode raw sign)")
    for k in sorted(set(a.files) & set(b.files) - RENUMBERED - set(SIGN_KEYS)):
        x, y = a[k], b[k]
        if x.dtype.kind in "US":
            if k.endswith("_json"):
                jd[k] = json_diffs(json.loads(str(x)), json.loads(str(y)))
                ok = not jd[k]
            else:
                ok = x.shape == y.shape and bool(np.all(x == y))
        elif x.dtype.kind in "fc":
            ok = x.shape == y.shape and x.dtype == y.dtype
            if ok and x.size:
                d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                kd[k] = d
                ok = d <= TOL and bool(np.array_equal(np.isnan(x), np.isnan(y)))
        else:
            ok = x.shape == y.shape and x.dtype == y.dtype and bool(np.array_equal(x, y))
        if not ok:
            bad.append(k)
    maxd = max(kd.values(), default=0.0)
    return {"new": new.name, "stored": old.name, "max_abs_diff": maxd, "differing_keys": bad,
            "modes_raw_sign_differs": n_raw_flipped,
            "max_abs_diff_per_key": {k: v for k, v in kd.items() if v > TOL},
            "json_leaves": {k: v[:12] for k, v in jd.items() if v}, "pass": not bad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-ng", type=int, default=10)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--keep", action="store_true", help="keep data/T5/_verify_<set>/ for inspection")
    a = ap.parse_args()
    report = {"per_ng": a.per_ng, "tolerance": TOL, "sets": {}}
    for s in a.sets:
        tmp = T5 / f"_verify_{s}"
        shutil.rmtree(tmp, ignore_errors=True)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
        # c_gen_fresh_test.main writes to data/T5/<set>: register a scratch set name with the same seed
        runner = REPO / "tools" / "_fresh_prefix_runner.py"
        runner.write_text(
            "import sys\n"
            f"sys.path[:0] = [r'{REPO}', r'{REPO / 'pipeline'}']\n"
            "import matplotlib\nmatplotlib.use('Agg')\n"
            "import c_gen_fresh_test as G\n"
            f"G.SETS = {{'_verify_{s}': G.SETS['{s}']}}\n"
            f"sys.argv = ['c_gen_fresh_test.py', '--set', '_verify_{s}', '--n-total', '{4 * a.per_ng}']\n"
            "sys.exit(G.main())\n", encoding="utf-8")
        rc = subprocess.run([sys.executable, "-u", str(runner)], cwd=str(REPO), env=env,
                            stdin=subprocess.DEVNULL).returncode
        runner.unlink(missing_ok=True)
        if rc:
            report["sets"][s] = {"rc": rc, "pass": False}
            continue
        new, old = by_ng(tmp / "samples_phase2_bandpass"), by_ng(T5 / s / "samples_phase2_bandpass")
        rows = []
        for ng in sorted(new):
            for k, p in enumerate(new[ng]):
                rows.append({"Ng": ng, "rank": k, **compare(p, old[ng][k])})
        report["sets"][s] = {"rc": rc, "structures": len(rows), "max_abs_diff": max(r["max_abs_diff"] for r in rows),
                             "modes_raw_sign_differs": sum(r["modes_raw_sign_differs"] for r in rows),
                             "modes_total": 4 * len(rows),
                             "failures": [r for r in rows if not r["pass"]],
                             "pass": len(rows) == 4 * a.per_ng and all(r["pass"] for r in rows)}
        print(f"{s}: {len(rows)} structures, max |diff| {report['sets'][s]['max_abs_diff']:.2e}, "
              f"{'PASS' if report['sets'][s]['pass'] else 'FAIL'}", flush=True)
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    report["pass"] = all(v["pass"] for v in report["sets"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fresh_test_prefix.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("FRESH TEST PREFIX PASS" if report["pass"] else "FRESH TEST PREFIX FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

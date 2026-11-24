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
compared numerically at the same tolerance.  Writes results/reproduction/fresh_test_prefix.json.
"""

from __future__ import annotations

import argparse
import json
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


def close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(b)))
    return a == b


def compare(new: Path, old: Path) -> dict:
    a, b = np.load(new, allow_pickle=False), np.load(old, allow_pickle=False)
    bad, maxd = [], 0.0
    if set(a.files) != set(b.files):
        bad.append(f"keys {sorted(set(a.files) ^ set(b.files))}")
    for k in sorted(set(a.files) & set(b.files) - RENUMBERED):
        x, y = a[k], b[k]
        if x.dtype.kind in "US":
            if k.endswith("_json"):
                ok = close(json.loads(str(x)), json.loads(str(y)))
            else:
                ok = x.shape == y.shape and bool(np.all(x == y))
        elif x.dtype.kind in "fc":
            ok = x.shape == y.shape and x.dtype == y.dtype
            if ok and x.size:
                d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                maxd = max(maxd, d)
                ok = d <= TOL and bool(np.array_equal(np.isnan(x), np.isnan(y)))
        else:
            ok = x.shape == y.shape and x.dtype == y.dtype and bool(np.array_equal(x, y))
        if not ok:
            bad.append(k)
    return {"new": new.name, "stored": old.name, "max_abs_diff": maxd, "differing_keys": bad, "pass": not bad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-ng", type=int, default=10)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
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
                             "failures": [r for r in rows if not r["pass"]],
                             "pass": len(rows) == 4 * a.per_ng and all(r["pass"] for r in rows)}
        print(f"{s}: {len(rows)} structures, max |diff| {report['sets'][s]['max_abs_diff']:.2e}, "
              f"{'PASS' if report['sets'][s]['pass'] else 'FAIL'}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    report["pass"] = all(v["pass"] for v in report["sets"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fresh_test_prefix.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("FRESH TEST PREFIX PASS" if report["pass"] else "FRESH TEST PREFIX FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

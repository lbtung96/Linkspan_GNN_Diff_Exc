"""Classify how each re-run result file differs from its committed (archive) copy.

    python tools/compare_results.py [--ref HEAD] [--tol 1e-5] [paths ...]

`git diff` says only *that* a file changed.  A re-run legitimately changes three kinds of content that carry no
result: timing columns (forward_ms_median, seconds, minutes: machine noise), path strings re-pointed to this
repository's layout, and row order in tables that accumulate across runs.  This tool separates those from
differences in the numbers.  For every changed file under results/ (or the paths given) it compares the
working-tree file with `git show <ref>:<path>`:

  CSV   rows matched on their key columns (every column that is not a float and not timing); then per column
        the maximum absolute difference.  Timing columns are reported, not judged.
  JSON  leaf by leaf; numbers at --tol (relative above 1), NaN == NaN; differing strings listed as pairs.
  other text (md, log, txt): number of differing lines; binary (png, pdf): identical or not.

Verdicts:
  identical          byte-identical (git sees no change)
  reordered          same rows, different order, nothing else
  timing-only        only timing columns / keys differ (possibly with reordering)
  strings-only       only strings differ (paths, module names; listed), possibly with timing
  numeric<=tol       numbers differ, all within tol (plus any of the above)
  text               a text file (md, log, txt) whose lines differ: not judged here, read the diff
  DIFFERENT          anything else: a number beyond tol, a row or key present on one side only, binary content

Writes results/reproduction/RESULTS_COMPARED.md and results/reproduction/compare.json.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "reproduction"
TIMING = re.compile(r"(forward_(ms|s)|_ms(_|$)|_s$|seconds|minutes|hours|elapsed|wall|runtime|duration|_sec$"
                    r"|_time$|^time|timestamp|started|finished|^date$|_utc$)", re.I)  # not _min: minima
NOT_TIMING = {"record_s", "t_total_s"}  # record lengths, not run times


def is_timing(name: str) -> bool:
    return name not in NOT_TIMING and bool(TIMING.search(name))


TEXT_EXT = {".md", ".log", ".txt", ".jsonl"}
BINARY_EXT = {".png", ".pdf", ".npz", ".gz"}
SKIP = {"results/RUN_LEDGER.jsonl"}


def git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def num(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def close(x: float, y: float, tol: float) -> bool:
    if x == y or (math.isnan(x) and math.isnan(y)):
        return True
    return abs(x - y) <= tol * max(1.0, abs(y))


def read_csv(b: bytes) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(io.StringIO(b.decode("utf-8-sig", "replace"))))
    return (rows[0], rows[1:]) if rows else ([], [])


def cmp_csv(new: bytes, old: bytes, tol: float) -> dict:
    hn, rn = read_csv(new)
    ho, ro = read_csv(old)
    if hn != ho:
        return {"verdict": "DIFFERENT", "why": f"header differs: only new {sorted(set(hn) - set(ho))}, "
                                              f"only old {sorted(set(ho) - set(hn))}"}
    timing = [i for i, c in enumerate(hn) if is_timing(c)]
    floaty = set()
    for i in range(len(hn)):
        for r in rn + ro:
            v = r[i] if i < len(r) else ""
            x = num(v)
            if x is not None and not float(x).is_integer():
                floaty.add(i)
                break
    key_cols = [i for i in range(len(hn)) if i not in floaty and i not in timing]

    def key(r):
        return tuple(r[i] if i < len(r) else "" for i in key_cols)

    reordered = [key(r) for r in rn] != [key(r) for r in ro]
    kn, ko = {}, {}
    for r in rn:
        kn.setdefault(key(r), []).append(r)
    for r in ro:
        ko.setdefault(key(r), []).append(r)
    only_new = [k for k in kn if k not in ko or len(kn[k]) != len(ko[k])]
    only_old = [k for k in ko if k not in kn]
    if only_new or only_old:
        return {"verdict": "DIFFERENT", "why": f"{len(only_new)} keys only (or with another multiplicity) in the "
                                              f"re-run, {len(only_old)} only in the archive",
                "example_keys": [list(k) for k in (only_new + only_old)[:3]], "key_columns": [hn[i] for i in key_cols]}
    maxd: dict[str, float] = {}
    strdiff: dict[str, set] = {}
    for k in kn:
        for a, b in zip(kn[k], ko[k]):
            for i in floaty | set(timing):
                x, y = num(a[i]), num(b[i])
                if x is None or y is None:
                    if a[i] != b[i]:
                        strdiff.setdefault(hn[i], set()).add((b[i], a[i]))
                    continue
                if not close(x, y, 0.0):
                    d = abs(x - y) if not (math.isinf(x) or math.isinf(y)) else math.inf
                    maxd[hn[i]] = max(maxd.get(hn[i], 0.0), d)
    timing_names = {hn[i] for i in timing}
    real = {c: d for c, d in maxd.items() if c not in timing_names}
    out = {"rows": len(rn), "reordered": reordered,
           "timing_columns_differing": sorted(c for c in maxd if c in timing_names),
           "max_abs_diff": real, "strings": {c: sorted(v)[:5] for c, v in strdiff.items()}}
    out["verdict"] = verdict(reordered, bool(out["timing_columns_differing"]), bool(strdiff), real, tol)
    return out


def verdict(reordered: bool, timing: bool, strings: bool, real: dict, tol: float) -> str:
    if any(not (d <= tol) for d in real.values()):
        return "DIFFERENT"
    if real:
        return "numeric<=tol"
    if strings:
        return "strings-only"
    if timing:
        return "timing-only"
    return "reordered" if reordered else "identical"


def json_leaves(a, b, path: str, tol: float, acc: dict) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) ^ set(b)):
            acc["missing"].append(f"{path}/{k}")
        for k in sorted(set(a) & set(b)):
            json_leaves(a[k], b[k], f"{path}/{k}", tol, acc)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["missing"].append(f"{path} (length {len(a)} vs {len(b)})")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            json_leaves(x, y, f"{path}[{i}]", tol, acc)
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            acc["strings"].append((path, b, a))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        x, y = float(a), float(b)
        if not close(x, y, 0.0):
            name = path.rsplit("/", 1)[-1]
            bucket = "timing" if is_timing(re.sub(r"\[\d+\]$", "", name)) else "numbers"
            d = abs(x - y) if not (math.isinf(x) or math.isinf(y)) else math.inf
            acc[bucket][path] = d
        return
    if a != b:
        acc["strings"].append((path, b, a))


def cmp_json(new: bytes, old: bytes, tol: float) -> dict:
    try:
        a, b = json.loads(new.decode("utf-8")), json.loads(old.decode("utf-8"))
    except ValueError as e:
        return {"verdict": "DIFFERENT", "why": f"not parseable: {e}"}
    acc = {"missing": [], "strings": [], "timing": {}, "numbers": {}}
    json_leaves(a, b, "", tol, acc)
    if acc["missing"]:
        return {"verdict": "DIFFERENT", "why": "keys or lengths differ", "examples": acc["missing"][:5]}
    worst = max(acc["numbers"].values(), default=0.0)
    out = {"timing_keys_differing": sorted(acc["timing"])[:10], "max_abs_diff": worst,
           "n_numbers_differing": len(acc["numbers"]),
           "examples": sorted(acc["numbers"], key=acc["numbers"].get, reverse=True)[:5],
           "strings": [list(s) for s in acc["strings"][:8]], "n_strings_differing": len(acc["strings"])}
    real = {"max": worst} if acc["numbers"] else {}
    out["verdict"] = verdict(False, bool(acc["timing"]), bool(acc["strings"]), real, tol)
    return out


def cmp_text(new: bytes, old: bytes) -> dict:
    ln, lo = new.decode("utf-8", "replace").splitlines(), old.decode("utf-8", "replace").splitlines()
    import difflib
    d = [l for l in difflib.unified_diff(lo, ln, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    return {"verdict": "text" if d else "identical", "differing_lines": len(d), "first": d[:6]}


def compare_file(rel: str, ref: str, tol: float) -> dict:
    p = REPO / rel
    old = git("show", f"{ref}:{rel}", binary=True)
    if not p.exists():
        return {"verdict": "DIFFERENT", "why": "deleted"}
    new = p.read_bytes()
    if new == old:
        return {"verdict": "identical"}
    if not old:
        return {"verdict": "new", "why": f"not in {ref}"}
    ext = p.suffix.lower()
    if ext == ".csv":
        return cmp_csv(new, old, tol)
    if ext == ".json":
        return cmp_json(new, old, tol)
    if ext in TEXT_EXT:
        return cmp_text(new, old)
    if new.replace(b"\r\n", b"\n") == old.replace(b"\r\n", b"\n"):
        return {"verdict": "identical", "why": "line endings only"}
    return {"verdict": "DIFFERENT", "why": "binary content differs" if ext in BINARY_EXT else "differs"}


def changed_files(ref: str) -> list[str]:
    tracked = git("diff", "--name-only", ref, "--", "results").splitlines()
    return [f for f in tracked if f not in SKIP and not f.startswith("results/reproduction/")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args()
    files = a.paths or changed_files(a.ref)
    res = {f: compare_file(f, a.ref, a.tol) for f in files}
    order = ["DIFFERENT", "new", "text", "numeric<=tol", "strings-only", "timing-only", "reordered", "identical"]
    lines = ["# How the re-run result files differ from the archive copies", "",
             f"Generated by `tools/compare_results.py` against `{a.ref}`, tolerance {a.tol:g}.  Only files whose "
             "bytes changed are listed; see the module docstring for the verdicts.", "",
             "| verdict | files |", "|---|---|"]
    for v in order:
        n = sum(r["verdict"] == v for r in res.values())
        if n:
            lines.append(f"| {v} | {n} |")
    lines += ["", "| file | verdict | detail |", "|---|---|---|"]
    for f in sorted(res, key=lambda f: (order.index(res[f]["verdict"]), f)):
        r = res[f]
        det = []
        if r.get("why"):
            det.append(r["why"])
        if r.get("timing_columns_differing") or r.get("timing_keys_differing"):
            det.append("timing: " + ", ".join(r.get("timing_columns_differing") or r.get("timing_keys_differing")))
        if isinstance(r.get("max_abs_diff"), dict) and r["max_abs_diff"]:
            det.append("max |diff| " + ", ".join(f"{c} {d:.2g}" for c, d in sorted(r["max_abs_diff"].items(),
                                                                                       key=lambda t: -t[1])[:4]))
        elif isinstance(r.get("max_abs_diff"), float) and r["max_abs_diff"]:
            det.append(f"max |diff| {r['max_abs_diff']:.2g} ({r['n_numbers_differing']} numbers)")
        if r.get("strings"):
            s = r["strings"]
            items = (s.items() if isinstance(s, dict) else [(x[0], [(x[1], x[2])]) for x in s])
            det.append("strings: " + "; ".join(f"{k}: {str(v[0][0])[:50]!r} -> {str(v[0][1])[:50]!r}"
                                               for k, v in list(items)[:2]))
        if r.get("reordered"):
            det.append("rows reordered")
        if r.get("differing_lines"):
            det.append(f"{r['differing_lines']} lines differ")
        lines.append(f"| `{f}` | {r['verdict']} | {' · '.join(det).replace('|', '/')} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS_COMPARED.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    (OUT / "compare.json").write_bytes(json.dumps(res, indent=1, default=str).encode("utf-8"))
    print("\n".join(lines[:12 + len(order)]))
    print(f"-> results/reproduction/RESULTS_COMPARED.md ({len(res)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""SHA-256 of every file this repository needs but does not keep in git (data, networks, field files).

    python tools/run_stage.py tools/hash_data.py [--workers 4]

Writes DATA_MANIFEST.tsv.gz (tracked: path, bytes, sha256 per file) and data_summary.csv (per folder: files,
bytes, and a folder hash = sha256 of the sorted "path<TAB>sha256" lines), which DATA.md quotes.  Anyone who has
the data can check it against the public repository with --check.

    python tools/hash_data.py --check        # re-hash and compare with DATA_MANIFEST.tsv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOTS = ["data", "runs", "field_measurements_raw", "SSI_modal_infor", "phase3_real_graphs",
         "experiments/E6_controls/task2_raw", "experiments/E19_mode_identification/cache",
         "experiments/E18_population_basis/prior.npz"]
SKIP_PARTS = {"__pycache__"}
MANIFEST = REPO / "DATA_MANIFEST.tsv.gz"
SUMMARY = REPO / "data_summary.csv"


def files() -> list[Path]:
    out = []
    for r in ROOTS:
        p = REPO / r
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file() and not SKIP_PARTS & set(f.parts)
                    and not f.name.endswith(".tmp") and "_verify_" not in f.as_posix()
                    and "/runs/repro/" not in f.as_posix()]
    return sorted(out)


def sha(p: str) -> tuple[str, int, str]:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return p, Path(p).stat().st_size, h.hexdigest()


def folder_of(rel: str) -> str:
    parts = rel.split("/")
    depth = 3 if parts[0] in ("data", "runs") else 2
    return "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1]) or rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fl = files()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [(Path(p).relative_to(REPO).as_posix(), n, h) for p, n, h in ex.map(sha, map(str, fl), chunksize=64)]
    if a.check:
        ref = {}
        with gzip.open(MANIFEST, "rt", encoding="utf-8") as fh:
            for r in csv.reader(fh, delimiter="\t"):
                if r[0] != "path":
                    ref[r[0]] = (int(r[1]), r[2])
        cur = {p: (n, h) for p, n, h in rows}
        missing = sorted(set(ref) - set(cur))
        extra = sorted(set(cur) - set(ref))
        bad = sorted(p for p in set(ref) & set(cur) if ref[p] != cur[p])
        print(f"{len(ref)} in manifest, {len(cur)} on disk: {len(missing)} missing, {len(extra)} not in manifest, "
              f"{len(bad)} differ")
        for p in (missing + bad)[:20]:
            print("  ", p)
        return 0 if not missing and not bad else 1
    with gzip.open(MANIFEST, "wt", newline="", encoding="utf-8", compresslevel=9) as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "bytes", "sha256"])
        w.writerows(rows)
    groups: dict[str, list] = {}
    for p, n, h in rows:
        groups.setdefault(folder_of(p), []).append((p, n, h))
    with SUMMARY.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["folder", "files", "bytes", "folder_sha256"])
        for g in sorted(groups):
            lines = "".join(f"{p}\t{h}\n" for p, _n, h in sorted(groups[g]))
            w.writerow([g, len(groups[g]), sum(n for _p, n, _h in groups[g]),
                        hashlib.sha256(lines.encode("utf-8")).hexdigest()])
    print(f"{len(rows)} files, {sum(n for _p, n, _h in rows) / 1e9:.1f} GB -> {MANIFEST.name}, {SUMMARY.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

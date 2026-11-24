"""Run one pipeline stage, tee its output, and record it in the run ledger.

    python tools/run_stage.py experiments/E5_real_transfer/stage3_zeroshot_corrected.py --no-plots

Why a ledger.  Verification needs to distinguish three states that look alike on
disk:

  * a stage was never re-run, so its CSV is still the copied file;
  * a stage was re-run and reproduced the copied file exactly;
  * a stage was re-run and produced something different.

Hashes cannot separate the first two - a perfect reproduction is byte-identical
to the file it reproduces - and mtimes are unreliable, because reading and
archiving tools touch them.  So the fact of execution is recorded explicitly.

Each run appends one JSON line to `results/RUN_LEDGER.jsonl` naming the stage,
its arguments, exit code, duration, interpreter, and every file in the stage's
own folder whose content changed or that was newly created.  Files the stage
rewrote byte-identically are listed under `reproduced`.

The interpreter defaults to $LINKSPAN_PY, else the venv named in
tools/INTERPRETER if present, else the current sys.executable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "results" / "RUN_LEDGER.jsonl"
LOGDIR = REPO / "logs"

# We re-emit the child's output on our own stdout, which is cp1252 whenever this
# is piped on Windows. Without this, a stage that prints "->" as U+2192 kills the
# *runner* with UnicodeEncodeError after the stage itself has already succeeded -
# which is indistinguishable, in the ledger, from the stage having failed.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - older/odd stdout objects
    pass


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(folder: Path) -> dict[str, tuple[float, str]]:
    """(mtime, sha256) for every result-shaped file under `folder`.

    Authorship is decided by the mtime advancing *during this run*: writing a
    file always moves its mtime, even when the bytes come out identical.  That
    is what separates "the stage rewrote this and got the same answer" from
    "this file was merely sitting in the same folder".  (mtime is unusable for
    the *other* question - whether a file is still the one copied in months ago
    - which is why staleness is read from this ledger instead.)
    """
    if not folder.exists():
        return {}
    out = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".npz", ".png", ".pdf"}:
            out[p.relative_to(REPO).as_posix()] = (p.stat().st_mtime, sha256(p))
    return out


def interpreter() -> str:
    if os.environ.get("LINKSPAN_PY"):
        return os.environ["LINKSPAN_PY"]
    pin = REPO / "tools" / "INTERPRETER"
    if pin.exists():
        cand = pin.read_text(encoding="utf-8").strip()
        if cand and Path(cand).exists():
            return cand
    return sys.executable


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    stage = Path(argv[0])
    args = argv[1:]
    script = REPO / stage
    if not script.exists():
        print(f"[fail] no such stage: {stage}")
        return 2

    py = interpreter()
    folder = script.parent
    before = snapshot(folder)

    LOGDIR.mkdir(exist_ok=True)
    # Include a slug of the arguments: run_all.py invokes some scripts several
    # times with different flags (E1/run_ablation.py --prepare/--train/--summary),
    # and a name derived from the script alone would have each overwrite the last,
    # leaving only the final invocation's log to diagnose an earlier failure.
    slug = "".join(c if c.isalnum() else "_" for c in "_".join(args))[:40].strip("_")
    log = LOGDIR / (stage.as_posix().replace("/", "__").removesuffix(".py")
                    + (f"__{slug}" if slug else "") + ".log")

    print(f"[run] {stage} {' '.join(args)}")
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        # -u: without it the child buffers stdout when it is not a TTY, so a
        # long stage shows nothing in its log until it finishes.
        # The child's stdout is a pipe, so Python picks the locale encoding for it
        # (cp1252 here) and any stage that prints a character outside it dies with
        # UnicodeEncodeError - only under run_stage, never when run by hand. We
        # already decode the pipe as UTF-8 below, so tell the child to use it too.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen([py, "-u", str(script), *args], cwd=str(REPO),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            sys.stdout.write(line)
        rc = proc.wait()
    dt = time.time() - t0

    after = snapshot(folder)
    # a file this stage wrote: newly created, or its mtime advanced during the run
    written = {k for k, (mt, _) in after.items()
               if k not in before or mt > before[k][0]}
    changed = sorted(k for k in written
                     if k not in before or before[k][1] != after[k][1])
    reproduced = sorted(k for k in written
                        if k in before and before[k][1] == after[k][1])

    entry = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage.as_posix(),
        "args": args,
        "exit_code": rc,
        "seconds": round(dt, 1),
        "python": py,
        "changed": changed,
        "reproduced": reproduced,
        "log": log.relative_to(REPO).as_posix(),
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    print(f"[{'ok' if rc == 0 else 'FAIL'}] {stage} exit={rc} in {dt:.1f}s "
          f"({len(changed)} file(s) changed) -> {log.relative_to(REPO)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
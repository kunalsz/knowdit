#!/usr/bin/env python3
"""Report progress of a C4 dataset build + learn run.

Cross-references two sources:

  * the historical KG database (ground truth for what has actually landed)
  * `<dataset>/manifest.json` + `<dataset>/skipped.txt`, written by
    `scripts/build_c4_dataset.py` (the denominator: what is queued vs skipped)

and prints:

  1. learned / remaining counts, with percentage
  2. learned content volume (findings by severity, semantics, links)
  3. the currently in-flight batch, parsed from the running process
  4. throughput + ETA, from samples this script records over successive runs
  5. the skipped breakdown by reason

The database is NEVER opened in place. A live learn run holds the WAL open, so
this script copies the db (plus the freshest WAL, including one held open as a
deleted inode by the running process) into a temp dir and queries the copy.
Read-only with respect to the run.

Usage:
    scripts/c4_progress.py
    scripts/c4_progress.py --db contest_kg.db --dataset-dir contests
    scripts/c4_progress.py --json
    scripts/c4_progress.py --watch 60          # refresh every 60s
    scripts/c4_progress.py --reset-history     # drop recorded throughput samples
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = "contest_kg.db"
DEFAULT_DATASET_DIR = "contests"

SEVERITIES = ("High", "Medium", "Low")


# ---------------------------------------------------------------------------
# Safe snapshot of a live SQLite database
# ---------------------------------------------------------------------------

def find_holders(db_path: Path) -> list[int]:
    """PIDs currently holding the db (or its WAL/SHM) open."""
    target = db_path.name
    pids: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if target in os.readlink(fd):
                        pids.append(int(proc.name))
                        break
                except OSError:
                    continue
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return sorted(set(pids))


def freshest(paths: list[Path]) -> Path | None:
    best, best_size = None, -1
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > best_size:
            best, best_size = path, size
    return best


def snapshot(db_path: Path, dest: Path) -> Path:
    """Copy db + best WAL into `dest` and return the path to query."""
    snap = dest / "snap.db"
    shutil.copy2(db_path, snap)

    holders = find_holders(db_path)
    wal_candidates = [db_path.with_name(db_path.name + "-wal")]
    for pid in holders:
        try:
            for fd in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    if os.readlink(fd).endswith(db_path.name + "-wal"):
                        wal_candidates.append(fd)
                except OSError:
                    continue
        except (PermissionError, FileNotFoundError):
            continue

    wal = freshest(wal_candidates)
    if wal is not None:
        try:
            shutil.copy2(wal, Path(str(snap) + "-wal"))
        except OSError:
            pass
    return snap


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def read_manifest(dataset_dir: Path) -> dict:
    """Status counts + skipped reasons from the dataset builder's outputs."""
    info = {"present": False, "counts": {}, "total": 0, "skipped": {}}
    manifest = dataset_dir / "manifest.json"
    if manifest.is_file():
        try:
            records = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            records = []
        counts: dict[str, int] = {}
        for record in records:
            status = record.get("status", "?")
            counts[status] = counts.get(status, 0) + 1
        info.update(present=True, counts=counts, total=len(records))
        info["skipped"] = {k: v for k, v in counts.items() if k != "ready"}

    skipped_file = dataset_dir / "skipped.txt"
    if skipped_file.is_file():
        reasons: dict[str, int] = {}
        for line in skipped_file.read_text(errors="replace").splitlines():
            if not line or line.startswith("#") or line.startswith("\t"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                reasons[parts[2]] = reasons.get(parts[2], 0) + 1
        if reasons:
            info["skipped"] = reasons
            info["skipped_file"] = str(skipped_file)
    return info


def db_stats(conn: sqlite3.Connection, dataset_dir: Path) -> dict:
    def scalar(sql: str, default=0):
        try:
            row = conn.execute(sql).fetchone()
            return row[0] if row and row[0] is not None else default
        except sqlite3.Error:
            return default

    stats: dict = {}
    stats["projects"] = scalar("SELECT count(*) FROM project WHERE status='completed'")
    stats["findings"] = scalar("SELECT count(*) FROM audit_finding")
    stats["semantics"] = scalar("SELECT count(*) FROM semantic_node")
    stats["links"] = scalar("SELECT count(*) FROM semantic_finding_link")
    stats["linked_findings"] = scalar("SELECT count(*) FROM finding_link_status")
    stats["checkpoints"] = scalar("SELECT count(*) FROM extraction_chunk")
    stats["in_flight_projects"] = scalar(
        "SELECT count(DISTINCT project_key) FROM extraction_chunk"
    )
    stats["severity"] = {
        sev: scalar(f"SELECT count(*) FROM audit_finding WHERE severity='{sev}'")
        for sev in SEVERITIES
    }
    stats["platform_ids"] = [
        row[0]
        for row in conn.execute(
            "SELECT platform_id FROM project_platform ORDER BY id"
        ).fetchall()
    ]
    return stats


# ---------------------------------------------------------------------------
# Running process
# ---------------------------------------------------------------------------

def process_info() -> dict:
    """The live `build_c4_dataset.py` / `knowdit learn c4` commands, if any."""
    info: dict = {"script": None, "knowdit": None, "knowdit_ids": [], "elapsed": None}
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,etime,cmd"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return info

    for line in out.splitlines():
        if "build_c4_dataset.py" in line and "grep" not in line:
            info["script"] = line.strip()
            match = re.search(r"\b(\d+):(\d{2})(?::(\d{2}))?\s", line)
            if match:
                info["elapsed"] = match.group(0).strip()
        elif "knowdit learn" in line and "c4" in line:
            info["knowdit"] = line.strip()
            ids = re.search(r"--c4-ids\s+([\d,]+)", line)
            if ids:
                info["knowdit_ids"] = ids.group(1).split(",")
    return info


# ---------------------------------------------------------------------------
# Throughput history
# ---------------------------------------------------------------------------

def history_path(db_path: Path, override: Path | None) -> Path:
    return override or db_path.with_suffix(db_path.suffix + ".progress.json")


def update_history(path: Path, completed: int, reset: bool) -> dict:
    """Append a (timestamp, completed) sample; return the derived rate.

    Samples are only useful within one continuous run, so a drop in the
    completed count (a wiped db / fresh start) resets the baseline.
    """
    samples: list[dict] = []
    if path.is_file() and not reset:
        try:
            samples = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            samples = []

    now = time.time()
    if samples and completed < samples[-1].get("completed", 0):
        samples = []  # counter went backwards -> new run
    samples.append({"ts": now, "completed": completed})

    # Keep the baseline sample plus the last few, to bound the file.
    if len(samples) > 50:
        samples = [samples[0]] + samples[-49:]
    try:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    except OSError:
        pass

    result = {"rate": None, "elapsed_seconds": None, "samples": len(samples),
              "delta": 0, "needs_more": len(samples) < 2}
    if len(samples) >= 2:
        first, last = samples[0], samples[-1]
        span = last["ts"] - first["ts"]
        done = last["completed"] - first["completed"]
        result["delta"] = done
        if span > 0:
            result["elapsed_seconds"] = span
            if done > 0:
                result["rate"] = done / (span / 3600.0)  # contests per hour
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def human_hours(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f}m"
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def build_report(db_path: Path, dataset_dir: Path, history_file: Path,
                 reset_history: bool, db_only: bool) -> dict:
    manifest = read_manifest(dataset_dir)
    proc = process_info()

    report: dict = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "db": str(db_path),
        "dataset_dir": str(dataset_dir),
        "manifest": manifest,
        "process": proc,
    }

    if not db_only and (db_path.is_file() or db_path.with_name(db_path.name + "-wal").is_file()):
        with tempfile.TemporaryDirectory(prefix="c4progress-") as tmp:
            snap = snapshot(db_path, Path(tmp))
            conn = sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
            try:
                report["db_stats"] = db_stats(conn, dataset_dir)
            finally:
                conn.close()

    completed = (report.get("db_stats") or {}).get("projects", 0)
    report["throughput"] = update_history(history_file, completed, reset_history)

    ready = manifest.get("counts", {}).get("ready", 0)
    learned_manifest = manifest.get("counts", {}).get("learned", 0)
    # A contest is only counted `learned` in the manifest once its whole batch
    # returns, so `ready` still includes anything mid-batch; the DB count can
    # therefore run slightly ahead. Use the larger for the numerator.
    done = max(learned_manifest, completed)
    report["progress"] = {
        "ready": ready,
        "remaining": ready,
        "learned_db": completed,
        "learned_manifest": learned_manifest,
        "percent": (100.0 * done / (done + ready)) if (done + ready) else 0.0,
    }
    return report


def print_report(report: dict) -> None:
    manifest = report["manifest"]
    proc = report["process"]
    db_stats_ = report.get("db_stats") or {}
    progress = report["progress"]
    throughput = report["throughput"]

    print(f"C4 learn progress — {report['timestamp']}")
    print(f"  db:      {report['db']}")
    print(f"  dataset: {report['dataset_dir']}")

    if proc["script"]:
        elapsed = f" up {proc['elapsed']}" if proc["elapsed"] else ""
        print(f"  RUNNING: build_c4_dataset.py{elapsed}")
    else:
        print("  not running (no build_c4_dataset.py process found)")

    if proc["knowdit_ids"]:
        ids = proc["knowdit_ids"]
        shown = ", ".join(ids[:6]) + (f" … (+{len(ids) - 6})" if len(ids) > 6 else "")
        print(f"  in-flight batch: {len(ids)} contests — {shown}")

    print()
    if manifest["present"]:
        pct = progress["percent"]
        print(f"  learned    {progress['learned_manifest']:>4} / {manifest['total']} reports"
              f"   ({pct:.1f}%)")
        print(f"  ready      {progress['ready']:>4} still to learn")
    else:
        print(f"  manifest not found under {report['dataset_dir']}")

    if db_stats_:
        print()
        print("  content in KG")
        print(f"    projects  {db_stats_.get('projects', 0)}")
        findings = db_stats_.get("findings", 0)
        sev = db_stats_.get("severity", {})
        sev_txt = " / ".join(f"{k} {sev.get(k, 0)}" for k in SEVERITIES)
        print(f"    findings  {findings}   ({sev_txt})")
        print(f"    semantics {db_stats_.get('semantics', 0)}")
        print(f"    links     {db_stats_.get('links', 0)}"
              f"   ({db_stats_.get('linked_findings', 0)} findings through linker)")
        if db_stats_.get("in_flight_projects"):
            print(f"    in-flight checkpoints: {db_stats_['in_flight_projects']} project(s),"
                  f" {db_stats_.get('checkpoints', 0)} chunk(s)")

    print()
    print("  throughput")
    rate = throughput.get("rate")
    if rate:
        eta_hours = progress["remaining"] / rate
        print(f"    {rate:.1f} contests/hour over {human_hours(throughput['elapsed_seconds'] / 3600)}")
        print(f"    ETA ~{human_hours(eta_hours)} for {progress['remaining']} remaining")
        print("    (optimistic — later contests are much larger than the early ones)")
    elif throughput.get("needs_more"):
        print("    first sample recorded; run this again for a rate/ETA")
    else:
        print(f"    no contest completed in the last "
              f"{human_hours((throughput.get('elapsed_seconds') or 0) / 3600)}"
              " — one is in flight; run this again later")

    skipped = manifest.get("skipped", {})
    if skipped:
        print()
        print(f"  skipped ({sum(skipped.values())})")
        for reason in sorted(skipped):
            print(f"    {reason:<24} {skipped[reason]}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="c4_progress.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB),
                        help=f"Historical KG database (default: {DEFAULT_DB}).")
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR),
                        help=f"Dataset dir with manifest.json/skipped.txt "
                             f"(default: {DEFAULT_DATASET_DIR}).")
    parser.add_argument("--history", type=Path, default=None,
                        help="Throughput-sample file (default: <db>.progress.json).")
    parser.add_argument("--reset-history", action="store_true",
                        help="Drop recorded throughput samples before recording this one.")
    parser.add_argument("--db-only", action="store_true",
                        help="Skip the database (manifest + process only).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--watch", type=int, metavar="SECONDS", default=None,
                        help="Refresh every N seconds until interrupted.")
    args = parser.parse_args(argv)

    history_file = history_path(args.db, args.history)

    def once() -> dict:
        report = build_report(args.db, args.dataset_dir, history_file,
                              args.reset_history, args.db_only)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_report(report)
        return report

    if args.watch is None:
        once()
        return 0

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            once()
            time.sleep(max(5, args.watch))
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

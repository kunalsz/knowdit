#!/usr/bin/env python3
"""Report progress of an Immunefi dataset build + learn run.

Cross-references two sources:

  * the historical KG database (ground truth for what has actually landed)
  * `<dataset>/manifest.json` + `<dataset>/skipped_immunefi.txt`, written by
    `scripts/build_immunefi_dataset.py` (the denominator: queued vs skipped)

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

The Immunefi run shares one KG with the C4, Sherlock, Cantina-competition,
Cantina-reviews, and Spearbit runs, and all write `c4-<id>` platform ids. Content
totals are therefore scoped to THIS dataset's ids (taken from its manifest)
rather than reported database-wide, or they would count the other runs' projects
too. Those ids live in the reserved `93000001+` band, so they never collide with
a genuine contest id.

Usage:
    scripts/immunefi_progress.py
    scripts/immunefi_progress.py --db contest_kg.db --dataset-dir immunefi
    scripts/immunefi_progress.py --json
    scripts/immunefi_progress.py --watch 60          # refresh every 60s
    scripts/immunefi_progress.py --reset-history     # drop recorded samples
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
DEFAULT_DATASET_DIR = "immunefi"

SEVERITIES = ("High", "Medium", "Low")

# The builder writes these; `skipped_immunefi.txt` is Immunefi-specific so an
# adjacent C4/Sherlock/Cantina/Spearbit dataset's skipped file is never picked
# up by mistake.
SKIPPED_FILENAME = "skipped_immunefi.txt"


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
    info: dict = {"present": False, "counts": {}, "total": 0, "skipped": {},
                  "ids": [], "ready_ids": []}
    manifest = dataset_dir / "manifest.json"
    if manifest.is_file():
        try:
            records = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            records = []
        counts: dict[str, int] = {}
        ids: list[int] = []
        ready_ids: list[int] = []
        for record in records:
            status = record.get("status", "?")
            counts[status] = counts.get(status, 0) + 1
            rid = record.get("id")
            if isinstance(rid, int) and rid > 0:
                ids.append(rid)
                if status == "ready":
                    ready_ids.append(rid)
        info.update(present=True, counts=counts, total=len(records),
                    ids=ids, ready_ids=ready_ids)
        info["skipped"] = {k: v for k, v in counts.items() if k != "ready"}

    skipped_file = dataset_dir / SKIPPED_FILENAME
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


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def db_stats(conn: sqlite3.Connection, platform_ids: list[str]) -> dict:
    """Content counts SCOPED to this dataset's platform ids.

    The KG is shared with the C4 and Sherlock runs, so an unscoped `count(*)`
    would report all three runs' content. Ids come from the manifest; an empty
    list means the dataset has not written anything yet.
    """
    def scalar(sql: str, params: tuple = (), default=0):
        try:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row and row[0] is not None else default
        except sqlite3.Error:
            return default

    stats: dict = {"scoped": bool(platform_ids), "platform_ids": len(platform_ids)}
    if not platform_ids:
        stats.update(projects=0, findings=0, semantics=0, links=0,
                     linked_findings=0, checkpoints=0, in_flight_projects=0,
                     severity={sev: 0 for sev in SEVERITIES})
        return stats

    # Resolve this dataset's ids -> internal project ids once, then reuse the
    # `IN (...)` list. Chunked to stay under SQLite's parameter ceiling.
    project_ids: list[int] = []
    for chunk in _chunks(platform_ids, 400):
        placeholders = ",".join("?" for _ in chunk)
        try:
            rows = conn.execute(
                f"SELECT project_id FROM project_platform WHERE platform_id IN ({placeholders})",
                chunk,
            ).fetchall()
        except sqlite3.Error:
            rows = []
        project_ids.extend(r[0] for r in rows)

    stats["scoped_projects_found"] = len(project_ids)
    if not project_ids:
        stats.update(projects=0, findings=0, semantics=0, links=0,
                     linked_findings=0, checkpoints=0, in_flight_projects=0,
                     severity={sev: 0 for sev in SEVERITIES})
        return stats

    id_chunks = list(_chunks(project_ids, 400))

    def sum_over(sql_template: str) -> int:
        total = 0
        for chunk in id_chunks:
            placeholders = ",".join("?" for _ in chunk)
            total += scalar(sql_template.format(ph=placeholders), tuple(chunk))
        return total

    stats["projects"] = len(project_ids)
    stats["findings"] = sum_over(
        "SELECT count(*) FROM project_finding WHERE project_id IN ({ph})")
    stats["semantics"] = sum_over(
        "SELECT count(*) FROM project_semantic WHERE project_id IN ({ph})")
    stats["links"] = sum_over(
        "SELECT count(*) FROM semantic_finding_link WHERE audit_finding_id IN "
        "(SELECT audit_finding_id FROM project_finding WHERE project_id IN ({ph}))")
    stats["linked_findings"] = sum_over(
        "SELECT count(*) FROM finding_link_status WHERE audit_finding_id IN "
        "(SELECT audit_finding_id FROM project_finding WHERE project_id IN ({ph}))")

    # Severity mix over this dataset's findings.
    stats["severity"] = {}
    for sev in SEVERITIES:
        total = 0
        for chunk in id_chunks:
            placeholders = ",".join("?" for _ in chunk)
            total += scalar(
                "SELECT count(*) FROM audit_finding WHERE severity=? AND id IN "
                f"(SELECT audit_finding_id FROM project_finding WHERE project_id IN ({placeholders}))",
                (sev, *chunk),
            )
        stats["severity"][sev] = total

    stats["checkpoints"] = scalar("SELECT count(*) FROM extraction_chunk")
    stats["in_flight_projects"] = scalar(
        "SELECT count(DISTINCT project_key) FROM extraction_chunk")
    return stats


# ---------------------------------------------------------------------------
# Running process
# ---------------------------------------------------------------------------

def process_info() -> dict:
    """The live `build_immunefi_dataset.py` / `knowdit learn c4` commands."""
    info: dict = {"script": None, "knowdit": None, "knowdit_ids": [], "elapsed": None}
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,etime,cmd"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return info

    for line in out.splitlines():
        if "build_immunefi_dataset.py" in line and "grep" not in line:
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
    return override or db_path.with_suffix(db_path.suffix + ".immunefi-progress.json")


def update_history(path: Path, completed: int, reset: bool) -> dict:
    """Append a (timestamp, completed) sample; return the derived rate.

    A drop in the completed count (wiped db / fresh start) resets the baseline,
    so a stale sample can't produce a negative or wild rate.
    """
    samples: list[dict] = []
    if path.is_file() and not reset:
        try:
            samples = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            samples = []

    now = time.time()
    if samples and completed < samples[-1].get("completed", 0):
        samples = []
    samples.append({"ts": now, "completed": completed})
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

    if not db_only and (db_path.is_file()
                        or db_path.with_name(db_path.name + "-wal").is_file()):
        platform_ids = [f"c4-{i}" for i in manifest["ids"]]
        with tempfile.TemporaryDirectory(prefix="immunefi-progress-") as tmp:
            snap = snapshot(db_path, Path(tmp))
            conn = sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
            try:
                report["db_stats"] = db_stats(conn, platform_ids)
            finally:
                conn.close()

    completed = (report.get("db_stats") or {}).get("projects", 0)
    report["throughput"] = update_history(history_file, completed, reset_history)

    ready = manifest.get("counts", {}).get("ready", 0)
    learned_manifest = manifest.get("counts", {}).get("learned", 0)
    # The manifest only flips a contest to `learned` once its whole batch
    # returns, so the DB count can run slightly ahead mid-batch; take the larger.
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

    print(f"Immunefi learn progress — {report['timestamp']}")
    print(f"  db:      {report['db']}")
    print(f"  dataset: {report['dataset_dir']}")

    if proc["script"]:
        elapsed = f" up {proc['elapsed']}" if proc["elapsed"] else ""
        print(f"  RUNNING: build_immunefi_dataset.py{elapsed}")
    else:
        print("  not running (no build_immunefi_dataset.py process found)")

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
        scope = ("scoped to this dataset" if db_stats_.get("scoped")
                 else "no ids written yet")
        print(f"  content in KG ({scope})")
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
        print(f"    {rate:.1f} contests/hour over "
              f"{human_hours(throughput['elapsed_seconds'] / 3600)}")
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
        prog="immunefi_progress.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB),
                        help=f"Historical KG database (default: {DEFAULT_DB}).")
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR),
                        help=f"Dataset dir with manifest.json/{SKIPPED_FILENAME} "
                             f"(default: {DEFAULT_DATASET_DIR}).")
    parser.add_argument("--history", type=Path, default=None,
                        help="Throughput-sample file "
                             "(default: <db>.immunefi-progress.json).")
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

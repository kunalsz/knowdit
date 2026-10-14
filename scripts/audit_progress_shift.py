#!/usr/bin/env python3
"""Report progress of a running (or finished) knowdit audit.

Reads the project SQLite database and prints:

  1. matched pairs by strength (High / Medium / Low)
  2. the planned links broken down by (match strength, link strength)
     bucket -- H+H, H+M, M+H, M+M
  3. links and pairs completed so far, with the same bucket breakdown
  4. a rate / ETA estimate from the on-disk link summaries

The database is NEVER opened in place. A live audit holds the WAL open, and
attaching to it directly unlinks the -wal/-shm files underneath the writer, so
this script copies the database (plus the freshest WAL it can find, including
one held open as a deleted inode by the running process) into a temp dir and
queries the copy. Read-only with respect to the audit.

Usage:
    scripts/audit_progress.py
    scripts/audit_progress.py --db path/to/knowdit.sqlite3 --summaries runs/paxos/summaries
    scripts/audit_progress.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_DB = "code4rena/contracts/shift-defi-platform/knowdit.sqlite3"
DEFAULT_SUMMARIES = "runs/shift-defi-platform/summaries"

# The planner defaults to --gen-specs-min-strength Medium and
# --gen-specs-min-link-strength Medium, so only these strengths become links.
KEPT = ("High", "Medium")
BUCKET_ORDER = [("High", "High"), ("High", "Medium"), ("Medium", "High"), ("Medium", "Medium")]


def _fd_target(link: str) -> str:
    """Normalize a /proc/PID/fd readlink result. Deleted inodes (e.g. a WAL
    the writer restarted past) carry a ' (deleted)' suffix; strip it so the
    link still matches the path it was created from."""
    if link.endswith(" (deleted)"):
        link = link[: -len(" (deleted)")]
    return os.path.realpath(link)


def find_holders(db_path: Path) -> list[int]:
    """PIDs that currently hold the database open. Match on the resolved
    full path -- several projects' databases are all named knowdit.sqlite3,
    and a basename match would claim another audit's WAL for this one."""
    target = os.path.realpath(db_path)
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        fddir = proc / "fd"
        try:
            for fd in fddir.iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                if _fd_target(link) == target:
                    pids.append(int(proc.name))
                    break
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return sorted(set(pids))


def freshest(candidates: list[Path]) -> Path | None:
    """Pick the largest readable file among candidates, deduped by target."""
    best, best_size = None, -1
    seen: set[str] = set()
    for c in candidates:
        try:
            size = c.stat().st_size
        except OSError:
            continue
        key = _fd_target(os.readlink(c)) if str(c).startswith("/proc") else os.path.realpath(c)
        if key in seen:
            continue
        seen.add(key)
        if size > best_size:
            best, best_size = c, size
    return best


def snapshot(db_path: Path, dest: Path, with_wal: bool = True) -> tuple[Path, list[str]]:
    """Copy the db (+ best WAL) into dest so we can read it safely."""
    notes: list[str] = []
    snap = dest / "snap.db"
    shutil.copy2(db_path, snap)

    holders = find_holders(db_path)
    if holders:
        notes.append(f"live writer pid(s): {', '.join(map(str, holders))}")

    wal_candidates = [db_path.with_name(db_path.name + "-wal")]
    wal_target = os.path.realpath(db_path.with_name(db_path.name + "-wal"))
    for pid in holders:
        fddir = Path(f"/proc/{pid}/fd")
        try:
            for fd in fddir.iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                if _fd_target(link) == wal_target:
                    wal_candidates.append(fd)
        except (PermissionError, FileNotFoundError):
            continue

    wal = freshest(wal_candidates) if with_wal else None
    if wal is not None:
        try:
            size = wal.stat().st_size
            shutil.copy2(wal, Path(str(snap) + "-wal"))
            notes.append(f"WAL: {wal} ({size} bytes)")
        except OSError as exc:
            notes.append(f"WAL copy failed ({exc}); reading committed state only")
    else:
        notes.append("no WAL found; reading committed state only")

    return snap, notes


def q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def fmt(n: int) -> str:
    return f"{n:,}"


def bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = int(width * done / total)
    filled = max(0, min(width, filled))
    pct = 100.0 * done / total
    return f"[{'#' * filled}{'.' * (width - filled)}] {pct:5.2f}%"


def collect(conn: sqlite3.Connection) -> dict:
    # ---- 1. matched pairs by strength ------------------------------------
    pairs_by_strength: dict[str, int] = {}
    for strength, n in q(
        conn, "SELECT strength, COUNT(*) FROM semantic_matched GROUP BY strength"
    ):
        pairs_by_strength[strength] = n

    # ---- plan: expand pairs x findings, mirroring select_pending ---------
    match_strength = {
        (e, h): s
        for e, h, s in q(conn, "SELECT extract_id, historical_id, strength FROM semantic_matched")
    }
    findings_by_hist: dict[int, list[tuple[int, str]]] = {}
    for h, f, s in q(
        conn,
        "SELECT historical_semantic_id, historical_finding_id, strength "
        "FROM historical_semantic_finding_link",
    ):
        findings_by_hist.setdefault(h, []).append((f, s))

    plan: dict[tuple[int, int, int], tuple[str, str]] = {}
    for (e, h), ms in match_strength.items():
        if ms not in KEPT:
            continue
        for f, ls in findings_by_hist.get(h, ()):
            if ls not in KEPT:
                continue
            plan[(e, h, f)] = (ms, ls)

    # ---- 3. completed links: any link with >=1 code_gen row --------------
    completed = set(
        q(
            conn,
            "SELECT DISTINCT s.semantic_id, s.historical_id, s.finding_id "
            "FROM code_gen c JOIN specification s ON c.spec_id = s.id",
        )
    )
    claimed = set(
        q(
            conn,
            "SELECT DISTINCT semantic_id, historical_id, finding_id FROM specification",
        )
    )

    def bucket_of(key: tuple[int, int, int]) -> tuple[str, str] | None:
        e, h, _ = key
        ms = match_strength.get((e, h))
        if ms is None:
            return None
        for f, ls in findings_by_hist.get(h, ()):
            if f == key[2]:
                return (ms, ls)
        return None

    plan_buckets = {b: 0 for b in BUCKET_ORDER}
    for b in plan.values():
        plan_buckets[b] = plan_buckets.get(b, 0) + 1

    done_in_plan = {k for k in completed if k in plan}
    done_buckets = {b: 0 for b in BUCKET_ORDER}
    done_offplan = 0
    for k in completed:
        b = bucket_of(k)
        if b is None or k not in plan:
            done_offplan += 1
        else:
            done_buckets[b] = done_buckets.get(b, 0) + 1

    plan_pairs = {(e, h) for (e, h, _) in plan}
    done_pairs = {(e, h) for (e, h, _) in done_in_plan}

    # ---- counters --------------------------------------------------------
    counts = {}
    for t in ("specification", "code_gen", "harness_run", "reflection", "valid_finding"):
        try:
            counts[t] = q(conn, f"SELECT COUNT(*) FROM {t}")[0][0]
        except sqlite3.Error:
            counts[t] = 0

    return {
        "pairs_by_strength": pairs_by_strength,
        "plan_total": len(plan),
        "plan_buckets": plan_buckets,
        "plan_pairs": len(plan_pairs),
        "completed_total": len(completed),
        "completed_in_plan": len(done_in_plan),
        "completed_offplan": done_offplan,
        "completed_buckets": done_buckets,
        "completed_pairs": len(done_pairs),
        "claimed_total": len(claimed),
        "counts": counts,
    }


def summarize(summaries_dir: Path) -> dict | None:
    if not summaries_dir.is_dir():
        return None
    rows = []
    for f in summaries_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        d["_mt"] = f.stat().st_mtime
        rows.append(d)
    if not rows:
        return None
    rows.sort(key=lambda r: r["_mt"])

    # Only measure the current invocation: summaries newer than the newest
    # total_links change, i.e. the contiguous tail with the same total_links.
    total_links = rows[-1].get("total_links")
    tail = [r for r in rows if r.get("total_links") == total_links]
    # Trim to the most recent contiguous run of equal total_links.
    run = [tail[-1]]
    for r in reversed(tail[:-1]):
        if r["_mt"] >= run[-1]["_mt"] - 60 * 60 * 6:
            run.append(r)
        else:
            break
    run.sort(key=lambda r: r["_mt"])

    span_min = (run[-1]["_mt"] - run[0]["_mt"]) / 60.0
    n = len(run)
    out = {
        "latched_total_links": total_links,
        "run_links": n,
        "run_span_min": span_min,
        "all_summaries": len(rows),
        "usage": {},
    }
    if n > 1 and span_min > 0:
        out["min_per_link"] = span_min / n
        out["links_per_hour"] = n / span_min * 60
    agg = {}
    for key in ("input_tokens", "output_tokens", "cache_tokens", "reasoning_tokens"):
        agg[key] = sum(
            ((r.get("usage") or {}).get("total") or {}).get(key, 0) for r in run
        )
    out["usage"] = agg
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB, help=f"project sqlite db (default: {DEFAULT_DB})")
    ap.add_argument("--summaries", default=DEFAULT_SUMMARIES, help=f"link summaries dir (default: {DEFAULT_SUMMARIES})")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 2

    last_exc: sqlite3.DatabaseError | None = None
    with tempfile.TemporaryDirectory(prefix="knowdit-progress-") as tmp:
        # A torn copy (checkpoint racing the file copy) is rare; retry, and on
        # the final attempt drop the WAL so we still report committed state.
        for attempt in range(3):
            snap, notes = snapshot(db_path, Path(tmp), with_wal=attempt < 2)
            conn = sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
            try:
                integrity = q(conn, "PRAGMA quick_check")[0][0]
                data = collect(conn)
                break
            except sqlite3.DatabaseError as exc:
                last_exc = exc
                continue
            finally:
                conn.close()
        else:
            print(f"error: could not read snapshot ({last_exc})", file=sys.stderr)
            print("hint: the audit may be mid-write; retry in a moment", file=sys.stderr)
            return 3

    stats = summarize(Path(args.summaries))
    data["snapshot_notes"] = notes
    data["integrity"] = integrity
    data["summary"] = stats

    if args.json:
        # Bucket keys are (match, link) tuples; JSON needs string keys.
        data["plan_buckets"] = {
            f"{k[0][0]}+{k[1][0]}": v for k, v in data["plan_buckets"].items()
        }
        data["completed_buckets"] = {
            f"{k[0][0]}+{k[1][0]}": v for k, v in data["completed_buckets"].items()
        }
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    pbs = data["pairs_by_strength"]
    plan_total = data["plan_total"]
    done = data["completed_in_plan"]

    print()
    print("=" * 64)
    print("  KNOWDIT AUDIT PROGRESS")
    print("=" * 64)
    print(f"  db: {db_path}")
    for note in notes:
        print(f"      {note}")
    print(f"      integrity: {integrity}")

    print()
    print("  MATCHED PAIRS  (one project semantic <-> one historical pattern)")
    print("  " + "-" * 60)
    for s in ("High", "Medium", "Low"):
        n = pbs.get(s, 0)
        mark = "  (planned)" if s in KEPT else "  (filtered out)"
        print(f"    {s:<8} {fmt(n):>8}{mark}")
    print(f"    {'TOTAL':<8} {fmt(sum(pbs.values())):>8}")

    print()
    print("  PLANNED LINKS  (pairs x attached findings; one link = one agent run)")
    print("  " + "-" * 60)
    print(f"    {'bucket':<10} {'links':>8}   {'done':>7}   progress")
    for b in BUCKET_ORDER:
        total_b = data["plan_buckets"].get(b, 0)
        done_b = data["completed_buckets"].get(b, 0)
        label = f"{b[0][0]}+{b[1][0]}"
        print(f"    {label:<10} {fmt(total_b):>8}   {fmt(done_b):>7}   {bar(done_b, total_b)}")
    print(f"    {'-' * 58}")
    print(f"    {'TOTAL':<10} {fmt(plan_total):>8}   {fmt(done):>7}   {bar(done, plan_total)}")
    print(f"    distinct (extract,historical) pairs in plan: {fmt(data['plan_pairs'])}")
    if stats and stats.get("latched_total_links"):
        print(f"    total_links latched by the running process: {fmt(stats['latched_total_links'])}")

    print()
    print("  COMPLETED")
    print("  " + "-" * 60)
    print(f"    links done (in plan)      {fmt(done)} / {fmt(plan_total)}")
    print(f"    links done (total)        {fmt(data['completed_total'])}")
    if data["completed_offplan"]:
        print(f"      of which not in plan    {fmt(data['completed_offplan'])}")
    print(f"    pairs done (>=1 link)     {fmt(data['completed_pairs'])} / {fmt(data['plan_pairs'])}")
    print(f"    links claimed (>=1 spec)  {fmt(data['claimed_total'])}")

    c = data["counts"]
    print()
    print("  ROWS")
    print("  " + "-" * 60)
    print(f"    specifications {fmt(c.get('specification', 0)):>8}    code_gen {fmt(c.get('code_gen', 0)):>8}")
    print(f"    harness_runs   {fmt(c.get('harness_run', 0)):>8}    reflections {fmt(c.get('reflection', 0)):>8}")
    print(f"    valid_findings {fmt(c.get('valid_finding', 0)):>8}")

    if stats and stats.get("min_per_link"):
        rem = plan_total - done
        mpl = stats["min_per_link"]
        print()
        print("  RATE / ETA  (from on-disk summaries, current invocation)")
        print("  " + "-" * 60)
        print(f"    {stats['run_links']} links in {stats['run_span_min']:.0f} min "
              f"= {mpl:.2f} min/link = {stats['links_per_hour']:.1f} links/h")
        u = stats.get("usage", {})
        if u.get("output_tokens"):
            print(f"    output {fmt(u['output_tokens'] // max(1, stats['run_links']))} tok/link, "
                  f"cache hit {100 * u.get('cache_tokens', 0) / max(1, u.get('input_tokens', 1)):.1f}%")
        print(f"    remaining {fmt(rem)} links -> ETA {rem * mpl / 60 / 24:.1f} days "
              f"at this rate")
        print("    (rate reflects the strongest bucket first; re-measure after the")
        print("     planner moves into the M+* buckets)")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

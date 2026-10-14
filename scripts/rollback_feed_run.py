#!/usr/bin/env python3
"""Scoped rollback of the last feed ingestion run (the hacken reports feed).

The hacken run was the newest ingestion, so all of its rows are the highest
allocated ids. Verified invariants (asserted before any delete):
  - every project   with id >= FIRST_PROJECT belongs to the run (no other feed
    project sits above the boundary)
  - no semantic_merge / finding_merge edge points from an old row into a new row
  - no old project's provenance references a new node/finding
so deleting by id boundary cannot orphan any pre-existing row.

Residual (unavoidable without a pre-run snapshot): canonicals owned by older
runs that this run merged into keep the generalization text the merge wrote onto
them. The original text is not stored anywhere, so it cannot be restored.
"""
import argparse
import sqlite3
import sys

FIRST_PROJECT = 1416   # first project admitted by the hacken feed
FIRST_SEMANTIC = 7217  # first semantic_node created by the hacken feed
FIRST_FINDING = 7482   # first audit_finding created by the hacken feed


def assert_safe(cur):
    q = lambda sql: cur.execute(sql).fetchone()[0]
    checks = {
        "non-hacken project above boundary": "SELECT COUNT(*) FROM project WHERE id>=%d AND id NOT IN (SELECT project_id FROM feed_report_source)" % FIRST_PROJECT,
        "old->new semantic_merge edge": "SELECT COUNT(*) FROM semantic_merge WHERE from_semantic_id<%d AND to_semantic_id>=%d" % (FIRST_SEMANTIC, FIRST_SEMANTIC),
        "old->new finding_merge edge": "SELECT COUNT(*) FROM finding_merge WHERE from_finding_id<%d AND to_finding_id>=%d" % (FIRST_FINDING, FIRST_FINDING),
        "old project -> new semantic": "SELECT COUNT(*) FROM project_semantic WHERE project_id<%d AND semantic_node_id>=%d" % (FIRST_PROJECT, FIRST_SEMANTIC),
        "old project -> new finding": "SELECT COUNT(*) FROM project_finding WHERE project_id<%d AND audit_finding_id>=%d" % (FIRST_PROJECT, FIRST_FINDING),
    }
    bad = {k: v for k, v in checks.items() if q(checks[k]) != 0}
    if bad:
        raise SystemExit(f"refusing to delete: boundary invariant violated: {bad}")
    print("boundary invariants OK")


def counts(cur):
    q = lambda sql: cur.execute(sql).fetchone()[0]
    p = FIRST_PROJECT
    s = FIRST_SEMANTIC
    f = FIRST_FINDING
    return {
        "project": f"SELECT COUNT(*) FROM project WHERE id>={p}",
        "project_platform": f"SELECT COUNT(*) FROM project_platform WHERE project_id>={p}",
        "project_category": f"SELECT COUNT(*) FROM project_category WHERE project_id>={p}",
        "project_semantic": f"SELECT COUNT(*) FROM project_semantic WHERE project_id>={p}",
        "project_finding": f"SELECT COUNT(*) FROM project_finding WHERE project_id>={p}",
        "feed_report_source": f"SELECT COUNT(*) FROM feed_report_source WHERE project_id>={p}",
        "semantic_node": f"SELECT COUNT(*) FROM semantic_node WHERE id>={s}",
        "semantic_function": f"SELECT COUNT(*) FROM semantic_function WHERE semantic_node_id>={s}",
        "semantic_node_category": f"SELECT COUNT(*) FROM semantic_node_category WHERE semantic_node_id>={s}",
        "audit_finding": f"SELECT COUNT(*) FROM audit_finding WHERE id>={f}",
        "audit_finding_category": f"SELECT COUNT(*) FROM audit_finding_category WHERE audit_finding_id>={f}",
        "semantic_merge": f"SELECT COUNT(*) FROM semantic_merge WHERE from_semantic_id>={s} OR to_semantic_id>={s}",
        "finding_merge": f"SELECT COUNT(*) FROM finding_merge WHERE from_finding_id>={f} OR to_finding_id>={f}",
        "semantic_finding_link": f"SELECT COUNT(*) FROM semantic_finding_link WHERE semantic_node_id>={s} OR audit_finding_id>={f}",
        "finding_link_status": f"SELECT COUNT(*) FROM finding_link_status WHERE audit_finding_id>={f}",
        "pending_semantic": f"SELECT COUNT(*) FROM pending_semantic WHERE semantic_node_id>={s}",
        "extraction_chunk": f"SELECT COUNT(*) FROM extraction_chunk WHERE project_key IN (SELECT platform_id FROM project_platform WHERE project_id>={p})",
    }


def apply(cur):
    p, s, f = FIRST_PROJECT, FIRST_SEMANTIC, FIRST_FINDING
    cur.execute(f"DELETE FROM semantic_finding_link WHERE semantic_node_id>={s} OR audit_finding_id>={f}")
    cur.execute(f"DELETE FROM finding_link_status WHERE audit_finding_id>={f}")
    cur.execute(f"DELETE FROM pending_semantic WHERE semantic_node_id>={s}")
    cur.execute(f"DELETE FROM semantic_function WHERE semantic_node_id>={s}")
    cur.execute(f"DELETE FROM semantic_node_category WHERE semantic_node_id>={s}")
    cur.execute(f"DELETE FROM audit_finding_category WHERE audit_finding_id>={f}")
    cur.execute(f"DELETE FROM semantic_merge WHERE from_semantic_id>={s} OR to_semantic_id>={s}")
    cur.execute(f"DELETE FROM finding_merge WHERE from_finding_id>={f} OR to_finding_id>={f}")
    cur.execute(f"DELETE FROM project_semantic WHERE project_id>={p}")
    cur.execute(f"DELETE FROM project_finding WHERE project_id>={p}")
    cur.execute(f"DELETE FROM project_category WHERE project_id>={p}")
    cur.execute(f"DELETE FROM extraction_chunk WHERE project_key IN (SELECT platform_id FROM project_platform WHERE project_id>={p})")
    cur.execute(f"DELETE FROM project_platform WHERE project_id>={p}")
    cur.execute(f"DELETE FROM feed_report_source WHERE project_id>={p}")
    cur.execute(f"DELETE FROM project WHERE id>={p}")
    cur.execute(f"DELETE FROM semantic_node WHERE id>={s}")
    cur.execute(f"DELETE FROM audit_finding WHERE id>={f}")

    # Remap residue: a run's in-project synthetic link can be moved onto a
    # pre-existing canonical pair, leaving a fabricated "in-project" edge whose
    # both endpoints predate the run. Such an edge has no project that actually
    # contains both endpoints, so drop it.
    cur.execute(f"""DELETE FROM semantic_finding_link
        WHERE semantic_node_id<{s} AND audit_finding_id<{f}
          AND evidence LIKE '%against the same project%'
          AND NOT EXISTS (
            SELECT 1 FROM project_semantic ps
            JOIN project_finding pf ON pf.project_id = ps.project_id
            WHERE ps.semantic_node_id = semantic_finding_link.semantic_node_id
              AND pf.audit_finding_id = semantic_finding_link.audit_finding_id)""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=OFF")
    assert_safe(cur)

    print("=== rows to delete ===")
    total = 0
    for k, sql in counts(cur).items():
        v = cur.execute(sql).fetchone()[0]
        print(f"  {k:26s} {v:>8d}")
        total += v
    print(f"  {'TOTAL':26s} {total:>8d}")

    if args.dry_run:
        print("\n[dry-run] no changes made")
        return

    cur.execute("BEGIN")
    apply(cur)
    conn.commit()
    cur.execute("PRAGMA foreign_keys=ON")
    print("\nrollback committed. leftover rows:")
    left = {k: cur.execute(sql).fetchone()[0] for k, sql in counts(cur).items()}
    print(" ", {k: v for k, v in left.items() if v})
    print("  integrity:", cur.execute("PRAGMA integrity_check").fetchone()[0])


if __name__ == "__main__":
    sys.exit(main())

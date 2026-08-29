#!/usr/bin/env python3
"""Offline merge cost report.

This intentionally evaluates prompt construction, not model decisions. It
compares the legacy unbounded historical-child rendering with the production
bounded rendering while keeping the canonical candidate set identical. LLM
quality still requires a replay against recorded prompts or a live model.
"""

import argparse
import json
import sqlite3
from collections import defaultdict


def bounded(value, cap):
    if len(value) <= cap:
        return value
    return value[:cap] + "...[truncated]"


def semantic_text(row, children, compact, cap, variants):
    ident, name, definition, description, category = row
    text = (
        f"ID: {ident}\nCategory: {category}\nName: {name}\n"
        f"Definition: {definition}\nDescription: {description}\n"
    )
    if children:
        text += "Merged from (historical raws):\n"
        for child in (children[:variants] if compact else children):
            text += (
                f"  - Name: {child[1]}\n    Definition: "
                f"{bounded(child[2], cap) if compact else child[2]}\n"
                f"    Description: {bounded(child[3], cap) if compact else child[3]}\n"
            )
    return text + "\n"


def finding_text(row, children, compact, cap, variants):
    ident, title, severity, root, description, patterns, exploits = row
    text = (
        f"ID: {ident}\nSeverity: {severity}\nCategory: category\n"
        f"Subcategory: subcategory\nTitle: {title}\nRoot Cause: {root}\n"
        f"Description: {description}\nPatterns: {patterns}\nExploits: {exploits}\n"
    )
    if children:
        text += "Merged from (historical raws):\n"
        for child in (children[:variants] if compact else children):
            text += (
                f"  - Title: {child[1]}\n    Severity: {child[2]}\n"
                f"    Root Cause: {bounded(child[3], cap) if compact else child[3]}\n"
                f"    Description: {bounded(child[4], cap) if compact else child[4]}\n"
                f"    Patterns: {bounded(child[5], cap) if compact else child[5]}\n"
                f"    Exploits: {bounded(child[6], cap) if compact else child[6]}\n"
            )
    return text + "\n"


def corpus_report(kg_path, cap, variants):
    db = sqlite3.connect(kg_path)
    semantics = {
        row[0]: row
        for row in db.execute(
            "select id,name,definition,description,category from semantic_node"
        )
    }
    semantic_children = defaultdict(list)
    for source, target in db.execute(
        "select from_semantic_id,to_semantic_id from semantic_merge "
        "order by to_semantic_id,from_semantic_id"
    ):
        if source in semantics and target in semantics:
            semantic_children[target].append(semantics[source])
    semantic_canonicals = [
        row
        for row in semantics.values()
        if not db.execute(
            "select 1 from semantic_merge where from_semantic_id=? limit 1", (row[0],)
        ).fetchone()
    ]

    findings = {
        row[0]: row
        for row in db.execute(
            "select id,title,severity,root_cause,description,patterns,exploits "
            "from audit_finding"
        )
    }
    finding_children = defaultdict(list)
    for source, target in db.execute(
        "select from_finding_id,to_finding_id from finding_merge "
        "order by to_finding_id,from_finding_id"
    ):
        if source in findings and target in findings:
            finding_children[target].append(findings[source])
    finding_canonicals = [
        row
        for row in findings.values()
        if not db.execute(
            "select 1 from finding_merge where from_finding_id=? limit 1", (row[0],)
        ).fetchone()
    ]

    def sizes(rows, children, render):
        full = sum(
            len(render(row, children[row[0]], False, cap, variants)) for row in rows
        )
        compact = sum(
            len(render(row, children[row[0]], True, cap, variants)) for row in rows
        )
        return full, compact, sum(bool(children[row[0]]) for row in rows)

    sem_full, sem_compact, sem_with_children = sizes(
        semantic_canonicals, semantic_children, semantic_text
    )
    finding_full, finding_compact, finding_with_children = sizes(
        finding_canonicals, finding_children, finding_text
    )
    return {
        "semantic": {
            "canonical_candidates": len(semantic_canonicals),
            "canonical_with_children": sem_with_children,
            "legacy_chars": sem_full,
            "bounded_chars": sem_compact,
            "reduction_percent": 100 * (sem_full - sem_compact) / sem_full
            if sem_full
            else 0,
        },
        "finding": {
            "canonical_candidates": len(finding_canonicals),
            "canonical_with_children": finding_with_children,
            "legacy_chars": finding_full,
            "bounded_chars": finding_compact,
            "reduction_percent": 100 * (finding_full - finding_compact) / finding_full
            if finding_full
            else 0,
        },
        "candidate_set_preserved": True,
    }


def llm_report(debug_path):
    db = sqlite3.connect(debug_path)
    rows = db.execute(
        "select cache_key,input_without_cached_tokens,cached_tokens,"
        "output_without_reasoning_tokens,reasoning_tokens,current_usage_usd "
        "from llm_debug where cache_key like '%-merge%'"
    ).fetchall()
    by_stage = {"semantic_merge": [], "finding_merge": []}
    for row in rows:
        stage = "finding_merge" if "finding-merge" in (row[0] or "") else "semantic_merge"
        by_stage[stage].append(row)
    result = {}
    for stage, values in by_stage.items():
        result[stage] = {
            "requests": len(values),
            "input_tokens": sum((v[1] or 0) + (v[2] or 0) for v in values),
            "cached_tokens": sum(v[2] or 0 for v in values),
            "output_tokens": sum(v[3] or 0 for v in values),
            "reasoning_tokens": sum(v[4] or 0 for v in values),
            "total_usd_at_run_end": max((v[5] or 0) for v in values),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", required=True, help="KG SQLite database")
    parser.add_argument("--llm-debug", help="llm_debug SQLite database")
    parser.add_argument("--raw-child-char-cap", type=int, default=1200)
    parser.add_argument("--raw-child-variant-cap", type=int, default=8)
    args = parser.parse_args()
    report = {
        "rendering": corpus_report(
            args.kg, args.raw_child_char_cap, args.raw_child_variant_cap
        ),
        "quality_evaluation": {
            "status": "not_run",
            "reason": "offline report does not make LLM merge decisions",
        },
    }
    if args.llm_debug:
        report["baseline_llm"] = llm_report(args.llm_debug)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

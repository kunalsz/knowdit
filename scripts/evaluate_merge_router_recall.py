#!/usr/bin/env python3
"""Replay the merge candidate router against historical merge edges.

Historical merge edges are treated as positive labels. The evaluator checks
whether the router's selected candidate subset still contains the final
canonical target. It does not call an LLM; this is a deterministic recall and
prompt-size safety check over the persisted KG.
"""

import argparse
import json
import sqlite3
from collections import defaultdict


TOP_K = 64
MIN_SCORE = 3
MIN_MARGIN = 1
CHILD_VARIANTS = 8
CHILD_CHARS = 1200
STOP = {
    "the", "and", "for", "with", "from", "that", "this", "into", "when",
    "then", "than", "can", "has", "have", "are", "not", "only", "same",
    "also", "new", "one", "all", "any", "its", "their", "during", "after",
    "before",
}


def terms(text):
    current = []
    out = set()
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif len(current) >= 3:
            word = "".join(current)
            if word not in STOP:
                out.add(word)
            current = []
        else:
            current = []
    if len(current) >= 3:
        word = "".join(current)
        if word not in STOP:
            out.add(word)
    return out


def bounded(value):
    value = value or ""
    return value if len(value) <= CHILD_CHARS else value[:CHILD_CHARS] + "...[truncated]"


def final_target(source, direct):
    seen = set()
    while source in direct and source not in seen:
        seen.add(source)
        source = direct[source]
    return source


def route(candidates, query):
    if len(candidates) <= TOP_K:
        return candidates, False
    query_terms = terms(query)
    scored = sorted(
        ((len(query_terms & candidate["terms"]), index)
         for index, candidate in enumerate(candidates)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    best = scored[0][0] if scored else 0
    second = scored[1][0] if len(scored) > 1 else 0
    if best < MIN_SCORE or best - second < MIN_MARGIN:
        return candidates, False
    return [candidates[index] for _, index in scored[:TOP_K]], True


def evaluate(kind, rows, edges, category_scoped, secondary_categories=None):
    by_id = {row[0]: row for row in rows}
    children = defaultdict(list)
    for source, target in edges:
        if source in by_id and target in by_id:
            children[target].append(source)
    direct = {source: target for source, target in edges}
    active = [row for row in rows if row[0] not in direct]
    active_by_category = defaultdict(list)
    if kind == "semantic":
        secondary = secondary_categories or defaultdict(set)
        for row in active:
            categories = {row[4]} | secondary[row[0]]
            for category in categories:
                active_by_category[category].append(row)
    else:
        active_by_category[None] = active

    candidate_by_id = {}
    for row in active:
        ident = row[0]
        child_rows = [by_id[source] for source in children[ident] if source in by_id]
        if kind == "semantic":
            text = f"{row[4]} {row[1]} {row[2]} {row[3]}"
            for child in child_rows:
                text += f" {child[1]} {child[2]} {child[3]}"
            rendered = (
                f"ID: {row[0]}\nCategory: {row[4]}\nName: {row[1]}\n"
                f"Definition: {row[2]}\nDescription: {row[3]}\n"
            )
            for child in child_rows[:CHILD_VARIANTS]:
                rendered += (
                    f"  - Name: {child[1]}\n    Definition: {bounded(child[2])}\n"
                    f"    Description: {bounded(child[3])}\n"
                )
        else:
            text = f"{row[1]} {row[2]} {row[3]} {row[4]} {row[5]} {row[6]}"
            for child in child_rows:
                text += " " + " ".join(str(value or "") for value in child[1:])
            rendered = (
                f"ID: {row[0]}\nSeverity: {row[2]}\nTitle: {row[1]}\n"
                f"Root Cause: {row[3]}\nDescription: {row[4]}\n"
                f"Patterns: {row[5]}\nExploits: {row[6]}\n"
            )
            for child in child_rows[:CHILD_VARIANTS]:
                rendered += (
                    f"  - Title: {child[1]}\n    Severity: {child[2]}\n"
                    f"    Root Cause: {bounded(child[3])}\n"
                    f"    Description: {bounded(child[4])}\n"
                    f"    Patterns: {bounded(child[5])}\n"
                    f"    Exploits: {bounded(child[6])}\n"
                )
        candidate_by_id[ident] = {
            "id": ident,
            "category": row[4] if kind == "semantic" else None,
            "terms": terms(text),
            "chars": len(rendered),
        }

    stats = {
        "edges": len(edges),
        "positive_edges_with_active_target": 0,
        "positive_edges_with_target_in_pool": 0,
        "target_absent_from_pool": 0,
        "routed": 0,
        "fallback": 0,
        "target_hits": 0,
        "target_misses": [],
        "candidate_count_full": 0,
        "candidate_count_selected": 0,
        "candidate_chars_full": 0,
        "candidate_chars_selected": 0,
    }
    for source, _ in edges:
        if source not in by_id:
            continue
        target = final_target(source, direct)
        if target not in candidate_by_id:
            continue
        stats["positive_edges_with_active_target"] += 1
        category = by_id[source][4] if kind == "semantic" else None
        pool_rows = active_by_category[category] if category_scoped else active
        pool = [candidate_by_id[row[0]] for row in pool_rows]
        if target not in {candidate["id"] for candidate in pool}:
            stats["target_absent_from_pool"] += 1
            continue
        stats["positive_edges_with_target_in_pool"] += 1
        query = " ".join(str(value or "") for value in by_id[source][1:])
        selected, was_routed = route(pool, query)
        stats["routed" if was_routed else "fallback"] += 1
        stats["candidate_count_full"] += len(pool)
        stats["candidate_count_selected"] += len(selected)
        stats["candidate_chars_full"] += sum(candidate["chars"] for candidate in pool)
        stats["candidate_chars_selected"] += sum(candidate["chars"] for candidate in selected)
        if any(candidate["id"] == target for candidate in selected):
            stats["target_hits"] += 1
        else:
            stats["target_misses"].append({"source": source, "target": target})

    evaluated = stats["positive_edges_with_target_in_pool"]
    stats["recall"] = stats["target_hits"] / evaluated if evaluated else None
    stats["route_rate"] = stats["routed"] / evaluated if evaluated else None
    stats["candidate_reduction_percent"] = (
        100 * (1 - stats["candidate_count_selected"] / stats["candidate_count_full"])
        if stats["candidate_count_full"] else 0
    )
    stats["char_reduction_percent"] = (
        100 * (1 - stats["candidate_chars_selected"] / stats["candidate_chars_full"])
        if stats["candidate_chars_full"] else 0
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", required=True, help="KG SQLite database")
    args = parser.parse_args()
    db = sqlite3.connect(args.kg)
    semantics = db.execute(
        "select id,name,definition,description,category from semantic_node"
    ).fetchall()
    semantic_edges = db.execute(
        "select from_semantic_id,to_semantic_id from semantic_merge"
    ).fetchall()
    findings = db.execute(
        "select id,title,severity,root_cause,description,patterns,exploits from audit_finding"
    ).fetchall()
    finding_edges = db.execute(
        "select from_finding_id,to_finding_id from finding_merge"
    ).fetchall()
    secondary = defaultdict(set)
    try:
        category_names = dict(db.execute("select id,name from category").fetchall())
        for node_id, category_id in db.execute(
            "select semantic_node_id,category_id from semantic_node_category"
        ):
            if category_id in category_names:
                secondary[node_id].add(category_names[category_id])
    except sqlite3.OperationalError:
        pass
    report = {
        "router": {"top_k": TOP_K, "min_score": MIN_SCORE, "min_margin": MIN_MARGIN},
        "semantic": {
            "all_active": evaluate("semantic", semantics, semantic_edges, False),
            "category_scoped": evaluate(
                "semantic", semantics, semantic_edges, True, secondary
            ),
        },
        "finding": {
            "all_active": evaluate("finding", findings, finding_edges, False),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

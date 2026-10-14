#!/usr/bin/env python3
"""Compare recorded merge runs before and after near-duplicate rescue.

The debug database contains both the original request and cached/retry rows,
so rows are deduplicated by cache key and normalized tool-call arguments. The
evaluator reports deterministic decision deltas and cost/latency proxies. A
gold JSON file can optionally label rescued decisions for precision scoring.
It never calls an LLM or mutates either database.
"""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    db = sqlite3.connect(path)
    return db.execute(
        "select id,cache_key,raw_resp,input_without_cached_tokens,cached_tokens,"
        "output_without_reasoning_tokens,reasoning_tokens,current_usage_usd,timestamp "
        "from llm_debug order by id"
    ).fetchall()


def stage(cache_key):
    key = cache_key or ""
    if "finding-merge" in key:
        return "finding_merge"
    if "near-duplicate-review" in key:
        return "near_duplicate_review"
    if "-merge-" in key or key.endswith("-merge"):
        return "semantic_merge"
    return None


def tool_arguments(value):
    """Yield decoded tool arguments from an llmy OpenAI response."""
    if not value:
        return
    try:
        response = json.loads(value)
    except (TypeError, ValueError):
        return
    for choice in response.get("choices", []):
        message = choice.get("message", {})
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            name = function.get("name", "")
            if name not in {
                "emit_semantic_merge_decision",
                "emit_finding_merge_decision",
            }:
                continue
            try:
                args = json.loads(function.get("arguments", "{}"))
            except (TypeError, ValueError):
                continue
            if isinstance(args, dict):
                yield name, args


def parse_debug(path):
    decisions = {"semantic": {}, "finding": {}}
    by_stage = defaultdict(lambda: {"rows": 0, "requests": 0, "input": 0,
                                    "cached": 0, "output": 0, "reasoning": 0,
                                    "timestamps": [], "cost_max": 0.0,
                                    "cost_spend_proxy": 0.0})
    seen_rows = set()
    debug_rows = rows(path)
    previous_cost = 0.0
    for row in debug_rows:
        ident, cache_key, raw_resp, input_tokens, cached, output, reasoning, cost, timestamp = row
        current_stage = stage(cache_key)
        row_cost = max(0.0, (cost or 0.0) - previous_cost)
        previous_cost = max(previous_cost, cost or 0.0)
        if current_stage is None:
            continue
        stats = by_stage[current_stage]
        stats["rows"] += 1
        stats["input"] += input_tokens or 0
        stats["cached"] += cached or 0
        stats["output"] += output or 0
        stats["reasoning"] += reasoning or 0
        stats["timestamps"].append(timestamp)
        stats["cost_max"] = max(stats["cost_max"], cost or 0.0)
        stats["cost_spend_proxy"] += row_cost

        calls = list(tool_arguments(raw_resp))
        # A retry/cache hit can contain the same response. Keep the first
        # normalized response for decision counts while retaining all usage.
        normalized = json.dumps(calls, sort_keys=True, separators=(",", ":"))
        request_key = (cache_key, normalized)
        if not calls or request_key in seen_rows:
            continue
        seen_rows.add(request_key)
        stats["requests"] += 1
        for function_name, args in calls:
            if function_name == "emit_semantic_merge_decision":
                kind = "semantic"
                name = args.get("new_semantic_name")
            else:
                kind = "finding"
                name = args.get("new_finding_title")
            if not name:
                continue
            key = str(name).strip().lower()
            target_field = args.get("merge_target_ids", [])
            targets = {int(value) for value in target_field if isinstance(value, int)}
            item = decisions[kind].setdefault(
                key, {"name": str(name), "targets": set(), "stages": set()}
            )
            item["targets"].update(targets)
            item["stages"].add(current_stage)

    merge_cost = 0.0
    for stats in by_stage.values():
        timestamps = stats.pop("timestamps")
        stats["time_span_seconds"] = max(timestamps) - min(timestamps) if timestamps else 0
        stats["cost_max"] = round(stats["cost_max"], 8)
        stats["cost_spend_proxy"] = round(stats["cost_spend_proxy"], 8)
        merge_cost += stats["cost_spend_proxy"]
    for kind in decisions:
        for item in decisions[kind].values():
            item["targets"] = sorted(item["targets"])
            item["stages"] = sorted(item["stages"])
    return {
        "decisions": decisions,
        "stages": dict(by_stage),
        "merge_cost_usd_proxy": round(merge_cost, 8),
        "total_cost_usd_proxy": max((row[7] or 0.0 for row in debug_rows), default=0.0),
    }


def kg_stats(path):
    db = sqlite3.connect(path)
    semantic_total = db.execute("select count(*) from semantic_node").fetchone()[0]
    semantic_edges = db.execute("select count(*) from semantic_merge").fetchone()[0]
    finding_total = db.execute("select count(*) from audit_finding").fetchone()[0]
    finding_edges = db.execute("select count(*) from finding_merge").fetchone()[0]
    semantic_active = db.execute(
        "select count(*) from semantic_node n where not exists "
        "(select 1 from semantic_merge m where m.from_semantic_id=n.id)"
    ).fetchone()[0]
    finding_active = db.execute(
        "select count(*) from audit_finding f where not exists "
        "(select 1 from finding_merge m where m.from_finding_id=f.id)"
    ).fetchone()[0]
    strength = Counter(row[0] for row in db.execute(
        "select strength from semantic_finding_link"
    ))
    return {
        "semantic_nodes": semantic_total,
        "semantic_active_canonicals": semantic_active,
        "semantic_merge_edges": semantic_edges,
        "finding_nodes": finding_total,
        "finding_active_canonicals": finding_active,
        "finding_merge_edges": finding_edges,
        "semantic_finding_links_by_strength": dict(sorted(strength.items())),
    }


def delta(before, after):
    result = {}
    for kind in ("semantic", "finding"):
        before_items = before["decisions"][kind]
        after_items = after["decisions"][kind]
        common = sorted(set(before_items) & set(after_items))
        before_new = {key for key in common if not before_items[key]["targets"]}
        after_new = {key for key in common if not after_items[key]["targets"]}
        rescued = sorted(key for key in common if not before_items[key]["targets"]
                         and after_items[key]["targets"])
        lost = sorted(key for key in common if before_items[key]["targets"]
                      and not after_items[key]["targets"])
        target_changed = sorted(key for key in common
                                if before_items[key]["targets"] != after_items[key]["targets"])
        before_target_signatures = {
            tuple(item["targets"])
            for item in before_items.values()
            if item["targets"]
        }
        after_target_signatures = {
            tuple(item["targets"])
            for item in after_items.values()
            if item["targets"]
        }
        target_signature_overlap = before_target_signatures & after_target_signatures
        result[kind] = {
            "before_decisions": len(before_items),
            "after_decisions": len(after_items),
            "before_merge_decisions": sum(bool(item["targets"]) for item in before_items.values()),
            "after_merge_decisions": sum(bool(item["targets"]) for item in after_items.values()),
            "before_new_decisions": sum(not item["targets"] for item in before_items.values()),
            "after_new_decisions": sum(not item["targets"] for item in after_items.values()),
            "common_decisions": len(common),
            "before_new_common": len(before_new),
            "after_new_common": len(after_new),
            "new_reduction_percent": (100 * (len(before_new) - len(after_new)) / len(before_new)
                                       if before_new else 0),
            "rescued_merge_count": len(rescued),
            "rescued_names": [after_items[key]["name"] for key in rescued],
            "lost_merge_count": len(lost),
            "target_changed_count": len(target_changed),
            "before_target_signatures": len(before_target_signatures),
            "after_target_signatures": len(after_target_signatures),
            "target_signature_overlap": len(target_signature_overlap),
            "target_signature_recall": (
                len(target_signature_overlap) / len(before_target_signatures)
                if before_target_signatures else None
            ),
        }
    return result


def gate_metrics(before, after, differences, gold):
    before_cost = before.get("merge_cost_usd_proxy", before.get("total_cost_usd_proxy", 0.0))
    after_cost = after.get("merge_cost_usd_proxy", after.get("total_cost_usd_proxy", 0.0))
    before_span = sum(item["time_span_seconds"] for item in before["stages"].values())
    after_span = sum(item["time_span_seconds"] for item in after["stages"].values())
    semantic = differences["semantic"]
    precision = None
    precision_labeled = 0
    if gold:
        labels = {str(key).strip().lower(): value for key, value in gold.items()}
        rescued = [key for key in differences["semantic"].get("rescued_names", [])]
        accepted = 0
        for name in rescued:
            expected = labels.get(name.strip().lower())
            if expected is None:
                continue
            precision_labeled += 1
            expected_ids = set(expected if isinstance(expected, list) else expected.get("target_ids", []))
            actual = set(after["decisions"]["semantic"][name.strip().lower()]["targets"])
            accepted += bool(expected_ids & actual) if expected_ids else bool(actual)
        precision = accepted / precision_labeled if precision_labeled else None
    return {
        "cost_usd_proxy_before": round(before_cost, 8),
        "cost_usd_proxy_after": round(after_cost, 8),
        "cost_change_percent": round(100 * (after_cost - before_cost) / before_cost, 3)
        if before_cost else None,
        "merge_time_span_seconds_before": before_span,
        "merge_time_span_seconds_after": after_span,
        "latency_change_percent": round(100 * (after_span - before_span) / before_span, 3)
        if before_span else None,
        "rescued_precision": precision,
        "rescued_precision_labeled": precision_labeled,
        "known_target_recall_proxy": semantic.get("target_signature_recall"),
        "gates": {
            "cost_below_five_percent": (after_cost - before_cost) / before_cost <= 0.05
            if before_cost else None,
            "latency_below_ten_percent": (after_span - before_span) / before_span <= 0.10
            if before_span else None,
            "known_target_recall_no_regression": semantic["lost_merge_count"] == 0,
            "new_reduction_at_least_25_percent": semantic["new_reduction_percent"] >= 25,
            "precision_measured": precision is not None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-kg", required=True)
    parser.add_argument("--after-kg", required=True)
    parser.add_argument("--before-debug")
    parser.add_argument("--after-debug")
    parser.add_argument("--gold", help="JSON object mapping rescued name to target ids")
    parser.add_argument("--output", help="Write JSON report to this path")
    args = parser.parse_args()
    empty = {"decisions": {"semantic": {}, "finding": {}}, "stages": {}, "total_cost_usd_proxy": 0.0}
    before = parse_debug(args.before_debug) if args.before_debug else empty
    after = parse_debug(args.after_debug) if args.after_debug else empty
    gold = json.loads(Path(args.gold).read_text()) if args.gold else None
    report = {
        "before_kg": kg_stats(args.before_kg),
        "after_kg": kg_stats(args.after_kg),
        "decision_delta": delta(before, after),
        "before_llm": before["stages"],
        "after_llm": after["stages"],
        "gates": gate_metrics(before, after, delta(before, after), gold),
        "limitations": [
            "cost uses stage-attributed llm_debug running-USD deltas as a proxy",
            "timestamp spans are coarse wall-time proxies and include retries/concurrency",
            "rescued precision requires --gold; target-ID recall assumes the compared runs share a corpus",
        ],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from main_pipeline import MediaDataPipeline


def overlaps(range_a: List[float], start_b: float | None, end_b: float | None) -> bool:
    if start_b is None and end_b is None:
        return False

    a_start, a_end = range_a
    b_start = start_b if start_b is not None else end_b
    b_end = end_b if end_b is not None else start_b

    if b_start is None or b_end is None:
        return False

    return not (b_end < a_start or b_start > a_end)


def evaluate_case_for_mode(
    pipeline: MediaDataPipeline,
    case: Dict[str, Any],
    content_type: Optional[str],
) -> Dict[str, Any]:
    query = case["query"]
    video_name = case["video_name"]
    target_time_range = case["target_time_range"]
    top_k = int(case.get("min_hit_at_k", 3))

    results = pipeline.search(
        query=query,
        top_k=top_k,
        video_name=video_name,
        content_type=content_type,
    )

    hit = False
    matched_rank = None

    for idx, item in enumerate(results, start=1):
        event_range = item.get("event_time_range") or {}
        meta = item.get("metadata", {}) or {}

        start_time = event_range.get("start")
        end_time = event_range.get("end")

        if start_time is None and meta.get("start_time") is not None:
            start_time = meta.get("start_time")
        if end_time is None and meta.get("end_time") is not None:
            end_time = meta.get("end_time")

        if overlaps(target_time_range, start_time, end_time):
            hit = True
            matched_rank = idx
            break

    return {
        "video_name": video_name,
        "query": query,
        "query_type": case.get("query_type", "generic"),
        "content_type": content_type or "all",
        "hit": hit,
        "matched_rank": matched_rank,
        "top_k": top_k,
    }


def evaluate_case(pipeline: MediaDataPipeline, case: Dict[str, Any]) -> Dict[str, Any]:
    preferred_content_types = case.get("preferred_content_types") or []
    reports: List[Dict[str, Any]] = []

    if preferred_content_types:
        for content_type in preferred_content_types:
            reports.append(evaluate_case_for_mode(pipeline, case, content_type))
    else:
        reports.append(evaluate_case_for_mode(pipeline, case, None))

    reports.append(evaluate_case_for_mode(pipeline, case, None))

    best_report = None
    for report in reports:
        if report["hit"]:
            if best_report is None:
                best_report = report
            elif (
                best_report["matched_rank"] is None
                or (
                    report["matched_rank"] is not None
                    and report["matched_rank"] < best_report["matched_rank"]
                )
            ):
                best_report = report

    if best_report is None:
        best_report = reports[-1]

    best_report["all_mode_reports"] = reports
    return best_report


def main():
    pipeline = MediaDataPipeline()
    benchmark_path = Path("evaluation/benchmark_cases.json")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    reports = [evaluate_case(pipeline, case) for case in cases]

    total = len(reports)
    hits = sum(1 for r in reports if r["hit"])
    hit_rate = hits / total if total else 0.0

    print("=== Evaluation Summary ===")
    print(f"Total cases: {total}")
    print(f"Hits: {hits}")
    print(f"Hit rate: {hit_rate:.2%}")
    print()

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for report in reports:
        qtype = report.get("query_type", "generic")
        by_type.setdefault(qtype, []).append(report)

    print("=== Breakdown by query_type ===")
    for qtype, items in sorted(by_type.items()):
        q_hits = sum(1 for item in items if item["hit"])
        q_rate = q_hits / len(items) if items else 0.0
        print(f"{qtype}: {q_hits}/{len(items)} ({q_rate:.2%})")
    print()

    print("=== Detailed Reports ===")
    for report in reports:
        print(
            {
                "video_name": report["video_name"],
                "query": report["query"],
                "query_type": report["query_type"],
                "best_content_type": report["content_type"],
                "hit": report["hit"],
                "matched_rank": report["matched_rank"],
                "top_k": report["top_k"],
            }
        )


if __name__ == "__main__":
    main()
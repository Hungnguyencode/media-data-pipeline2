from __future__ import annotations

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_pipeline import MediaDataPipeline
from src.utils import get_config


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
    search_mode = str(case.get("search_mode", "auto")).strip().lower() or "auto"

    results = pipeline.search(
        query=query,
        top_k=top_k,
        video_name=video_name,
        content_type=content_type,
        search_mode=search_mode,
    )

    hit = False
    matched_rank = None
    top_result = results[0] if results else None

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
        "search_mode": search_mode,
        "content_type": content_type or "all",
        "hit": hit,
        "matched_rank": matched_rank,
        "top_k": top_k,
        "result_count": len(results),
        "top_result_preview": {
            "fusion_score": top_result.get("fusion_score") if top_result else None,
            "query_type": top_result.get("query_type") if top_result else None,
            "search_mode": top_result.get("search_mode") if top_result else None,
            "matched_signals": top_result.get("matched_signals") if top_result else [],
            "ranking_explanation": top_result.get("ranking_explanation") if top_result else "",
            "display_text": top_result.get("display_text") if top_result else "",
        },
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

    # Quan trọng: copy dict để tránh circular reference
    final_report = dict(best_report)
    final_report["all_mode_reports"] = [dict(r) for r in reports]
    return final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument(
        "--reranker",
        choices=["on", "off"],
        default="off",
        help="Enable or disable cross-encoder reranker during evaluation",
    )
    parser.add_argument(
        "--benchmark",
        default=str(PROJECT_ROOT / "evaluation" / "benchmark_cases.json"),
        help="Path to benchmark_cases.json",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "evaluation" / "latest_report.json"),
        help="Path to output JSON report",
    )
    return parser.parse_args()


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    config = get_config(force_reload=True)
    config.setdefault("reranker", {})
    config["reranker"]["enabled"] = args.reranker == "on"

    pipeline = MediaDataPipeline(config)

    benchmark_path = Path(args.benchmark).resolve()
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    indexed_videos = set(pipeline.vector_indexer.list_videos())

    valid_cases: List[Dict[str, Any]] = []
    skipped_cases: List[Dict[str, Any]] = []

    for case in cases:
        video_name = str(case.get("video_name", "")).strip()
        if not video_name or video_name not in indexed_videos:
            skipped_cases.append(
                {
                    "video_name": video_name,
                    "query": case.get("query", ""),
                    "reason": "video_not_indexed",
                }
            )
            continue
        valid_cases.append(case)

    reports = [evaluate_case(pipeline, case) for case in valid_cases]

    total = len(reports)
    hits = sum(1 for r in reports if r["hit"])
    hit_rate = hits / total if total else 0.0

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for report in reports:
        qtype = report.get("query_type", "generic")
        by_type.setdefault(qtype, []).append(report)

    type_breakdown: Dict[str, Dict[str, Any]] = {}
    for qtype, items in sorted(by_type.items()):
        q_hits = sum(1 for item in items if item["hit"])
        q_rate = q_hits / len(items) if items else 0.0
        type_breakdown[qtype] = {
            "hits": q_hits,
            "total": len(items),
            "hit_rate": round(q_rate, 4),
        }

    final_report = {
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "benchmark_file": str(benchmark_path),
        "output_file": str(Path(args.output).resolve()),
        "reranker_enabled": config["reranker"]["enabled"],
        "indexed_videos_available": len(indexed_videos),
        "valid_cases": total,
        "skipped_cases_count": len(skipped_cases),
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "type_breakdown": type_breakdown,
        "skipped_cases": skipped_cases,
        "detailed_reports": reports,
    }

    output_path = Path(args.output).resolve()
    save_report(final_report, output_path)

    print("=== Evaluation Summary ===")
    print(f"Benchmark file: {benchmark_path}")
    print(f"Output report: {output_path}")
    print(f"Reranker enabled: {config['reranker']['enabled']}")
    print(f"Indexed videos available: {len(indexed_videos)}")
    print(f"Valid cases: {total}")
    print(f"Skipped cases: {len(skipped_cases)}")
    print(f"Hits: {hits}")
    print(f"Hit rate: {hit_rate:.2%}")
    print()

    if skipped_cases:
        print("=== Skipped Cases ===")
        for item in skipped_cases:
            print(item)
        print()

    print("=== Breakdown by query_type ===")
    for qtype, stats in type_breakdown.items():
        print(f"{qtype}: {stats['hits']}/{stats['total']} ({stats['hit_rate']:.2%})")
    print()

    print("=== Detailed Reports ===")
    for report in reports:
        print(
            {
                "video_name": report["video_name"],
                "query": report["query"],
                "query_type": report["query_type"],
                "search_mode": report["search_mode"],
                "best_content_type": report["content_type"],
                "hit": report["hit"],
                "matched_rank": report["matched_rank"],
                "top_k": report["top_k"],
                "result_count": report["result_count"],
                "top_result_preview": report["top_result_preview"],
            }
        )


if __name__ == "__main__":
    main()
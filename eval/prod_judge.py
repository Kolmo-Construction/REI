"""
Async production monitoring script.
Pulls recent traces from Langfuse, samples them, judges via Claude Opus 4.6,
writes scores back as Langfuse scores, and alerts on safety violations.

Usage:
    uv run python eval/prod_judge.py --sample-rate 0.05
    uv run python eval/prod_judge.py --lookback-minutes 60 --sample-rate 0.10

Run as a cron job every 15 minutes.
Exit code 1 if any safety alert is triggered (for alerting integration).
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta, timezone

from greenvest.config import settings
from eval.judge import SAFETY_ALERT_THRESHOLD, COMPOSITE_ALERT_THRESHOLD, judge_recommendation
from eval.langfuse_client import langfuse_client, langfuse_api_client


async def _judge_traces(args: argparse.Namespace) -> int:
    """Pull traces, judge them, write scores back. Returns safety alert count."""
    lf_api = langfuse_api_client()
    if lf_api is None:
        print(
            "[WARN] Langfuse not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing).",
            file=sys.stderr,
        )
        return 0

    if not settings.ANTHROPIC_API_KEY:
        print("[WARN] ANTHROPIC_API_KEY not set — cannot run judge scoring.", file=sys.stderr)
        return 0

    start_time = datetime.now(timezone.utc) - timedelta(minutes=args.lookback_minutes)
    print(f"Pulling traces from Langfuse since {start_time.isoformat()}")

    try:
        resp = await lf_api.trace.list(
            from_timestamp=start_time,
            limit=1000,
            name="greenvest_invoke",
        )
        traces = resp.data if resp and resp.data else []
    except Exception as exc:
        print(f"[WARN] Failed to list traces: {exc}", file=sys.stderr)
        return 0

    print(f"Found {len(traces)} traces. Sampling at {args.sample_rate:.0%}...")
    sampled = [t for t in traces if random.random() < args.sample_rate]
    print(f"Sampled {len(sampled)} traces for judging.")

    if not sampled:
        print("No traces sampled — nothing to judge.")
        return 0

    lf = langfuse_client()
    safety_alerts = 0

    for trace in sampled:
        trace_id = trace.id

        # Extract query and recommendation from trace input/output
        inp = getattr(trace, "input", {}) or {}
        out = getattr(trace, "output", {}) or {}

        query = inp.get("query", "")
        recommendation = out.get("recommendation", "")

        if not query or not recommendation:
            print(f"  [{trace_id[:8]}] Skipping — missing query or recommendation.")
            continue

        scores = await judge_recommendation(query, recommendation)
        if scores is None:
            continue

        print(
            f"  [{trace_id[:8]}] composite={scores.composite:.3f} "
            f"persona={scores.persona:.2f} accuracy={scores.accuracy:.2f} "
            f"safety={scores.safety:.2f} relevance={scores.relevance:.2f}"
        )

        # Write scores back to Langfuse
        if lf:
            for name, value in [
                ("greenvest_persona", scores.persona),
                ("greenvest_accuracy", scores.accuracy),
                ("greenvest_safety", scores.safety),
                ("greenvest_relevance", scores.relevance),
                ("greenvest_composite", scores.composite),
            ]:
                try:
                    lf.create_score(
                        trace_id=trace_id,
                        name=name,
                        value=value,
                        data_type="NUMERIC",
                        comment=scores.reasoning if name == "greenvest_composite" else None,
                    )
                except Exception as exc:
                    print(f"  [WARN] Failed to write score {name}: {exc}", file=sys.stderr)

        # Alert checks
        if scores.safety < SAFETY_ALERT_THRESHOLD:
            safety_alerts += 1
            print(
                f"  [ALERT] SAFETY score {scores.safety:.3f} < {SAFETY_ALERT_THRESHOLD} "
                f"for trace {trace_id[:8]} | query: {query[:80]!r}"
            )
        if scores.composite < COMPOSITE_ALERT_THRESHOLD:
            print(
                f"  [WARN] Composite score {scores.composite:.3f} < {COMPOSITE_ALERT_THRESHOLD} "
                f"for trace {trace_id[:8]}"
            )

    return safety_alerts


def main():
    parser = argparse.ArgumentParser(description="Production judge for Greenvest traces")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.05,
        help="Fraction of traces to sample (0.0–1.0, default: 0.05)",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=15,
        help="How many minutes back to pull traces (default: 15)",
    )
    args = parser.parse_args()

    safety_alerts = asyncio.run(_judge_traces(args))

    print(f"\nJudging complete. Safety alerts: {safety_alerts}")
    if safety_alerts > 0:
        print(f"[ALERT] {safety_alerts} safety violation(s) detected — exiting with code 1.")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

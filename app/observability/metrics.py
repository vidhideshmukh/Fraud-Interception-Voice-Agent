"""Per-LLM-call latency metrics — for real model evaluation.

The audit trail's per-turn `latency_ms` only timed the DIALOG call. But each
customer turn actually fires THREE LLM calls when NeMo Guardrails is live —
input rail, dialog, output rail — and to evaluate/compare models you need each
one timed SEPARATELY, tagged with its model + stage, then aggregated with
PERCENTILES (p50/p95/p99), not a single blended average (the tail is what a
voice call feels).

This module is that metrics pipeline: `record_llm_call()` logs one sample per
LLM inference to a durable, date-partitioned file (logs/metrics/llm-YYYY-MM-DD
.jsonl) plus an in-memory ring for the live view; `summarize()` groups by
(model, stage) into count/p50/p95/p99/avg/max. In production this is where you'd
emit to Prometheus/CloudWatch instead — same shape, different sink.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

METRICS_DIR = Path(os.getenv("LOG_DIR", "logs")) / "metrics"
METRICS_PERSIST = os.getenv("METRICS_PERSIST", "true").lower() == "true"

# Live view — recent samples in memory; the durable files hold the full history.
_RING: deque[dict] = deque(maxlen=8000)


def record_llm_call(stage: str, model: str, latency_ms: float, *,
                    tokens_in: int | None = None, tokens_out: int | None = None,
                    session_id: str | None = None) -> dict:
    """Record ONE LLM inference. `stage` is input_rail | dialog | output_rail;
    `model` is the model id (or 'mock'). Never raises."""
    sample = {
        "ts": int(time.time() * 1000), "stage": stage, "model": model,
        "latency_ms": round(float(latency_ms), 1),
        "tokens_in": tokens_in, "tokens_out": tokens_out, "session_id": session_id,
    }
    _RING.append(sample)
    if METRICS_PERSIST:
        try:
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with (METRICS_DIR / f"llm-{day}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
        except Exception:  # noqa: BLE001 — metrics must never break a live call
            pass
    return sample


def read_all(limit: int = 8000) -> list[dict]:
    """All durable latency samples across every day file (oldest first, tailed).
    Falls back to the in-memory ring if nothing has been persisted yet."""
    if not METRICS_DIR.exists():
        return list(_RING)[-limit:]
    out = []
    for f in sorted(METRICS_DIR.glob("llm-*.jsonl")):
        try:
            for raw in f.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    out.append(json.loads(raw))
        except Exception:  # noqa: BLE001
            pass
    out.sort(key=lambda s: s.get("ts", 0))
    return out[-limit:]


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1))))
    return round(sorted_vals[k], 1)


def summarize(samples: list[dict]) -> list[dict]:
    """Group samples by (model, stage) -> count + p50/p95/p99/avg/max latency.
    This is the model-evaluation view: how fast each model is, per stage, at the
    median AND the tail."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for s in samples:
        groups[(s.get("model"), s.get("stage"))].append(s.get("latency_ms", 0.0))
    rows = []
    for (model, stage), lats in groups.items():
        s = sorted(lats)
        rows.append({
            "model": model, "stage": stage, "count": len(s),
            "p50": _percentile(s, 50), "p95": _percentile(s, 95), "p99": _percentile(s, 99),
            "avg": round(sum(s) / len(s), 1), "max": round(max(s), 1),
        })
    # order: dialog first, then rails; then by model
    stage_order = {"dialog": 0, "input_rail": 1, "output_rail": 2}
    rows.sort(key=lambda r: (stage_order.get(r["stage"], 9), r["model"] or ""))
    return rows

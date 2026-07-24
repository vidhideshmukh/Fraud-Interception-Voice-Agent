"""Investigation Agent — the genuine agentic component.

WHERE THIS SITS in the architecture (and why it, not the trigger, is the agent):
the deterministic scorer (app/core/fraud_trigger.py) resolves the obvious ~65% of
traffic in microseconds and must stay a fixed, auditable rule — an LLM has no
business computing a risk score. But it only looks at TWO dimensions (amount +
geography), so on the ambiguous "messy middle" it drops to a coin-flip: it
over-flags legitimate travel and MISSES device/velocity fraud, because it can't
see those signals. That gap is where reasoning genuinely adds value.

This agent handles exactly those ambiguous cases. It calls deterministic EVIDENCE
TOOLS across four dimensions — device, location, velocity, amount — and then
REASONS OVER THE COMBINATION (no single signal decides: one suspicious signal is
usually explainable, two-or-more together is the fraud pattern). In live mode the
reasoning is a Nemotron call; offline it's a transparent multi-signal heuristic.
Either way it returns a verdict + rationale + recommendation — turning a raw alert
into the reasoned recommendation a human analyst would otherwise have to produce.

The tools are deterministic and the money decision stays gated (its output is a
RECOMMENDATION — block_and_escalate / human_review / clear — not an autonomous
action). Same principle as everywhere else here: tools and policy are code; the
agent reasons and advises.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.agents import prompts
from app.llm import nemotron

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
VELOCITY_SUSPICIOUS_MIN = 30      # a second charge under 30 min after the last is implausibly fast
AMOUNT_ELEVATED_RATIO = 1.5       # spend >= 1.5x the customer's normal ceiling is elevated


@dataclass
class InvestigationResult:
    predicted_fraud: bool
    confidence: float
    recommendation: str           # block_and_escalate | human_review | clear
    rationale: str
    evidence: list[dict]
    signals_flagged: int
    model_id: str


# --- Evidence tools. Each examines ONE dimension of a case and returns a finding
#     {dimension, suspicious, detail}. Deterministic — they read the signals the
#     dataset carries inline (known vs current device/location, velocity, amount). ---

def device_evidence(case: dict) -> dict:
    known = case.get("current_device") in (case.get("known_devices") or [])
    return {"dimension": "device", "suspicious": not known,
            "detail": (f"device '{case.get('current_device')}' is "
                       f"{'a recognised device' if known else 'NOT a recognised device for this customer'}")}


def location_evidence(case: dict) -> dict:
    loc = case.get("current_location")
    known = loc in (case.get("known_locations") or [])
    return {"dimension": "location", "suspicious": not known,
            "detail": f"location '{loc}' is {'a usual location' if known else 'outside the customer usual locations'}"}


def velocity_evidence(case: dict) -> dict:
    v = case.get("velocity_minutes")
    suspicious = v is not None and v < VELOCITY_SUSPICIOUS_MIN
    return {"dimension": "velocity", "suspicious": suspicious,
            "detail": (f"{v} minutes since the previous transaction"
                       + (" — implausibly fast" if suspicious else " — a normal gap"))}


def amount_evidence(case: dict, p95_gbp: float | None) -> dict:
    amt = float(case.get("amount", 0))
    if not p95_gbp:
        return {"dimension": "amount", "suspicious": False,
                "detail": f"£{amt:,.0f} (no baseline available to compare)"}
    ratio = amt / max(p95_gbp, 1.0)
    return {"dimension": "amount", "suspicious": ratio >= AMOUNT_ELEVATED_RATIO,
            "detail": f"£{amt:,.0f} is {ratio:.1f}x the customer's normal ceiling (£{p95_gbp:,.0f})"}


def gather_evidence(case: dict, p95_gbp: float | None = None) -> list[dict]:
    """Run all four evidence tools on a case."""
    return [device_evidence(case), location_evidence(case),
            velocity_evidence(case), amount_evidence(case, p95_gbp)]


def _deterministic_verdict(evidence: list[dict]) -> tuple[bool, float, str, str]:
    """Multi-signal reasoning (the offline stand-in for the LLM): no single signal
    decides. One suspicious signal is usually explainable; two-or-more independent
    ones together is the fraud pattern. This is exactly the combination the blunt
    amount+geo scorer cannot make, because it never sees device or velocity."""
    flagged = [e for e in evidence if e["suspicious"]]
    n = len(flagged)
    predicted_fraud = n >= 2
    if n >= 3:
        confidence, rec = 0.95, "block_and_escalate"
    elif n == 2:
        confidence, rec = 0.80, "block_and_escalate"
    elif n == 1:
        confidence, rec = 0.60, "human_review"     # one flag — genuinely borderline, ask a human
    else:
        confidence, rec = 0.90, "clear"
    rationale = ("; ".join(e["detail"] for e in flagged)
                 if flagged else "all four signals are within the customer's normal pattern")
    return predicted_fraud, confidence, rec, rationale


def _format_for_llm(case: dict, evidence: list[dict]) -> str:
    lines = [f"Ambiguous transaction for {case.get('customer_name')}:",
             f"  amount £{float(case.get('amount', 0)):,.0f} at {case.get('merchant')} "
             f"({case.get('merchant_category')})", "Evidence findings:"]
    lines += [f"  - {e['dimension']}: {e['detail']}  [{'SUSPICIOUS' if e['suspicious'] else 'ok'}]"
              for e in evidence]
    return "\n".join(lines)


def investigate(case: dict, p95_gbp: float | None = None, *, session_id: str | None = None) -> InvestigationResult:
    """Investigate one ambiguous case. Gathers evidence deterministically, then
    reasons over it — Nemotron in live mode, a transparent heuristic offline (or
    if the model call fails). The verdict is a recommendation; code stays the
    authority on the action."""
    evidence = gather_evidence(case, p95_gbp)
    fraud, confidence, rec, rationale = _deterministic_verdict(evidence)
    model_id = "reasoner:deterministic"

    if not MOCK_MODE:
        data = nemotron.complete_json(prompts.INVESTIGATION_SYSTEM, _format_for_llm(case, evidence),
                                      stage="investigation", session_id=session_id)
        if data and "is_fraud" in data:      # trust the model only if it returned the contract
            fraud = bool(data["is_fraud"])
            confidence = float(data.get("confidence", confidence))
            rec = data.get("recommendation", rec)
            rationale = data.get("rationale", rationale)
            model_id = nemotron.NIM_MODEL_REALTIME

    return InvestigationResult(predicted_fraud=fraud, confidence=confidence, recommendation=rec,
                               rationale=rationale, evidence=evidence,
                               signals_flagged=sum(1 for e in evidence if e["suspicious"]),
                               model_id=model_id)

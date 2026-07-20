"""Async resolution agent — runs OFF the critical path after the customer says
'not me'. In live mode this is where the heavier Nemotron Super model plans the
remediation; in mock mode the plan is fixed. Either way the tools it calls are
the same mock bank services, and every action is audited.

Pitch-deck mapping: this is the "Banking Agent" in the architecture diagram.
Its tool calls are also what gets registered as NeMo Agent Toolkit functions
for the `nat run` async flow (see nat/functions.py and config/workflow.yaml)
— this module's `resolve_fraud`/`resolve_legit` are the inline-demo version
of the same tool sequence a NAT react_agent would plan and execute off the
critical path in the fuller build.

Every bank call here is idempotency-keyed by (session_id, txn_id,
action) — see bank.py — so a retried resolution (crashed worker, at-least-
once delivery, whatever) can never double-block or double-release.
"""
from __future__ import annotations

from app.database.models import CallSession
from app.security import audit
from app.database import bank


def resolve_fraud(session: CallSession) -> dict:
    txn = session.event.txn
    actions = [
        bank.block_card(session.session_id, txn.txn_id, session.customer.customer_id),
        bank.open_chargeback(session.session_id, txn.txn_id),
    ]
    case = bank.upsert_case(
        session.session_id,
        outcome="fraud_confirmed",
        note=(f"Customer denied txn {txn.txn_id} ({txn.merchant}, £{txn.amount_gbp:,.0f}). "
              f"Card blocked, reissue ordered, chargeback opened. RCA: {session.event.rca_reason}"),
    )
    for a in actions:
        audit.log_turn(session.session_id, "tool", a)
    audit.log_turn(session.session_id, "state", {"case": case["case_id"], "outcome": "fraud_confirmed"})
    return {"actions": actions, "case": case}


def resolve_legit(session: CallSession) -> dict:
    txn = session.event.txn
    action = bank.release_hold(session.session_id, txn.txn_id)
    case = bank.upsert_case(
        session.session_id,
        outcome="false_positive_recovered",
        note=f"Customer confirmed txn {txn.txn_id}. Hold released. RCA: {session.event.rca_reason}",
    )
    audit.log_turn(session.session_id, "tool", action)
    audit.log_turn(session.session_id, "state", {"case": case["case_id"], "outcome": "false_positive_recovered"})
    return {"actions": [action], "case": case}

"""Drives the live transaction-stream demo surface.

This is the upgrade from "click a button to trigger a fraud event" to
"watch a real-looking feed of transactions, most legitimate, one flagged,
and the call fires on its own" — the mentor's "real-time transaction
happening, best-case-scenario" ask. Poll-based (GET /stream/transactions,
advance the replay by N each call) rather than a websocket/SSE push,
because a hackathon dashboard polling once a second is simpler to get right
under time pressure than managing a long-lived server connection, and at
this scale (hundreds of transactions, a handful of viewers) there's no
real cost to polling over pushing.
"""
from __future__ import annotations

from typing import Optional

from app.database.models import Transaction
from app.database import bank
from app.core import fraud_trigger

_materialized: Optional[list[Transaction]] = None
_position = 0
_triggered_txn_ids: set[str] = set()


def _stream() -> list[Transaction]:
    """Lazily materializes the sorted transaction list once per process —
    bank's contents don't change after load, so there's no reason to
    re-sort on every poll."""
    global _materialized
    if _materialized is None:
        _materialized = list(fraud_trigger.iter_transaction_stream())
    return _materialized


def reset():
    """Rewinds the transaction_stream — used by tests and by a fresh demo run."""
    global _materialized, _position, _triggered_txn_ids
    _materialized = None
    _position = 0
    _triggered_txn_ids = set()


def advance(count: int = 1) -> dict:
    """Advances the replay by `count` transactions. Any transaction whose
    score crosses fraud_trigger.RISK_THRESHOLD auto-starts a call session
    (imported lazily to avoid a circular import — orchestrator imports
    several services, none of which should need to import transaction_stream back).
    Each txn_id triggers at most once even across repeated polls."""
    from app.core import orchestrator  # deferred: avoids a circular import at module load

    stream = _stream()
    global _position
    start = _position
    end = min(start + count, len(stream))
    items = []

    for txn in stream[start:end]:
        score = fraud_trigger.score_transaction(txn)
        flagged = score >= fraud_trigger.RISK_THRESHOLD
        session_id = None
        verification_code = None  # only ever populated on the one response that auto-triggers a call

        if flagged and txn.txn_id not in _triggered_txn_ids:
            _triggered_txn_ids.add(txn.txn_id)
            event = fraud_trigger.emit_event(txn.txn_id)
            session = orchestrator.start_call(event)
            session_id = session.session_id
            # Symmetric with POST /trigger's response: the verification code
            # is surfaced exactly once, at the moment of push, for the
            # Barclays app UI to display. /session/{id} deliberately does
            # NOT re-expose it on later polls — see main.py's docstring.
            verification_code = session.verification_code

        items.append({
            "txn_id": txn.txn_id, "customer_id": txn.customer_id,
            "amount_gbp": txn.amount_gbp, "merchant": txn.merchant, "city": txn.city,
            "channel": txn.channel, "flagged": flagged, "risk_score": round(score, 3),
            "auto_triggered_session_id": session_id,
            "verification_code": verification_code,
        })

    _position = end
    return {"items": items, "position": _position, "total": len(stream), "done": _position >= len(stream)}

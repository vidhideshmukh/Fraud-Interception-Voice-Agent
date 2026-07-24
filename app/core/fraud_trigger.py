"""History-based fraud scoring — the bank's fraud model, simplified.

In production this is a Kafka/Redis stream of scored transactions feeding an
RCA engine. Here, `assess()` scores a NEW transaction against the specific
CUSTOMER'S OWN behavioral baseline (app/database/bank.py builds ~45 days of
settled history per customer; CustomerBehavior summarizes it). That's the
production shape: fraud is a deviation from *this account's* normal, not a
global rule. An £800 spend is routine for a Premier customer and alarming for
a student; a card used in Lagos is fraud for a London customer who never
travels, but expected for one who does.

Every risk score here is EXPLAINABLE — it comes with the exact signals that
produced it (impossible travel, amount spike, unusual location), which become
the RCA reason the voice agent reads to the customer.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Iterator, Optional

from app.database import bank
from app.database.models import FraudEvent, Transaction, new_id

# A transaction must clear this score to trigger a call (a "high-risk event").
RISK_THRESHOLD = float(os.getenv("FRAUD_RISK_THRESHOLD", "0.85"))


@dataclass
class Assessment:
    score: float
    signals: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.signals:
            return "Transaction matches the customer's established spending pattern; no anomaly detected."
        return "Risk engine flags: " + "; ".join(self.signals) + "."


def assess(txn: Transaction) -> Assessment:
    """Score `txn` against its customer's history/baseline. Returns the score
    AND the human-readable signals behind it. Uses the MAX of the triggered
    signal weights (so one decisive signal, like card-present abroad, sets the
    score) while collecting every signal's text for the RCA reason."""
    customer = bank.CUSTOMERS.get(txn.customer_id)
    if customer is None or customer.behavior is None:
        return Assessment(0.10)  # unknown customer — can't compare to a baseline

    b = customer.behavior
    first_name = customer.name.split()[0]
    foreign = txn.city.strip().lower() not in bank.UK_CITIES
    unusual_city = txn.city not in b.cities
    amount_ratio = txn.amount_gbp / max(b.p95_gbp, 1.0)

    weight = 0.05
    signals: list[str] = []

    # --- geography, weighed against this customer's OWN travel history ---
    if foreign:
        if b.intl_travel:
            # This customer does travel abroad — a foreign transaction is
            # unusual but not alarming on its own.
            weight = max(weight, 0.55)
            signals.append(f"transaction in {txn.city} (outside the UK); {first_name} does travel "
                           f"internationally, so treated as lower risk")
        elif txn.channel == "card_present":
            weight = max(weight, 0.96)
            signals.append(f"card-present transaction in {txn.city}, outside the UK — impossible travel: "
                           f"no foreign-travel history on this account")
        else:
            weight = max(weight, 0.88)
            signals.append(f"card-not-present transaction from {txn.city}, outside the UK and "
                           f"{first_name}'s usual locations")
    elif unusual_city:
        weight = max(weight, 0.55)
        signals.append(f"transaction in {txn.city}, outside {first_name}'s usual areas "
                       f"({', '.join(b.cities)})")

    # --- amount, weighed against this customer's OWN typical high spend ---
    if amount_ratio >= 3:
        weight = max(weight, 0.90)
        signals.append(f"£{txn.amount_gbp:,.0f} is {amount_ratio:.1f}x this customer's typical high "
                       f"spend (£{b.p95_gbp:,.0f})")
    elif amount_ratio >= 1.5:
        # A moderate spike is only decisive combined with an unusual location.
        weight = max(weight, 0.82 if unusual_city else 0.5)
        signals.append(f"£{txn.amount_gbp:,.0f} is above this customer's usual ceiling (£{b.p95_gbp:,.0f})")
    if len(signals) >= 2 and weight < 0.85:
        weight = min(weight + 0.20, 0.92)  # two moderate signals together are more concerning
        signals.append(f"multiple independent anomalies signals detected together.")
    return Assessment(round(min(weight, 0.98), 3), signals)


def score_transaction(txn: Transaction) -> float:
    return assess(txn).score


def emit_event(txn_id: str) -> FraudEvent:
    """Explicit trigger for one transaction (used by /trigger/{txn_id} and
    tests). Always produces a FraudEvent regardless of score."""
    txn = bank.TRANSACTIONS[txn_id]
    a = assess(txn)
    return FraudEvent(event_id=new_id("evt"), txn=txn, risk_score=a.score, rca_reason=a.reason)


def iter_transaction_stream() -> Iterator[Transaction]:
    """All non-settled (new) transactions in timestamp order. History txns are
    the baseline, not part of the live feed, so they're excluded here."""
    yield from sorted((t for t in bank.TRANSACTIONS.values() if t.status != "settled"),
                      key=lambda t: t.ts_ms)


def iter_flagged_events() -> Iterator[FraudEvent]:
    """New transactions crossing RISK_THRESHOLD, as FraudEvents."""
    for txn in iter_transaction_stream():
        a = assess(txn)
        if a.score >= RISK_THRESHOLD:
            yield FraudEvent(event_id=new_id("evt"), txn=txn, risk_score=a.score, rca_reason=a.reason)


def random_fraud_event() -> Optional[FraudEvent]:
    """Pick a RANDOM fraud transaction from ANY customer and build a FraudEvent
    for it — so the real-time demo rings a different customer each time instead
    of always the first-in-order one. Prefers ground-truth `label.is_fraud` when
    the loaded data carries it; otherwise falls back to any new transaction that
    crosses the risk threshold. Returns None if there are no fraud candidates."""
    active = [t for t in bank.TRANSACTIONS.values() if t.status != "settled"]
    labelled = [t for t in active if t.label and t.label.is_fraud]
    pool = labelled or [t for t in active if score_transaction(t) >= RISK_THRESHOLD]
    if not pool:
        return None
    txn = random.choice(pool)
    a = assess(txn)
    return FraudEvent(event_id=new_id("evt"), txn=txn, risk_score=a.score, rca_reason=a.reason)

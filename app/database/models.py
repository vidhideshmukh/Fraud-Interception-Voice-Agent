"""Pydantic entities shared across the app. Keep flat and boring.

Schema note: `Transaction`, `CustomerProfile` and `TxnLabel` intentionally
mirror the shape NeMo Data Designer is configured to emit (see
`data/generate_datasets.py`) — not just what the demo's two hardcoded
transactions needed. That way generated data loads with zero translation
layer between "what the generator produces" and "what the orchestrator
consumes". If you add a field to the generator prompt, add it here first.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import time
import uuid


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class CustomerBehavior(BaseModel):
    """A customer's behavioral BASELINE — the statistical summary of their
    normal spending, derived from their transaction history. This is what the
    fraud model compares a NEW transaction against: fraud is only meaningful
    as a deviation from *this* baseline (an £800 spend is normal for a Premier
    customer, alarming for a student). See app/core/fraud_trigger.py.

    Barclays is a UK retail bank, so amounts are GBP and `cities` are the UK
    places this customer normally transacts in.
    """
    avg_gbp: float                        # typical transaction size
    p95_gbp: float                        # 95th-percentile spend — the "high but normal" ceiling
    merchant_mix: dict[str, float]        # e.g. {"grocery": 0.4, "fuel": 0.2, ...}
    active_hours: list[int] = Field(default_factory=lambda: [8, 23])  # [start, end]
    cities: list[str] = Field(default_factory=list)  # UK cities the customer normally uses their card in
    intl_travel: bool = False             # do they normally travel/spend abroad?


class Customer(BaseModel):
    """Runtime customer record. Carries the behavioral baseline inline so the
    fraud model can score an incoming transaction against this customer's own
    history without a second lookup."""
    customer_id: str
    name: str
    phone: str
    card_last4: str
    home_city: str
    account_type: str = "Standard"        # Barclays UK: Standard | Premier | Student | Basic
    behavior: Optional[CustomerBehavior] = None


class CustomerProfile(BaseModel):
    """Generator-input record for one customer (NeMo Data Designer, Dataset 1).
    Distinct from `Customer` only in that it's the generation-time seed shape.
    """
    customer_id: str
    name: str
    home_city: str
    card_last4: str
    segment: str
    behavior: CustomerBehavior

    def to_customer(self, phone: str) -> Customer:
        return Customer(customer_id=self.customer_id, name=self.name, phone=phone,
                        card_last4=self.card_last4, home_city=self.home_city,
                        behavior=self.behavior)


class TxnLabel(BaseModel):
    """Ground truth that only exists in synthetic data — never available in prod.

    `scenario_id` ties back to the fraud scenario catalog (FRAUD__2 doc,
    section 1) so generated injects double as a coverage checklist: 21
    scenarios in the catalog, at least one inject each.
    """
    is_fraud: bool
    scenario_id: Optional[str] = None  # e.g. "1A-1" (cloned card, card-present abroad)
    inject_note: Optional[str] = None  # human-readable reason for the label


class Transaction(BaseModel):
    txn_id: str
    customer_id: str
    amount_gbp: float                     # Barclays is a UK bank — amounts are GBP (£)
    merchant: str
    city: str
    channel: str  # card_present | online
    category: str = "other"               # grocery | dining | retail | fuel | travel | electronics | ...
    mcc: Optional[str] = None             # merchant category code
    device_id: Optional[str] = None       # None = no device signal (e.g. card-present); present = online/app
    ts_ms: int = Field(default_factory=now_ms)
    status: str = "held"  # settled (history) | held (new, awaiting review) | released | blocked
    label: Optional[TxnLabel] = None      # ground truth — present on generated data, absent in "prod"


class FraudEvent(BaseModel):
    event_id: str
    txn: Transaction
    risk_score: float
    rca_reason: str  # structured explanation from the (mock) RCA engine


class CallState(str, Enum):
    PUSH_SENT = "push_sent"
    DIALING = "dialing"
    AWAITING_VERIFICATION = "awaiting_verification"
    VERIFIED = "verified"
    RESOLVED_LEGIT = "resolved_legit"
    RESOLVED_FRAUD = "resolved_fraud"
    ESCALATED = "escalated"
    NO_ANSWER = "no_answer"
    CHANNEL_FROZEN = "channel_frozen"  # 3x failed verification — possible attacker, never soften auth


class Intent(str, Enum):
    CONFIRM_LEGIT = "confirm_legit"
    DENY = "deny"
    UNSURE = "unsure"
    DISTRESS = "distress"


class AgentTurn(BaseModel):
    """The dialog agent's only output contract. The model writes prose and
    picks an intent; the orchestrator's policy code decides what happens
    next — a jailbroken model still can't release a top-risk transaction,
    because it never gets a vote on that, only on `intent`/`confidence`."""
    reply: str
    intent: Optional[Intent] = None
    confidence: float = 1.0


class CallSession(BaseModel):
    session_id: str
    event: FraudEvent
    customer: Customer
    state: CallState = CallState.PUSH_SENT
    verification_code: str = ""  # one-time numeric OTP pushed to the app (see security/anti_vishing.py)
    verification_attempts: int = 0  # 3 wrong attempts -> CHANNEL_FROZEN, not infinite re-prompts
    transcript: list[dict] = Field(default_factory=list)
    outcome: Optional[str] = None

    def agent_utterances(self) -> list[str]:
        """Prior agent replies this call, oldest first — used by the repetition
        guard to detect the agent looping instead of progressing the dialog."""
        return [t["text"] for t in self.transcript if t["role"] == "assistant"]

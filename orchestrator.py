"""In-memory mock of Barclays' core banking systems: customers, transaction
HISTORY, card actions, case management. Barclays is a UK retail bank (~20M UK
personal-banking customers), so customers are British, amounts are GBP, and
cities are UK locations. FastAPI-ready but callable directly for speed.

Why real transaction HISTORY, not just two flagged transactions: fraud is only
meaningful as a deviation from a customer's normal behaviour. So seed() builds
each customer ~45 days of settled, in-pattern transactions (their baseline),
and app/core/fraud_trigger.py scores a NEW incoming transaction against THAT
history — an £800 spend is routine for a Premier customer and alarming for a
student. This is the production shape: score the anomaly against the account's
own past, not a global rule.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from app.database.models import (Customer, CustomerBehavior, CustomerProfile,
                                 Transaction, TxnLabel, new_id, now_ms)
from app.observability.logging_setup import get_logger

log = get_logger("bank")


def _db_exec(sql: str, params: tuple = ()) -> None:
    """Best-effort WRITE-THROUGH to the SQLite DB so runtime state (card actions,
    cases, incidents, transaction status) survives a restart. The in-memory
    working set stays authoritative for the live call — the DB is the durable
    mirror — so a persist failure is logged and swallowed rather than breaking a
    card action mid-call. No-op until the DB has been built."""
    if not DB_PATH.exists():
        return
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(sql, params)
        con.commit()
        con.close()
    except Exception as e:  # noqa: BLE001 — durability must never break a live action
        log.warning("db write-through failed (%s): %s", sql.split()[0] if sql else "?", e)

CUSTOMERS: dict[str, Customer] = {}
TRANSACTIONS: dict[str, Transaction] = {}
CARD_ACTIONS: list[dict] = []
CASES: dict[str, dict] = {}
# Human-review work items raised on escalation — the "request for a human fraud
# analyst to look into this" that a person actually works from (open -> resolved).
# In production this is a row in a case-management/ticketing system.
INCIDENTS: dict[str, dict] = {}

# UK cities Barclays customers transact in — used by the fraud model to tell a
# domestic anomaly from a foreign one (card used abroad = far higher risk).
UK_CITIES = {"london", "manchester", "birmingham", "leeds", "glasgow", "edinburgh",
             "bristol", "liverpool", "sheffield", "cardiff", "newcastle", "nottingham",
             "leicester", "brighton", "oxford", "cambridge", "reading", "york"}

# The banking data now lives in a REAL SQLite database (data/barclays.db), built
# from the NeMo Data Designer dataset by data/build_db.py. seed() loads it into
# the in-memory working set below — this replaced the old hardcoded customer
# seed + Python-generated history. Customers and their behavioural baselines are
# now DERIVED from real transaction data in the DB, the production-shaped
# direction (history in -> baseline out).
# Overridable via BARCLAYS_DB so tests point at a throwaway DB and never touch
# the real data/barclays.db (write-through would otherwise pollute it).
DB_PATH = (Path(os.environ["BARCLAYS_DB"]) if os.getenv("BARCLAYS_DB")
           else Path(__file__).resolve().parent.parent.parent / "data" / "barclays.db")

# Mock human-agent roster — stand-in for a real routing/ACD (automatic call
# distribution) system. A warm handoff needs an actual person to hand off to.
# UK fraud desks, matching the Barclays geography.
HUMAN_AGENTS = [
    {"agent_id": "spec_001", "name": "Rhys Morgan", "desk": "London Fraud Desk"},
    {"agent_id": "spec_002", "name": "Aisha Khan", "desk": "Manchester Fraud Desk"},
    {"agent_id": "spec_003", "name": "Daniel Fletcher", "desk": "Glasgow Fraud Desk"},
]
_next_agent_idx = 0

# (session_id, txn_id, action) -> the action dict that fired the first time.
# Retries of the same triple return the original result instead of firing again.
_EXECUTED_ACTIONS: dict[tuple[str, str, str], dict] = {}


def _ensure_db(force_rebuild: bool = False) -> None:
    """Build data/barclays.db from the dataset if it doesn't exist yet (so a
    fresh checkout / the test harness works with no manual step), or REBUILD it
    from scratch when force_rebuild=True — the reset path used by seed().
    Loaded by path (not `import data.build_db`) so it works regardless of how
    the process was launched."""
    if DB_PATH.exists() and not force_rebuild:
        return
    import importlib.util
    # build_db.py always lives in the project's data/ dir (next to bank.py's
    # package root), NOT next to the DB file — DB_PATH may be a temp/test path.
    build_script = Path(__file__).resolve().parent.parent.parent / "data" / "build_db.py"
    spec = importlib.util.spec_from_file_location("build_db", build_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_database(db_path=DB_PATH)  # build at OUR configured path (may be a throwaway test DB)


def _load_from_db() -> None:
    """Load customers + their transaction history from the SQLite DB into the
    in-memory working set. The baseline is stored flattened (cities /
    merchant_mix as JSON columns) — reassemble it into a CustomerBehavior here.
    Everything downstream keeps reading the same in-memory dicts, so only THIS
    function knows the data now comes from a database."""
    import sqlite3
    _ensure_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    for row in con.execute("SELECT * FROM customers"):
        behavior = CustomerBehavior(
            avg_gbp=row["avg_gbp"], p95_gbp=row["p95_gbp"],
            merchant_mix=json.loads(row["merchant_mix"]),
            cities=json.loads(row["cities"]), intl_travel=bool(row["intl_travel"]))
        CUSTOMERS[row["customer_id"]] = Customer(
            customer_id=row["customer_id"], name=row["name"], phone=row["phone"],
            card_last4=row["card_last4"], home_city=row["home_city"],
            account_type=row["account_type"], behavior=behavior)
    for row in con.execute("SELECT * FROM transactions"):
        TRANSACTIONS[row["txn_id"]] = Transaction(
            txn_id=row["txn_id"], customer_id=row["customer_id"], amount_gbp=row["amount_gbp"],
            merchant=row["merchant"], city=row["city"], channel=row["channel"],
            category=row["category"], mcc=row["mcc"], device_id=row["device_id"],
            ts_ms=row["ts_ms"], status=row["status"])

    # Load the persisted mutable state (may be absent in an old DB — guard it).
    # This is what makes runtime state survive a restart when RESET_ON_BOOT=false.
    try:
        for row in con.execute("SELECT * FROM card_actions ORDER BY id"):
            action = json.loads(row["detail"])
            CARD_ACTIONS.append(action)
            _EXECUTED_ACTIONS[(action["session_id"], action["txn_id"], action["action"])] = action
        for row in con.execute("SELECT * FROM cases"):
            CASES[row["session_id"]] = json.loads(row["data"])
        for row in con.execute("SELECT * FROM incidents"):
            INCIDENTS[row["incident_id"]] = json.loads(row["data"])
    except sqlite3.OperationalError:
        pass  # pre-persistence DB schema — nothing to load

    # Put a REAL phone number on one customer for live SMS/WhatsApp testing.
    # Read from the environment so the real number never lives in source or the
    # committed DB file. DEMO_PHONE_CUSTOMER chooses which customer (default the
    # flagship fraud customer, Emily) so triggering her fraud reaches your phone.
    demo_phone = os.getenv("DEMO_PHONE")
    if demo_phone:
        target = os.getenv("DEMO_PHONE_CUSTOMER", "cust_emily")
        if target in CUSTOMERS:
            CUSTOMERS[target].phone = demo_phone
    con.close()


def seed():
    """Reset to a clean slate: REBUILD the DB from the dataset (drops + recreates
    every table, re-derives baselines, and re-inserts the two fixed 'held' demo
    transactions — the Lagos fraud + the Currys false-positive), then reload it
    into memory. Used by the test harness and by an explicit demo reset.

    In production you run this ONCE (or `python data/build_db.py`) then start the
    server with RESET_ON_BOOT=false, so real activity persists across restarts
    instead of being wiped. The mutable-state tables come back empty on a reset."""
    global _next_agent_idx
    _ensure_db(force_rebuild=True)  # drop + recreate the whole DB from the dataset
    CUSTOMERS.clear(); TRANSACTIONS.clear(); CARD_ACTIONS.clear(); CASES.clear()
    INCIDENTS.clear(); _EXECUTED_ACTIONS.clear()
    _next_agent_idx = 0
    _load_from_db()


def get_customer_history(customer_id: str) -> list[Transaction]:
    """This customer's SETTLED past transactions (their baseline), newest first.
    This is the 'history data' the fraud model reads to decide if a new
    transaction is anomalous for THIS person."""
    return sorted(
        (t for t in TRANSACTIONS.values() if t.customer_id == customer_id and t.status == "settled"),
        key=lambda t: t.ts_ms, reverse=True)


def load_generated_dataset(path: str | Path, *, phone_prefix: str = "+44 7700 9000") -> int:
    """Loads NeMo Data Designer's Dataset 1 output (customer profiles +
    transactions with mcc/device_id/label) on top of whatever's already
    seeded. Returns the number of transactions loaded. Additive by design —
    call seed() first if you want a clean slate instead of a merge.

    Generated customer profiles don't carry a phone number (see
    CustomerProfile docstring in models.py) — `phone_prefix` plus the
    customer's position in the file fills in a synthetic one, since nothing
    downstream actually dials it in mock mode.
    """
    data = json.loads(Path(path).read_text())
    for i, raw_profile in enumerate(data.get("customers", [])):
        profile = CustomerProfile.model_validate(raw_profile)
        customer = profile.to_customer(phone=f"{phone_prefix}{i:04d}")
        CUSTOMERS[customer.customer_id] = customer

    loaded = 0
    for raw_txn in data.get("transactions", []):
        txn = _transaction_from_generated(raw_txn)
        TRANSACTIONS[txn.txn_id] = txn
        loaded += 1
    return loaded


def _transaction_from_generated(raw: dict) -> Transaction:
    """Bridges the generator's shape (ISO `ts`, nested `label`) onto the
    runtime Transaction model (epoch-ms `ts_ms`). Kept as one small function
    so there's exactly one place that knows about this schema difference —
    see the note in models.py's module docstring."""
    from datetime import datetime

    fields = dict(raw)
    ts_iso = fields.pop("ts", None)
    if ts_iso:
        fields["ts_ms"] = int(datetime.fromisoformat(ts_iso).timestamp() * 1000)
    if "amount_inr" in fields and "amount_gbp" not in fields:  # legacy generator output
        fields["amount_gbp"] = fields.pop("amount_inr")
    label_raw = fields.pop("label", None)
    if label_raw:
        fields["label"] = TxnLabel.model_validate(label_raw)
    return Transaction.model_validate(fields)


def insert_transaction(txn: Transaction) -> Transaction:
    """Record a NEW transaction (a real-world swipe/checkout) in the working set
    AND persist it, so an inserted transaction survives a restart too."""
    TRANSACTIONS[txn.txn_id] = txn
    _db_exec("INSERT OR REPLACE INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
             (txn.txn_id, txn.customer_id, txn.amount_gbp, txn.merchant, txn.city, txn.channel,
              txn.category, txn.mcc, txn.device_id, txn.ts_ms, txn.status,
              int(txn.label.is_fraud) if txn.label else 0))
    return txn


def _idempotency_key(session_id: str, txn_id: str, action: str) -> tuple[str, str, str]:
    return (session_id, txn_id, action)


def _persist_action(action: dict) -> None:
    """Write-through a card action to the durable card_actions table."""
    _db_exec("INSERT INTO card_actions (action, session_id, txn_id, customer_id, ts, detail) "
             "VALUES (?,?,?,?,?,?)",
             (action["action"], action.get("session_id"), action.get("txn_id"),
              action.get("customer_id"), action.get("ts"), json.dumps(action)))


def release_hold(session_id: str, txn_id: str) -> dict:
    key = _idempotency_key(session_id, txn_id, "release_hold")
    if key in _EXECUTED_ACTIONS:
        return _EXECUTED_ACTIONS[key]
    TRANSACTIONS[txn_id].status = "released"
    action = {"action": "release_hold", "txn_id": txn_id, "session_id": session_id, "ts": now_ms()}
    CARD_ACTIONS.append(action)
    _EXECUTED_ACTIONS[key] = action
    _db_exec("UPDATE transactions SET status='released' WHERE txn_id=?", (txn_id,))
    _persist_action(action)
    return action


def block_card(session_id: str, txn_id: str, customer_id: str) -> dict:
    key = _idempotency_key(session_id, txn_id, "block_card")
    if key in _EXECUTED_ACTIONS:
        return _EXECUTED_ACTIONS[key]
    action = {"action": "block_card", "customer_id": customer_id, "txn_id": txn_id,
              "session_id": session_id, "reissue_ordered": True, "ts": now_ms()}
    CARD_ACTIONS.append(action)
    _EXECUTED_ACTIONS[key] = action
    _persist_action(action)
    return action


def open_chargeback(session_id: str, txn_id: str) -> dict:
    key = _idempotency_key(session_id, txn_id, "open_chargeback")
    if key in _EXECUTED_ACTIONS:
        return _EXECUTED_ACTIONS[key]
    TRANSACTIONS[txn_id].status = "blocked"
    action = {"action": "open_chargeback", "txn_id": txn_id, "session_id": session_id, "ts": now_ms()}
    CARD_ACTIONS.append(action)
    _EXECUTED_ACTIONS[key] = action
    _db_exec("UPDATE transactions SET status='blocked' WHERE txn_id=?", (txn_id,))
    _persist_action(action)
    return action


def upsert_case(session_id: str, **fields) -> dict:
    case = CASES.setdefault(session_id, {"case_id": new_id("case"), "session_id": session_id})
    case.update(fields, updated_ts=now_ms())
    _db_exec("INSERT OR REPLACE INTO cases (session_id, case_id, data, updated_ts) VALUES (?,?,?,?)",
             (session_id, case["case_id"], json.dumps(case, default=str), case["updated_ts"]))
    return case


def raise_incident(session_id: str, *, customer_id: str, customer_name: str, txn_id: str,
                   amount_gbp: float, merchant: str, city: str, rca_reason: str,
                   handoff_reason: str, specialist: dict, card_already_blocked: bool) -> dict:
    """Raise a durable INCIDENT — the request for a human fraud analyst to look
    into an escalated case (adopted from Neha's `_raise_incident`). Our live
    call can't block on a person mid-call, so escalation warm-hands-off AND
    creates this OPEN work item; a human actions it out-of-band via
    resolve_incident(). That mirrors the real flow: the call ends, a ticket
    sits in the analyst's queue with the full context, they decide later."""
    incident_id = new_id("INC")
    INCIDENTS[incident_id] = {
        "incident_id": incident_id, "session_id": session_id, "status": "open",
        "customer_id": customer_id, "customer_name": customer_name,
        "txn_id": txn_id, "amount_gbp": amount_gbp, "merchant": merchant, "city": city,
        "rca_reason": rca_reason, "handoff_reason": handoff_reason,
        "assigned_to": specialist["name"], "desk": specialist["desk"],
        "card_already_blocked": card_already_blocked,
        "opened_ts": now_ms(), "resolution": None,
    }
    _db_exec("INSERT OR REPLACE INTO incidents (incident_id, session_id, status, data, opened_ts) "
             "VALUES (?,?,?,?,?)",
             (incident_id, session_id, "open", json.dumps(INCIDENTS[incident_id], default=str),
              INCIDENTS[incident_id]["opened_ts"]))
    return INCIDENTS[incident_id]


def resolve_incident(incident_id: str, decision: str) -> dict:
    """A human analyst resolves an OPEN incident with 'block' or 'approve' — the
    completion of the human-in-the-loop review. Fires the corresponding bank
    action (idempotency-keyed like every other action) and marks the incident
    resolved. Returns {"incident", "actions"} so the caller can audit the
    actions in the right layer (bank.py stays audit-free)."""
    inc = INCIDENTS.get(incident_id)
    if inc is None:
        raise KeyError(incident_id)
    if inc["status"] == "resolved":
        return {"incident": inc, "actions": []}  # idempotent

    sid, txn_id, cust_id = inc["session_id"], inc["txn_id"], inc["customer_id"]
    if "block" in decision.lower():
        actions = [block_card(sid, txn_id, cust_id), open_chargeback(sid, txn_id)]
        outcome = "fraud_confirmed_by_analyst"
    else:
        actions = [release_hold(sid, txn_id)]
        outcome = "approved_by_analyst"

    inc["status"] = "resolved"
    inc["resolution"] = {"decision": decision.lower(), "outcome": outcome, "resolved_ts": now_ms()}
    _db_exec("UPDATE incidents SET status='resolved', data=? WHERE incident_id=?",
             (json.dumps(inc, default=str), incident_id))
    upsert_case(sid, outcome=outcome, note=f"Analyst resolved incident {incident_id} -> {decision}")
    return {"incident": inc, "actions": actions}


def assign_specialist() -> dict:
    """Round-robin pick from HUMAN_AGENTS — see that list's docstring for
    why this exists. Not randomized on purpose: round-robin is
    deterministic and testable, and spreads escalations evenly across the
    roster the same way a real ACD's least-busy-agent routing would,
    without needing to model actual agent availability for a demo."""
    global _next_agent_idx
    agent = HUMAN_AGENTS[_next_agent_idx % len(HUMAN_AGENTS)]
    _next_agent_idx += 1
    return agent


# How the process starts up:
#   RESET_ON_BOOT=true  (default) — clean slate every boot: rebuild the DB from
#       the dataset and reload. Repeatable, pristine demos + the test harness.
#   RESET_ON_BOOT=false — PRODUCTION: load whatever's persisted (customers,
#       transactions with their current status, card actions, cases, incidents)
#       so real activity survives restarts instead of being wiped. Run
#       `python data/build_db.py` once to initialise, then deploy with this off.
if os.getenv("RESET_ON_BOOT", "true").lower() == "true":
    seed()
else:
    _load_from_db()

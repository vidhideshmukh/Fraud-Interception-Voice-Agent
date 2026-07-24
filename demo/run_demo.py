"""End-to-end demo driver (mock mode). Runs all three paths:
  A. card saved   — customer confirms the purchase (false positive recovered)
  B. fraud caught — customer denies, card blocked + reissue + chargeback
  C. no answer    — timeout, hold kept (the judges WILL ask about this one)
Then prints the audit chain summary and verifies integrity.
Usage: python demo/run_demo.py"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MOCK_MODE", "true")

from app.core import orchestrator                      # noqa: E402
from app.security import audit
from app.database import bank
from app.core import fraud_trigger  # noqa: E402


def banner(title):
    print("\n" + "=" * 62 + f"\n  {title}\n" + "=" * 62)


def turn(session, text):
    print(f"\n  CUSTOMER: {text}")
    reply = orchestrator.customer_says(session, text)
    print(f"  AGENT:    {reply}")


def path_a():
    banner("PATH A — false positive recovered (card saved)")
    event = fraud_trigger.emit_event("txn_gadget_02")
    print(f"  [fraud model] risk={event.risk_score}  RCA: {event.rca_reason[:70]}...")
    session = orchestrator.start_call(event)
    print(f"  [app push] verification code on the customer's phone: {session.verification_code}")
    print(f"  AGENT:    {session.transcript[-1]['text']}")
    turn(session, session.verification_code)               # customer reads the code back
    turn(session, "Yes, I did make that purchase, that's my new laptop")
    assert session.state.value == "resolved_legit", session.state
    print(f"  >> outcome: {session.outcome}")
    return session


def path_b():
    banner("PATH B — fraud caught (card blocked before money moves)")
    event = fraud_trigger.emit_event("txn_lagos_01")
    print(f"  [fraud model] risk={event.risk_score}  RCA: {event.rca_reason[:70]}...")
    session = orchestrator.start_call(event)
    print(f"  [app push] verification code on the customer's phone: {session.verification_code}")
    print(f"  AGENT:    {session.transcript[-1]['text']}")
    turn(session, session.verification_code)
    turn(session, "No, that was not me, I have never been to Lagos!")
    assert session.state.value == "resolved_fraud", session.state
    print(f"  >> outcome: {session.outcome}")
    return session


def path_c():
    banner("PATH C — no answer (hold kept, retry scheduled)")
    event = fraud_trigger.emit_event("txn_lagos_01")
    session = orchestrator.start_call(event)
    result = orchestrator.no_answer(session)
    print(f"  >> {result}")
    return session


if __name__ == "__main__":
    bank.seed()
    audit.reset()
    path_a()
    path_b()
    path_c()

    banner("AUDIT CHAIN (what the judges see)")
    for r in audit.get_log():
        print(f"  #{r['seq']:>2} {r['kind']:<8} {r['prev_hash'][:8]} -> {r['hash'][:8]}  "
              f"{str(r['payload'])[:58]}")
    ok = audit.verify_chain()
    print(f"\n  chain integrity: {'VALID' if ok else '*** BROKEN ***'}")
    print(f"  card actions: {bank.CARD_ACTIONS}")
    assert ok
    print("\n  All demo paths passed.")

"""Investigation uplift eval — the number that justifies the investigation agent.

On the AMBIGUOUS cases, compares two classifiers against ground truth:
  1. the deterministic amount+geo scorer (app/core/fraud_trigger.assess), and
  2. the Investigation Agent (app/agents/investigation_agent.investigate), which
     reasons over device + velocity + location + amount together.

If the agent is worth its cost, its accuracy on the ambiguous class should be far
above the scorer's (which is near a coin-flip there, by construction). Runs in
MOCK mode by default (the transparent multi-signal heuristic); set MOCK_MODE=false
to have Nemotron do the reasoning instead.

Run:  python eval/investigation_uplift.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MOCK_MODE", "true")

from app.database import bank                       # noqa: E402
from app.database.models import Transaction         # noqa: E402
from app.core import fraud_trigger                   # noqa: E402
from app.agents import investigation_agent           # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "generated" / "fraud_signals_dataset.json"


def _scorer_predicts_fraud(case: dict, cust) -> bool:
    """The deterministic amount+geo scorer's verdict (score >= alert threshold)."""
    cat = case["merchant_category"].lower()
    chan = "online" if cat in ("online", "travel", "electronics") else "card_present"
    txn = Transaction(txn_id="t", customer_id=cust.customer_id, amount_gbp=float(case["amount"]),
                      merchant=case["merchant"], city=case["current_location"], channel=chan, category=cat)
    return fraud_trigger.assess(txn).score >= fraud_trigger.RISK_THRESHOLD


def _confusion(preds: list[bool], labels: list[bool]) -> dict:
    tp = sum(1 for p, y in zip(preds, labels) if p and y)
    tn = sum(1 for p, y in zip(preds, labels) if not p and not y)
    fp = sum(1 for p, y in zip(preds, labels) if p and not y)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y)
    n = len(labels)
    return {"acc": (tp + tn) / n if n else 0, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main() -> None:
    records = json.loads(DATA.read_text(encoding="utf-8"))
    by_name = {c.name: c for c in bank.CUSTOMERS.values()}
    ambiguous = [r for r in records if r.get("scenario") == "ambiguous" and r["customer_name"] in by_name]

    labels, scorer_preds, agent_preds = [], [], []
    for r in ambiguous:
        cust = by_name[r["customer_name"]]
        p95 = cust.behavior.p95_gbp if cust.behavior else None
        labels.append(bool(r["isFraud"]))
        scorer_preds.append(_scorer_predicts_fraud(r, cust))
        agent_preds.append(investigation_agent.investigate(r, p95).predicted_fraud)

    s, a = _confusion(scorer_preds, labels), _confusion(agent_preds, labels)
    print(f"Ambiguous cases evaluated: {len(ambiguous)}  (mode: {'mock heuristic' if investigation_agent.MOCK_MODE else 'Nemotron'})\n")
    print(f"{'classifier':<28}{'accuracy':>10}{'false-pos':>11}{'false-neg':>11}")
    print(f"{'-'*60}")
    print(f"{'deterministic amount+geo':<28}{s['acc']*100:>9.0f}%{s['fp']:>11}{s['fn']:>11}")
    print(f"{'Investigation Agent':<28}{a['acc']*100:>9.0f}%{a['fp']:>11}{a['fn']:>11}")
    print(f"\nUplift on the ambiguous class: {s['acc']*100:.0f}%  ->  {a['acc']*100:.0f}%  "
          f"(+{(a['acc']-s['acc'])*100:.0f} points)")


if __name__ == "__main__":
    main()

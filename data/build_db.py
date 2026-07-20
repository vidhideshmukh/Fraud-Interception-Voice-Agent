"""Build the SQLite banking database from the NeMo Data Designer dataset.

    data/generated/fraud_signals_dataset.json  ──►  data/barclays.db
                                                     (customers + transactions)

This is the INGESTION step — the bridge from "generated data at rest" to "the
real DB the backend runs on". It does two production-shaped things:

  1. NORMALISES the flat, per-event dataset into two relational tables:
     `customers` (one row per person) and `transactions` (one row per event).

  2. DERIVES each customer's behavioural BASELINE from their own transactions —
     average + 95th-percentile spend, usual cities, merchant mix. This is the
     real-world direction: history in → baseline computed → fraud scored against
     it (the demo's old bank.py did it backwards, inventing a baseline first).

Re-run any time to rebuild the DB from the dataset (it drops + recreates). When
your colleague's corrected NeMo file lands, point --json at it and re-run — the
rest of the app doesn't change.

Run:  python data/build_db.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "data" / "generated" / "fraud_signals_dataset.json"
DB_PATH = ROOT / "data" / "barclays.db"

# The 5 customers the live demo + test suite reference by a STABLE id/card
# (e.g. the flagship transaction txn_lagos_01 -> cust_emily). We map the
# dataset's people onto those identities by name so nothing downstream breaks;
# everyone else gets an id/card derived from their own record. In a real
# deployment this map is the join between the analytics dataset's customer_ids
# and the core-banking account records.
_KNOWN_IDENTITY = {
    "Emily Clarke":    ("cust_emily",     "4821", "Premier"),
    "James Whitmore":  ("cust_james",     "7710", "Standard"),
    "Sophie Bennett":  ("cust_sophie",    "3390", "Student"),
    "Oliver Grant":    ("cust_oliver",    "6654", "Standard"),
    "Charlotte Reid":  ("cust_charlotte", "9043", "Premier"),
}

# UK cities, to derive intl_travel: a customer whose usual locations are all in
# the UK has intl_travel=False (a foreign charge is then high-risk for them).
_UK = {"london", "manchester", "birmingham", "leeds", "glasgow", "edinburgh", "bristol",
       "liverpool", "cardiff", "newcastle", "sheffield", "nottingham", "leicester"}

_MCC = {"grocery": "5411", "dining": "5812", "food": "5812", "retail": "5651", "fuel": "5541",
        "travel": "4722", "online": "5969", "transport": "4111", "electronics": "5732"}


def _percentile(values: list[float], p: float) -> float:
    """Simple p-th percentile (no numpy). Good enough for a per-customer p95
    over a few dozen transactions."""
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((p / 100) * (len(s) - 1)))
    return s[k]


def _account_type_from_spend(p95: float) -> str:
    """Infer a Barclays account tier from the customer's high-spend level —
    a stand-in for the real product flag on the account record."""
    if p95 < 200:
        return "Student"
    if p95 < 600:
        return "Standard"
    return "Premier"


def _identity(name: str, phone: str) -> tuple[str, str, str | None]:
    """(customer_id, card_last4, account_type|None). Known demo customers keep
    their fixed identity; others get a slug id + last-4-of-phone card, with the
    account type derived from spend later."""
    if name in _KNOWN_IDENTITY:
        return _KNOWN_IDENTITY[name]
    cust_id = "cust_" + name.split()[0].lower()
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    return cust_id, (digits[-4:] or "0000"), None


def _ts_ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def build_database(json_path: Path = JSON_PATH, db_path: Path = DB_PATH) -> tuple[int, int]:
    if not Path(json_path).exists():
        raise SystemExit(f"[build_db] dataset not found: {json_path}\n"
                         f"  Generate it first: python data/generate_fraud_signals_dataset.py")
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # Group the flat events by their dataset customer_id.
    by_customer: dict[str, list[dict]] = {}
    for r in records:
        by_customer.setdefault(r["customer_id"], []).append(r)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # rebuild from scratch every run
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE customers (
            customer_id   TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            phone         TEXT,
            card_last4    TEXT,
            home_city     TEXT,
            account_type  TEXT,
            avg_gbp       REAL,          -- baseline: typical spend
            p95_gbp       REAL,          -- baseline: high-but-normal ceiling
            cities        TEXT,          -- baseline: usual cities (JSON array)
            merchant_mix  TEXT,          -- baseline: category fractions (JSON object)
            intl_travel   INTEGER        -- baseline: 0/1 does this customer spend abroad
        );
        CREATE TABLE transactions (
            txn_id        TEXT PRIMARY KEY,
            customer_id   TEXT NOT NULL,
            amount_gbp    REAL,
            merchant      TEXT,
            city          TEXT,
            channel       TEXT,          -- card_present | online
            category      TEXT,
            mcc           TEXT,
            device_id     TEXT,
            ts_ms         INTEGER,
            status        TEXT,          -- settled (history) | held | released | blocked
            is_fraud      INTEGER,       -- ground-truth label from the dataset
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        );
        -- Mutable-state tables — written through at runtime so state survives a
        -- restart (see bank.py). Empty at build time; filled as calls resolve.
        CREATE TABLE card_actions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            action      TEXT,            -- block_card | release_hold | open_chargeback
            session_id  TEXT,
            txn_id      TEXT,
            customer_id TEXT,
            ts          INTEGER,
            detail      TEXT             -- JSON of the full action dict
        );
        CREATE TABLE cases (
            session_id  TEXT PRIMARY KEY,
            case_id     TEXT,
            data        TEXT,            -- JSON of the full case dict
            updated_ts  INTEGER
        );
        CREATE TABLE incidents (
            incident_id TEXT PRIMARY KEY,
            session_id  TEXT,
            status      TEXT,            -- open | resolved
            data        TEXT,            -- JSON of the full incident dict
            opened_ts   INTEGER
        );
    """)

    tx_counter = 0
    for rows in by_customer.values():
        name, phone = rows[0]["customer_name"], rows[0]["phone_number"]
        cust_id, last4, account_type = _identity(name, phone)

        # Baseline is computed from the customer's LEGIT transactions only —
        # including known fraud would inflate their "normal" ceiling.
        legit = [r for r in rows if not r["isFraud"]] or rows
        amounts = [float(r["amount"]) for r in legit]
        avg_gbp = round(statistics.mean(amounts), 2)
        p95_gbp = round(_percentile(amounts, 95), 2)
        if account_type is None:
            account_type = _account_type_from_spend(p95_gbp)

        cities = rows[0]["known_locations"]                         # baseline cities
        mix = Counter(r["merchant_category"].lower() for r in legit)
        total = sum(mix.values()) or 1
        merchant_mix = {k: round(v / total, 3) for k, v in mix.items()}
        intl_travel = int(any(c.strip().lower() not in _UK for c in cities))
        home_counter = Counter(r["current_location"] for r in legit)
        home_city = home_counter.most_common(1)[0][0] if home_counter else (cities[0] if cities else "London")

        con.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cust_id, name, phone, last4, home_city, account_type, avg_gbp, p95_gbp,
             json.dumps(cities), json.dumps(merchant_mix), intl_travel))

        # Every event becomes a settled transaction (this customer's history).
        for r in rows:
            tx_counter += 1
            cat = r["merchant_category"].lower()
            online = cat in ("online", "travel", "electronics")
            con.execute(
                "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"txn_db_{tx_counter:05d}", cust_id, float(r["amount"]), r["merchant"],
                 r["current_location"], "online" if online else "card_present", cat,
                 _MCC.get(cat, "5999"), f"dev_{cust_id}_known" if online else None,
                 _ts_ms(r["current_transaction_time"]), "settled", int(r["isFraud"])))

    # Two fixed 'held' demo transactions with stable ids the demo + tests rely
    # on (the flagship Lagos fraud + the Currys false-positive). In the DB now
    # (was added in-memory before) so they load and persist like any other txn.
    demo_txns = [
        ("txn_lagos_01", "cust_emily", 680.0, "Electronics Bazaar", "Lagos",
         "card_present", "electronics", "5732", None),
        ("txn_gadget_02", "cust_james", 1299.0, "Currys Online", "Manchester",
         "online", "electronics", "5732", "dev_cust_james_known"),
    ]
    for tid, cid, amt, merch, city, chan, cat, mcc, dev in demo_txns:
        if con.execute("SELECT 1 FROM customers WHERE customer_id=?", (cid,)).fetchone():
            con.execute("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (tid, cid, amt, merch, city, chan, cat, mcc, dev, 0, "held", 0))

    con.commit()
    n_cust = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    n_txn = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    con.close()
    print(f"[build_db] wrote {db_path}")
    print(f"[build_db] {n_cust} customers, {n_txn} transactions "
          f"(baselines derived from each customer's legit history)")
    return n_cust, n_txn


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(JSON_PATH), help="source dataset JSON")
    ap.add_argument("--db", default=str(DB_PATH), help="output SQLite path")
    args = ap.parse_args()
    build_database(Path(args.json), Path(args.db))

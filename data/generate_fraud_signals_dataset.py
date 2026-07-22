"""Generate the fraud-signals dataset in the schema the team standardised on.

Each row is ONE self-contained transaction event carrying, inline, everything a
fraud model needs to judge it: the customer's known devices/locations (their
baseline) PLUS the current device/location/amount/timing — and the ground-truth
`isFraud` label. Exactly this shape (Barclays UK, GBP):

    {
      "customer_name": "Emily Clarke",
      "customer_id": "C481900231",
      "phone_number": "+44 7700 900341",
      "merchant": "Currys Online",
      "merchant_category": "Electronics",
      "amount": 4500.0,
      "known_devices": ["iPhone 14", "MacBook Air"],
      "current_device": "Samsung Galaxy S22",
      "known_locations": ["London", "Manchester"],
      "current_location": "Lagos",
      "previous_transaction_time": "2026-07-18T09:45:00Z",
      "current_transaction_time": "2026-07-18T10:05:00Z",
      "velocity_minutes": 20,
      "isFraud": true
    }

Why the label is trustworthy: fraud rows are built by DELIBERATELY breaking the
customer's baseline (new device, impossible location, impossible travel-velocity,
or amount spike), and legit rows stay entirely within it. So `isFraud` is not a
guess — it's true by construction, which is what makes this dataset usable to
measure a classifier honestly.

Two generation engines (same idea as data/generate_datasets.py):

  --engine python  (default) — pure, deterministic-with-seed, offline. The
    guaranteed-available path; labels are correct by construction.

  --engine datadesigner — the REAL NeMo Data Designer (`data-designer` package)
    generating against the NVIDIA Build API. Data Designer samples the base
    columns and writes each merchant name with a Nemotron LLM column; we then
    stamp the device/location/velocity signals so the `isFraud` label stays
    coherent (a sampler alone can't guarantee "fraud row ⇒ signals actually
    broken"). This keeps "we used NVIDIA data generation" literally true.

Run:
    python data/generate_fraud_signals_dataset.py --n 120
    python data/generate_fraud_signals_dataset.py --n 120 --engine datadesigner
Outputs data/generated/fraud_signals_dataset.json  (+ .csv for eyeballing).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_DIR = Path(__file__).parent / "generated"

# ---------------------------------------------------------------------------
# The Barclays UK customer base — same people as the live app (app/database/
# bank.py), so the dataset and the demo tell one consistent story. Each carries
# the baseline a transaction is judged against: usual devices, usual UK cities,
# typical spend, and the merchant categories they actually use.
#   avg_gbp / high_gbp : normal spend and their typical HIGH (p95-ish) spend.
#   A fraud amount-spike is generated well above high_gbp; legit stays under it.
# ---------------------------------------------------------------------------
CUSTOMERS = [
    dict(name="Emily Clarke",     cid="C481900231", phone="+44 7700 900341",
         devices=["iPhone 14", "MacBook Air"],            locations=["London", "Manchester"],
         avg_gbp=85,  high_gbp=650,  cats=["Dining", "Retail", "Travel", "Grocery"]),
    dict(name="James Whitmore",   cid="C771002884", phone="+44 7700 900112",
         devices=["Samsung Galaxy S22"],                  locations=["Manchester", "Leeds"],
         avg_gbp=40,  high_gbp=280,  cats=["Grocery", "Fuel", "Dining", "Online"]),
    dict(name="Sophie Bennett",   cid="C339045170", phone="+44 7700 900733",
         devices=["iPhone 12"],                           locations=["Edinburgh"],
         avg_gbp=22,  high_gbp=140,  cats=["Dining", "Online", "Transport"]),
    dict(name="Oliver Grant",     cid="C665401938", phone="+44 7700 900558",
         devices=["Google Pixel 7", "iPad Air"],          locations=["Birmingham", "London"],
         avg_gbp=55,  high_gbp=430,  cats=["Retail", "Fuel", "Dining", "Grocery"]),
    dict(name="Charlotte Reid",   cid="C904377612", phone="+44 7700 900920",
         devices=["iPhone 15 Pro", "MacBook Pro"],        locations=["Bristol", "London"],
         avg_gbp=130, high_gbp=1100, cats=["Travel", "Dining", "Retail", "Grocery"]),
    dict(name="Liam Foster",      cid="C210567433", phone="+44 7700 900274",
         devices=["OnePlus 11"],                          locations=["Leeds"],
         avg_gbp=38,  high_gbp=300,  cats=["Grocery", "Online", "Dining"]),
    dict(name="Grace Turner",     cid="C158823406", phone="+44 7700 900651",
         devices=["iPhone 13"],                           locations=["Glasgow", "Edinburgh"],
         avg_gbp=60,  high_gbp=500,  cats=["Retail", "Dining", "Travel"]),
    dict(name="Noah Patel",       cid="C337719205", phone="+44 7700 900388",
         devices=["Samsung Galaxy S23", "iPad Pro"],      locations=["London"],
         avg_gbp=75,  high_gbp=600,  cats=["Online", "Dining", "Retail", "Electronics"]),
    dict(name="Amelia Hughes",    cid="C902144870", phone="+44 7700 900517",
         devices=["iPhone 14 Pro"],                       locations=["Cardiff", "Bristol"],
         avg_gbp=48,  high_gbp=360,  cats=["Grocery", "Retail", "Dining"]),
    dict(name="Jack Wilson",      cid="C556201947", phone="+44 7700 900806",
         devices=["Google Pixel 8"],                      locations=["Liverpool", "Manchester"],
         avg_gbp=52,  high_gbp=410,  cats=["Fuel", "Grocery", "Online", "Dining"]),
]

# UK merchants per category — realistic names the current_location can host.
MERCHANTS = {
    "Grocery":     ["Tesco", "Sainsbury's", "Waitrose", "M&S Food", "Co-op", "Aldi"],
    "Dining":      ["Nando's", "Pret A Manger", "Wagamama", "Greggs", "The Ivy"],
    "Retail":      ["John Lewis", "Next", "ASOS", "Selfridges", "Zara", "Marks & Spencer"],
    "Fuel":        ["Shell", "BP", "Esso", "Texaco"],
    "Travel":      ["Trainline", "British Airways", "Booking.com", "easyJet"],
    "Online":      ["Amazon UK", "eBay", "Argos", "Currys Online"],
    "Transport":   ["TfL", "Lothian Buses", "Uber", "Trainline"],
    "Electronics": ["Currys", "Apple Store", "Argos", "Currys Online"],
}

# Devices a customer would NOT own — used to synthesize the "new device" signal.
UNKNOWN_DEVICES = ["Samsung Galaxy S22", "Xiaomi Redmi Note 12", "Motorola Edge 40",
                   "Huawei P60", "Nokia G60", "unknown Android device", "Windows PC (new)"]
# Places outside the customer's usual set — foreign = strongest location signal.
FOREIGN_CITIES = ["Lagos", "Dubai", "Bangkok", "Manila", "Kuala Lumpur", "Istanbul", "Accra"]
# Far-flung UK cities used for a softer domestic-location anomaly.
FAR_UK_CITIES = ["Aberdeen", "Plymouth", "Norwich", "Swansea", "Inverness", "Belfast"]

# The four fraud signal patterns we inject. Each returns the fields it overrides
# on top of an otherwise-normal transaction, so a fraud row is a legit row with
# one (or more) baselines deliberately broken.
FRAUD_PATTERNS = ["new_device", "impossible_location", "impossible_travel", "amount_spike"]


def _iso(dt: datetime) -> str:
    """UTC ISO-8601 with a trailing Z, matching the schema's time fields."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_merchant(raw: str, category: str, rng: random.Random) -> str:
    """Sanitize an LLM merchant-column value. The model is asked for ONLY a
    merchant name, but occasionally returns prose ("Amazon.uk\\n\\nNote: since
    you asked..."). Take the first line, strip quote/markdown noise, and if it
    still looks like a sentence (too long, or contains explainer words), fall
    back to a real sampled merchant so the dataset never carries a paragraph as
    a merchant name."""
    first = str(raw).strip().splitlines()[0].strip().strip('"\'*` ').strip()
    looks_like_prose = len(first) > 40 or any(
        w in first.lower() for w in ("note:", "here is", "here's", "since you", "alternative", "sorry"))
    if not first or looks_like_prose:
        return rng.choice(MERCHANTS.get(category, ["Merchant"]))
    return first


def _legit_amount(cust: dict, rng: random.Random) -> float:
    """A normal spend for this customer: lognormal-ish around avg, capped under
    their typical high so it never accidentally looks like a spike."""
    amt = abs(rng.lognormvariate(0, 0.5)) * cust["avg_gbp"]
    return round(min(amt, cust["high_gbp"] * 0.9), 2)


def _build_record(cust: dict, is_fraud: bool, rng: random.Random) -> dict:
    """Build one transaction event for `cust`, fraudulent or not.

    Legit  : current device & location are from the customer's known set, amount
             is in-pattern, and the gap since the previous transaction is a
             believable few hours to a couple of days.
    Fraud  : start from that same legit shape, then apply ONE random fraud
             pattern that breaks a baseline — which is what makes isFraud=True
             true by construction rather than by opinion.
    """
    category = rng.choice(cust["cats"])
    merchant = rng.choice(MERCHANTS[category])
    current_device = rng.choice(cust["devices"])
    current_location = rng.choice(cust["locations"])
    amount = _legit_amount(cust, rng)

    # Previous transaction: a few hours to ~2 days before "now"; current follows
    # it by a normal gap. velocity_minutes = minutes between the two.
    now = datetime.now(timezone.utc)
    prev_time = now - timedelta(minutes=rng.randint(30, 2 * 24 * 60))
    velocity_minutes = rng.randint(90, 26 * 60)  # legit: 1.5h .. ~1 day
    curr_time = prev_time + timedelta(minutes=velocity_minutes)

    if is_fraud:
        pattern = rng.choice(FRAUD_PATTERNS)
        if pattern == "new_device":
            # Card-not-present from a device the customer has never used.
            current_device = rng.choice([d for d in UNKNOWN_DEVICES if d not in cust["devices"]])
            category = rng.choice(["Online", "Electronics"])
            merchant = rng.choice(MERCHANTS[category])
        elif pattern == "impossible_location":
            # A charge somewhere the customer simply does not transact.
            current_location = rng.choice(FOREIGN_CITIES + FAR_UK_CITIES)
        elif pattern == "impossible_travel":
            # Physically-impossible: a second charge in a different country only
            # minutes after the last one — no one travels that fast.
            current_location = rng.choice(FOREIGN_CITIES)
            velocity_minutes = rng.randint(3, 25)
            curr_time = prev_time + timedelta(minutes=velocity_minutes)
            current_device = rng.choice(cust["devices"])  # cloned card, card-present abroad
            category = rng.choice(["Retail", "Electronics"])
            merchant = rng.choice(MERCHANTS[category])
        elif pattern == "amount_spike":
            # Spend far above anything in this account's history.
            amount = round(cust["high_gbp"] * rng.uniform(3.0, 8.0), 2)
            category = rng.choice(["Electronics", "Retail", "Online"])
            merchant = rng.choice(MERCHANTS[category])

    return {
        "customer_name": cust["name"],
        "customer_id": cust["cid"],
        "phone_number": cust["phone"],
        "merchant": merchant,
        "merchant_category": category,
        "amount": amount,
        "known_devices": list(cust["devices"]),
        "current_device": current_device,
        "known_locations": list(cust["locations"]),
        "current_location": current_location,
        "previous_transaction_time": _iso(prev_time),
        "current_transaction_time": _iso(curr_time),
        "velocity_minutes": velocity_minutes,
        "isFraud": is_fraud,
    }


def generate_python(n: int, fraud_rate: float, seed: int) -> list[dict]:
    """Pure-Python engine: n records, ~fraud_rate fraudulent, labels correct by
    construction. Deterministic for a given seed so a run is reproducible."""
    rng = random.Random(seed)
    n_fraud = round(n * fraud_rate)
    flags = [True] * n_fraud + [False] * (n - n_fraud)
    rng.shuffle(flags)
    records = [_build_record(rng.choice(CUSTOMERS), is_fraud, rng) for is_fraud in flags]
    return records


def generate_datadesigner(n: int, fraud_rate: float, seed: int, model: str) -> list[dict]:
    """REAL NeMo Data Designer engine. Data Designer samples the customer and the
    fraud flag and writes each merchant name with a Nemotron LLM column on NVIDIA
    Build; we then stamp the device/location/velocity/amount signals so the
    `isFraud` label stays coherent (a sampler can't guarantee a fraud row's
    baselines are actually broken — that's the one thing that must be exact)."""
    from dotenv import load_dotenv  # load .env so NVIDIA_API_KEY works when run standalone
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("[datadesigner] no NVIDIA_API_KEY in env — use --engine python for offline.")

    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    provider = dd.ModelProvider(name="nvidia_build", endpoint="https://integrate.api.nvidia.com/v1",
                                provider_type="openai", api_key=api_key)
    b = dd.DataDesignerConfigBuilder(
        model_configs=[dd.ModelConfig(alias="nvidia-text", model=model, provider="nvidia_build")])

    b.add_column(dd.SamplerColumnConfig(name="cid", sampler_type=dd.SamplerType.CATEGORY,
                 params=dd.CategorySamplerParams(values=[c["cid"] for c in CUSTOMERS])))
    b.add_column(dd.SamplerColumnConfig(name="merchant_category", sampler_type=dd.SamplerType.CATEGORY,
                 params=dd.CategorySamplerParams(values=list(MERCHANTS.keys()))))
    b.add_column(dd.SamplerColumnConfig(name="is_fraud", sampler_type=dd.SamplerType.BERNOULLI,
                 params=dd.BernoulliSamplerParams(p=fraud_rate)))
    # The LLM column: a real Nemotron call per row for a plausible UK merchant
    # name in the sampled category — the "generated by an LLM" part of the data.
    b.add_column(dd.LLMTextColumnConfig(name="merchant", model_alias="nvidia-text",
                 system_prompt="You name UK high-street/online merchants. Reply with ONLY the merchant name, no quotes, no preamble.",
                 prompt="Give the name of one realistic UK merchant in the '{{ merchant_category }}' category."))

    print(f"[datadesigner] generating {n} rows via {model} on NVIDIA Build…")
    dsgn = DataDesigner(model_providers=[provider])
    results = dsgn.create(config_builder=b, num_records=n)
    df = results.load_dataset()

    by_cid = {c["cid"]: c for c in CUSTOMERS}
    rng = random.Random(seed)
    records = []
    for row in df.itertuples(index=False):
        r = row._asdict()
        cust = by_cid[r["cid"]]
        is_fraud = bool(int(r["is_fraud"]))
        rec = _build_record(cust, is_fraud, rng)              # coherent signals + label
        rec["merchant_category"] = r["merchant_category"]
        rec["merchant"] = _clean_merchant(r["merchant"], rec["merchant_category"], rng)  # LLM-written, sanitized
        records.append(rec)
    return records


def _write(records: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "fraud_signals_dataset.json"
    json_path.write_text(json.dumps(records, indent=2))

    # A flat CSV too (lists become "a | b") so the data is easy to eyeball / open
    # in Excel when showing judges what was generated.
    csv_path = OUT_DIR / "fraud_signals_dataset.csv"
    fields = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in records:
            w.writerow([" | ".join(v) if isinstance(v, list) else v for v in (r[k] for k in fields)])

    n_fraud = sum(1 for r in records if r["isFraud"])
    print(f"[done] {len(records)} records ({n_fraud} fraud / {len(records) - n_fraud} legit) "
          f"across {len({r['customer_id'] for r in records})} customers")
    print(f"       JSON -> {json_path}")
    print(f"       CSV  -> {csv_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=120, help="number of records (schema wants ~100-150)")
    p.add_argument("--fraud-rate", type=float, default=0.35,
                   help="fraction fraudulent (0.35 gives a balanced eval set; real traffic is far lower)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine", choices=["python", "datadesigner"], default="python")
    p.add_argument("--dd-model", default="nvidia/llama-3.3-nemotron-super-49b-v1",
                   help="NVIDIA Build model for Data Designer's LLM merchant column")
    args = p.parse_args()

    if args.engine == "datadesigner":
        records = generate_datadesigner(args.n, args.fraud_rate, args.seed, args.dd_model)
    else:
        records = generate_python(args.n, args.fraud_rate, args.seed)
    _write(records)


if __name__ == "__main__":
    main()

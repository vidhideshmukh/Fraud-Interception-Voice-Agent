"""Hash-chained append-only audit log. Each record includes the SHA-256 of the
previous record, so tampering breaks the chain — a cheap, demo-visible stand-in
for NemoClaw's immutable audit trail (presented as production hardening).
Every dialog turn, tool call, and state change goes through here.

Schema unification note: Jyotika's eval-harness output schema (per-run
records with predicted_intent/confidence/rail_verdicts/latency_ms/model_id/
prompt_version/seed) and this module's per-turn audit record turned out to
be the same object wearing two names — one written during a live call, the
other written during an offline eval replay. `log_agent_turn()` below is the
one place that shape is assembled, so the compliance panel judges watch
*is* the eval record: no separate logging path to keep in sync, and
model_id/prompt_version/seed get stamped on the critical path for free
instead of as a Day-2 afterthought.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.database.models import now_ms

_LOG: list[dict] = []

# --- durable, date-partitioned persistence of the audit trail ---------------
# The in-memory _LOG is the live view (fast, powers /audit + the ops console);
# but a compliance trail must survive restarts. So every record is ALSO appended
# to logs/audit/audit-YYYY-MM-DD.jsonl — one JSON record per line, partitioned by
# date (the standard way audit/event logs are archived). Records never contain
# secrets (the code digits are never in a payload — only code_length/channels),
# so writing them to disk is safe. Toggle with AUDIT_PERSIST=false for tests
# that don't want disk I/O.
AUDIT_DIR = Path(os.getenv("LOG_DIR", "logs")) / "audit"
AUDIT_PERSIST = os.getenv("AUDIT_PERSIST", "true").lower() == "true"

# Per-call artefacts (adopted from Jennifer's audit_log, which the mentor liked):
#  - logs/audit/sessions/<session_id>.jsonl : one file PER CALL, so a reviewer
#    opens a single file and sees that whole call — nothing else mixed in.
#  - logs/audit/sessions_index.jsonl        : a browsable rollup, one line per
#    call open/close, carrying the OUTCOME and TIME-TO-INTERCEPT metric.
# Both survive restarts, so the ops console can show past calls "like production".
SESSIONS_DIR = AUDIT_DIR / "sessions"
SESSIONS_INDEX = AUDIT_DIR / "sessions_index.jsonl"

# The per-session file is written HUMAN-READABLE (adapted from jenjam007 /
# the phase-1 repo): a session header box, then one indented JSON block per
# event separated by a '=' rule — so a reviewer opens one file and reads the
# whole call top-to-bottom. The day-partitioned master stays compact JSONL
# (machine-readable) because it powers the ops dashboard's aggregate view.
_HDR = "#" * 72
_SEP = "=" * 72

_SESSION_START: dict[str, int] = {}   # session_id -> start epoch-ms (for time-to-intercept)
_CLOSED: set[str] = set()             # sessions already closed (idempotency)


def _persist(record: dict) -> None:
    """Write a record to the day-partitioned master AND its per-call file."""
    if not AUDIT_PERSIST:
        return
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Day-partitioned master: compact one-record-per-line (machine-readable,
        # powers read_all_persisted + the ops dashboard).
        with (AUDIT_DIR / f"audit-{day}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        sid = record.get("session_id")
        if sid:  # per-call file — one file per session, HUMAN-READABLE
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            path = SESSIONS_DIR / f"{sid}.jsonl"
            new_file = not path.exists()   # write the header once, and never twice after a restart
            with path.open("a", encoding="utf-8") as f:
                if new_file:
                    f.write(f"{_HDR}\n# SESSION {sid}\n{_HDR}\n")
                f.write(f"\n{_SEP}\n{json.dumps(record, indent=2, default=str)}\n")
    except Exception:  # noqa: BLE001 — a disk hiccup must never break a live call
        pass


def _index_append(entry: dict) -> None:
    if not AUDIT_PERSIST:
        return
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with SESSIONS_INDEX.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def log_turn(session_id: str, kind: str, payload: dict) -> dict:
    """Generic append — used directly for push/asr/tts/tool/state/handoff
    entries, whose payloads don't need a fixed shape. See log_agent_turn()
    for the one kind (`agent`) that does."""
    prev_hash = _LOG[-1]["hash"] if _LOG else "GENESIS"
    record = {
        "seq": len(_LOG),
        "ts": now_ms(),
        "session_id": session_id,
        "kind": kind,          # push | asr | agent | tool | state | tts | handoff
        "payload": payload,
        "prev_hash": prev_hash,
    }
    record["hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    _LOG.append(record)
    _persist(record)  # durable, date-partitioned copy (survives restarts)
    return record


def log_agent_turn(session_id: str, *, predicted_intent: Optional[str], confidence: float,
                   reply_text: str, state_after: str, model_id: str, prompt_version: str,
                   latency_ms: float, seed: Optional[int] = None,
                   rail_verdicts: Optional[dict] = None,
                   actions_fired: Optional[list[str]] = None) -> dict:
    """The standardized shape for every live-loop dialog turn. Matches
    Jyotika's eval-record schema field-for-field on purpose — a scripted
    Dataset-3 replay and a real call both produce this same structure, so
    the regression gate and the live audit panel are one code path, not two."""
    payload = {
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "reply_text": reply_text,
        "state_after": state_after,
        "rail_verdicts": rail_verdicts or {"input": "not_configured", "output": "not_configured"},
        "actions_fired": actions_fired or [],
        "latency_ms": latency_ms,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "seed": seed,
    }
    return log_turn(session_id, "agent", payload)


def get_log(session_id: str | None = None) -> list[dict]:
    return [r for r in _LOG if session_id is None or r["session_id"] == session_id]


# --- session lifecycle: the browsable index + time-to-intercept metric -------

def open_session(session_id: str, *, customer_name: str, amount_gbp: float,
                 city: str, rca_reason: str) -> None:
    """Call this when a call starts. Records the call's start time (for
    time-to-intercept) and an 'open' line in the durable sessions index."""
    _SESSION_START[session_id] = now_ms()
    _index_append({
        "session_id": session_id, "status": "open", "started_ts": _SESSION_START[session_id],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "customer_name": customer_name, "amount_gbp": amount_gbp, "city": city,
        "rca_reason": rca_reason,
    })


def close_session(session_id: str, outcome: str) -> None:
    """Call this when a call reaches a terminal outcome. Computes TIME-TO-
    INTERCEPT (call start -> resolution) and appends a 'complete' line to the
    index. Idempotent — only the first terminal transition closes the call."""
    if session_id in _CLOSED:
        return
    _CLOSED.add(session_id)
    started = _SESSION_START.get(session_id)
    tti = round((now_ms() - started) / 1000, 1) if started else None
    _index_append({
        "session_id": session_id, "status": "complete", "ended_ts": now_ms(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome, "time_to_intercept_seconds": tti,
    })


def read_sessions_index() -> list[dict]:
    """Read the durable sessions index, merging each call's open + complete lines
    into one record. Survives restarts, so this is 'all calls ever', newest
    first — what the ops console shows as persisted history."""
    if not SESSIONS_INDEX.exists():
        return []
    merged: dict[str, dict] = {}
    try:
        for raw in SESSIONS_INDEX.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            entry = json.loads(raw)
            merged.setdefault(entry["session_id"], {}).update(entry)
    except Exception:  # noqa: BLE001 — a malformed line shouldn't break the view
        pass
    return sorted(merged.values(), key=lambda e: e.get("started_ts", 0), reverse=True)


def read_session_file(session_id: str) -> list[dict]:
    """The full per-call audit trail for one session, straight from its own
    file — works even after a restart (the in-memory _LOG is gone by then)."""
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    out = []
    if _SEP in text:
        # Current human-readable format: header box + '=' rules + indented JSON
        # blocks — split on the rule and parse each block that is JSON.
        for block in text.split(_SEP):
            block = block.strip()
            if block.startswith("{"):
                try:
                    out.append(json.loads(block))
                except Exception:  # noqa: BLE001
                    pass
    else:
        # Older/compact format: one JSON object per line (plain JSONL). Without
        # this branch these files parsed to 0 records and the ops console showed
        # "No records" when a reviewer clicked the call, even though the data is
        # all there. Parse line-by-line, skipping any non-JSON (e.g. a header).
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    return out


def read_all_persisted(limit: int = 3000) -> list[dict]:
    """EVERY durable audit record across every day-partitioned file — i.e. the
    complete history spanning all runs, oldest first, tailed to the last
    `limit`. This is what the monitoring dashboard renders so its panels +
    metrics (guardrails, agent turns, tools, latency, screening) show the full
    history and survive restarts, instead of just the current run's in-memory
    view. `ts` is globally increasing (epoch-ms at write), so it orders cleanly
    even though per-run `seq` resets."""
    if not AUDIT_DIR.exists():
        return []
    records = []
    for f in sorted(AUDIT_DIR.glob("audit-*.jsonl")):
        try:
            for raw in f.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    records.append(json.loads(raw))
        except Exception:  # noqa: BLE001 — a bad line/file shouldn't break the view
            pass
    records.sort(key=lambda r: (r.get("ts", 0), r.get("seq", 0)))
    return records[-limit:]


def verify_chain() -> bool:
    prev = "GENESIS"
    for r in _LOG:
        if r["prev_hash"] != prev:
            return False
        body = {k: v for k, v in r.items() if k != "hash"}
        if hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16] != r["hash"]:
            return False
        prev = r["hash"]
    return True


def reset():
    """Clear the in-memory live view + session-lifecycle tracking. The durable
    files (day master, per-session, index) are NOT deleted — you never delete an
    audit trail; a reset just starts a fresh live view."""
    _LOG.clear()
    _SESSION_START.clear()
    _CLOSED.clear()

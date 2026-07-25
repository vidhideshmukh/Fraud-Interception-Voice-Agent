"""FastAPI surface — the app server the whole demo talks to.

Endpoint groups:
- POST /trigger/{txn_id}        manually simulate one fraud event -> starts a call session
- GET  /stream/transactions     poll-driven live transaction transaction_stream (auto-triggers calls)
- POST /stream/reset            rewind the transaction_stream for a fresh demo run
- POST /converse/{session_id}   one customer utterance -> agent reply
- GET  /session/{session_id}    call state + transcript, for the Barclays app UI to poll
- POST /no_answer/{session_id}  timeout path
- GET  /audit                   the live hash-chained audit log (the compliance panel)
- GET  /bank/state              transactions, card actions, cases
- POST /dataset/load            load a NeMo Data Designer-generated dataset file
- GET  /app/*                   the Barclays app demo page (static files)

Run: uvicorn app.api.server:app --reload

Why the local voice loop (demo/voice_loop.py) is an HTTP client of this
server rather than something that imports orchestrator.py directly: it
mirrors the documented production topology (SIP trunk -> app server; Riva
is "just the media leg" per the team's own Judge Q&A doc) and means the
Barclays app UI and the audit panel — both polling this same server — see
the live call's state update in real time, without needing to share
in-process memory across two separate processes.
"""
from __future__ import annotations

# Must run before any `app.*` import below — several modules (adapters/llm.py,
# tts.py, asr.py, guardrails.py) read MOCK_MODE/NVIDIA_API_KEY/etc. at import
# time via os.getenv(), not lazily. python-dotenv has been a core dependency
# since the start of this project but was never actually wired in anywhere —
# every command in this README instead relied on manually running
# `set -a; source .env; set +a` first, which is bash-only and breaks in
# PowerShell/cmd. This makes `.env` load automatically in any shell.
import os
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Configure structured, date-partitioned logging before anything else logs.
from app.observability.logging_setup import setup_logging, get_logger
setup_logging()
log = get_logger("server")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core import orchestrator
from app.database.models import Transaction, new_id
from app.security import audit
from app.database import bank
from app.core import fraud_trigger, transaction_stream

app = FastAPI(title="Fraud-Interception Voice Agent (demo)")

# Permissive on purpose: this is a local hackathon demo server, and the
# Barclays app page may be opened from a different origin/port than uvicorn's
# during development. Not a production posture — see security-review notes
# in README.md before this touches anything real.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Utterance(BaseModel):
    text: str


@app.on_event("startup")
def _warm_live_connections():
    """Pays the TLS/connection-setup cost (measured live at ~1.3s extra)
    once, at server boot, instead of letting it land on whichever call
    happens to be first — which on stage is very possibly the flagship
    demo call in front of judges. No-ops in mock mode. Wrapped in
    try/except: a warm-up failure (bad key, flaky wifi at the exact
    moment the server starts) should degrade to a cold first call, not
    prevent the server from starting at all."""
    import os
    if os.getenv("MOCK_MODE", "true").lower() == "true":
        return
    try:
        # llm lives at app.llm.nemotron since the enterprise folder restructure
        # (it used to be app/security/llm.py). Warming both the NIM connection
        # pool and the NeMo Guardrails rails here is what keeps the first REAL
        # call — often the flagship demo call — off the cold-start path.
        from app.security import guardrails
        from app.llm import nemotron
        nemotron.complete_turn(system="Reply with JSON only: {\"reply\": \"ok\", \"intent\": \"unsure\", \"confidence\": 0.5}",
                               history=[], user_text="warmup", context={"last4": "0000"})
        guardrails.check_input("warmup")
        log.info("startup: live LLM + guardrails connections warmed")
    except Exception as e:  # noqa: BLE001 — startup warm-up must never block boot
        log.warning("startup: warm-up call failed (%s) — first real call pays the cold-start cost", e)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/")


@app.get("/health")
def health():
    """Liveness probe — the process is up and serving (k8s/load-balancer liveness)."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe — dependencies are usable: the SQLite DB is reachable and
    the customer base is loaded. Returns 503 until ready, so a load balancer can
    hold traffic until the backend is actually serving (k8s readiness)."""
    n = len(bank.CUSTOMERS)
    db_ok = n > 0
    body = {
        "ready": db_ok,
        "db": "ok" if db_ok else "unavailable",
        "customers_loaded": n,
        "mode": "mock" if os.getenv("MOCK_MODE", "true").lower() == "true" else "live",
        "persistence": "clean-slate" if os.getenv("RESET_ON_BOOT", "true").lower() == "true" else "durable",
    }
    if not db_ok:
        raise HTTPException(503, body)
    return body


@app.post("/trigger/{txn_id}")
def trigger(txn_id: str):
    """Manual trigger — still here for the scripted-demo fallback (per the
    risk register: if the live transaction_stream misbehaves on stage, click a button
    instead and the flow is identical)."""
    if txn_id not in bank.TRANSACTIONS:
        raise HTTPException(404, "unknown txn")
    event = fraud_trigger.emit_event(txn_id)
    session = orchestrator.start_call(event)
    return {"session_id": session.session_id,
            "verification_code": session.verification_code,  # demo UI shows the 'phone'
            "opening": session.transcript[-1]["text"],
            "rca_reason": event.rca_reason}


@app.get("/stream/transactions")
def stream_transactions(advance: int = 1):
    """Poll-driven live transaction_stream. The Barclays dashboard calls this roughly
    once a second with advance=1 (or a few) to replay the loaded transaction
    stream; any transaction crossing the risk threshold auto-starts a call
    — see app/services/transaction_stream.py."""
    return transaction_stream.advance(advance)


@app.post("/stream/reset")
def stream_reset():
    transaction_stream.reset()
    return {"ok": True}


@app.get("/sessions")
def list_sessions():
    """Summary of every call this run, newest first — the data behind the Fraud
    Operations Console (demo/ops_dashboard). Read-only projection of the live
    sessions; the full turn-by-turn detail is in /audit."""
    out = []
    for sid, s in orchestrator.SESSIONS.items():
        out.append({
            "session_id": sid,
            "customer_name": s.customer.name,
            "customer_id": s.customer.customer_id,
            "card_last4": s.customer.card_last4,
            "amount_gbp": s.event.txn.amount_gbp,
            "merchant": s.event.txn.merchant,
            "city": s.event.txn.city,
            "rca_reason": s.event.rca_reason,
            "risk_score": s.event.risk_score,
            "state": s.state.value,
            "outcome": s.outcome,
            "verification_attempts": s.verification_attempts,
            "turns": sum(1 for t in s.transcript if t["role"] == "user"),
        })
    return list(reversed(out))  # newest call first


@app.get("/history")
def call_history():
    """Persisted call history from the durable audit index — survives restarts,
    so the ops console shows past calls even after a reboot. Each entry carries
    the outcome + time-to-intercept; newest first."""
    return audit.read_sessions_index()


@app.get("/history/{session_id}")
def call_detail(session_id: str):
    """The full per-call audit trail from that session's own file — works after a
    restart, unlike the in-memory /audit."""
    return {"session_id": session_id, "records": audit.read_session_file(session_id)}


# --- Log-file browser: read the durable logs FROM the dashboard --------------
# So the ops team can browse/open log files without SSHing into the cluster.
# NOTE: like the rest of the ops console this is unauthenticated in the demo —
# it exposes logs (which contain transcripts/PII). Gate it behind auth before a
# real deployment (see README "Production readiness").
LOG_ROOT = Path(os.getenv("LOG_DIR", "logs")).resolve()


@app.get("/logs")
def list_logs():
    """List the durable log files (app log + date-partitioned audit + per-call
    session files) with sizes — the file tree the dashboard browses."""
    if not LOG_ROOT.exists():
        return {"root": str(LOG_ROOT), "files": []}
    files = []
    # Be defensive: on a shared filesystem (e.g. the cluster's lustre mount) a
    # single unreadable file, broken symlink, or permission-denied subdir would
    # otherwise raise and 500 this endpoint — which froze the whole ops console
    # (its poll fetches every panel together). Skip anything we can't stat.
    try:
        entries = sorted(LOG_ROOT.rglob("*"))
    except OSError:
        entries = []
    for p in entries:
        try:
            if p.is_file():
                st = p.stat()
                files.append({"path": p.relative_to(LOG_ROOT).as_posix(),
                              "size": st.st_size, "modified_ms": int(st.st_mtime * 1000)})
        except OSError:
            continue
    return {"root": str(LOG_ROOT), "files": files}


@app.get("/logs/view")
def view_log(path: str, limit: int = 1000):
    """Return a log file's content — parsed as JSONL records if it is one, else
    raw text. Big files are tailed to the last `limit` lines. The path is
    validated to stay inside LOG_DIR (no traversal outside the logs tree)."""
    target = (LOG_ROOT / path).resolve()
    try:
        target.relative_to(LOG_ROOT)          # raises if `path` escapes the logs dir
    except ValueError:
        raise HTTPException(400, "invalid path")
    if not target.is_file():
        raise HTTPException(404, "not found")

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-limit:]
    records, is_jsonl = [], True
    for ln in tail:
        if not ln.strip():
            continue
        try:
            records.append(json.loads(ln))
        except Exception:  # noqa: BLE001 — not JSONL, fall back to raw text
            is_jsonl = False
            break
    if is_jsonl:
        return {"path": path, "format": "jsonl", "total_lines": len(lines), "records": records}
    return {"path": path, "format": "text", "total_lines": len(lines), "text": "\n".join(tail)}


@app.post("/answer/{session_id}")
def answer(session_id: str):
    """The customer picks up. Transitions a RINGING call to 'answered' so the
    agent greets and asks for verification — this is what the 'Answer' button
    in the Barclays app calls. Idempotent (answering an already-answered call
    is a harmless no-op). Returns the updated call state + the greeting so the
    UI/handset can render/speak it."""
    session = orchestrator.SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "unknown session")
    orchestrator.answer_call(session)
    return {"session_id": session_id, "state": session.state,
            "opening": session.transcript[-1]["text"] if session.transcript else ""}


class WebRTCOffer(BaseModel):
    sdp: str
    type: str
    session_id: str


@app.get("/ice-servers")
def ice_servers():
    """ICE servers (STUN/TURN) for the BROWSER peer, sourced from .env
    (TURN_URLS/TURN_USERNAME/TURN_CREDENTIAL). The browser fetches this before
    creating its RTCPeerConnection so it too gets a relay candidate — without it,
    only the server side has TURN and the connection can't complete across a
    firewall. Returns {"iceServers": []} when TURN isn't configured."""
    from app.media import webrtc
    return {"iceServers": webrtc.ice_servers_json()}


@app.post("/offer")
async def webrtc_offer(offer: WebRTCOffer):
    """WebRTC signaling: the browser sends its SDP offer for a ringing call; we
    establish the peer connection (mic in -> Riva ASR -> orchestrator -> Riva TTS
    -> speaker out) and return the SDP answer. This is the browser-as-phone media
    leg — the alternative to demo/voice_loop.py. See app/media/webrtc.py."""
    from app.media import webrtc
    try:
        return await webrtc.handle_offer(offer.sdp, offer.type, offer.session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001 — surface a clean 500, keep the server up
        log.warning("webrtc offer failed: %s", e)
        raise HTTPException(500, f"webrtc setup failed: {e}")


@app.post("/converse/{session_id}")
def converse(session_id: str, utt: Utterance):
    session = orchestrator.SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "unknown session")
    reply = orchestrator.customer_says(session, utt.text)
    return {"reply": reply, "state": session.state, "outcome": session.outcome}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Full call state for the Barclays app UI / any dashboard to poll —
    deliberately omits verification_code (see orchestrator.start_call's
    docstring: the code is customer-facing via /trigger's response, not
    something re-exposed on every poll of call state)."""
    session = orchestrator.SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "unknown session")
    return {
        "session_id": session.session_id,
        "state": session.state,
        "outcome": session.outcome,
        "customer_name": session.customer.name,
        "customer_card_last4": session.customer.card_last4,  # for the dashboard's card visual only
        "txn": session.event.txn,
        "rca_reason": session.event.rca_reason,
        "transcript": session.transcript,
        "verification_attempts": session.verification_attempts,
    }


# A call the handset should auto-attach to via --latest = one that is RINGING
# and not yet answered. Deliberately NOT awaiting_verification/verified: once a
# call has been answered it's mid-conversation (someone's handling it), so a
# freshly-started handset must NOT grab it. This is the fix for "--latest keeps
# jumping onto an old call" — a half-finished call left in awaiting_verification
# used to linger as "active" and get grabbed instead of the next real ring. To
# re-attach to a specific in-progress call (e.g. after a handset crash), use
# --session <id>, not --latest.
_RINGING_STATES = {"push_sent", "dialing"}


@app.get("/sessions/latest")
def latest_active_session():
    """The most recently started call that is RINGING (unanswered). Lets the
    voice handset (demo/voice_loop.py --latest) sit 'waiting for an incoming
    call' and auto-attach the instant a UI-inserted fraud transaction fires one
    — so the presenter pre-starts the handset once, then just inserts a
    transaction and the phone rings for real, no session id to copy mid-demo.
    Returns session_id: null when nothing is currently ringing (so a stale,
    already-answered call can never be grabbed).

    SESSIONS is insertion-ordered (plain dict), so iterating in reverse yields
    the newest call first."""
    for sid in reversed(list(orchestrator.SESSIONS)):
        session = orchestrator.SESSIONS[sid]
        if session.state.value in _RINGING_STATES:
            return {"session_id": sid, "state": session.state.value}
    return {"session_id": None, "state": None}


@app.post("/no_answer/{session_id}")
def no_answer(session_id: str):
    session = orchestrator.SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "unknown session")
    return {"result": orchestrator.no_answer(session)}


@app.get("/audit")
def get_audit(session_id: str | None = None):
    return {"chain_valid": audit.verify_chain(), "records": audit.get_log(session_id)}


@app.get("/audit/all")
def get_audit_all(limit: int = 3000):
    """ALL persisted audit records across every run (durable) — the monitoring
    dashboard's source, so its panels + metrics show full history and survive
    restarts, not just the current run's in-memory chain."""
    return {"records": audit.read_all_persisted(limit)}


@app.get("/metrics/latency")
def latency_metrics():
    """Per-LLM-call latency for MODEL EVALUATION. Every inference — the dialog
    call AND each guardrail rail — is timed separately and tagged with its
    model + stage; this returns the p50/p95/p99/avg/max per (model, stage) plus
    recent raw samples for the time-series chart. Durable across restarts."""
    from app.observability import metrics
    samples = metrics.read_all()
    return {"summary": metrics.summarize(samples), "samples": samples[-600:]}


@app.get("/bank/state")
def bank_state():
    return {"transactions": list(bank.TRANSACTIONS.values()),
            "card_actions": bank.CARD_ACTIONS,
            "cases": list(bank.CASES.values())}


@app.get("/bank/customers")
def bank_customers():
    return list(bank.CUSTOMERS.values())


@app.get("/incidents")
def list_incidents():
    """Human-review incidents raised on escalation — the data behind the ops
    console's 'Human review queue'. Open ones first, then most recent."""
    return sorted(bank.INCIDENTS.values(),
                  key=lambda i: (i["status"] != "open", -i["opened_ts"]))


class IncidentResolution(BaseModel):
    decision: str  # "block" or "approve"


@app.post("/incidents/{incident_id}/resolve")
def resolve_incident(incident_id: str, res: IncidentResolution):
    """A human fraud analyst resolves an open incident — the completion of the
    human-in-the-loop review. Fires the bank action (block+chargeback, or
    release) and audits it in the escalated call's trail."""
    try:
        result = bank.resolve_incident(incident_id, res.decision)
    except KeyError:
        raise HTTPException(404, "unknown incident")
    inc = result["incident"]
    for a in result["actions"]:
        audit.log_turn(inc["session_id"], "tool", a)
    audit.log_turn(inc["session_id"], "state",
                   {"incident_resolved": incident_id, "analyst_decision": res.decision,
                    "outcome": inc["resolution"]["outcome"]})
    return inc


class NewTransactionRequest(BaseModel):
    customer_id: str
    amount_gbp: float
    merchant: str
    city: str
    channel: str = "card_present"
    category: str = "other"
    mcc: str | None = None
    device_id: str | None = None


@app.post("/bank/transactions")
def new_transaction(req: NewTransactionRequest):
    """Simulates a real-world transaction landing on the bank's own
    transaction stream — a card swipe or online checkout actually
    happening, NOT a 'report this as fraud' button. There is no direct
    fraud flag anywhere in this request; the bank's (heuristic, for the
    demo) fraud model scores it exactly like any other transaction, and
    the call pipeline fires automatically ONLY if it crosses the risk
    threshold — the same auto-trigger path the live transaction_stream uses. The
    customer's Barclays app has no visibility into this endpoint at all;
    it only ever learns about the outcome via the push notification it
    receives if-and-when a call actually starts, same as a real
    cardholder. Keeping this endpoint entirely separate from anything the
    phone-panel UI can call is deliberate — see the 'Bank transaction
    feed' panel in demo/barclays_app/index.html, which is visually and
    functionally a different system from the phone panel next to it."""
    if req.customer_id not in bank.CUSTOMERS:
        raise HTTPException(404, "unknown customer")

    txn = Transaction(txn_id=new_id("txn"), customer_id=req.customer_id, amount_gbp=req.amount_gbp,
                      merchant=req.merchant, city=req.city, channel=req.channel,
                      category=req.category, mcc=req.mcc, device_id=req.device_id)
    bank.insert_transaction(txn)  # in-memory + write-through to the DB

    # Score against THIS customer's history/baseline — assess() returns the
    # score AND the explainable reason (why it's anomalous for this account).
    assessment = fraud_trigger.assess(txn)
    flagged = assessment.score >= fraud_trigger.RISK_THRESHOLD
    session_id = None
    verification_code = None
    if flagged:
        event = fraud_trigger.emit_event(txn.txn_id)
        # auto_answer=False: the call RINGS in the app and stays silent until
        # the presenter taps 'Answer' (POST /answer). The Riva voice then
        # starts exactly on pickup — never talking to an unanswered phone.
        session = orchestrator.start_call(event, auto_answer=False)
        session_id = session.session_id
        verification_code = session.verification_code  # surfaced once, same rule as /trigger and the transaction_stream

    # Log EVERY screening decision to the tamper-evident audit chain, not just
    # the ones that start a call. In a real bank the compliance record is the
    # fraud-model's verdict on every transaction it scores — a cleared "no
    # action" decision is just as auditable as a flagged one (you must be able
    # to prove later WHY a transaction was let through). session_id is None
    # here because this is a bank-level screening event, distinct from any call
    # session it may spawn; it therefore surfaces in the global compliance view.
    audit.log_turn(None, "screening", {
        "action": "fraud_screen",
        "txn_id": txn.txn_id,
        "customer_id": txn.customer_id,
        "customer_name": bank.CUSTOMERS[txn.customer_id].name,
        "amount_gbp": txn.amount_gbp,
        "merchant": txn.merchant,
        "city": txn.city,
        "channel": txn.channel,
        "risk_score": assessment.score,
        "flagged": flagged,
        "decision": "flagged_call_initiated" if flagged else "cleared_no_action",
        "reason": assessment.reason,
        "call_session_id": session_id,  # linkage to the call, if one started
    })

    return {"txn_id": txn.txn_id, "risk_score": assessment.score, "flagged": flagged,
            "reason": assessment.reason, "auto_triggered_session_id": session_id,
            "verification_code": verification_code}


class DatasetLoadRequest(BaseModel):
    path: str


@app.post("/dataset/load")
def load_dataset(req: DatasetLoadRequest):
    """Loads a NeMo Data Designer output file (see data/generate_datasets.py)
    on top of the seeded demo data, and rewinds the transaction_stream so the newly
    loaded transactions are what gets replayed."""
    count = bank.load_generated_dataset(req.path)
    transaction_stream.reset()
    return {"transactions_loaded": count}


# Mount the two demo pages last, so they don't shadow the API routes above.
# Paths are resolved ABSOLUTELY from this file's location, not the process's
# working directory — otherwise `uvicorn app.api.server:app` only serves the
# UIs when launched from the project root, and silently 404s the pages when
# launched from anywhere else (found live: the pages vanished when the server
# was started from the parent folder). A missing directory is logged, not
# swallowed, so the failure is visible instead of mysterious.
_DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "demo"


def _mount_ui(route: str, folder: str, name: str) -> None:
    path = _DEMO_DIR / folder
    if not path.is_dir():
        log.warning("UI not mounted at %s — directory missing: %s", route, path)
        return
    app.mount(route, StaticFiles(directory=str(path), html=True), name=name)
    log.info("UI mounted: %s -> %s", route, path)


_mount_ui("/app", "barclays_app", "barclays_app")   # the cardholder's phone app
# The Fraud Operations Console — the judges' observability view (all calls,
# guardrail decisions, latency, fraud screening, and the tamper-evident audit
# chain). Separate page from the customer phone app on purpose: one is what a
# cardholder sees, this is what the bank's fraud-ops team sees.
_mount_ui("/ops", "ops_dashboard", "ops_dashboard")

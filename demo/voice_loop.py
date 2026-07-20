"""The live 'phone' — this is what a presenter actually talks to on stage.

Architecture note (why this is a separate process talking HTTP, not an
import of app.core.orchestrator): see app/main.py's module docstring. Short
version — this script plays the role of the telephony/media leg (Riva ASR +
TTS against the local mic/speaker), while the orchestrator's state lives in
the FastAPI server process. That means the Barclays app UI and the /audit
panel, both polling the same server, update live while this script runs —
exactly what the demo needs judges to see.

Usage (live mode — set MOCK_MODE=false in .env, this loads it automatically
in any shell, see the load_dotenv() note below):
    python demo/voice_loop.py txn_lagos_01

  Or attach to a session the live transaction_stream / bank transaction feed already
  auto-triggered instead of a specific txn — see --session below (this is
  what the dashboard's "Copy" button under the phone panel hands you).

Mock mode: this script is not useful (there's no real audio to capture) —
use demo/run_demo.py or POST to /converse directly for the text-mode path.
"""
from __future__ import annotations

import argparse
import os
import sys

# Windows' default console codepage (cp1252) can't encode characters Riva
# sometimes returns (accents, smart punctuation), and print() then crashes
# the whole call loop mid-turn — found live 2026-07-18: a customer's second
# utterance killed the script before it reached the server. Force UTF-8 with
# a safe fallback so a stray character never takes the demo down.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 — older stdout objects; not worth failing over
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same reasoning as app/main.py's module docstring: python-dotenv has been a
# core dependency since the start but was never wired in, so every command
# relied on manually sourcing .env first (bash-only, breaks in
# PowerShell/cmd). Must run before MOCK_MODE is read below.
from dotenv import load_dotenv
load_dotenv()

TERMINAL_STATES = {"resolved_legit", "resolved_fraud", "escalated", "channel_frozen", "no_answer"}


def _api_base() -> str:
    return os.getenv("API_BASE", "http://localhost:8000")


def _wait_for_incoming_call(base: str) -> str:
    """Poll the server for the newest live call and return its id as soon as
    one appears. This is what makes the on-stage flow seamless: the presenter
    starts the handset FIRST (it prints 'waiting…'), then inserts a suspicious
    transaction in the dashboard — the moment the fraud model flags it and
    starts a call, this picks it up and attaches, no session id to copy."""
    import time
    import requests

    print("[handset ready] waiting for an incoming fraud call — "
          "insert a suspicious transaction in the Barclays dashboard…  (Ctrl+C to cancel)")
    while True:
        try:
            r = requests.get(f"{base}/sessions/latest", timeout=10)
            r.raise_for_status()
            sid = r.json().get("session_id")
            if sid:
                print(f"[incoming call] session={sid}")
                return sid
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001 — transient poll error; keep waiting
            pass
        time.sleep(0.5)


def _next_ringing_call(base: str) -> str | None:
    """The OLDEST ringing (unanswered) call — FIFO queue order — or None if the
    queue is empty. Used by --queue operator mode to work through MULTIPLE
    detected frauds in the order they were flagged."""
    import requests
    try:
        sessions = requests.get(f"{base}/sessions", timeout=10).json()
    except Exception:  # noqa: BLE001 — transient; treat as empty this tick
        return None
    ringing = [s for s in sessions if s["state"] in ("push_sent", "dialing")]
    # /sessions is newest-first, so the LAST ringing entry is the oldest -> FIFO.
    return ringing[-1]["session_id"] if ringing else None


def _converse_loop(base, session_id, mic, tts, requests) -> None:
    """mic -> /converse -> speak, until the call reaches a terminal state.
    Lets KeyboardInterrupt propagate so an operator can Ctrl+C out of the queue."""
    while True:
        text = mic.listen()
        if not text:
            continue
        print(f"Customer: {text}")
        data = requests.post(f"{base}/converse/{session_id}", json={"text": text}).json()
        print(f"Agent: {data['reply']}  [state={data['state']}]")
        tts.speak(data["reply"])
        if data["state"] in TERMINAL_STATES:
            print(f"[call ended] outcome={data['outcome']}")
            return


def _await_answer_and_greet(base, session_id, tts) -> None:
    """Attach to a call: if it's still RINGING, wait for it to be answered in the
    app (the greeting doesn't exist until pickup), then speak the greeting.
    KeyboardInterrupt propagates so the caller can close the handset."""
    import time
    import requests
    ringing_states = {"push_sent", "dialing"}
    data = requests.get(f"{base}/session/{session_id}").json()
    if data["state"] in ringing_states:
        print(f"[ringing] session={session_id} — tap 'Answer call' in the Barclays app to take the call…")
        while data["state"] in ringing_states:
            time.sleep(0.5)
            try:
                data = requests.get(f"{base}/session/{session_id}").json()
            except Exception:  # noqa: BLE001 — transient poll error; keep waiting
                pass
    print(f"[answered] session={session_id} state={data['state']}")
    if data["transcript"]:
        tts.speak(data["transcript"][-1]["text"])  # greeting, on pickup


def _run_queue_mode(base, requests, asr, tts) -> None:
    """Operator mode: handle MULTIPLE detected frauds one after another. Picks
    the oldest ringing call (FIFO), the operator (this handset) picks it up
    itself — auto-answering, no per-call browser click — talks it to a
    resolution, then moves to the next, until the queue is empty. This is the
    'a burst of frauds just fired, work through them' scenario."""
    import time
    mic = asr.MicStreamASR()
    print("[operator mode] working the fraud queue FIFO — each flagged transaction is a call.")
    print("Insert suspicious transactions in the dashboard; I'll pick them up in order.  (Ctrl+C to stop)")
    announced_empty = False
    try:
        while True:
            sid = _next_ringing_call(base)
            if not sid:
                if not announced_empty:
                    print("[queue empty] waiting for the next incoming fraud call…")
                    announced_empty = True
                time.sleep(1.0)
                continue
            announced_empty = False
            requests.post(f"{base}/answer/{sid}")   # operator picks up the queued call
            time.sleep(0.3)
            data = requests.get(f"{base}/session/{sid}").json()
            print(f"\n=== picked up {sid} — {data['customer_name']} · "
                  f"£{data['txn']['amount_gbp']:,.0f} · {data['txn']['city']} ===")
            if data["transcript"]:
                tts.speak(data["transcript"][-1]["text"])  # greeting, on pickup
            _converse_loop(base, sid, mic, tts, requests)
    except KeyboardInterrupt:
        print("\n[operator handset closed]")


def run(txn_id: str | None, session_id: str | None,
        wait_latest: bool = False, wait_queue: bool = False) -> None:
    if os.getenv("MOCK_MODE", "true").lower() == "true":
        print("MOCK_MODE=true — there's no real audio to capture in mock mode. "
              "Use demo/run_demo.py (text-mode, offline) instead, or set "
              "MOCK_MODE=false with real NVIDIA_API_KEY + Riva credentials.")
        return

    import requests  # deferred: only needed for this live-mode script

    # Deferred import: asr.MicStreamASR and tts.speak both require the
    # `riva.client` / `sounddevice` packages, which mock mode doesn't need.
    from app.speech import asr, tts

    base = _api_base()

    if wait_queue:
        _run_queue_mode(base, requests, asr, tts)
        return

    # --latest: wait for calls and handle them one after another, KEEPING the
    # browser 'Answer' step, and LOOP so you never re-run the script — after a
    # call ends it goes straight back to waiting for the next incoming call.
    if wait_latest and not session_id:
        mic = asr.MicStreamASR()
        try:
            while True:
                sid = _wait_for_incoming_call(base)
                _await_answer_and_greet(base, sid, tts)
                _converse_loop(base, sid, mic, tts, requests)
                print("\n[call ended — waiting for the next incoming fraud call…]\n")
        except KeyboardInterrupt:
            print("\n[handset closed]")
        return

    # --session <id>: attach to ONE specific call, exit when it ends.
    if session_id:
        _await_answer_and_greet(base, session_id, tts)
    else:
        # txn_id: trigger one fresh call.
        resp = requests.post(f"{base}/trigger/{txn_id}")
        resp.raise_for_status()
        data = resp.json()
        session_id = data["session_id"]
        print(f"[call started] session={session_id}")
        print(f"[Barclays app would show verification code] {data['verification_code']}")
        tts.speak(data["opening"])

    try:
        _converse_loop(base, session_id, asr.MicStreamASR(), tts, requests)
    except KeyboardInterrupt:
        print("\n[call ended by presenter]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("txn_id", nargs="?", default="txn_lagos_01",
                       help="Transaction to trigger a fresh call for (ignored if --session is given).")
    parser.add_argument("--session", default=None,
                       help="Attach to a session the live transaction_stream already auto-triggered, instead of "
                            "triggering a new one.")
    parser.add_argument("--latest", action="store_true",
                       help="Wait for the next fraud call, auto-attach (no session id to copy), keep the browser "
                            "'Answer' step — and STAY RUNNING across calls: handle one, then wait for the next. "
                            "Run it ONCE for the whole demo (Ctrl+C to stop). The recommended live-demo mode.")
    parser.add_argument("--queue", action="store_true",
                       help="Operator mode for MULTIPLE frauds: work through the queue of ringing calls FIFO, "
                            "auto-answering and resolving each in turn, until you Ctrl+C. Use this to demo a burst "
                            "of detected frauds being handled one after another.")
    args = parser.parse_args()

    if args.session is None and args.txn_id is None and not args.latest and not args.queue:
        print("Need a txn_id, --session, --latest, or --queue.", file=sys.stderr)
        sys.exit(1)

    run(txn_id=args.txn_id, session_id=args.session, wait_latest=args.latest, wait_queue=args.queue)

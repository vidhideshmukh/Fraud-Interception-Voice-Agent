"""Real-time dialog agent — the critical path. ONE LLM call per customer turn.
Anything heavier (RCA elaboration, resolution planning) belongs in the async
layer. Latency budget for the whole turn: ASR + LLM + TTS < 1s.

Pitch-deck mapping: this module is the "Fraud Agent" in the architecture
diagram (slide 9 / Neha's README). Voice I/O lives in app/adapters/asr.py
and tts.py — that's the "Voice Agent" — and never touches the LLM, which is
exactly why the pitch's "three agents" story doesn't contradict the
single-hop latency NFR: only this module calls a model on the live path.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.llm import nemotron
from app.agents import prompts
from app.database.models import CallSession, Intent

# ---------------------------------------------------------------------------
# Latency fast-path (adapted from Jennifer's repo). An exact, unambiguous
# "yes"/"no" answer does not need the LLM to classify it or draft a reply — we
# short-circuit to a deterministic AgentTurn, which lets the orchestrator skip
# BOTH the dialog model AND the output-rail model on the commonest turns. Our
# NeMo Guardrails path adds two LLM calls per turn Jennifer's doesn't have, so
# this is exactly where that overhead is clawed back.
#
# Safety by design: we match the WHOLE normalised utterance against these sets,
# never a substring. "no but actually yes" is a different string from "no", so
# it fails the match and falls through to the full grounded LLM + guardrails
# path — nuance is never silently mis-handled. And because the fast-path
# produces the same {intent, confidence} shape the LLM would, the orchestrator's
# policy (high-value-deny escalation, etc.) runs identically: this changes
# latency only, never a decision.
# ---------------------------------------------------------------------------
_FAST_CONFIRM = {"yes", "yeah", "yep", "yup", "correct", "i did", "it was me",
                 "that was me", "yes i did", "yes that was me", "that's right",
                 "yeah that was me", "yes it was me", "that's me", "that is me", 
                 "yes that's me","yep that's me", "yup that's me", "yes it was", 
                 "that's mine","i made that","i did that","yes i made that","definitely","absolutely","sure","of course","certainly","affirmative","indeed","exactly","correctamundo","roger that","you got it","you bet","for sure","without a doubt"}
_FAST_DENY = {"no", "nope", "not me", "wasn't me", "that wasn't me", "no i didn't",
              "i didn't", "no that wasn't me", "no it wasn't me", "never","no way","not at all","negative","nah","no way jose","no sir","no ma'am","no thanks","no thank you","not really","not in a million years","absolutely not","no chance","nope nope nope","no way no how","i didn't do it","i didn't do that","not mine","wasn't me","that's not mine","i never did that","no it wasn't me","no it wasn't me","no i didn't do that","no i didn't do it","no i never did that","no i never did it","no i never did that before","no i never did it before","no i never did that in my life"}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace — so 'Yes.',
    ' yes ', and 'yes!' all match, while 'yes, but...' (a longer, different
    utterance) deliberately does not."""
    t = re.sub(r"[^a-z' ]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def fast_path_intent(user_text: str) -> Intent | None:
    """Exact-match intent for the latency fast-path, or None if the utterance
    is anything less than unambiguous (which then gets the full LLM path)."""
    norm = _normalize(user_text)
    if norm in _FAST_CONFIRM:
        return Intent.CONFIRM_LEGIT
    if norm in _FAST_DENY:
        return Intent.DENY
    return None


def opening_line(session: CallSession) -> str:
    """The SINGLE opening message spoken on pickup: who we are (Barclays fraud
    team, recorded line), why (a possibly fraudulent transaction) AND an
    immediate request to read back the app verification code — no separate
    'am I speaking with X / is now a good time' turns and no waiting on a
    yes/no. Model-phrased so it varies naturally, but tightly constrained: at
    most 1-2 sentences, no invented personal agent name, no small talk. The
    goal's recorded+fraud wording keeps the mandatory-disclosure rail satisfied
    (answer_call prepends the canonical line only if the model somehow drops
    it); the safe fallback carries the same content offline."""
    first = session.customer.name.split()[0]
    return nemotron.generate_line(
        goal=(f"In AT MOST 1-2 short sentences: greet {first} by first name, say this is the Barclays "
              f"fraud-prevention team on a recorded line about a possibly fraudulent transaction on their account, "
              f"then ask them to open their Barclays app and read back the verification code shown there. Do NOT "
              f"invent or give any personal agent name (never say 'I am <name>'), do NOT make small talk or ask how "
              f"they are. Briefly reassure you'll never ask for their PIN or card number."),
        context={"session_id": session.session_id, "customer_first_name": first},
        fallback=(f"Hello {first} — this is the Barclays fraud-prevention team on a recorded line, about a possibly "
                  f"fraudulent transaction on your account. Please open your Barclays app and read me the "
                  f"verification code shown there — I'll never ask for your PIN or card number."))


# ---------------------------------------------------------------------------
# Phase-1 style LLM-DRIVEN investigation loop. Instead of the model only
# CLASSIFYING the utterance into a fixed intent (which the orchestrator then
# maps to an action), the model DRIVES the call: each turn it reads the flagged
# txn + full transcript and returns ONE decision — ask another question, or
# conclude (approve / block / escalate). The orchestrator executes whatever it
# decides (validated by the action rail). This is what makes the dialog feel
# investigative and human rather than a one-shot classify-and-act.
# ---------------------------------------------------------------------------
_VALID_ACTIONS = {"ask", "approve", "block", "escalate"}


@dataclass
class Decision:
    action: str          # ask | approve | block | escalate
    message: str         # what the agent says this turn
    confidence: float
    model_id: str
    latency_ms: float


def _dialog_history(session: CallSession) -> str:
    lines = [f"{'Agent' if t['role'] == 'assistant' else 'Customer'}: {t['text']}"
             for t in session.transcript]
    return "\n".join(lines) if lines else "(no conversation yet)"


def _fallback_decision(session: CallSession, user_text: str, latency_ms: float) -> Decision:
    """Keyword decision used ONLY in mock mode or if the LLM is unreachable —
    keeps the call alive with the same {action, message} contract the loop uses."""
    t = user_text.lower()
    first = session.customer.name.split()[0]
    txn = session.event.txn
    if any(w in t for w in ("scared", "help", "human", "person", "agent", "scam", "panic")):
        return Decision("escalate", "I understand — let me pass this to our internal team who will look into "
                        "it further. Take care, and have a good day.", 0.5, "fallback", latency_ms)
    if any(w in t for w in ("not me", "didn't", "did not", "wasn't", "no i", "never", "fraud")):
        return Decision("block", f"Thank you, {first}. I'm blocking your card ending {session.customer.card_last4} "
                        "now and a replacement is on its way. Take care, and have a good day.", 0.9, "fallback", latency_ms)
    if any(w in t for w in ("yes", "i did", "that was me", "i made", "authorised", "authorized", "my purchase")):
        return Decision("approve", f"Thanks for confirming, {first}. I've released the hold, so your card works "
                        "normally again. Take care, and have a good day.", 0.9, "fallback", latency_ms)
    return Decision("ask", f"Thanks, {first}. Just to check — do you recognise the £{txn.amount_gbp:,.0f} "
                    f"payment at {txn.merchant}?", 0.6, "fallback", latency_ms)


def investigate_turn(session: CallSession, user_text: str) -> Decision:
    """One turn of the LLM-driven investigation loop (phase-1 logic). The model
    returns {action, message, confidence}; on mock mode or any failure a keyword
    fallback keeps the call alive."""
    txn = session.event.txn
    user_msg = (
        f"Flagged transaction: £{txn.amount_gbp:,.0f} at {txn.merchant}, {txn.city}. "
        f"Why it was flagged: {session.event.rca_reason}. "
        f"Customer: {session.customer.name} (card ending {session.customer.card_last4}, "
        f"home city {session.customer.home_city}).\n\n"
        f"Conversation so far:\n{_dialog_history(session)}\n\n"
        f"The customer just said: \"{user_text}\"\n\n"
        f"Decide your next step and reply with the JSON contract."
    )
    t0 = time.perf_counter()
    data = nemotron.complete_json(prompts.INVESTIGATION_DIALOG_SYSTEM, user_msg,
                                  stage="dialog", session_id=session.session_id, max_tokens=200)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    action = (data.get("action") or "").strip().lower()
    message = (data.get("message") or "").strip()
    if action not in _VALID_ACTIONS or not message:
        return _fallback_decision(session, user_text, latency_ms)
    try:
        confidence = float(data.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    return Decision(action, message, confidence, nemotron.NIM_MODEL_REALTIME, latency_ms)

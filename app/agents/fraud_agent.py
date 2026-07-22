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

from app.llm import nemotron
from app.agents import prompts
from app.database.models import AgentTurn, CallSession, Intent

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


def fast_path_turn(session: CallSession, user_text: str) -> AgentTurn | None:
    """A deterministic AgentTurn for an exact confirm/deny (no LLM call), or
    None. Confidence is 1.0 because an exact match is a certainty, not a
    probabilistic guess. The reply mirrors what the LLM path says for the same
    intent; on a high-value deny the orchestrator will override this reply with
    the escalation message anyway (policy is unchanged), so the canned text is
    only ever spoken on the paths where the LLM would have said the same thing."""
    intent = fast_path_intent(user_text)
    if intent is None:
        return None
    last4 = session.customer.card_last4
    if intent == Intent.CONFIRM_LEGIT:
        return AgentTurn(intent=intent, confidence=1.0,
                         reply=(f"Great, thank you. I've released the hold and your card ending "
                                f"{last4} is working normally again. Sorry for the interruption."))
    return AgentTurn(intent=intent, confidence=1.0,
                     reply=(f"Thank you for confirming. I'm blocking card ending {last4} now and a "
                            f"replacement is on its way. You will not be liable for this transaction."))


def opening_line(session: CallSession) -> str:
    txn = session.event.txn
    return (f"Hello {session.customer.name.split()[0]}, this is Barclays' automated "
            f"fraud-prevention assistant; this call is recorded. We paused a payment of "
            f"£{txn.amount_gbp:,.0f} at {txn.merchant}. To verify this call is "
            f"genuine, please read the verification code we just sent to your Barclays app.")


def handle_turn(session: CallSession, user_text: str) -> nemotron.TurnResult:
    """Returns the full TurnResult (not just the AgentTurn) so the orchestrator
    can stamp model_id/prompt_version/latency_ms onto the audit record without
    this module needing to know anything about audit logging itself."""
    context = {
        "name": session.customer.name,
        "last4": session.customer.card_last4,
        "home_city": session.customer.home_city,
        "merchant": session.event.txn.merchant,
        "amount": session.event.txn.amount_gbp,
        "city": session.event.txn.city,
        "rca": session.event.rca_reason,
        "verified": session.state.value,
        "session_id": session.session_id,   # for per-call latency metrics
    }
    history = [{"role": m["role"], "content": m["text"]} for m in session.transcript]
    return nemotron.complete_turn(prompts.REALTIME_SYSTEM, history, user_text, context)

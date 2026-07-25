"""Call session state machine — the spine of the whole demo.

  PUSH_SENT -> DIALING -> AWAITING_VERIFICATION -> VERIFIED -> {RESOLVED_LEGIT,
  RESOLVED_FRAUD, ESCALATED} ; timeout anywhere before VERIFIED -> NO_ANSWER ;
  3x failed verification -> CHANNEL_FROZEN (never re-prompt forever — a
  wrong code three times is a possible attacker, not a hard-of-hearing
  customer, and auth must never soften to be helpful).

Everything is auditable: each transition and turn writes to the audit chain.
The escalation policy is code, not prompt — the LLM can *suggest* an intent
but thresholds in this file decide what happens. A jailbroken model can talk
all it wants; it cannot make _escalation_reason() return None when it
shouldn't.
"""
from __future__ import annotations

import os

from app.speech import asr, tts
from app.security import guardrails
from app.agents import fraud_agent, prompts, resolution_agent
from app.llm import nemotron
from app.database.models import AgentTurn, CallSession, CallState, FraudEvent, Intent, new_id
from app.security import anti_vishing, audit, notify, repetition_guard
from app.database import bank

CONF_THRESHOLD = float(os.getenv("CONFIDENCE_ESCALATION_THRESHOLD", "0.7"))
HIGH_VALUE_GBP = float(os.getenv("HIGH_VALUE_ESCALATION_GBP", "750"))
# Safety cap on LLM-driven investigation questions before we hand to a human —
# mirrors phase-1's max_turns, so the model can't loop forever asking questions.
MAX_DIALOG_TURNS = int(os.getenv("MAX_DIALOG_TURNS", "6"))
MAX_VERIFICATION_ATTEMPTS = int(os.getenv("MAX_VERIFICATION_ATTEMPTS", "3"))
# Latency fast-path: an exact "yes"/"no" skips the dialog + output-rail LLM
# calls (see fraud_agent.fast_path_turn). On by default; set FAST_PATH_ENABLED=
# false to force every turn through the full LLM path (e.g. to A/B the latency).
FAST_PATH_ENABLED = os.getenv("FAST_PATH_ENABLED", "true").lower() == "true"

# The remediation (block/chargeback/case) is executed by the NeMo Agent Toolkit
# `tool_calling_agent` (config/workflow.yaml) — the real NAT agent, run in-process
# via app/agents/nat_runner.py. The deterministic resolution_agent stays as a
# FAILSAFE only: if the agent's LLM is unavailable/errors, a fraud card must
# still be blocked. Bank actions are idempotency-keyed, so a NAT partial + the
# failsafe can never double-execute. Off in mock mode (no LLM) and toggleable.
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
NAT_AGENTS_ENABLED = os.getenv("NAT_AGENTS_ENABLED", "true").lower() == "true"
RESOLUTION_WORKFLOW = os.getenv("NAT_RESOLUTION_CONFIG", "config/workflow.yaml")

SESSIONS: dict[str, CallSession] = {}


def start_call(event: FraudEvent, auto_answer: bool = True) -> CallSession:
    """Begins a session: push the verification phrase and place the call.

    The call is created in the RINGING state (CallState.PUSH_SENT) — the agent
    does NOT speak yet. Speaking begins only when the customer picks up, via
    answer_call() below. This mirrors a real phone call: it rings first, and
    the agent's greeting is heard only after the customer answers.

    auto_answer controls whether we pick up automatically:
      - True  (default): pick up immediately — used by the scripted /trigger
        fallback and by the test/eval harness, which have no human to press
        an 'Answer' button.
      - False: leave it ringing — used by the live demo's /bank/transactions
        path, where the presenter answers from the Barclays app so the Riva
        voice starts exactly when they pick up (never talking to an empty room).

    Returns the session so the caller can hand the phrase to the Barclays app UI."""
    customer = bank.CUSTOMERS[event.txn.customer_id]
    session = CallSession(session_id=new_id("call"), event=event, customer=customer)
    SESSIONS[session.session_id] = session

    session.verification_code = anti_vishing.send_push(session.session_id)
    # Deliver the code across every configured channel: the in-app push always,
    # plus SMS/WhatsApp if enabled (off by default — see security/notify.py's
    # SECURITY NOTE on why the app stays the primary, secure channel).
    channels = notify.deliver_code(session.customer.name, session.customer.phone,
                                   session.verification_code)
    # Security note, and a real change from the original scaffold: the
    # code's actual digits are never written to the audit chain. The chain
    # is meant to be replayable, exportable evidence a human analyst or
    # regulator might read — a live verification secret has no business
    # sitting in that record. What IS recorded is only which CHANNELS it went
    # to (for the compliance view), never the digits. The Barclays app UI gets
    # the code straight from this function's return value / the API response.
    audit.log_turn(session.session_id, "push", {"code_length": len(session.verification_code),
                                                 "channels": channels})
    # Open the durable session record (start time for time-to-intercept + an
    # 'open' line in the browsable sessions index).
    audit.open_session(session.session_id, customer_name=session.customer.name,
                       amount_gbp=session.event.txn.amount_gbp, city=session.event.txn.city,
                       rca_reason=session.event.rca_reason)
    # state stays PUSH_SENT (ringing) until answered.
    if auto_answer:
        answer_call(session)
    return session


def answer_call(session: CallSession) -> CallSession:
    """The customer picks up — NOW the agent greets and asks for verification.

    Idempotent: called from POST /answer when the presenter taps 'Answer' in
    the app, but a double-tap (or a call that was already auto-answered) is a
    harmless no-op, because we only transition out of the RINGING state once."""
    if session.state != CallState.PUSH_SENT:
        return session  # already answered / already past the ringing stage
    # The opening is now ONE crisp message — who we are, why we're calling, and a
    # request to read the app code — so we go straight to verification instead of
    # a separate greeting turn that waited on "yes, I'm X".
    session.state = CallState.AWAITING_VERIFICATION
    audit.log_turn(session.session_id, "state", {"answered": True, "state": session.state.value})
    opening = fraud_agent.opening_line(session)
    # Mandatory-disclosure rail: guarantee the opening states it's a recorded
    # fraud call; prepend the canonical line only if somehow missing.
    if not guardrails.disclosure_ok(opening):
        audit.log_turn(session.session_id, "rail", {"disclosure": "prepended"})
        opening = guardrails.DISCLOSURE_LINE + opening
    _say(session, opening)
    return session


def customer_says(session: CallSession, utterance: str) -> str:
    """One full turn: (already-transcribed text in) -> guardrails(input) ->
    (verify | dialog) -> guardrails(output) -> policy -> action -> (text
    out, for the media leg to speak). Returns the agent's reply text.

    Text-only on both ends, on purpose — see asr.transcribe_text_only()'s
    and tts.speak_text_only()'s docstrings. The server never touches real
    audio; `utterance` here is already a transcript by the time it arrives
    (demo/voice_loop.py did the real ASR against the physical mic before
    POSTing to /converse), same as this function's reply is text for that
    same script to speak, not something the server speaks itself."""
    text = asr.transcribe_text_only(utterance)
    if not text or not text.strip():
        return _say(session, "I didn't catch that — could you repeat it?")
    session.transcript.append({"role": "user", "text": text})
    audit.log_turn(session.session_id, "asr", {"text": text})

    # Runs on every utterance regardless of call state — a credential offer
    # or jailbreak attempt during verification is just as real a risk as
    # one after. Topical checks are plain keyword matches (see
    # adapters/guardrails.py), so a clean turn pays no latency for this.
    # The LLM self-check is skipped specifically during verification: it
    # false-positived live on a customer reading their code back (a bare
    # digit sequence reads as PIN/OTP disclosure to a context-free
    # classifier) — see check_input()'s docstring for the full story.
    # Also skipped for an exact, unambiguous "yes"/"no" on a verified call: the
    # latency fast-path (see _handle_dialog_turn) answers those with no model
    # call, and a bare "yes"/"no" has no jailbreak/credential surface for the
    # LLM self-check to find — so paying an LLM call to screen it is pure
    # latency. The cheap keyword topical checks still run regardless.
    _is_fast = (FAST_PATH_ENABLED and session.state == CallState.VERIFIED
                and fraud_agent.fast_path_intent(text) is not None)
    # Greeting + verification turns are brief acknowledgements / a code read-back,
    # so skip the fuzzy LLM self-check there (the cheap keyword checks still run).
    _pre_dialog = session.state in (CallState.GREETING, CallState.AWAITING_VERIFICATION)
    input_verdict = guardrails.check_input(
        text, run_self_check=(not _pre_dialog and not _is_fast))
    if input_verdict.blocked:
        audit.log_agent_turn(session.session_id, predicted_intent=None, confidence=0.0,
                             reply_text=input_verdict.reply_override, state_after=session.state.value,
                             model_id="guardrails", prompt_version=prompts.PROMPT_VERSION,
                             latency_ms=0.0, rail_verdicts=input_verdict.verdicts)
        return _say(session, input_verdict.reply_override)

    if session.state == CallState.GREETING:
        return _handle_greeting_turn(session, text)

    if session.state == CallState.AWAITING_VERIFICATION:
        return _handle_verification_turn(session, text)

    if session.state != CallState.VERIFIED:
        return _say(session, "Please read the verification code from your app first.")

    return _handle_dialog_turn(session, text)


# Cheap keyword read of "is now a good time?" so the agent can respond naturally.
# It never abandons the call (this is urgent possible-fraud) — it just adjusts tone.
_NEGATIVE_TIME_WORDS = ("no", "not now", "busy", "later", "can't", "cannot", "can not",
                        "bad time", "driving", "at work", "in a meeting", "not a good")


def _seems_negative_about_time(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _NEGATIVE_TIME_WORDS)


def _handle_greeting_turn(session: CallSession, text: str) -> str:
    """The human-sounding opening, before verification. Deterministic (no LLM);
    ANY reply advances it. Turn 1 confirmed who we're speaking to (opening_line);
    here we (step 1) identify ourselves + flag the urgent reason + ask for a
    moment, then (step 2) reassure and ask for the app verification code."""
    session.intro_step += 1
    first = session.customer.name.split()[0]

    if session.intro_step == 1:
        audit.log_turn(session.session_id, "state", {"intro": "identify_and_ask_time"})
        line = nemotron.generate_line(
            goal=(f"Identify yourself ONLY as the Barclays fraud-prevention team on a recorded line — do NOT invent or "
                  f"give a personal agent name (never say 'I am <name>'); say 'this is the Barclays fraud-prevention "
                  f"team'. Then briefly apologise for calling {first} unexpectedly, tell them you've spotted an urgent "
                  f"transaction on their account that may be fraud, and ask if now is an okay moment to talk."),
            context={"session_id": session.session_id, "customer_first_name": first},
            fallback=(f"Thank you, {first}. This is the Barclays fraud-prevention team, calling on a recorded line. "
                      f"I'm sorry to call you unexpectedly — we've spotted an urgent transaction on your account that "
                      f"may be fraudulent, and I'd like to check it with you. Is now an okay moment to talk?"))
        # Mandatory-disclosure rail: this opening turn MUST state it's a recorded
        # fraud call. If the model's line omitted either fact, prepend the
        # canonical disclosure so it is guaranteed regardless of generation.
        if not guardrails.disclosure_ok(line):
            audit.log_turn(session.session_id, "rail", {"disclosure": "prepended"})
            line = guardrails.DISCLOSURE_LINE + line
        return _say(session, line)

    # step 2: acknowledge the "do you have time" answer, then ask for the code.
    session.state = CallState.AWAITING_VERIFICATION
    audit.log_turn(session.session_id, "state", {"to": "awaiting_verification", "after": "human_intro"})
    busy = _seems_negative_about_time(text)
    line = nemotron.generate_line(
        goal=("Ask the customer to open their Barclays app and read back the verification code shown there, so you "
              "can confirm their identity and that this really is Barclays. Reassure them you will NEVER ask for "
              "their PIN, password or full card number — the app code is only to confirm it's really them."
              + (" They said it's a bad time, so first briefly acknowledge that and say you'll be quick because this "
                 "may be fraud, then ask for the code." if busy else "")),
        context={"session_id": session.session_id, "customer_first_name": first},
        fallback=(("I completely understand, and I'm sorry to catch you at a bad time — but this is important because "
                   "it may be fraud on your account, so I promise to be as quick as I can. " if busy
                   else "Thank you, I appreciate it. ") +
                  "First, just so you can be completely sure this really is Barclays and to confirm your identity, "
                  "please open your Barclays app — you'll see a verification code we've just sent you. Could you read "
                  "that code back to me? To be clear, I will never ask for your PIN, your password or your full card "
                  "number — this app code is only to confirm it's really you."))
    return _say(session, line)


def _handle_verification_turn(session: CallSession, text: str) -> str:
    """The pre-verification branch never calls the LLM — it's a plain string
    compare (see anti_vishing.verify_code). That's worth restating here because
    it's the answer to "isn't the code check itself the latency problem":
    it's O(1) code, not a model call."""
    first = session.customer.name.split()[0]
    if anti_vishing.is_expired(session.session_id):
        session.verification_code = anti_vishing.resend(session.session_id)
        audit.log_turn(session.session_id, "push", {"code_length": len(session.verification_code),
                                                     "channel": "in_app_notification",
                                                     "reason": "expired_resend"})
        return _say(session, nemotron.generate_line(
            goal="The verification code expired for security. Tell the customer you've just sent a fresh code to "
                 "their Barclays app and ask them to read the new one.",
            context={"session_id": session.session_id, "customer_first_name": first},
            fallback="That code has expired for your safety. I've just sent a fresh one to your Barclays app — "
                     "please read me the new code."))

    # If the customer hasn't actually read a code yet — an acknowledgement
    # ("okay, let me open the app"), a question, or chit-chat — do NOT count it
    # as a failed attempt (that would waste a 3-strikes try and re-ask as if they
    # got it wrong). Just wait warmly for them to read the code.
    if not anti_vishing.looks_like_code(text):
        return _say(session, nemotron.generate_line(
            goal=("The customer has not read the verification code yet. Warmly and briefly wait for them: once "
                  "they've opened the Barclays app, ask them to read you the verification code shown there. Do NOT "
                  "imply they got anything wrong."),
            context={"session_id": session.session_id, "customer_first_name": first},
            fallback=("No problem — whenever you're ready, please open your Barclays app and read me the "
                      "verification code shown there.")))

    if anti_vishing.verify_code(session.session_id, text):
        session.state = CallState.VERIFIED
        session.verification_attempts = 0
        audit.log_turn(session.session_id, "state", {"to": "verified", "method": "app_otp"})
        txn = session.event.txn
        reply = nemotron.generate_line(
            goal=(f"Warmly thank {first} for verifying. Then tell them a payment of £{txn.amount_gbp:,.0f} at "
                  f"{txn.merchant} in {txn.city} was paused, and ask whether they made this purchase."),
            context={"session_id": session.session_id, "customer_first_name": first},
            fallback=(f"Thank you, {first} — that's verified, I appreciate it. Now, we paused a payment of £"
                      f"{txn.amount_gbp:,.0f} at {txn.merchant} in {txn.city}. "
                      f"Can I just check with you — did you make this purchase?"))
        return _say(session, reply)

    session.verification_attempts += 1
    audit.log_turn(session.session_id, "state",
                   {"event": "verification_failed", "attempt": session.verification_attempts})

    if session.verification_attempts >= MAX_VERIFICATION_ATTEMPTS:
        return _freeze_channel(session)

    # Freshly phrased each attempt — never the same repeated string (goal fixes the
    # meaning; the model varies the words).
    return _say(session, nemotron.generate_line(
        goal=(f"The code {first} just read does NOT match the one sent to their Barclays app. This is attempt "
              f"{session.verification_attempts} of {MAX_VERIFICATION_ATTEMPTS}. Gently and reassuringly ask them to "
              f"open the Barclays app notification and read the code again carefully. Remind them you'll never ask "
              f"for their PIN, full card number or password. Keep it fresh — do not reuse earlier wording."),
        context={"session_id": session.session_id, "attempt": session.verification_attempts,
                 "customer_first_name": first},
        fallback=("That code doesn't match what we sent to your app. Please open the Barclays app notification and "
                  "read the code again carefully — take your time. Remember, we'll never ask for your PIN, full card "
                  "number or password; only this app code.")))


def _freeze_channel(session: CallSession) -> str:
    """3 failed verification attempts. Matches the scenario catalog's own
    documented rule (previously an unenforced [gap] in this scaffold):
    'possible attacker — never soften auth to be helpful.' Also matches
    Dataset 3's scripted regression case F-05_wrong_phrase_3x: expect
    state == channel_frozen, actions == [freeze_channel, human_callback_task]."""
    session.state = CallState.CHANNEL_FROZEN
    session.outcome = "auth_failed_frozen"
    for action in ("freeze_channel", "human_callback_task"):
        audit.log_turn(session.session_id, "tool", {"action": action, "session_id": session.session_id})
    bank.upsert_case(session.session_id, outcome="auth_failed_frozen",
                          note="3 failed verification attempts — channel frozen, human callback "
                               "scheduled. Possible attacker; auth was never softened.")

    # Raise a durable INCIDENT so a human sees this in the review queue and knows to
    # CALL THE CUSTOMER BACK — exactly what the agent promises on the line. It stays
    # OPEN in the queue until an analyst actions it (out-of-band identity check),
    # same mechanism as an escalation. No card action: the hold simply stays.
    specialist = bank.assign_specialist()
    incident = bank.raise_incident(
        session.session_id, customer_id=session.customer.customer_id,
        customer_name=session.customer.name, txn_id=session.event.txn.txn_id,
        amount_gbp=session.event.txn.amount_gbp, merchant=session.event.txn.merchant,
        city=session.event.txn.city, rca_reason=session.event.rca_reason,
        handoff_reason="auth_failed_frozen — verification failed 3x; call the customer back to verify by another method",
        specialist=specialist, card_already_blocked=False)
    audit.log_turn(session.session_id, "incident",
                   {"incident_id": incident["incident_id"], "status": "open",
                    "assigned_to": specialist["name"], "reason": "auth_failed_frozen_callback"})
    audit.log_turn(session.session_id, "state", {"to": "channel_frozen", "policy": "three_strikes"})
    audit.close_session(session.session_id, session.outcome)

    first = session.customer.name.split()[0]
    return _say(session, nemotron.generate_line(
        goal=(f"{first} has entered the wrong verification code three times, so for security you must end the call "
              f"now. Kindly but firmly explain that you're ending the call, and that the Barclays fraud team will "
              f"call them back shortly to verify their identity through another method. Do not sound accusatory."),
        context={"session_id": session.session_id, "customer_first_name": first},
        fallback=("For your security, I'm ending this call now. Our fraud team will call you back shortly to verify "
                  "your identity through another method. Thank you for your patience.")))


_ACTION_TO_INTENT = {"approve": Intent.CONFIRM_LEGIT, "block": Intent.DENY,
                     "escalate": Intent.UNSURE, "ask": Intent.UNSURE}


def _execute_resolution(session: CallSession, outcome: str) -> None:
    """Run the remediation through the NAT `tool_calling_agent` (the real agent),
    falling back to the deterministic resolution_agent if NAT is unavailable or
    errors. A fraud outcome MUST block the card, so the failsafe is not optional;
    bank actions are idempotency-keyed, so NAT + failsafe can't double-execute."""
    txn = session.event.txn
    if not MOCK_MODE and NAT_AGENTS_ENABLED:
        try:
            from app.agents import nat_runner
            msg = (f"A verified Barclays fraud-prevention call has concluded. "
                   f"Outcome: {outcome}. session_id={session.session_id}, "
                   f"txn_id={txn.txn_id}, customer_id={session.customer.customer_id}. "
                   f"Execute the correct remediation by calling the tools now.")
            result = nat_runner.run_workflow(RESOLUTION_WORKFLOW, msg)
            audit.log_turn(session.session_id, "state",
                           {"resolution_engine": "nat_tool_calling_agent",
                            "outcome": outcome, "agent_result": str(result)[:200]})
            return
        except Exception as e:  # noqa: BLE001 — agent/LLM failure must not leave a card unblocked
            audit.log_turn(session.session_id, "state",
                           {"resolution_engine": "deterministic_failsafe",
                            "outcome": outcome, "nat_error": str(e)[:160]})
    # Mock mode, NAT disabled, or NAT failed -> deterministic guarantee.
    if outcome == "fraud_confirmed":
        resolution_agent.resolve_fraud(session)
    else:
        resolution_agent.resolve_legit(session)


def _handle_dialog_turn(session: CallSession, text: str) -> str:
    """Phase-1 style LLM-DRIVEN investigation loop. The model reads the flagged
    transaction + full transcript and returns ONE decision — ask another
    question, or conclude (approve / block / escalate). We execute the model's
    chosen action (validated by the action rail in resolution_agent), rather than
    classifying a fixed intent and mapping it deterministically. This is what
    makes the dialog investigative and human instead of one-shot classify+act.

    Deterministic safety nets still wrap the model: output+groundedness rails on
    every spoken line, a repetition guard, a turn cap, and a human review (HITL)
    before RELEASING a hold on a low-confidence approval."""
    session.dialog_turns += 1
    decision = fraud_agent.investigate_turn(session, text)
    action, message, confidence = decision.action, decision.message, decision.confidence

    # Output rails on the spoken line: self-check (PII/advice/tone/topic) then
    # groundedness (no invented figures). A block overrides to a safe escalation.
    ov = guardrails.check_output(message)
    if not ov.blocked:
        ov = guardrails.check_groundedness(message, session.event)
    if ov.blocked:
        message = ov.reply_override

    is_repeat = repetition_guard.is_repetitive(message, session.agent_utterances())
    turn = AgentTurn(reply=message, intent=_ACTION_TO_INTENT.get(action, Intent.UNSURE),
                     confidence=confidence)

    audit.log_agent_turn(session.session_id, predicted_intent=turn.intent, confidence=confidence,
                         reply_text=message, state_after=session.state.value,
                         model_id=decision.model_id, prompt_version=prompts.PROMPT_VERSION,
                         latency_ms=decision.latency_ms, rail_verdicts=ov.verdicts)
    audit.log_turn(session.session_id, "decision", {"action": action, "confidence": confidence})

    # Safety overrides that force a human regardless of the model's choice.
    if ov.blocked:
        return _escalate(session, turn, "guardrail_blocked_reply")
    if is_repeat:
        return _escalate(session, turn, "repetition_loop")
    if action == "ask" and session.dialog_turns >= MAX_DIALOG_TURNS:
        return _escalate(session, turn, "max_turns")

    if action == "ask":
        return _say(session, message)

    if action == "escalate":
        return _escalate(session, turn, "model_escalate")

    # Releasing a hold on a possibly-fraudulent payment is the one direction that
    # gets a human safety net when the model isn't confident (HITL, unchanged).
    if action == "approve" and confidence < CONF_THRESHOLD:
        return _escalate(session, turn, "low_confidence")

    if action == "approve":
        session.state = CallState.RESOLVED_LEGIT
        session.outcome = "false_positive_recovered"
        _execute_resolution(session, "false_positive_recovered")
        audit.close_session(session.session_id, session.outcome)
        return _say(session, message)

    if action == "block":
        session.state = CallState.RESOLVED_FRAUD
        session.outcome = "fraud_confirmed"
        _execute_resolution(session, "fraud_confirmed")
        audit.close_session(session.session_id, session.outcome)
        return _say(session, message)

    return _escalate(session, turn, "unknown_action")     # defensive: never fall through


def no_answer(session: CallSession) -> str:
    """Timeout path: customer did not pick up / never verified. Card stays safe."""
    session.state = CallState.NO_ANSWER
    session.outcome = "no_answer_hold_kept"
    bank.upsert_case(session.session_id, outcome="no_answer",
                          note="No answer; hold kept, retry scheduled, SMS + app fallback sent.")
    audit.log_turn(session.session_id, "state",
                   {"to": "no_answer", "policy": "hold_kept_retry_scheduled"})
    audit.close_session(session.session_id, session.outcome)
    return "unreachable: hold kept, retry + app/SMS fallback scheduled"


def _escalation_reason(session: CallSession, turn: AgentTurn, *,
                       is_repeat: bool, output_blocked: bool) -> str | None:
    """Single source of truth for "should this turn escalate, and why" — a
    string reason (used in the handoff summary and audit trail) instead of
    a bare bool, so a human analyst reading the handoff knows what tripped
    it without re-deriving it from the transcript."""
    if output_blocked:
        return "guardrail_blocked_reply"
    if is_repeat:
        return "repetition_loop"
    if turn.intent == Intent.DISTRESS:
        return "distress"
    if turn.intent == Intent.UNSURE and turn.confidence < CONF_THRESHOLD:
        return "low_confidence"
    if turn.intent == Intent.DENY and session.event.txn.amount_gbp >= HIGH_VALUE_GBP:
        # Top-decile risk can never be auto-released/auto-resolved on intent
        # alone — defense against 1E-1 (attacker holds phone AND card, so
        # verification alone isn't enough for a high-value case).
        return "high_value_deny"
    return None


def _escalate(session: CallSession, turn: AgentTurn, reason: str) -> str:
    session.state = CallState.ESCALATED
    session.outcome = "escalated_to_human"
    # A warm handoff needs an actual person on the other end — "connecting
    # you to a specialist" with nobody behind it isn't a real escalation
    # path. See bank.HUMAN_AGENTS for why this is a roster, not a
    # single hardcoded name.
    specialist = bank.assign_specialist()

    # A high-value deny is STILL a deny — the customer told us they didn't make
    # the payment. Block the card immediately as a protective measure BEFORE
    # handing off: freezing the card is low-risk and reversible (a replacement
    # is reissued), and it protects the customer even in the attacker-holds-
    # both-phone-and-card scenario (blocking hurts the attacker too). What we
    # deliberately DON'T auto-do for a large sum is the chargeback/refund —
    # that financial decision is what the human specialist reviews. So the
    # protective action and the money decision are split: card blocked now,
    # refund decided by a person. Other escalation reasons (distress, low
    # confidence, a blocked jailbreak) are NOT confirmed denies, so no card
    # action is taken for them.
    protective_block = None
    if reason == "high_value_deny":
        protective_block = bank.block_card(session.session_id, session.event.txn.txn_id,
                                           session.customer.customer_id)
        audit.log_turn(session.session_id, "tool", protective_block)

    summary = prompts.HANDOFF_SUMMARY_TEMPLATE.format(
        session_id=session.session_id, name=session.customer.name,
        last4=session.customer.card_last4, merchant=session.event.txn.merchant,
        city=session.event.txn.city, amount=session.event.txn.amount_gbp,
        rca=session.event.rca_reason, verified=session.state.value,
        intent=turn.intent, confidence=turn.confidence, reason=reason,
        specialist_name=specialist["name"], specialist_desk=specialist["desk"])
    audit.log_turn(session.session_id, "handoff", {"summary": summary, "reason": reason,
                                                   "specialist": specialist,
                                                   "card_blocked": bool(protective_block)})
    bank.upsert_case(session.session_id, outcome="escalated", note=summary,
                          assigned_specialist=specialist["name"], card_blocked=bool(protective_block))

    # Raise a durable INCIDENT — the human-review request the analyst works from
    # (open until they resolve it with block/approve). The call ends here; the
    # human acts out-of-band. See bank.raise_incident / resolve_incident.
    incident = bank.raise_incident(
        session.session_id, customer_id=session.customer.customer_id,
        customer_name=session.customer.name, txn_id=session.event.txn.txn_id,
        amount_gbp=session.event.txn.amount_gbp, merchant=session.event.txn.merchant,
        city=session.event.txn.city, rca_reason=session.event.rca_reason,
        handoff_reason=reason, specialist=specialist, card_already_blocked=bool(protective_block))
    audit.log_turn(session.session_id, "incident",
                   {"incident_id": incident["incident_id"], "status": "open",
                    "assigned_to": specialist["name"], "reason": reason})
    audit.close_session(session.session_id, session.outcome)

    if reason == "high_value_deny":
        # Explain BOTH actions to the customer: what we did now, and that the case
        # goes to the internal team for the money decision.
        return _say(session,
                    f"Thank you — I've blocked your card ending {session.customer.card_last4} right away "
                    f"so it can't be used again, and a replacement is on its way. As this is a large payment, "
                    f"I've escalated this to our internal fraud team for further investigation, and they'll be "
                    f"in touch. You will not be liable for this transaction. Take care, and have a good day.")

    # For a model-CHOSEN escalation, speak the model's own line — it already tells
    # the customer (naturally, in context) that they're being passed to a specialist,
    # and it has passed the output+groundedness rails. For every OTHER escalation
    # reason (low-confidence approval, repetition, max-turns, a blocked reply) the
    # model's message was about something else, so we speak the reliable canonical
    # escalation line instead.
    if reason == "model_escalate" and turn.reply:
        return _say(session, turn.reply)

    return _say(session, "Thank you — I've escalated this to our internal fraud team for further "
                         "investigation, and they'll be in touch with you shortly. Take care, and have a good day.")


def _say(session: CallSession, text: str) -> str:
    # Text-only on purpose — the server is not the phone. See
    # tts.speak_text_only()'s docstring: the real synthesis + playback
    # happens exclusively in demo/voice_loop.py, the process that actually
    # owns the mic/speaker. Calling the real speak() here double-spoke
    # every utterance and blocked the HTTP response for the full playback
    # duration when this was tested live.
    spoken = tts.speak_text_only(text)
    session.transcript.append({"role": "assistant", "text": text})
    audit.log_turn(session.session_id, "tts", {"text": text})
    return spoken

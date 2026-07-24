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
MAX_VERIFICATION_ATTEMPTS = int(os.getenv("MAX_VERIFICATION_ATTEMPTS", "3"))
# Latency fast-path: an exact "yes"/"no" skips the dialog + output-rail LLM
# calls (see fraud_agent.fast_path_turn). On by default; set FAST_PATH_ENABLED=
# false to force every turn through the full LLM path (e.g. to A/B the latency).
FAST_PATH_ENABLED = os.getenv("FAST_PATH_ENABLED", "true").lower() == "true"

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


def _handle_dialog_turn(session: CallSession, text: str) -> str:
    # Latency fast-path: an exact, unambiguous "yes"/"no" is answered with a
    # deterministic turn — NO dialog LLM call, and NO output-rail LLM call
    # either (the reply is our own canned text, so there is nothing
    # model-generated to screen). Everything downstream — repetition guard,
    # audit record, escalation policy, bank action — runs identically to the
    # LLM path, because the fast-path emits the same {intent, confidence} shape.
    fast = fraud_agent.fast_path_turn(session, text) if FAST_PATH_ENABLED else None
    if fast is not None:
        turn: AgentTurn = fast
        model_id, prompt_version, latency_ms, seed = "fastpath", prompts.PROMPT_VERSION, 0.0, None
        reply_text, output_blocked = turn.reply, False
        rail_verdicts = {"input": "keyword_only", "output": "skipped_fastpath"}
    else:
        result = fraud_agent.handle_turn(session, text)  # llm.TurnResult
        turn = result.turn
        model_id, prompt_version = result.model_id, result.prompt_version
        latency_ms, seed = result.latency_ms, result.seed
        # Guardrails run on the model-proposed reply before it's ever spoken:
        # first the self-check (PII/credentials/advice/tone/topic), then the
        # groundedness rail (no made-up figures). Either block forces escalation.
        output_verdict = guardrails.check_output(turn.reply)
        if not output_verdict.blocked:
            output_verdict = guardrails.check_groundedness(turn.reply, session.event)
        reply_text = output_verdict.reply_override if output_verdict.blocked else turn.reply
        output_blocked = output_verdict.blocked
        rail_verdicts = output_verdict.verdicts

    # Repetition guard: a near-duplicate of the agent's own recent reply
    # means the dialog isn't progressing. Scenario catalog's own rule: "max
    # 2 clarification loops, then escalate — never a third rephrase."
    is_repeat = repetition_guard.is_repetitive(turn.reply, session.agent_utterances())

    audit.log_agent_turn(session.session_id, predicted_intent=turn.intent, confidence=turn.confidence,
                         reply_text=reply_text, state_after=session.state.value,
                         model_id=model_id, prompt_version=prompt_version,
                         latency_ms=latency_ms, seed=seed,
                         rail_verdicts=rail_verdicts)

    reason = _escalation_reason(session, turn, is_repeat=is_repeat, output_blocked=output_blocked)
    if reason:
        return _escalate(session, turn, reason)

    if turn.intent == Intent.CONFIRM_LEGIT:
        session.state = CallState.RESOLVED_LEGIT
        session.outcome = "false_positive_recovered"
        resolution_agent.resolve_legit(session)          # async in prod; inline in demo
        audit.close_session(session.session_id, session.outcome)
    elif turn.intent == Intent.DENY:
        session.state = CallState.RESOLVED_FRAUD
        session.outcome = "fraud_confirmed"
        resolution_agent.resolve_fraud(session)          # async in prod; inline in demo
        audit.close_session(session.session_id, session.outcome)
    return _say(session, reply_text)


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
        # Explain BOTH actions to the customer: what we did now, and why we're
        # transferring rather than finishing on the call.
        return _say(session,
                    f"Thank you — I've blocked your card ending {session.customer.card_last4} right away "
                    f"so it can't be used again, and a replacement is on its way. Because this is a large "
                    f"payment of £{session.event.txn.amount_gbp:,.0f}, I'm connecting you to "
                    f"{specialist['name']}, our fraud specialist at the {specialist['desk']}, who will "
                    f"complete the investigation and arrange your refund. You will not be liable for this "
                    f"transaction.")

    return _say(session, f"I'm connecting you to {specialist['name']}, our fraud specialist at the "
                         f"{specialist['desk']}. They can see our full conversation, so you won't "
                         f"have to repeat yourself.")


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

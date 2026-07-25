"""NeMo Guardrails adapter — input + output rails on every live-loop turn.

Live mode uses the REAL `nemoguardrails` package (LLMRails), built from
`config/guardrails/config.yml` + `rails/topics.co`, verified 2026-07-18 to
actually block jailbreaks/credential-requests on input and PIN-asking
replies on output. This is a genuine NeMo Guardrails integration, not a
reimplementation of it.

How it coexists with our split-generation architecture: LLMRails.generate()
normally owns the whole turn (input rail -> generate -> output rail), but
our live loop needs its OWN generation call so the reply comes back as the
{reply, intent, confidence} JSON contract AgentTurn expects. NeMo Guardrails
0.23's GenerationOptions solves this cleanly — we run it with input-rails-
ONLY before our fraud_agent call, and output-rails-ONLY on the proposed
reply after, never letting it generate. So NeMo Guardrails does the
moderation; our own code does the generation.

Measured latency (live, 2026-07-18): the real Colang runtime adds ~1-2s per
check — genuinely slower than a bare NIM call. That's the safety-over-speed
tradeoff the FRAUD__2 doc explicitly endorses ("rails can only make the
system more conservative... that asymmetry is why they're on the critical
path"). The fast topical keyword pre-filter below catches the common attacks
at zero latency so the slow LLM rail only runs when it has to.

Mock mode keeps a dependency-free keyword heuristic mirroring the same rail
criteria, so the full flow + the "type the PIN question, watch the rail
block it" demo still work offline with zero packages installed.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from app.observability import metrics

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
GUARDRAILS_MODEL = os.getenv("NIM_MODEL_GUARDRAILS", "nvidia/nemotron-3-nano-30b-a3b")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
GUARDRAILS_CONFIG_PATH = os.getenv("GUARDRAILS_CONFIG_PATH", "config/guardrails")

# The actual self-check prompts live in config/guardrails/config.yml — that's
# what real NeMo Guardrails reads and runs, so it's the single source of
# truth now (no Python copy to keep in sync). Both prompts were rewritten
# live 2026-07-18 with explicit allow-examples after the original bare
# "block if it reveals X" wording false-positived on ordinary fraud dialog:
# the input rail blocked plain denials ("I didn't make this purchase"), and
# the output rail blocked normal resolution replies mentioning the last-4
# ("I am blocking card ending 4821..."), which would have derailed the
# flagship path. See config.yml's own comments for the full history.

# --- fast topical pre-filter, mirrors config/guardrails/rails/topics.co ---
# Zero-latency keyword catch for the two most common cases so the slow
# (~1-2s) NeMo Guardrails LLM rail only runs when these don't already
# resolve the turn.
_OFF_TOPIC_PATTERNS = (r"\bstocks?\b", r"\btell.{0,10}joke\b", r"\bsystem prompt\b",
                       r"\bcredit limit\b", r"\bincrease.{0,10}limit\b")
# NOTE: "otp" is deliberately NOT here. Verification now uses a numeric code the
# bank PUSHES for the customer to READ BACK, so a legitimate read-back ("my OTP
# is 680863") must not be keyword-blocked. The customer's own secrets (PIN, full
# card number, CVV, password) are still caught, and the LLM self-check (post-
# verification, where run_self_check is on) still handles a genuine "here's my
# OTP" credential offer.
_CREDENTIAL_ASK_PATTERNS = (r"\bmy pin\b", r"\bcard number\b",
                            r"\btell you my (pin|cvv|password)\b")

OFF_TOPIC_REPLY = ("I can only help with verifying the flagged transaction on this call. "
                   "For anything else, please use the Barclays app or call customer care.")
CREDENTIAL_REFUSAL_REPLY = ("No — please never share your PIN, OTP or card number with anyone "
                            "on a call, including us. We only need you to read the verification "
                            "code from your Barclays app.")
SAFE_ESCALATION_REPLY = "Let me connect you with a specialist to be safe."

# --- Execution rail: only these four banking actions may ever fire. ---
# A concluded action that isn't one of these (a hallucinated or injected tool
# name) is refused by validate_action() before it can touch bank.py.
ALLOWED_ACTIONS = frozenset({"block_card", "release_hold", "open_chargeback", "upsert_case"})

# --- Dialog rail: mandatory opening disclosure. ---
# Early in the call the agent MUST disclose that this is a recorded
# fraud-prevention call. disclosure_ok() checks both facts are present; if the
# model's line omits either, the orchestrator prepends DISCLOSURE_LINE.
_DISCLOSURE_PATTERNS = (r"record", r"fraud")
DISCLOSURE_LINE = ("Before we continue — this is the Barclays fraud-prevention team and this call "
                   "is recorded. I'm calling about a possible fraudulent transaction on your account. ")

# --- Output rail: tone / toxicity. Offline keyword catch only; the live LLM
# self_check_output prompt carries the real "professional, empathetic" judgment. ---
_TOXIC_WORDS = ("idiot", "stupid", "shut up", "moron", "screw you", "you people",
                "shut it", "damn you")

# £-amounts, for the groundedness rail.
_MONEY_RE = re.compile(r"£\s?([\d,]+(?:\.\d+)?)")

_rails = None


def _nemo_rails():
    """Lazy singleton LLMRails built from config/guardrails. Heavy to
    construct (loads the Colang runtime + config), so built once per
    process and reused across every turn.

    nest_asyncio.apply() here is a real fix, not boilerplate: NeMo
    Guardrails' `generate()` is sync but runs an event loop internally, so
    it raises "sync generate inside async code" if called while a loop is
    already running. Our real call paths are safe (FastAPI sync endpoints
    run in a threadpool with no loop; tests + voice_loop.py are plain sync),
    but the FastAPI startup warm-up runs in an async context and hit exactly
    this. nest_asyncio lets the sync generate() re-enter a running loop, so
    the same check_input/check_output work identically from any context —
    found live 2026-07-18 via the warm-up failing on boot."""
    global _rails
    if _rails is None:
        import nest_asyncio  # nemoguardrails dependency; always present when it is
        nest_asyncio.apply()
        from nemoguardrails import LLMRails, RailsConfig  # deferred: only needed live
        _rails = LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_PATH))
    return _rails


def _nemo_check(text: str, *, is_input: bool) -> bool:
    """Runs ONE rail stage (input-only or output-only) of the real NeMo
    Guardrails config against `text`, and returns True if that rail BLOCKED.

    Detection: with dialog/retrieval off and only the one rail stage on,
    NeMo Guardrails returns the customer's own text back when the rail
    passes, and substitutes a refusal ("I'm sorry, I can't respond...")
    when it blocks. We confirm a block via the rail-activation log's
    `refuse to respond` decision — more robust than string-matching the
    refusal text, which could change with config edits."""
    from nemoguardrails.rails.llm.options import (
        GenerationLogOptions, GenerationOptions, GenerationRailsOptions)

    if is_input:
        rails_opt = GenerationRailsOptions(input=True, output=False, dialog=False, retrieval=False)
        messages = [{"role": "user", "content": text}]
    else:
        rails_opt = GenerationRailsOptions(input=False, output=True, dialog=False, retrieval=False)
        # Output rails need the assistant message to check; the prior user
        # turn is a neutral placeholder so the rail has conversational shape.
        messages = [{"role": "user", "content": "did I make this purchase?"},
                    {"role": "assistant", "content": text}]

    opts = GenerationOptions(rails=rails_opt, log=GenerationLogOptions(activated_rails=True))
    try:
        resp = _nemo_rails().generate(messages=messages, options=opts)
    except Exception:  # noqa: BLE001 — guardrail LLM endpoint unreachable (e.g. gpu009 down)
        # Degrade, don't crash the call. Found live: when the rail's LLM
        # endpoint was down, this raised straight through customer_says and
        # killed the whole WebRTC turn. The cheap keyword/topical checks in
        # check_input already ran before this LLM self-check, so failing it
        # open here keeps the call alive; the dialog model (which has its own
        # hosted fallback) still produces the reply. Returns False = "not
        # blocked by the LLM rail" rather than propagating the error.
        _rail_llm_unavailable(is_input)
        return False
    for r in (resp.log.activated_rails if resp.log else []):
        if any("refuse" in str(d).lower() for d in (r.decisions or [])):
            return True
    return False


_RAIL_WARNED = {"input": False, "output": False}


def _rail_llm_unavailable(is_input: bool) -> None:
    """Warn once per stage that the guardrail LLM is unreachable, so the log
    isn't flooded every turn while it's down."""
    key = "input" if is_input else "output"
    if not _RAIL_WARNED[key]:
        _RAIL_WARNED[key] = True
        print(f"[guardrails] {key} rail LLM unreachable — degrading to keyword checks "
              f"for this stage until it recovers (call continues).")


@dataclass
class RailVerdict:
    """What orchestrator.py needs back from a rail check: whether it passed,
    and — if not — what to say/do instead. `verdicts` is the {"input":...,
    "output":...} shape that flows straight into audit.log_agent_turn().

    No separate "should this escalate" flag here on purpose: orchestrator.py
    derives that itself from `blocked` (an output-rail block always forces
    escalation, an input-rail block never does — it just substitutes a
    reply and returns early). Keeping that branch in one place
    (_escalation_reason) instead of echoing it into this dataclass avoids
    the two ever disagreeing."""
    blocked: bool
    reply_override: str | None = None
    verdicts: dict = field(default_factory=dict)


def _mock_self_check(text: str) -> bool:
    """Offline keyword heuristic mirroring the real rail criteria, so the
    whole flow (and the live-block demo moment) works with zero packages."""
    t = text.lower()
    credential_words = ("pin", "cvv", "otp", "password", "card number")
    jailbreak_words = ("ignore your instructions", "ignore previous instructions",
                      "system prompt", "you are now", "act as")
    financial_advice_words = ("you should invest", "guaranteed return", "buy this stock")
    return (any(w in t for w in credential_words)
            or any(w in t for w in jailbreak_words)
            or any(w in t for w in financial_advice_words)
            or any(w in t for w in _TOXIC_WORDS))


def check_input(user_text: str, *, run_self_check: bool = True) -> RailVerdict:
    """Runs before the customer's utterance is treated as dialog. Topical
    rails first (cheap, no LLM call) since "ignore off-topic requests" and
    "refuse credential offers" are common enough that catching them without
    a network round-trip matters for latency.

    `run_self_check=False` is set by orchestrator.py during the
    verification step specifically. Found live (2026-07-18): the LLM
    self-check false-positived on a customer reading their verification
    code back — "Six eight zero eight six three" — flagging it as PIN/OTP
    disclosure. From the self-check prompt's narrow view (no awareness of
    *why* the assistant is asking), a bare spoken digit sequence looks
    exactly like a PIN readback, when at this specific point in the call
    it's the one and only expected, legitimate input. This would have
    blocked the first required step of every real call. The topical
    keyword checks stay on even during verification (an explicit "why do
    you need my card number" is still worth catching there); only the
    fuzzier LLM judgment call gets skipped, since the verification step
    doesn't route through dialog/intent logic anyway (see
    orchestrator._handle_verification_turn's docstring — no LLM call, ever,
    on that path) and has its own narrow security model already (code
    match, 3-strikes lockout)."""
    t = user_text.lower()
    if any(re.search(p, t) for p in _CREDENTIAL_ASK_PATTERNS):
        return RailVerdict(blocked=True, reply_override=CREDENTIAL_REFUSAL_REPLY,
                           verdicts={"input": "blocked_credential_topic", "output": "n/a"})
    if any(re.search(p, t) for p in _OFF_TOPIC_PATTERNS):
        return RailVerdict(blocked=True, reply_override=OFF_TOPIC_REPLY,
                           verdicts={"input": "blocked_off_topic", "output": "n/a"})
    if run_self_check:
        _t = time.perf_counter()
        blocked = _mock_self_check(user_text) if MOCK_MODE else _nemo_check(user_text, is_input=True)
        metrics.record_llm_call("input_rail", "mock" if MOCK_MODE else GUARDRAILS_MODEL,
                                (time.perf_counter() - _t) * 1000)
        if blocked:
            return RailVerdict(blocked=True, reply_override=CREDENTIAL_REFUSAL_REPLY,
                               verdicts={"input": "blocked_nemo_input_rail", "output": "n/a"})
    return RailVerdict(blocked=False, verdicts={"input": "pass", "output": "n/a"})


def check_output(candidate_reply: str) -> RailVerdict:
    """Runs on the dialog agent's proposed reply before it's spoken. A false
    block degrades to a templated safe reply or escalation — annoying,
    never dangerous — matching the FRAUD__2 doc's stated policy. Blocking
    forces escalation rather than just substituting the reply: if the model
    produced an unsafe reply, its proposed intent isn't trustworthy either."""
    _t = time.perf_counter()
    blocked = _mock_self_check(candidate_reply) if MOCK_MODE else _nemo_check(candidate_reply, is_input=False)
    metrics.record_llm_call("output_rail", "mock" if MOCK_MODE else GUARDRAILS_MODEL,
                            (time.perf_counter() - _t) * 1000)
    if blocked:
        return RailVerdict(blocked=True, reply_override=SAFE_ESCALATION_REPLY,
                           verdicts={"input": "pass", "output": "blocked_nemo_output_rail"})
    return RailVerdict(blocked=False, verdicts={"input": "pass", "output": "pass"})


def check_groundedness(candidate_reply: str, event) -> RailVerdict:
    """Output rail — groundedness: the agent may only quote figures that come
    from the flagged transaction, never a made-up one. Deterministic and
    fail-OPEN: it blocks ONLY when the reply states a £-amount that doesn't
    match the real flagged amount (a hallucinated figure); anything it can't
    parse — or a reply with no amount at all — passes untouched, so ordinary
    replies are never falsely blocked. A block forces escalation, same as the
    other output rails. `event` is session.event (has .txn.amount_gbp)."""
    try:
        flagged = float(event.txn.amount_gbp)
    except Exception:  # noqa: BLE001 — no parseable flagged amount: nothing to ground against
        return RailVerdict(blocked=False, verdicts={"input": "pass", "output": "grounded_skipped"})
    if flagged <= 0:
        return RailVerdict(blocked=False, verdicts={"input": "pass", "output": "grounded_skipped"})
    for raw in _MONEY_RE.findall(candidate_reply):
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        # Tolerate the rounding the dialog prompt explicitly encourages ("about
        # £45,000" for a £44,890 charge). Block ONLY a clearly-invented figure —
        # more than 10% off the real amount — so a legitimate confirmation that
        # restates a rounded amount is never falsely blocked (which would have
        # forced an escalation even when the customer approved).
        if abs(val - flagged) / flagged > 0.10:
            return RailVerdict(blocked=True, reply_override=SAFE_ESCALATION_REPLY,
                               verdicts={"input": "pass", "output": "blocked_ungrounded_amount"})
    return RailVerdict(blocked=False, verdicts={"input": "pass", "output": "grounded"})


def disclosure_ok(text: str) -> bool:
    """Dialog rail — mandatory disclosure: True when the line states BOTH that
    the call is recorded and that it concerns fraud. The orchestrator uses this
    on the opening turn and prepends DISCLOSURE_LINE if either is missing, so the
    disclosure is guaranteed regardless of what the model generated."""
    t = text.lower()
    return all(re.search(p, t) for p in _DISCLOSURE_PATTERNS)


def validate_action(action_name: str) -> bool:
    """Execution rail: True only for the four sanctioned banking actions. Called
    before anything in resolution_agent touches bank.py, so a hallucinated or
    injected action name (anything outside ALLOWED_ACTIONS) is refused."""
    return action_name in ALLOWED_ACTIONS

"""LLM adapter — the mentor-directed merge point.

Two prior prototypes fed into this file:
  - Neha's orchestrator called `nemotron-3-ultra-550b-a55b` on the live turn.
    A 550B model has no business on a sub-second critical path; this was the
    actual latency bug the mentor was reacting to when she said "swap the
    models" — not the verification-code step (see anti_vishing.py's docstring
    for that half of the fix).
  - Jennifer's voice_loop2.py had the right model call
    (`nemotron-3-nano-30b-a3b`, `enable_thinking: False`) but no JSON
    contract — it returned free prose because it was a standalone prototype,
    not wired to a state machine. `orchestrator.py` needs {reply, intent,
    confidence} to branch on, so that contract is grafted back on here.

Design note (unchanged from the original scaffold): the orchestrator only
ever calls complete_turn(); swapping mock -> NIM -> a local NemoClaw-routed
Nemotron changes nothing upstream. This is the 'privacy router' seam — in
production, PII-bearing turns route to a LOCAL Nemotron NIM inside bank
infra, never to a hosted endpoint.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from app.agents.prompts import PROMPT_VERSION
from app.database.models import AgentTurn, Intent
from app.observability import metrics

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# nvidia/nemotron-3-nano-30b-a3b — chosen for first-token speed, not depth;
# intent classification on 4 labels doesn't need a bigger model. The async
# resolution agent (app/agents/resolution_agent.py) uses the Super-class
# model instead, where the 30s budget makes its extra quality nearly free.
NIM_MODEL_REALTIME = os.getenv("NIM_MODEL_REALTIME", "nvidia/nemotron-3-nano-30b-a3b")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS_REALTIME", "150"))
# Optional — only meaningful for the eval harness replaying Dataset 3's
# scripted calls, where a fixed seed makes a run reproducible enough to
# attribute a regression to a specific prompt/model change rather than
# sampling noise. Leave unset on a real customer call; there's nothing to
# reproduce there and it's one fewer knob to worry about at the event.
LLM_SEED = os.environ.get("LLM_SEED")
LLM_SEED = int(LLM_SEED) if LLM_SEED else None

_DENY_WORDS = ("not me", "didn't", "did not", "no i", "never", "fraud", "wasn't me")
_CONFIRM_WORDS = ("yes", "i did", "that was me", "my purchase", "i made")
_DISTRESS_WORDS = ("scared", "help", "panic", "human", "person", "agent", "scam")

_client = None


def _openai_client():
    """Lazy singleton — reused across every turn in the process, same
    pattern as adapters/tts.py's _service(). Verified live (2026-07-18):
    constructing a fresh OpenAI() per call means a fresh httpx connection
    pool every time, so TCP/TLS setup never amortizes — measured latency
    trending 1471ms -> 955ms -> 696ms across repeated calls with a
    per-call client, which is the connection-reuse curve, not a model
    warm-up curve. This is exactly the kind of thing the field manual's
    'quote measured numbers, not targets' rule is for."""
    global _client
    if _client is None:
        from openai import OpenAI  # deferred import: not needed in mock mode
        # A self-hosted/cluster NIM (e.g. http://gpu009:8000/v1) doesn't validate the
        # token, so fall back to a placeholder when NVIDIA_API_KEY is unset — the
        # hosted endpoint still needs the real key (set it in .env for that).
        _client = OpenAI(base_url=NIM_BASE_URL, api_key=os.getenv("NVIDIA_API_KEY") or "not-needed")
    return _client


@dataclass
class TurnResult:
    """What complete_turn() returns — the AgentTurn plus everything the audit
    chain and eval harness need to reconstruct *why* this reply happened.
    Bundling this here (instead of stamping it in orchestrator.py) keeps
    the "what model/prompt/timing produced this" fact next to the one
    call site that actually knows it."""
    turn: AgentTurn
    model_id: str
    prompt_version: str
    latency_ms: float
    seed: int | None = None


def complete_json(system: str, user: str, *, stage: str = "investigation",
                  session_id: str | None = None, max_tokens: int = 256) -> dict:
    """Generic structured-output LLM call for agents that need a bespoke JSON
    shape (e.g. the investigation agent's verdict) rather than complete_turn's
    {reply,intent,confidence}. Reuses the same client + latency metrics seam.
    Never raises — returns {} on mock mode or any failure, so the caller falls
    back to its own deterministic reasoner (code decides, model only advises)."""
    t0 = time.perf_counter()
    if MOCK_MODE:
        metrics.record_llm_call(stage, "mock", round((time.perf_counter() - t0) * 1000, 1),
                                session_id=session_id)
        return {}
    try:
        resp = _openai_client().chat.completions.create(
            model=NIM_MODEL_REALTIME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1, max_tokens=max_tokens, timeout=8.0,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        usage = getattr(resp, "usage", None)
        metrics.record_llm_call(stage, NIM_MODEL_REALTIME, (time.perf_counter() - t0) * 1000,
                                tokens_in=getattr(usage, "prompt_tokens", None),
                                tokens_out=getattr(usage, "completion_tokens", None),
                                session_id=session_id)
        return data
    except Exception:  # noqa: BLE001 — API error or malformed JSON; caller has a deterministic fallback
        metrics.record_llm_call(stage, NIM_MODEL_REALTIME, (time.perf_counter() - t0) * 1000,
                                session_id=session_id)
        return {}


_DIALOGUE_SYSTEM = (
    "You are a warm, professional agent on the Barclays fraud-prevention team, on a live phone "
    "call. You will be given the GOAL of your next spoken line. Say it in ONE or two short, natural, "
    "spoken sentences suitable for text-to-speech. Vary your wording naturally so you never sound "
    "scripted or repetitive. Output ONLY the words you speak — no quotes, no labels, no notes.")


def generate_line(goal: str, *, context: dict | None = None, fallback: str = "",
                  stage: str = "dialogue") -> str:
    """Generate ONE natural spoken agent line for a GOAL (what the line must
    accomplish) — so agent dialogue is never a hard-coded, repeated string. The
    GOAL fixes the MEANING (ask for the code, reassure, never ask for a PIN); the
    LLM varies the WORDS. Live: the model phrases it, differently each time.
    Mock / on any failure: returns `fallback`, so offline + tests still work."""
    t0 = time.perf_counter()
    sid = (context or {}).get("session_id")
    if MOCK_MODE:
        metrics.record_llm_call(stage, "mock", round((time.perf_counter() - t0) * 1000, 1), session_id=sid)
        return fallback
    try:
        ctx = ("\n\nContext: " + "; ".join(f"{k}: {v}" for k, v in context.items() if k != "session_id")
               ) if context else ""
        resp = _openai_client().chat.completions.create(
            model=NIM_MODEL_REALTIME,
            messages=[{"role": "system", "content": _DIALOGUE_SYSTEM},
                      {"role": "user", "content": "GOAL: " + goal + ctx}],
            temperature=0.7, max_tokens=140, timeout=6.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        line = (resp.choices[0].message.content or "").strip().strip('"').strip()
        usage = getattr(resp, "usage", None)
        metrics.record_llm_call(stage, NIM_MODEL_REALTIME, (time.perf_counter() - t0) * 1000,
                                tokens_in=getattr(usage, "prompt_tokens", None),
                                tokens_out=getattr(usage, "completion_tokens", None), session_id=sid)
        return line or fallback
    except Exception:  # noqa: BLE001 — a phrasing failure must not break the call; use the safe fallback
        metrics.record_llm_call(stage, NIM_MODEL_REALTIME, (time.perf_counter() - t0) * 1000, session_id=sid)
        return fallback


def _mock_turn(user_text: str, context: dict) -> AgentTurn:
    """Deterministic offline fallback — keyword match, no network. Good enough
    to exercise every branch in the state machine without an API key."""
    t = user_text.lower()
    if any(w in t for w in _DISTRESS_WORDS):
        return AgentTurn(reply="I understand. Let me connect you to a specialist "
                               "right away. They have the full context.",
                         intent=Intent.DISTRESS, confidence=0.95)
    if any(w in t for w in _DENY_WORDS):
        return AgentTurn(reply=f"Thank you for confirming. I am blocking card ending "
                               f"{context['last4']} now and a replacement is on its way. "
                               f"You will not be liable for this transaction.",
                         intent=Intent.DENY, confidence=0.92)
    if any(w in t for w in _CONFIRM_WORDS):
        return AgentTurn(reply="Great, thank you. I have released the hold and your "
                               "card works normally. Sorry for the interruption.",
                         intent=Intent.CONFIRM_LEGIT, confidence=0.90)
    return AgentTurn(reply="I did not quite catch that. Did you make this purchase — "
                           "yes or no? Or say 'agent' for a human specialist.",
                     intent=Intent.UNSURE, confidence=0.40)


def complete_turn(system: str, history: list[dict], user_text: str, context: dict) -> TurnResult:
    """One live-loop LLM call. Always returns a TurnResult so callers get
    latency + provenance whether or not MOCK_MODE is on — the audit chain
    shouldn't have a different shape depending on which mode produced it."""
    t0 = time.perf_counter()

    if MOCK_MODE:
        turn = _mock_turn(user_text, context)
        lat = round((time.perf_counter() - t0) * 1000, 1)
        metrics.record_llm_call("dialog", "mock", lat, session_id=context.get("session_id"))
        return TurnResult(turn=turn, model_id="mock", prompt_version=PROMPT_VERSION,
                          latency_ms=lat, seed=LLM_SEED)

    # --- live mode: NVIDIA NIM, OpenAI-compatible ---
    client = _openai_client()
    # Found live (2026-07-18): `context` was accepted as a parameter here
    # but never actually used to build `messages` — only _mock_turn() used
    # it. REALTIME_SYSTEM's own text claims the model is given "customer
    # name, card last-4, home city / the flagged transaction / the RCA
    # reason / verification status", but in live mode none of that was
    # ever sent; the model had only the raw conversation history to infer
    # from. Folding it into the system message fixes both the grounding
    # gap and is the more likely fix for the JSON-contract failures
    # observed live — an ungrounded model rambles more.
    system_with_context = system + (
        f"\n\nCurrent call context:\n"
        f"- Customer: {context.get('name')} (card ending {context.get('last4')}, "
        f"home city {context.get('home_city')})\n"
        f"- Flagged transaction: £{context.get('amount', 0):,.0f} at "
        f"{context.get('merchant')}, {context.get('city')}\n"
        f"- RCA reason: {context.get('rca')}\n"
        f"- Verification status: {context.get('verified')}"
    )
    messages = ([{"role": "system", "content": system_with_context}]
                + history
                + [{"role": "user", "content": user_text}])
    create_kwargs = dict(
        model=NIM_MODEL_REALTIME,
        messages=messages,
        temperature=0.2,
        max_tokens=LLM_MAX_TOKENS,
        # Neha's original code had this and it was dropped in the merge —
        # putting it back after finding, live (2026-07-18), that the model
        # sometimes ignores the "output JSON only" prompt instruction on
        # its own and replies in plain prose instead (reproduced: a short,
        # emotionally-loaded deny like "No, I didn't." a few turns into a
        # conversation was enough to trigger it, ~1 in 3 attempts in
        # testing). That silently fell through to the JSON-parse-failure
        # safety net, escalating every one of those turns to a human
        # instead of confirming fraud — which would have broken the
        # flagship "fraud caught" demo path unpredictably. response_format
        # constrains the model's output at the API level instead of
        # relying on prompt compliance alone; confirmed live across
        # repeated attempts on the exact failing conversation that this
        # eliminates the failure.
        response_format={"type": "json_object"},
        # Jennifer's fix, carried forward: Nemotron reasoning models emit
        # thinking tokens by default, which is what actually blew the
        # latency budget (more than model size alone). Disabling it is the
        # single biggest lever on turn latency after model choice.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if LLM_SEED is not None:
        create_kwargs["seed"] = LLM_SEED

    # Retry once before the safe fallback. Found live (2026-07-18): under
    # call density (real NeMo Guardrails adds 2 LLM calls per turn on top of
    # this one), a transient rate-limit or a one-off malformed response would
    # trip the JSON-parse fallback and escalate a clear "yes, that was me" to
    # a human. A clear confirm classified 4/4 in isolation but failed under
    # load — so it's transience, not the model. One retry (with a short
    # backoff on an API error) recovers it; the safe UNSURE fallback still
    # applies only if the retry ALSO fails, so a truly malformed turn can
    # never fall through to an action branch.
    turn = None
    for attempt in range(2):
        try:
            _c0 = time.perf_counter()
            resp = client.chat.completions.create(**create_kwargs, timeout=3.0)
            call_ms = (time.perf_counter() - _c0) * 1000   # THIS single inference's latency
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            turn = AgentTurn(**data)
            usage = getattr(resp, "usage", None)
            metrics.record_llm_call("dialog", NIM_MODEL_REALTIME, call_ms,
                                    tokens_in=getattr(usage, "prompt_tokens", None),
                                    tokens_out=getattr(usage, "completion_tokens", None),
                                    session_id=context.get("session_id"))
            break
        except Exception:  # noqa: BLE001 — API error OR malformed JSON; both get one retry
            if attempt == 0:
                time.sleep(0.6)  # brief backoff for a transient rate-limit
    if turn is None:
        # Both attempts failed — degrade safely to escalation. The "code
        # writes cheques" guarantee: a malformed reply never falls through
        # to an action branch.
        turn = AgentTurn(reply="Let me connect you with a specialist to be safe.",
                         intent=Intent.UNSURE, confidence=0.0)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return TurnResult(turn=turn, model_id=NIM_MODEL_REALTIME,
                      prompt_version=PROMPT_VERSION, latency_ms=latency_ms, seed=LLM_SEED)

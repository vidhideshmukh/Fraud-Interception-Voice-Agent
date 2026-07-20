"""Loop detector for the live dialog agent.

Mentor feedback: "model refinement — no customer response repetition." Two
related failure modes hide behind that one sentence — the agent echoing the
customer's own words back verbatim, and the agent re-asking a question it
already asked. `prompts.py` now tells the model not to do either (rule 6),
but per this project's own "policy in code, not prompt" rule (see
orchestrator.py's module docstring), a prompt instruction is not a
guarantee — a model can still drift into a loop under a weird conversation
path. This module is the code-level backstop: cheap, deterministic,
independent of whether the model is behaving that turn.

Deliberately not using embeddings or an extra LLM call for this — that would
put a second model call back on the critical path, which is exactly what the
single-hop design forbids. `difflib.SequenceMatcher` is stdlib, O(n*m) on
short sentences, and good enough to catch "the agent is stuck," which is a
binary trigger, not a precision-graded judgment.
"""
from __future__ import annotations

from difflib import SequenceMatcher

SIMILARITY_THRESHOLD = 0.82  # near-duplicate, not just "on the same topic"
LOOKBACK_TURNS = 2  # matches the scenario catalog's "max 2 clarification loops"


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a.lower().strip(), b=b.lower().strip()).ratio()


def is_repetitive(candidate_reply: str, prior_agent_utterances: list[str],
                  lookback: int = LOOKBACK_TURNS,
                  threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """True if `candidate_reply` is a near-duplicate of any of the agent's
    last `lookback` utterances — i.e. the dialog isn't progressing."""
    for prior in prior_agent_utterances[-lookback:]:
        if _similarity(candidate_reply, prior) >= threshold:
            return True
    return False

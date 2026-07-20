"""Smoke tests: if these are green the demo cannot embarrass you.
Run: pytest tests/ -q

Extended from the original scaffold's 6 tests to cover everything added in
this merge: digit-code verification (renamed from phrase), the
channel-freeze policy (previously a documented but unenforced [gap]), the
repetition guard, guardrails, and action idempotency. Original tests are
kept, just updated for the renamed field/state (session.phrase ->
verification_code, AWAITING_PHRASE -> AWAITING_VERIFICATION).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MOCK_MODE"] = "true"
os.environ["AUDIT_PERSIST"] = "false"  # tests exercise the in-memory chain; no disk writes
os.environ["METRICS_PERSIST"] = "false"  # don't write metric files during tests
# Isolate tests from the real DB — write-through goes to a throwaway file so the
# suite never pollutes data/barclays.db.
import tempfile  # noqa: E402
os.environ["BARCLAYS_DB"] = os.path.join(tempfile.gettempdir(), "barclays_test.db")

import pytest  # noqa: E402
from app.core import orchestrator  # noqa: E402
from app.agents import resolution_agent  # noqa: E402
from app.security import anti_vishing, audit, repetition_guard
from app.database import bank
from app.core import fraud_trigger  # noqa: E402
from app.observability import metrics  # noqa: E402


@pytest.fixture(autouse=True)
def fresh():
    bank.seed()
    audit.reset()
    orchestrator.SESSIONS.clear()


def _wrong_code(correct: str) -> str:
    """A code guaranteed not to match `correct` — safer than a hardcoded
    literal, which has a (tiny but real) chance of colliding with a
    randomly generated code."""
    return "0" * len(correct) if correct != "0" * len(correct) else "9" * len(correct)


def _verified_session(txn_id):
    session = orchestrator.start_call(fraud_trigger.emit_event(txn_id))
    orchestrator.customer_says(session, session.verification_code)
    assert session.state.value == "verified"
    return session


# --- original scaffold coverage, updated for the rename -------------------

def test_wrong_code_stays_awaiting_verification():
    session = orchestrator.start_call(fraud_trigger.emit_event("txn_gadget_02"))
    orchestrator.customer_says(session, _wrong_code(session.verification_code))
    assert session.state.value == "awaiting_verification"
    assert session.verification_attempts == 1


def test_path_legit_releases_hold():
    session = _verified_session("txn_gadget_02")
    orchestrator.customer_says(session, "Yes I did make that purchase")
    assert session.state.value == "resolved_legit"
    assert bank.TRANSACTIONS["txn_gadget_02"].status == "released"


def test_path_fraud_blocks_card():
    session = _verified_session("txn_lagos_01")
    orchestrator.customer_says(session, "No, that was not me")
    assert session.state.value == "resolved_fraud"
    assert bank.TRANSACTIONS["txn_lagos_01"].status == "blocked"
    assert any(a["action"] == "block_card" for a in bank.CARD_ACTIONS)


def test_distress_escalates_to_human():
    session = _verified_session("txn_lagos_01")
    orchestrator.customer_says(session, "I'm scared, I want to talk to a person")
    assert session.state.value == "escalated"


def test_no_answer_keeps_hold():
    session = orchestrator.start_call(fraud_trigger.emit_event("txn_lagos_01"))
    orchestrator.no_answer(session)
    assert session.state.value == "no_answer"
    assert bank.TRANSACTIONS["txn_lagos_01"].status == "held"


def test_audit_chain_integrity():
    session = _verified_session("txn_gadget_02")
    orchestrator.customer_says(session, "yes that was me")
    assert audit.verify_chain()
    assert len(audit.get_log(session.session_id)) >= 6


# --- latency fast-path (Jennifer's idea, adapted) -------------------------
# The fast-path must be a latency optimisation ONLY: same decisions as the LLM
# path, just without the model call on exact "yes"/"no". These lock that in.

def _last_agent_model(session):
    recs = [r for r in audit.get_log(session.session_id) if r["kind"] == "agent"]
    return recs[-1]["payload"]["model_id"]


def test_fast_path_exact_yes_resolves_legit_without_llm():
    session = _verified_session("txn_lagos_01")
    orchestrator.customer_says(session, "Yes.")           # exact, punctuation stripped
    assert session.state.value == "resolved_legit"
    assert _last_agent_model(session) == "fastpath"       # no LLM call was made


def test_fast_path_exact_no_low_value_blocks_without_llm():
    session = _verified_session("txn_lagos_01")            # £680, below high-value threshold
    orchestrator.customer_says(session, "no")
    assert session.state.value == "resolved_fraud"
    assert bank.TRANSACTIONS["txn_lagos_01"].status == "blocked"
    assert _last_agent_model(session) == "fastpath"


def test_fast_path_high_value_deny_blocks_card_then_escalates():
    """A high-value deny blocks the card immediately (protective) AND escalates
    the refund decision to a human — the card freeze is safe to auto-do, the
    chargeback is not. Policy is unchanged whether the deny came from the
    fast-path or the LLM."""
    session = _verified_session("txn_gadget_02")           # £1299, above high-value threshold
    orchestrator.customer_says(session, "No")
    assert session.state.value == "escalated"
    assert _last_agent_model(session) == "fastpath"        # intent came from fast-path...
    # protective: the card IS blocked right away, even though we escalate
    assert any(a["action"] == "block_card" and a["session_id"] == session.session_id
               for a in bank.CARD_ACTIONS)
    # ...but the chargeback/refund is left to the human (no auto-chargeback)
    assert bank.TRANSACTIONS["txn_gadget_02"].status != "blocked"
    assert not any(a["action"] == "open_chargeback" and a["session_id"] == session.session_id
                   for a in bank.CARD_ACTIONS)


def test_fast_path_verbose_answer_falls_through_to_llm():
    session = _verified_session("txn_lagos_01")
    orchestrator.customer_says(session, "No, it was never me, I've not been there")
    assert session.state.value == "resolved_fraud"
    assert _last_agent_model(session) != "fastpath"        # nuanced -> full LLM path


def test_fast_path_can_be_disabled(monkeypatch):
    monkeypatch.setattr(orchestrator, "FAST_PATH_ENABLED", False)
    session = _verified_session("txn_lagos_01")
    orchestrator.customer_says(session, "yes")
    assert session.state.value == "resolved_legit"         # same outcome...
    assert _last_agent_model(session) != "fastpath"        # ...but via the LLM path


# --- new coverage: channel-freeze (previously an unenforced [gap]) --------

def test_three_wrong_codes_freezes_channel():
    session = orchestrator.start_call(fraud_trigger.emit_event("txn_lagos_01"))
    wrong = _wrong_code(session.verification_code)
    orchestrator.customer_says(session, wrong)
    orchestrator.customer_says(session, wrong)
    assert session.state.value == "awaiting_verification"  # still 2 strikes, not frozen yet
    orchestrator.customer_says(session, wrong)
    assert session.state.value == "channel_frozen"
    assert session.outcome == "auth_failed_frozen"
    fired = {a["payload"]["action"] for a in audit.get_log(session.session_id) if a["kind"] == "tool"}
    assert fired == {"freeze_channel", "human_callback_task"}


def test_correct_code_after_two_wrong_still_verifies():
    """Two strikes don't burn the customer — only three does."""
    session = orchestrator.start_call(fraud_trigger.emit_event("txn_lagos_01"))
    wrong = _wrong_code(session.verification_code)
    orchestrator.customer_says(session, wrong)
    orchestrator.customer_says(session, wrong)
    orchestrator.customer_says(session, session.verification_code)
    assert session.state.value == "verified"


# --- OTP verification (numeric code, matching Jennifer's latest auth) --------
# The point of these: a correct read-back must verify no matter HOW the digits
# were spoken/transcribed (raw digits, digit words, "oh", "double one"), and a
# wrong or stale code must fail. The "double one" case is the exact live bug
# that made us abandon codes before — it's covered here now.

def _set_code(session_id, code):
    anti_vishing._active[session_id] = {"code": code, "sent_at": time.time()}


def test_verify_code_exact_digits():
    _set_code("sess_exact", "680863")
    assert anti_vishing.verify_code("sess_exact", "680863") is True


def test_verify_code_spaced_and_punctuated():
    _set_code("sess_fmt", "680863")
    assert anti_vishing.verify_code("sess_fmt", "6 8 0 8 6 3.") is True


def test_verify_code_spoken_as_words_with_oh_for_zero():
    """ASR sometimes returns digit WORDS, and reads 0 as 'oh'."""
    _set_code("sess_words", "680863")
    assert anti_vishing.verify_code("sess_words", "six eight oh eight six three") is True


def test_verify_code_double_and_triple():
    """THE past bug: '11' spoken as 'double one', '777' as 'triple seven'.
    Both must normalise back to the repeated digits."""
    _set_code("sess_dbl", "114777")
    assert anti_vishing.verify_code("sess_dbl", "double one four triple seven") is True


def test_verify_code_ignores_leading_filler():
    _set_code("sess_filler", "680863")
    assert anti_vishing.verify_code("sess_filler", "okay the code is 680863") is True


def test_verify_code_rejects_wrong_code():
    _set_code("sess_wrong", "680863")
    assert anti_vishing.verify_code("sess_wrong", "111111") is False


def test_verify_code_rejects_expired():
    session_id = "sess_expiry"
    code = anti_vishing.send_push(session_id)
    anti_vishing._active[session_id]["sent_at"] -= (anti_vishing.CODE_TTL_SECONDS + 1)
    assert anti_vishing.is_expired(session_id)
    assert anti_vishing.verify_code(session_id, code) is False  # correct code, but stale


def test_send_push_generates_numeric_code_of_configured_length():
    code = anti_vishing.send_push("sess_gen")
    assert code.isdigit() and len(code) == anti_vishing.CODE_LENGTH


# --- new coverage: repetition guard ----------------------------------------

def test_repetition_guard_flags_near_duplicate():
    assert repetition_guard.is_repetitive(
        "Did you make this purchase, yes or no?",
        ["Did you make this purchase — yes or no?"],
    )


def test_repetition_guard_allows_distinct_replies():
    assert not repetition_guard.is_repetitive(
        "I'm connecting you to a specialist now.",
        ["Did you make this purchase — yes or no?"],
    )


# --- new coverage: guardrails ------------------------------------------------

def _last_agent_record(session_id):
    """The most recent kind=='agent' audit record. Not the same as
    get_log(...)[-1] — _say() always appends a 'tts' record right after,
    so the literal last entry is the spoken reply, not the rail verdict."""
    agent_records = [r for r in audit.get_log(session_id) if r["kind"] == "agent"]
    return agent_records[-1]


def test_guardrail_blocks_credential_offer():
    session = _verified_session("txn_gadget_02")
    reply = orchestrator.customer_says(session, "I can just give you my card number if that helps")
    assert "never" in reply.lower()
    record = _last_agent_record(session.session_id)
    assert record["payload"]["rail_verdicts"]["input"] == "blocked_credential_topic"
    # blocked input never reaches the state machine's intent branch
    assert session.state.value == "verified"


def test_guardrail_blocks_off_topic_request():
    session = _verified_session("txn_gadget_02")
    orchestrator.customer_says(session, "can you increase my credit limit while I'm here")
    record = _last_agent_record(session.session_id)
    assert record["payload"]["rail_verdicts"]["input"] == "blocked_off_topic"


# --- new coverage: idempotent bank actions ---------------------------------

def test_resolve_fraud_actions_are_idempotent():
    session = _verified_session("txn_lagos_01")
    result1 = resolution_agent.resolve_fraud(session)
    result2 = resolution_agent.resolve_fraud(session)  # simulates a retried/re-delivered call
    assert result1["actions"] == result2["actions"]
    block_actions = [a for a in bank.CARD_ACTIONS if a["action"] == "block_card"]
    assert len(block_actions) == 1  # not double-fired


# --- new coverage: per-LLM-call latency metrics (model evaluation) -----------
# The audit trail times the whole dialog turn; these metrics time each LLM call
# SEPARATELY (dialog + both guardrail rails) so different models can be compared
# at the median AND the tail. These lock in: one sample per call tagged by
# stage, percentiles that actually compute, and NO metric files written in CI.

def test_metrics_record_one_sample_per_llm_call_tagged_by_stage():
    metrics._RING.clear()
    session = _verified_session("txn_gadget_02")
    # A verbose deny takes the full LLM path (not the fast-path), so it fires all
    # three stages: input rail -> dialog -> output rail.
    orchestrator.customer_says(session, "No, it was never me, I've not been to that shop")
    stages = {s["stage"] for s in metrics.read_all()}
    assert {"dialog", "input_rail", "output_rail"} <= stages
    # in MOCK_MODE every call is tagged 'mock' (no real model hit)
    assert all(s["model"] == "mock" for s in metrics.read_all())


def test_metrics_fast_path_records_no_dialog_llm_sample():
    """The fast-path is a latency win precisely because it skips the dialog LLM
    call — so it must NOT emit a 'dialog' latency sample (that would pollute the
    model-eval numbers with a 0ms call the model never actually made)."""
    metrics._RING.clear()
    session = _verified_session("txn_lagos_01")
    orchestrator.customer_says(session, "yes")             # exact match -> fast-path
    assert session.state.value == "resolved_legit"
    assert not any(s["stage"] == "dialog" for s in metrics.read_all())


def test_metrics_summarize_computes_percentiles_per_model_stage():
    samples = [{"model": "m", "stage": "dialog", "latency_ms": v}
               for v in (100, 200, 300, 400, 1000)]
    (row,) = metrics.summarize(samples)
    assert row["model"] == "m" and row["stage"] == "dialog" and row["count"] == 5
    assert row["p50"] == 300 and row["max"] == 1000
    assert row["p95"] >= row["p50"] and row["p99"] >= row["p95"]  # tail >= median


def test_metrics_do_not_write_files_during_tests(tmp_path, monkeypatch):
    """METRICS_PERSIST=false is set at the top of this module — recording must
    stay in memory and touch no disk, so CI never leaves metric files behind.
    Point METRICS_DIR at a throwaway dir to prove nothing lands there even if the
    project's real logs/metrics/ already exists from a prior live run."""
    assert metrics.METRICS_PERSIST is False
    monkeypatch.setattr(metrics, "METRICS_DIR", tmp_path / "metrics")
    metrics._RING.clear()
    metrics.record_llm_call("dialog", "mock", 12.3)
    assert len(metrics._RING) == 1                       # recorded in memory...
    assert not (tmp_path / "metrics").exists()           # ...but no file written

"""NeMo Agent Toolkit (NAT) function registrations for the Banking Agent's
tool allowlist. `config/workflow.yaml` references these by name
(`nemo_agent_toolkit/block_card` etc.); `nat run --config_file config/workflow.yaml`
wires them into a **tool_calling_agent** that plans and executes the async
resolution flow off the critical path — the NVIDIA-native version of what
app/agents/resolution_agent.py does inline for the demo.

VERIFIED LIVE 2026-07-18 against nvidia-nat==1.8.0. What working end-to-end
(not just reading docs) actually took:

  1. Package rename `nat/` -> `nemo_agent_toolkit/` — `nvidia-nat`'s own top-level
     import name IS `nat` (a namespace package), and our regular package
     silently shadowed it whenever the project root was on sys.path, exactly
     how `nat run` loads things.

  2. `FunctionBaseConfig`'s `name=` class-kwarg is the SHORT name only —
     TypedBaseModel.__init_subclass__ builds `full_type` as
     `f"{package}/{name}"` itself, so `name="block_card"` yields
     `nemo_agent_toolkit/block_card`. Passing the full slash path doubled it.

  3. **Explicit Pydantic input schemas with Field descriptions (below),
     not bare `def _fn(session_id: str, txn_id: str)`.** With bare type
     hints, the agent's model had no description of what each argument was
     and its first tool call sent empty `{}` args, failing validation and
     burning retries. A described input schema is what makes the model
     fill `session_id`/`txn_id`/etc. correctly on the first attempt — the
     difference between a flaky multi-retry loop and a clean one-shot call.

  4. **tool_calling_agent, not react_agent** (see config/workflow.yaml's
     comment): our tools take multiple structured args, and react_agent's
     text-blob "Action Input:" format got mangled by the model. The
     tool_calling_agent uses the model's native structured tool-calling.

Every tool wraps the *same* bank functions resolution_agent.py calls
inline — one implementation of "what block_card actually does," two ways of
orchestrating it (deterministic inline for the demo's guaranteed path, NAT
tool_calling_agent for the NVIDIA-native orchestration story). No read-PAN
tool, no transfer tool, no limit-change tool exists here or anywhere in the
codebase — an agent (or a prompt-injected model) cannot call a tool that
was never registered.
"""
# NOTE: deliberately NO `from __future__ import annotations` here. NAT calls
# typing.get_type_hints() on the inner tool functions to derive their input
# type; with stringized (future) annotations, resolving the `inp:
# BlockCardInput` forward-ref inside NAT's evaluation context raised
# NameError at workflow-build time. Real annotation objects resolve
# immediately. Found live 2026-07-18.

from pydantic import BaseModel, Field

from app.security import audit
from app.database import bank

try:
    from nat.builder.builder import Builder
    from nat.builder.function_info import FunctionInfo
    from nat.cli.register_workflow import register_function
    from nat.data_models.function import FunctionBaseConfig
    _NAT_AVAILABLE = True
except ImportError:
    # `nvidia-nat` isn't installed in mock-mode / offline dev — the rest of
    # the app must not fail to import because of that, so this module
    # degrades to a no-op instead of raising at import time.
    _NAT_AVAILABLE = False


# --- Input schemas. These are what the agent's model reads to fill tool
#     arguments correctly on the first call; the descriptions are load-bearing,
#     not decorative (see point 3 in the module docstring). ---

class BlockCardInput(BaseModel):
    session_id: str = Field(description="The call session id this action belongs to, e.g. 'call_ab12'.")
    txn_id: str = Field(description="The flagged transaction id the customer denied, e.g. 'txn_lagos_01'.")
    customer_id: str = Field(description="The customer id whose card should be blocked, e.g. 'cust_priya'.")


class ReleaseHoldInput(BaseModel):
    session_id: str = Field(description="The call session id this action belongs to.")
    txn_id: str = Field(description="The flagged transaction id the customer confirmed as legitimate.")


class OpenChargebackInput(BaseModel):
    session_id: str = Field(description="The call session id this action belongs to.")
    txn_id: str = Field(description="The denied transaction id to open a chargeback against.")


class UpsertCaseInput(BaseModel):
    session_id: str = Field(description="The call session id this case belongs to.")
    outcome: str = Field(description="Final outcome, e.g. 'fraud_confirmed' or 'false_positive_recovered'.")
    note: str = Field(description="A short analyst note summarizing what was done and why.")


if _NAT_AVAILABLE:

    class BlockCardConfig(FunctionBaseConfig, name="block_card"):
        pass

    @register_function(config_type=BlockCardConfig)
    async def block_card_tool(config: BlockCardConfig, builder: Builder):
        async def _block_card(inp: BlockCardInput) -> str:
            action = bank.block_card(inp.session_id, inp.txn_id, inp.customer_id)
            audit.log_turn(inp.session_id, "tool", action)
            return f"Card blocked for customer {inp.customer_id}, txn {inp.txn_id}; reissue ordered."

        yield FunctionInfo.from_fn(
            _block_card, input_schema=BlockCardInput,
            description="Block a customer's card and order a reissue when they deny a transaction. "
                        "Idempotent per (session_id, txn_id).",
        )

    class ReleaseHoldConfig(FunctionBaseConfig, name="release_hold"):
        pass

    @register_function(config_type=ReleaseHoldConfig)
    async def release_hold_tool(config: ReleaseHoldConfig, builder: Builder):
        async def _release_hold(inp: ReleaseHoldInput) -> str:
            action = bank.release_hold(inp.session_id, inp.txn_id)
            audit.log_turn(inp.session_id, "tool", action)
            return f"Hold released on txn {inp.txn_id}."

        yield FunctionInfo.from_fn(
            _release_hold, input_schema=ReleaseHoldInput,
            description="Release the hold on a flagged transaction the customer confirmed as legitimate.",
        )

    class OpenChargebackConfig(FunctionBaseConfig, name="open_chargeback"):
        pass

    @register_function(config_type=OpenChargebackConfig)
    async def open_chargeback_tool(config: OpenChargebackConfig, builder: Builder):
        async def _open_chargeback(inp: OpenChargebackInput) -> str:
            action = bank.open_chargeback(inp.session_id, inp.txn_id)
            audit.log_turn(inp.session_id, "tool", action)
            return f"Chargeback opened for txn {inp.txn_id}."

        yield FunctionInfo.from_fn(
            _open_chargeback, input_schema=OpenChargebackInput,
            description="Open a chargeback on a transaction the customer denied making.",
        )

    class UpsertCaseConfig(FunctionBaseConfig, name="upsert_case"):
        pass

    @register_function(config_type=UpsertCaseConfig)
    async def upsert_case_tool(config: UpsertCaseConfig, builder: Builder):
        async def _upsert_case(inp: UpsertCaseInput) -> str:
            case = bank.upsert_case(inp.session_id, outcome=inp.outcome, note=inp.note)
            return f"Case {case['case_id']} updated: {inp.outcome}."

        yield FunctionInfo.from_fn(
            _upsert_case, input_schema=UpsertCaseInput,
            description="Create or update the case record for a call session with an outcome and analyst note.",
        )

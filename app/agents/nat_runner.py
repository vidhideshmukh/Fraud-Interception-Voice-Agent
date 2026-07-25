"""In-process runner for the NeMo Agent Toolkit (NAT) workflows.

This makes the NAT agents the ACTUAL engine — the same `tool_calling_agent`
(resolution) and `rewoo_agent` (investigation) you'd launch with `nat run`, but
invoked INSIDE the app instead of shelling out to the CLI. So the running
product genuinely executes the NAT agents; they are not a side demonstration.

How it works:
  - Each workflow config is BUILT ONCE and cached for the process. Building an
    agent + its LLM client is expensive, so we never rebuild per call.
  - NAT is fully async; the orchestrator is sync (it runs in a FastAPI thread /
    a WebRTC executor). So we run every NAT workflow on ONE dedicated background
    event-loop thread and block the caller for the result. That keeps a single
    long-lived builder context alive (required — the workflow cleans itself up
    the moment its `load_workflow` context exits).

Live mode only: the agents call real LLMs. In mock mode / if NAT isn't
installed, `run_workflow` raises so callers fall back to their deterministic
path — a fraud card must ALWAYS be blocked even if the agent's LLM is down.
"""
from __future__ import annotations

import asyncio
import threading

from app.observability.logging_setup import get_logger

log = get_logger("nat")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_stack = None                      # AsyncExitStack holding the live workflow contexts
_sessions: dict[str, object] = {}  # config_file -> SessionManager (built once)
_build_lock: asyncio.Lock | None = None
_start_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Start (once) a dedicated daemon event loop the NAT workflows run on."""
    global _loop, _thread
    with _start_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, name="nat-loop", daemon=True)
            _thread.start()
    return _loop


async def _get_session(config_file: str):
    """Build+cache the workflow SessionManager for a config, keeping its context
    open for the process lifetime via a shared AsyncExitStack."""
    global _stack, _build_lock
    if _build_lock is None:
        _build_lock = asyncio.Lock()
    async with _build_lock:
        if config_file not in _sessions:
            from contextlib import AsyncExitStack
            from nat.runtime.loader import load_workflow
            if _stack is None:
                _stack = AsyncExitStack()
            _sessions[config_file] = await _stack.enter_async_context(load_workflow(config_file))
            log.info("nat: built workflow session for %s", config_file)
    return _sessions[config_file]


async def _run(config_file: str, message: str) -> str:
    sm = await _get_session(config_file)
    async with sm.run(message) as runner:
        return await runner.result(to_type=str)


def run_workflow(config_file: str, message: str, *, timeout: float = 45.0) -> str:
    """Run a NAT workflow in-process and return its string result. Blocks the
    calling (sync) thread. Raises on any failure so callers can fall back."""
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(_run(config_file, message), loop)
    return fut.result(timeout=timeout)


def shutdown() -> None:
    """Close the cached workflow contexts + stop the loop (call on app shutdown)."""
    global _loop, _stack, _sessions
    if _loop is None:
        return

    async def _close():
        global _stack
        if _stack is not None:
            await _stack.aclose()
            _stack = None

    try:
        asyncio.run_coroutine_threadsafe(_close(), _loop).result(timeout=10)
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass
    _sessions = {}
    _loop.call_soon_threadsafe(_loop.stop)
    _loop = None

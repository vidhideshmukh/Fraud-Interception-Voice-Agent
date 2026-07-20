# Fraud-Interception Voice Agent

Team BEAMers · India Agentic AI Open Hackathon · Track A: Agentic Workflows

When the fraud model flags a suspicious transaction, an autonomous agent calls the
customer within seconds, authenticates them via a one-time app **code** (anti-vishing),
walks them through the flagged transaction, and resolves it live — release the hold,
block-and-reissue, or warm handoff to a human — before the money moves.

## What this is

This is a merge of three things the team built separately, done after a mentor review:

- The **original hackathon scaffold** — the state machine, mock bank, hash-chained
  audit log, and guardrails config that carry the three differentiators judges care
  about (anti-vishing handshake, compliance spine, single-hop latency budget).
- **Neha's** working Riva ASR/TTS integration (streaming, interim results, the
  `Magpie-Multilingual.EN-US.Aria` voice) and multi-agent framing
  (Fraud/Voice/Banking Agent, matching the pitch deck's architecture diagram).
- **Jennifer's** live-loop LLM config (`nemotron-3-nano-30b-a3b`,
  `enable_thinking: False`) — the actual fix for the latency problem the mentor
  flagged, which turned out to be Neha's 550B model on the critical path, not the
  verification step itself.

If you're picking this up cold: every non-obvious design decision has a comment at
its call site explaining *why*, not just what — start with `app/orchestrator.py`'s
module docstring, then `app/adapters/llm.py`'s for the mentor-directed merge
specifically.

## NVIDIA stack status — real vs. designed-in (honest inventory)

Every technology named in the pitch deck, and whether it's *actually running*
(installed, wired, verified live 2026-07-18) or *designed-in*:

| PPT tech | Status | Where / how verified |
|---|---|---|
| **Nemotron NIMs** (Nano live loop, Super async) | ✅ **Live** | `app/adapters/llm.py`; real completions verified, JSON contract enforced |
| **Riva** streaming ASR + TTS (Aria voice) | ✅ **Live** | `app/adapters/asr.py`, `tts.py`; verified with a real spoken phone call end-to-end |
| **NeMo Guardrails** | ✅ **Live (real `nemoguardrails` LLMRails)** | `app/adapters/guardrails.py`; real input+output rails, verified blocking jailbreaks/credential-asks and correctly passing normal fraud dialog. ~1-2s/check (real Colang runtime) — fast keyword pre-filter catches common attacks at 0ms |
| **NeMo Agent Toolkit (NAT / AI-Q)** | ✅ **Live (`nat run` verified)** | `nat_plugin/functions.py` + `config/workflow.yaml`; `tool_calling_agent` drives all 4 banking tools via real Nemotron Super, verified executing a full fraud remediation (`nat validate` passes, `nat run` mutates real state) |
| **NeMo Data Designer** | ✅ **Live (`data-designer` pkg, verified generating)** | `data/generate_datasets.py --engine datadesigner`; real sampler columns + LLM-generated RCA reasons via Nemotron on NVIDIA Build, verified generating and round-tripping through the live loader. Uses the standalone client-side package (not the `nemo-microservices` platform SDK, which would need a GPU deployment) |
| **NemoClaw** | ⚙️ **Designed-in (by NVIDIA's constraint)** | Alpha, Ubuntu/GPU-only — cannot run on the demo machine. The hash-chained audit log + privacy-router seam mirror its interfaces; presented as production hardening, per the team's own field manual |

So the only part of the stack that is *not* actually running is NemoClaw, and
that's because NVIDIA ships it Ubuntu/GPU-only alpha — not a shortcut on our side.
(NeMo Data Designer *looked* like the same kind of constraint — the
`nemo-microservices` SDK needs a deployed platform — but the standalone
`data-designer` package runs client-side against NVIDIA Build, so it's real.)

Generate Dataset 1 with the real Data Designer:
```bash
python data/generate_datasets.py --dataset1 --engine datadesigner --dd-records 200
# (--engine python, the default, is the offline statistical backend — richer
#  per-profile distributions + full 21-scenario coverage, no API key needed)
```

**Two agent paths, on purpose (not redundancy):** the live orchestrator resolves
money actions *deterministically* (`app/agents/resolution_agent.py` — exact known
IDs, no LLM guessing what to block), because that's the more robust choice for
irreversible actions. NAT's `tool_calling_agent` is the NVIDIA-native *agentic*
orchestration of the same allow-listed tools, verified runnable via `nat run`. NAT
is used where an agent adds value (planning/observability/profiling), not forced
onto the one path where determinism is safer.

To run the NAT agent yourself:
```bash
pip install -e .                                   # registers nat_plugin/ tools as NAT entry-points
nat validate --config_file config/workflow.yaml    # schema check, no API key needed
nat run --config_file config/workflow.yaml --input "Customer cust_priya denied txn_lagos_01 in session s1. Confirmed fraud. Block the card, open a chargeback, update the case."
```

## What changed in this merge (if you know the original scaffold)

- **Word-phrase → 6-digit code.** The phrase (`amber-lotus-42`) was the actual
  latency risk — ASR mishears on multi-word phrases forced retry loops. A digit
  code is what ASR is most reliable on; the verification check itself was always
  an O(1) string compare, never the bottleneck. See `app/services/app_push.py`.
- **3 wrong codes now actually freezes the channel.** This was a documented rule
  ("possible attacker — never soften auth") that the original state machine never
  enforced. See `CallState.CHANNEL_FROZEN` and `orchestrator._freeze_channel()`.
- **A repetition guard.** Catches the agent echoing itself or re-asking an answered
  question, forces escalation instead of a third rephrase. See
  `app/services/repetition_guard.py`.
- **Guardrails are actually wired in now**, not just a config file nobody called.
  `app/adapters/guardrails.py` runs input/output self-checks (credential requests,
  jailbreaks, off-topic, financial advice) on every turn, in both mock and live mode.
- **Idempotent bank actions.** `block_card`/`release_hold`/`open_chargeback` are keyed
  by `(session_id, txn_id, action)` — a retried call can't double-fire. This was a
  stated NFR the original code didn't actually implement.
- **A unified audit/eval schema.** `audit.log_agent_turn()` stamps `model_id`,
  `prompt_version`, `seed`, and `rail_verdicts` on every turn — the same shape
  Jyotika's eval-harness plan calls for, so the live audit panel *is* the eval
  record, not a second logging path to keep in sync.
- **A live transaction-stream demo surface**, so the flagship case can fire on its
  own instead of a manual button click. See `app/services/ticker.py` and
  `demo/barclays_app/index.html`.
- **The Barclays app is now an actual page**, not just a backend function returning
  a string. Phone-styled, shows the push notification + live call transcript, next
  to the transaction ticker and the audit panel — one browser window for the whole
  demo.
- **Synthetic data generation** (`data/generate_datasets.py`) — Dataset 1 (customer
  profiles + ~500 transactions + fraud injects) is pure statistical generation, no
  API needed. Dataset 3 (scripted regression calls) is hand-authored against the
  real state machine. Dataset 2 (labelled utterances) needs an LLM and refuses to
  run against a Nemotron model unless explicitly overridden — see that script's
  module docstring for why.

## Quick start (mock mode — works anywhere, now, zero API keys)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python demo/run_demo.py            # runs both scripted demo paths end to end
pytest tests/ -q                   # state machine + audit + guardrails + idempotency
uvicorn app.main:app --reload      # opens the Barclays app dashboard at /app/
```

> **Note on this repo's current state:** everything here was written and
> statically reviewed (cross-checked every call site against every function
> signature, checked for import cycles, re-read the dense files end to end) but
> **not executed** — there's no Python interpreter in the environment this was
> built in. Run the two commands above first thing and fix whatever surfaces;
> treat this as a thorough first draft, not a verified-green build.

Open `http://localhost:8000/` — it redirects to the Barclays app dashboard. Click
**"Start live transaction stream"** to watch the ticker replay and auto-trigger a
call, or use the manual trigger buttons (the documented stage fallback if the ticker
misbehaves — flow is identical either way).

## Flip to live mode at the event

1. Get an API key at https://build.nvidia.com (free) — get four, one per teammate,
   round-robin in `.env` if you hit rate limits mid-demo.
2. `cp .env.example .env`, set `NVIDIA_API_KEY=nvapi-...`, `MOCK_MODE=false`, and the
   Riva `RIVA_ASR_FUNCTION_ID` / `RIVA_TTS_FUNCTION_ID` mentors give you.
3. `pip install openai nvidia-riva-client numpy sounddevice requests` (see the
   commented block in `requirements.txt`).
4. Run the API server (`uvicorn app.main:app --reload`), then in another terminal:
   `MOCK_MODE=false python demo/voice_loop.py txn_lagos_01` — this is the actual
   live phone: real mic in, real Riva/Nemotron round-trip, real speaker out. The
   Barclays dashboard and audit panel update live in the browser while it runs,
   because the voice loop talks to the same server over HTTP (see
   `app/main.py`'s module docstring for why it's a separate process, not an import).
5. Guardrails: works today via `app/adapters/guardrails.py` without any extra
   install. `pip install nemoguardrails` only if you want the full Colang engine —
   see that module's docstring for the tradeoff.
6. NeMo Agent Toolkit: `pip install nvidia-nat`, then
   `nat run --config_file config/workflow.yaml --input "..."` — the Banking/Resolution
   Agent's tools are registered in `nat/functions.py`. **Do not** point the live
   voice loop at NAT without benchmarking first — see `config/workflow.yaml`'s
   comment on why the live loop stays on the custom orchestrator until that's measured.

## Layout

```
config/
  workflow.yaml            # NeMo Agent Toolkit config (nat run) — async Banking Agent
  guardrails/               # NeMo Guardrails config.yml + Colang rails
data/
  generate_datasets.py      # Dataset 1 (transactions), 2 (utterances), 3 (scripted calls)
  generated/                 # generator output lands here
nat/
  functions.py               # NAT tool registrations for the Banking Agent's tool allowlist
app/
  main.py                    # FastAPI: trigger, live ticker, converse, session, audit
  orchestrator.py            # call session state machine (the critical path — start here)
  models.py                  # pydantic entities, incl. the Dataset-1-compatible schema
  agents/
    prompts.py                # system prompts — tune these, they are the product
    dialog_agent.py            # real-time intent+dialog — the "Fraud Agent"
    resolution_agent.py        # async block/reissue/chargeback — the "Banking Agent"
  adapters/
    llm.py                     # NIM client (Nemotron Nano, thinking disabled); mock fallback
    asr.py, tts.py              # Riva streaming ASR/TTS — the "Voice Agent"; mock fallback
    guardrails.py                # input/output self-check + topical rails; mock fallback
  services/
    bank_mock.py                # customers, transactions, cards, cases — idempotent actions
    fraud_trigger.py             # fraud event scoring + RCA reason
    ticker.py                     # drives the live transaction-stream demo surface
    app_push.py                   # one-time digit-code handshake (anti-vishing)
    audit.py                      # hash-chained audit log, unified with the eval schema
    repetition_guard.py            # loop detector — the code-level backstop on the prompt rule
demo/
  run_demo.py                  # CLI driver: both paths, prints audit chain (offline)
  voice_loop.py                  # the actual live phone — talks to the API server over HTTP
  demo_script.md                  # word-for-word talk track for the 5-min demo
  barclays_app/index.html         # phone UI + transaction ticker + audit panel, one page
tests/
  test_flow.py                  # green = safe to keep hacking
```

## Design rules (do not break these during the event)

- **Single-hop live loop.** One LLM call per customer turn. RCA and resolution run
  async — if you add agent hops to the live loop the customer hears dead air. This
  is why only `dialog_agent.py` (the "Fraud Agent") touches an LLM on the critical
  path; `asr.py`/`tts.py` (the "Voice Agent") are I/O only.
- **The agent never asks for credentials.** Auth = customer reads back the code the
  bank pushed to their app. That is the anti-vishing differentiator — see
  `app/services/app_push.py`'s docstring before "fixing" this again.
- **Every turn goes through `audit.log_turn()` / `log_agent_turn()`.** The audit
  panel is a first-class demo artefact and the eval record, not a debug log.
- **Uncertain → human.** Low confidence, high value, repetition, or a blocked
  guardrail always escalates — see `orchestrator._escalation_reason()`, the single
  place this policy lives.
- **Policy is code, not prompt.** The LLM proposes an intent; thresholds in
  `orchestrator.py` decide what happens. A jailbroken model cannot release a
  high-value transaction — it never has a tool that could.

## Production readiness — done vs. remaining (honest)

This is a hackathon build, not a certified banking system. What's been made
production-*shaped*, and what a real rollout still needs:

**Done (production-shaped):**
- **Real database.** Customers + transactions + card actions + cases + incidents
  live in SQLite (`data/barclays.db`); build/reset with `python data/build_db.py`.
  Runtime mutations write-through, so with `RESET_ON_BOOT=false` state survives
  restarts. `RESET_ON_BOOT=true` (default) gives repeatable demos.
- **Durable, partitioned logs.** App log rotates daily (`logs/app/`); the
  tamper-evident audit trail is appended to date-partitioned files
  (`logs/audit/audit-YYYY-MM-DD.jsonl`) — survives restarts, exportable for
  regulators. Secrets are never logged.
- **Health probes.** `/health` (liveness) + `/ready` (readiness — DB reachable,
  customers loaded) for load-balancer / k8s gating.
- **Human-in-the-loop.** Escalations raise durable incidents an analyst resolves
  (`/incidents`, ops console) — the review queue is real, not cosmetic.
- **Idempotent bank actions**, config/secrets via env (`.env`, git-ignored),
  graceful degradation (a failed TTS/SMS/DB-write never breaks a call).

**Still needed for true production (not built here):**
- **Data store:** SQLite → managed **Postgres** with replication/backups; the
  audit trail → an append-only/immutable ledger or SIEM (the code notes NemoClaw
  as the target). SQLite is fine for a single-instance demo, not for HA.
- **AuthN/AuthZ** on every endpoint (the ops console + `/incidents/resolve` are
  currently open); secrets in a **vault** (not `.env`); mTLS between services.
- **Verification:** evolve the spoken code → in-app **tap-to-approve** (SCA /
  PSD2, device-bound); SMS/WhatsApp need DLT registration + WhatsApp Business API.
- **Ops:** metrics/tracing/alerting, rate-limiting, autoscaling behind a load
  balancer, CI/CD, pen-testing, DR, data-residency + retention policies.

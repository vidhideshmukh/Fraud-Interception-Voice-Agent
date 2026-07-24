# Setup & Run — Fraud-Interception Voice Agent

From a fresh clone to a live voice call with both UIs. Two modes:

- **Mock mode** (default) — the whole flow runs offline, **zero API keys**. Best for setup, tests, and the UI walkthrough.
- **Live mode** — real NVIDIA Nemotron + Riva speech. Needs API keys and a mic/speaker.

---

## 1. Prerequisites

- **Python 3.13** (`python --version`)
- **git**
- For **live voice only**: a working **microphone + speaker**, and an NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) (free).

---

## 2. Clone

```bash
git clone <your-repo-url>
cd fraud-voice-agent
```

## 3. Set up the environment

```bash
# create + activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# install dependencies
pip install -r requirements.txt

# create your config from the template (runs in MOCK mode out of the box)
cp .env.example .env              # Windows: copy .env.example .env
```

> The database (`data/barclays.db`) is committed and ready. If you ever want to
> rebuild it from the dataset: `python data/build_db.py`.

## 4. Verify it works

```bash
pytest tests/ -q                  # expect: 30 passed
python demo/run_demo.py           # scripted end-to-end walkthrough, no mic needed
```

---

## 5. Run — UI walkthrough (mock mode, no mic)

**Terminal 1 — start the server:**

```bash
uvicorn app.api.server:app --reload
```

**Open the two UIs in your browser:**

| UI | URL | What it is |
|----|-----|------------|
| Customer app | http://localhost:8000/app | The customer's Barclays app — receives the call + verification code |
| Ops console | http://localhost:8000/ops | Operator dashboard — live calls, incidents, audit chain, latency metrics |

**Trigger a fraud call:** in the **customer app**, insert a suspicious transaction
(high-value / foreign). The fraud model flags it → the phone **rings** in the app,
and a verification code is pushed on screen. Watch the **ops console** update live.

---

## 6. Run — LIVE voice call (the real demo)

This is the part a presenter actually talks to.

**a. Put live credentials in `.env`:**

```bash
MOCK_MODE=false
NVIDIA_API_KEY=nvapi-...           # from build.nvidia.com
RIVA_ASR_FUNCTION_ID=...           # Riva ASR / TTS function IDs (from the event)
RIVA_TTS_FUNCTION_ID=...
```

**b. Install the live speech deps** (mic/TTS + guardrails engine):

```bash
pip install nvidia-riva-client sounddevice numpy openai nemoguardrails
```

**c. Start the server** (Terminal 1):

```bash
uvicorn app.api.server:app --reload
```

**d. Be the phone** (Terminal 2) — this captures your mic and speaks the agent's replies:

```bash
python demo/voice_loop.py --latest
```

**e. The call flow:**
1. Trigger a fraud transaction from the **customer app** (http://localhost:8000/app).
2. The app **rings** — click **Answer**.
3. Speak to the agent: **read the verification code** shown in the app, then answer
   its question (**"yes"** = I made it / **"no"** = fraud).
4. Watch the **ops console** (http://localhost:8000/ops) log the whole call — decision,
   card action, audit chain, and per-LLM-call latency — in real time.

**Voice-loop options:**

| Command | Use |
|---------|-----|
| `python demo/voice_loop.py --latest` | Answer calls one after another, keep the handset open (recommended) |
| `python demo/voice_loop.py --queue` | Operator mode — auto-answer a FIFO queue of multiple simultaneous frauds |
| `python demo/voice_loop.py txn_lagos_01` | Trigger + handle one specific transaction |
| `python demo/voice_loop.py --session <id>` | Attach to one existing call (the ops console's "Copy" button gives the id) |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `dataset not found` on boot | Rebuild: `python data/build_db.py` (needs `data/generated/fraud_signals_dataset.json`, which is committed) |
| Port 8000 in use | `uvicorn app.api.server:app --reload --port 8001` (and set `API_BASE` in `.env` to match) |
| Voice loop silent / no audio | You're in mock mode — set `MOCK_MODE=false` and install the live speech deps (step 6b) |
| `barclays.db` shows as modified in git | Expected — it rebuilds on boot. Silence it: `git update-index --skip-worktree data/barclays.db` |

## Security note

`.env` holds secrets (API keys, phone numbers) and is **git-ignored** — never commit it.
The ops console and its endpoints are **unauthenticated** in this build; run it on
localhost/trusted networks only, not exposed to the internet.

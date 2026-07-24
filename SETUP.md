# Copy to .env and fill in at the event. Never commit the real .env —
# see the memory note on Neha's original repo for exactly why this matters:
# three live nvapi- keys got committed there and had to be rotated.
MOCK_MODE=false
RESET_ON_BOOT=true

# --- NVIDIA NIM (build.nvidia.com -> free API key) ---
# ============================================================
# ACTIVE PROFILE: self-hosted cluster NIM on gpu009
# (OpenAI-compatible; no API key validation on this endpoint)
# Model: NVIDIA Nemotron 3 Nano 30B-A3B served as nvidia/nemotron-3-nano
# Smoke test:
#   curl http://gpu009:8000/v1/chat/completions -H "Content-Type: application/json" \
#     -d '{"model":"nvidia/nemotron-3-nano","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
# ============================================================
NIM_BASE_URL=http://gpu009:8000/v1
NIM_MODEL_REALTIME=nvidia/nemotron-3-nano
NIM_MODEL_GUARDRAILS=nvidia/nemotron-3-nano

# Placeholder only — cluster ignores it, but app/speech/asr.py + tts.py raise
# KeyError if unset. Hosted Riva (grpc.nvcf.nvidia.com) will NOT work with this
# placeholder: put a real (rotated!) nvapi- key here before any voice testing.
NVIDIA_API_KEY=not-needed-on-cluster

# --- hosted-NIM profile (comment cluster block above, uncomment these) ---
# NIM_BASE_URL=https://integrate.api.nvidia.com/v1
# NIM_MODEL_REALTIME=nvidia/nemotron-3-nano-30b-a3b
# NIM_MODEL_GUARDRAILS=nvidia/nemotron-3-nano-30b-a3b
# NVIDIA_API_KEY=nvapi-YOUR-ROTATED-KEY   # old key was committed: ROTATE IT

# Four keys across the team, not one — rate limits are per key. Round-robin
# in your own .env if you hit limits mid-demo; MOCK_MODE=true is the
# worst-case fallback and is visually indistinguishable in a 5-min demo.

# Live loop: Nano-class, thinking disabled (see adapters/llm.py) — this is
# the mentor's latency fix. Async resolution agent: Super-class, no latency
# constraint (30s budget). Guardrails self-check: Nano is enough, it's a
# 5-token yes/no judgment.
LLM_MAX_TOKENS_REALTIME=150
# Optional — only set this for reproducible eval replays (Dataset 3), not
# for a real customer call. See adapters/llm.py's docstring.
# LLM_SEED=42

# --- Riva speech (mentors will give the server URI / NGC function IDs) ---
RIVA_SERVER_URI=grpc.nvcf.nvidia.com:443
RIVA_ASR_FUNCTION_ID=71203149-d3b7-4460-8231-1be2543a1fca
RIVA_TTS_FUNCTION_ID=877104f7-e885-42b9-8de8-f6e4c6303969
RIVA_TTS_VOICE=Magpie-Multilingual.EN-US.Aria
RIVA_SAMPLE_RATE=16000
RIVA_TTS_SAMPLE_RATE=22050
ASR_SILENCE_TIMEOUT_S=2.0

# --- Anti-vishing handshake ---
VERIFICATION_CODE_LENGTH=6
VERIFICATION_CODE_TTL_SECONDS=600

# --- Escalation / risk thresholds (policy in code, these just parameterize it) ---
CONFIDENCE_ESCALATION_THRESHOLD=0.7
HIGH_VALUE_ESCALATION_INR=100000
MAX_VERIFICATION_ATTEMPTS=3
FRAUD_RISK_THRESHOLD=0.85

# --- demo/voice_loop.py (talks to the FastAPI server over HTTP) ---
API_BASE=http://localhost:8000



"""Riva TTS adapter.

Merge note (the mentor's exact ask — "Neha's voice, Jennifer's model"):
Neha's TTS call named an explicit voice — `Magpie-Multilingual.EN-US.Aria` —
which is why her demo sounded better than Jennifer's (whose call let the
server pick a default voice). That voice selection is kept as-is below.
What's *not* kept is Neha's playback: she calls `synthesize()` (blocking,
waits for the whole clip) then `sd.play()` the full buffer. That means
nothing is heard until synthesis finishes — the opposite of what a <1s
turn budget needs. Riva also exposes `synthesize_online()`, a streaming
call that yields audio chunks as they're generated; this adapter plays each
chunk the moment it arrives via a persistent `sounddevice.OutputStream`, so
first-audio latency is bounded by the first chunk, not the whole utterance.
"""
from __future__ import annotations

import os
import time

from app.observability.logging_setup import get_logger

_log = get_logger("tts")

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
# Escape hatch for when the hosted Riva/NVCF TTS function is DEGRADED on
# NVIDIA's side (a real outage hit 2026-07-18). Set RIVA_TTS_DISABLED=true and
# speak() returns immediately without touching Riva — no wasted retry/backoff
# on a dead function — so the handset stays snappy while the browser's Web
# Speech preview carries the agent voice instead. Flip it back off the moment
# the function recovers to get the premium Riva Aria voice back.
RIVA_TTS_DISABLED = os.getenv("RIVA_TTS_DISABLED", "false").lower() == "true"
RIVA_SERVER_URI = os.getenv("RIVA_SERVER_URI", "grpc.nvcf.nvidia.com:443")
RIVA_TTS_FUNCTION_ID = os.getenv("RIVA_TTS_FUNCTION_ID", "")
# Dedicated TTS auth key (Riva/NVCF). Falls back to NVIDIA_API_KEY if unset.
RIVA_TTS_API_KEY = os.getenv("RIVA_TTS_API_KEY") or os.getenv("NVIDIA_API_KEY", "")
RIVA_TTS_VOICE = os.getenv("RIVA_TTS_VOICE", "Magpie-Multilingual.EN-US.Aria")
TTS_SAMPLE_RATE = int(os.getenv("RIVA_TTS_SAMPLE_RATE", "22050"))

_auth = None
_tts_service = None


def _service():
    """Lazy singleton — one gRPC auth/service handle reused across turns
    instead of reconnecting on every `speak()` call (that reconnect cost is
    exactly the kind of thing that quietly blows a sub-second turn budget)."""
    global _auth, _tts_service
    if _tts_service is None:
        import riva.client  # deferred import: not needed in mock mode
        _auth = riva.client.Auth(
            uri=RIVA_SERVER_URI, use_ssl=True,
            metadata_args=[["authorization", f"Bearer {RIVA_TTS_API_KEY}"],
                           ["function-id", RIVA_TTS_FUNCTION_ID]],
        )
        _tts_service = riva.client.SpeechSynthesisService(_auth)
    return _tts_service


def _reset_service() -> None:
    """Drop the cached gRPC auth/service handles so the next speak() rebuilds
    them from scratch. Used after a synthesis failure — if the channel went
    bad, a stale handle would just keep failing; a fresh connect can recover."""
    global _auth, _tts_service
    _auth = None
    _tts_service = None


def speak(text: str) -> str:
    """Mock mode: returns the text unchanged (demo driver prints it as the
    'agent voice', tests assert on it). Live mode: streams synthesis and
    plays each chunk as it arrives; still returns `text` either way so
    callers (orchestrator, audit log) always get a string back.

    Never raises. A voice failure — most importantly the hosted Riva/NVCF
    function being DEGRADED or rate-limited on NVIDIA's side (which is out of
    our control) — must NOT crash a live call. Found live 2026-07-18: a
    'DEGRADED function cannot be invoked' gRPC error while synthesizing the
    greeting killed the whole handset with a traceback. Now we retry a couple
    of times (a degraded NVCF function often recovers, or a retry routes to a
    healthy instance) and, if it still fails, degrade gracefully to on-screen
    text so the demo continues. The returned string is unchanged either way."""
    if MOCK_MODE or RIVA_TTS_DISABLED:
        return text

    import numpy as np
    import sounddevice as sd  # deferred import: only needed live

    last_err = None
    for attempt in range(3):
        try:
            responses = _service().synthesize_online(
                text=text, voice_name=RIVA_TTS_VOICE,
                language_code="en-US", sample_rate_hz=TTS_SAMPLE_RATE,
            )
            with sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16") as stream:
                for resp in responses:
                    if resp.audio:
                        stream.write(np.frombuffer(resp.audio, dtype=np.int16))
            return text
        except Exception as e:  # noqa: BLE001 — any synth/playback failure is non-fatal
            last_err = e
            _reset_service()  # drop the gRPC handle so the next try reconnects cleanly
            if attempt < 2:
                time.sleep(0.8)  # brief backoff; a DEGRADED function may recover

    print(f"\n[tts] voice unavailable — {type(last_err).__name__}: {last_err}\n"
          f"[tts] continuing WITHOUT audio (this is a hosted NVIDIA function issue, not the app).\n"
          f'[tts] AGENT SAID: "{text}"\n')
    return text


def synthesize_pcm(text: str) -> tuple[bytes, int]:
    """Return raw PCM16 mono audio bytes + its sample rate for `text`, WITHOUT
    playing it locally — for the WebRTC media path, which sends the audio over a
    peer connection to the browser instead of a local speaker. Returns empty bytes
    in mock mode / if the hosted function is degraded, so the caller degrades to
    on-screen text (same policy as speak()). Never raises."""
    if MOCK_MODE or RIVA_TTS_DISABLED:
        return b"", TTS_SAMPLE_RATE
    # Retry across a couple of attempts: the hosted NVCF TTS function scales to
    # zero and the FIRST call can fail with DEADLINE_EXCEEDED / "failed to
    # establish link to worker" while a GPU worker cold-starts. A retry often
    # catches the now-warm worker. (Startup also warms it — see server.py.)
    last = None
    for attempt in range(3):
        try:
            buf = bytearray()
            for resp in _service().synthesize_online(
                    text=text, voice_name=RIVA_TTS_VOICE,
                    language_code="en-US", sample_rate_hz=TTS_SAMPLE_RATE):
                if resp.audio:
                    buf.extend(resp.audio)
            if buf:
                _log.info("tts.synthesize_pcm: %d bytes @ %dHz (attempt %d) for %r",
                          len(buf), TTS_SAMPLE_RATE, attempt + 1, text[:40])
                return bytes(buf), TTS_SAMPLE_RATE
            last = "Riva returned 0 audio bytes"
        except Exception as e:  # noqa: BLE001 — synth failure must not crash the call
            last = f"{type(e).__name__}: {str(e)[:160]}"
            _reset_service()  # drop the channel so the next attempt reconnects
        if attempt < 2:
            time.sleep(1.2)   # give the NVCF worker a moment to come up
    _log.warning("tts.synthesize_pcm: no audio after 3 attempts (%s) server=%s function=%s voice=%s",
                 last, RIVA_SERVER_URI, RIVA_TTS_FUNCTION_ID, RIVA_TTS_VOICE)
    return b"", TTS_SAMPLE_RATE


def speak_text_only(text: str) -> str:
    """Same signature as speak(), but NEVER touches real Riva synthesis or
    local audio hardware, regardless of MOCK_MODE.

    Why this exists (found live, 2026-07-18): app/orchestrator.py runs
    inside the FastAPI server, and demo/voice_loop.py talks to that server
    over HTTP rather than importing the orchestrator directly (see
    main.py's docstring — it mirrors "SIP trunk -> app server, Riva is
    just the media leg"). That means the server itself is not the phone.
    When orchestrator._say() called the real speak() in live mode, every
    utterance got synthesized and played through the SERVER's speaker —
    then voice_loop.py synthesized and played the SAME text again through
    the CLIENT's speaker on receiving the HTTP reply. On one laptop that's
    an audible double-speak; it also blocked every /trigger and /converse
    HTTP response for the full spoken duration of the sentence (measured:
    an 18s response for one opening line), since the server was sitting
    there talking to an empty room before it even answered the request.

    The server should only ever deal in text. Real synthesis + playback is
    exclusively demo/voice_loop.py's job, because it's the one process
    that actually owns a microphone and a speaker."""
    return text

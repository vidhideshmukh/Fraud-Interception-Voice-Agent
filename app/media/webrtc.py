"""WebRTC media leg — makes the BROWSER the phone.

This is the drop-in that replaces demo/voice_loop.py's role: instead of a
separate process owning a local mic/speaker via sounddevice, the customer's
browser captures the mic and plays the agent's voice over a WebRTC peer
connection, and THIS module is the server-side peer.

Crucially, nothing in the brain changes. The pipeline is still:
    inbound audio -> Riva ASR -> orchestrator.customer_says -> Riva TTS -> outbound audio
The orchestrator, fraud logic, guardrails, agents and audit are untouched — the
server already speaks only text; WebRTC just changes how the audio arrives.

Design notes:
  - aiortc gives/takes 48 kHz audio frames; Riva ASR wants 16 kHz and Riva TTS
    emits 22.05 kHz, so PyAV (av.AudioResampler) bridges the sample rates.
  - Endpointing is a simple energy VAD on the 16 kHz stream: once speech starts,
    a run of silence (ASR_SILENCE_TIMEOUT_S) marks the end of the utterance —
    the same "listen until they stop" behaviour MicStreamASR had, over frames.
  - Riva ASR/TTS calls are blocking gRPC, so they run in a thread executor to
    keep the aiortc event loop responsive.
  - STUN/TURN is optional and env-driven (see _ice_config): set TURN_URLS/
    TURN_USERNAME/TURN_CREDENTIAL to traverse NAT/firewall (e.g. laptop browser ->
    cluster server). With those unset it stays localhost/LAN only, as before.

Full voice needs a live Riva (MOCK_MODE=false + RIVA_* env). In mock mode the
transport still establishes and audio frames flow (verified by a loopback
client), but there is no real ASR/TTS content.
"""
from __future__ import annotations

import asyncio
import fractions
import os
import time

from app.observability.logging_setup import get_logger

log = get_logger("webrtc")

ASR_RATE = 16000
TTS_OUT_RATE = 48000
FRAME_MS = 0.02                                   # 20 ms frames
SILENCE_TIMEOUT_S = float(os.getenv("ASR_SILENCE_TIMEOUT_S", "1.2"))
VAD_RMS_THRESHOLD = float(os.getenv("WEBRTC_VAD_RMS", "500"))   # int16 RMS above this = speech

_pcs: set = set()                                 # keep peer connections alive (aiortc GCs unref'd ones)


def _import_rtc():
    """Deferred imports so the app loads even if aiortc/av aren't installed."""
    import av  # noqa: F401
    import numpy as np  # noqa: F401
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamTrack
    return av, np, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack


def _ice_config():
    """Build an RTCConfiguration (STUN + TURN) from env, so WebRTC can traverse a
    firewall/NAT (e.g. browser on a laptop -> server on the DGX cluster). Reads:
        STUN_URL         - optional STUN url (default: metered public STUN)
        TURN_URLS        - comma-separated TURN url(s), e.g.
                           "turn:global.relay.metered.ca:443,turns:global.relay.metered.ca:443?transport=tcp"
        TURN_USERNAME    - TURN username
        TURN_CREDENTIAL  - TURN credential
    Returns None when no TURN is configured, so aiortc keeps its default host-only
    (localhost/LAN) behaviour — the app runs unchanged when TURN isn't set."""
    turn_urls = [u.strip() for u in os.getenv("TURN_URLS", "").split(",") if u.strip()]
    user = os.getenv("TURN_USERNAME")
    cred = os.getenv("TURN_CREDENTIAL")
    if not (turn_urls and user and cred):
        return None
    from aiortc import RTCConfiguration, RTCIceServer
    servers = []
    stun = os.getenv("STUN_URL", "stun:stun.relay.metered.ca:80")
    if stun:
        servers.append(RTCIceServer(urls=stun))
    servers.append(RTCIceServer(urls=turn_urls, username=user, credential=cred))
    log.info("webrtc: using ICE config with %d TURN url(s)", len(turn_urls))
    return RTCConfiguration(iceServers=servers)


def _make_playback_track():
    """A MediaStreamTrack that plays queued TTS PCM (48 kHz mono) and silence when
    idle. Timing modelled on aiortc's own AudioStreamTrack so frames are paced to
    real time via pts/time_base."""
    av, np, _, _, MediaStreamTrack = _import_rtc()

    class TTSPlaybackTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self.rate = TTS_OUT_RATE
            self.samples = int(FRAME_MS * self.rate)          # 960
            self._buf = bytearray()
            self._lock = asyncio.Lock()
            self._start = None
            self._timestamp = 0

        async def push(self, pcm48_mono_s16: bytes):
            if pcm48_mono_s16:
                async with self._lock:
                    self._buf.extend(pcm48_mono_s16)

        async def recv(self):
            if self._start is None:
                self._start = time.time()
                self._timestamp = 0
            else:
                self._timestamp += self.samples
                wait = self._start + self._timestamp / self.rate - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            need = self.samples * 2
            async with self._lock:
                if len(self._buf) >= need:
                    chunk = bytes(self._buf[:need]); del self._buf[:need]
                else:
                    chunk = bytes(self._buf) + b"\x00" * (need - len(self._buf)); self._buf.clear()
            frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples)
            frame.sample_rate = self.rate
            frame.pts = self._timestamp
            frame.time_base = fractions.Fraction(1, self.rate)
            frame.planes[0].update(chunk)
            return frame

    return TTSPlaybackTrack()


def _resample_pcm(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    """Resample mono s16 PCM between sample rates via PyAV."""
    if not pcm or in_rate == out_rate:
        return pcm
    av, np, *_ = _import_rtc()
    src = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
    src.sample_rate = in_rate
    src.planes[0].update(pcm)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=out_rate)
    out = bytearray()
    for f in resampler.resample(src):
        out.extend(f.to_ndarray().astype("<i2").tobytes())
    return bytes(out)


async def _speak(text: str, outbound) -> None:
    """Synthesize `text` (Riva, in a thread) and queue it on the outbound track."""
    if not text:
        return
    from app.speech import tts
    loop = asyncio.get_event_loop()
    pcm, sr = await loop.run_in_executor(None, tts.synthesize_pcm, text)
    if pcm:
        await outbound.push(_resample_pcm(pcm, sr, TTS_OUT_RATE))


_TERMINAL_STATES = {"resolved_legit", "resolved_fraud", "escalated", "no_answer", "channel_frozen"}


async def _drain_and_close(outbound, pc) -> None:
    """After the agent's FINAL line is queued, wait for it to actually finish
    streaming (the outbound buffer empties) before closing the connection — so the
    last sentence is heard in full, not cut off the instant the call resolves.
    Capped with a safety timeout so a stuck buffer can't hold the call open."""
    for _ in range(600):                 # up to ~60 s safety cap
        await asyncio.sleep(0.1)
        if len(outbound._buf) == 0:
            break
    await asyncio.sleep(0.8)             # small tail so the last frames + jitter buffer play out
    try:
        await pc.close()
    except Exception:  # noqa: BLE001
        pass


async def _consume_inbound(track, session, outbound, pc) -> None:
    """Read the customer's mic frames, endpoint on silence, transcribe, run the
    turn through the orchestrator, and speak the reply back. When the turn resolves
    the call, wait for the final sentence to finish playing, then close."""
    _, np, *_ = _import_rtc()
    import av
    from app.speech import asr
    from app.core import orchestrator

    resampler = av.AudioResampler(format="s16", layout="mono", rate=ASR_RATE)
    loop = asyncio.get_event_loop()
    collected = bytearray()
    in_speech = False
    silence_since = None

    while True:
        try:
            frame = await track.recv()
        except Exception:  # track ended / connection closed
            break
        for rf in resampler.resample(frame):
            arr = rf.to_ndarray().flatten().astype(np.int16)
            if arr.size == 0:
                continue
            rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
            now = time.time()
            if rms > VAD_RMS_THRESHOLD:
                in_speech, silence_since = True, None
                collected.extend(arr.tobytes())
            elif in_speech:
                collected.extend(arr.tobytes())
                silence_since = silence_since or now
                if now - silence_since > SILENCE_TIMEOUT_S:
                    utter, collected[:] = bytes(collected), b""
                    in_speech, silence_since = False, None
                    text = await loop.run_in_executor(None, asr.transcribe, utter)
                    if text and text.strip():
                        log.info("webrtc heard: %s", text)
                        reply = await loop.run_in_executor(None, orchestrator.customer_says, session, text)
                        await _speak(reply, outbound)
                        # If that turn resolved the call, let the final line play out
                        # fully, then close — instead of the browser cutting it off.
                        if session.state.value in _TERMINAL_STATES:
                            await _drain_and_close(outbound, pc)
                            return


async def handle_offer(sdp: str, offer_type: str, session_id: str | None) -> dict:
    """Signaling entry point (POST /offer). Establishes the peer connection for an
    existing ringing call, wires the audio pipeline, greets the customer, and
    returns the SDP answer."""
    from app.core import orchestrator
    _, _, RTCPeerConnection, RTCSessionDescription, _ = _import_rtc()

    session = orchestrator.SESSIONS.get(session_id) if session_id else None
    if session is None:
        raise KeyError(f"no active call session '{session_id}'")

    ice = _ice_config()
    pc = RTCPeerConnection(configuration=ice) if ice else RTCPeerConnection()
    _pcs.add(pc)
    outbound = _make_playback_track()

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.ensure_future(_consume_inbound(track, session, outbound, pc))

    @pc.on("connectionstatechange")
    async def on_state():
        log.info("webrtc connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            _pcs.discard(pc)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
    # Add the outbound (agent voice) track AFTER setRemoteDescription so it attaches
    # to the browser's EXISTING audio transceiver (making that line sendrecv) rather
    # than a separate m-line the browser never offered to receive on. Without this,
    # the browser's audio line answers recvonly and the agent's voice never reaches
    # it — the "no audio, no Tap-to-hear button" symptom.
    pc.addTrack(outbound)

    # The customer just "answered" — transition the call and speak the greeting.
    orchestrator.answer_call(session)                       # idempotent
    greeting = session.transcript[-1]["text"] if session.transcript else ""
    asyncio.ensure_future(_speak(greeting, outbound))

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def shutdown() -> None:
    """Close all peer connections (call on server shutdown)."""
    await asyncio.gather(*[pc.close() for pc in list(_pcs)], return_exceptions=True)
    _pcs.clear()

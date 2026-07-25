"""Riva streaming ASR adapter.

Merge note: Jennifer's voice_loop.py recorded a fixed-length clip with `sox`
then submitted it as one blocking recognize call — simple, but there's no
interim signal to build barge-in on. Neha's implementation streams the mic
into Riva continuously on a background thread and surfaces interim results
as they arrive, which is what barge-in and low ASR-latency both need. That
implementation is carried forward here (as `MicStreamASR`), cross-platform
via `sounddevice` — Neha already used it, so no macOS-only `sox`/`afplay`
dependency to strip out.

Two entry points, because "transcribe already-captured audio" and "listen to
the mic until the customer stops talking" are genuinely different jobs:
  - `transcribe(bytes)`      — given audio, return text. Interface-compatible
                                with the original scaffold; useful if an HTTP
                                audio-upload path is ever added.
  - `MicStreamASR().listen()` — the real demo path. Captures from the local
                                mic, streams to Riva, returns once the
                                customer goes quiet. Used by demo/voice_loop.py.

Honest limitation: `listen()` surfaces interim results (see `on_interim`)
which is the *signal* barge-in needs, but this adapter does not itself
cancel an in-flight TTS playback — that requires the caller to run
TTS-playback and ASR-listening concurrently (two threads racing), which
demo/voice_loop.py does not yet do. Flagging this here rather than claiming
full barge-in works when only its precondition does.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Optional

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
RIVA_SERVER_URI = os.getenv("RIVA_SERVER_URI", "grpc.nvcf.nvidia.com:443")
RIVA_ASR_FUNCTION_ID = os.getenv("RIVA_ASR_FUNCTION_ID", "")
# Dedicated ASR auth key (Riva/NVCF). Falls back to NVIDIA_API_KEY if unset.
RIVA_ASR_API_KEY = os.getenv("RIVA_ASR_API_KEY") or os.getenv("NVIDIA_API_KEY", "")
SAMPLE_RATE = int(os.getenv("RIVA_SAMPLE_RATE", "16000"))
SILENCE_TIMEOUT_S = float(os.getenv("ASR_SILENCE_TIMEOUT_S", "2.0"))
# Force English by default. Found live 2026-07-18: with language_code=RIVA_ASR_LANGUAGE
# (auto-detect), Riva mis-detected a plain English denial ("No, that wasn't
# me...") as Spanish and returned garbage ("Los vi en el vinclar"), which the
# LLM then MISCLASSIFIED as a confirm — a fraud got marked legit. Pin the
# language for the demo; set RIVA_ASR_LANGUAGE=multi only if you genuinely
# need multilingual and accept the mis-detection risk.
RIVA_ASR_LANGUAGE = os.getenv("RIVA_ASR_LANGUAGE", "en-US")


def transcribe_text_only(utterance: str) -> str:
    """Text passthrough, regardless of MOCK_MODE — used by orchestrator.py.

    Why this exists (found live, 2026-07-18, the input-side twin of
    tts.speak_text_only()): orchestrator.customer_says() used to call the
    real transcribe() unconditionally. That's correct for mock mode and
    for a hypothetical "POST raw audio to the server" design — but our
    actual architecture (see main.py's docstring) has demo/voice_loop.py
    doing REAL transcription client-side via MicStreamASR against the
    physical mic, then POSTing the resulting TEXT to /converse. By the
    time the server sees `utterance`, it is already a transcript, never
    raw audio. Calling the live transcribe() on it meant gRPC-streaming a
    Python string to Riva as if it were PCM audio — which fails immediately
    ("Exception iterating requests!") the first time a real customer
    utterance came through /converse. The server should only ever deal in
    text; real ASR is exclusively demo/voice_loop.py's job, same reasoning
    as the TTS side."""
    return str(utterance)


def transcribe(utterance_or_audio) -> str:
    """Mock mode: text passthrough (the demo driver / tests feed typed
    'utterances' directly, so the whole flow runs offline). Live mode:
    single-shot recognize over already-captured RAW AUDIO BYTES — NOT for
    orchestrator.py's use (see transcribe_text_only() above); this remains
    available for a future direct-audio-upload endpoint if one gets built."""
    if MOCK_MODE:
        return str(utterance_or_audio)

    import riva.client  # deferred import: not needed in mock mode

    auth = riva.client.Auth(
        uri=RIVA_SERVER_URI, use_ssl=True,
        metadata_args=[["authorization", f"Bearer {RIVA_ASR_API_KEY}"],
                       ["function-id", RIVA_ASR_FUNCTION_ID]],
    )
    asr_service = riva.client.ASRService(auth)
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE, language_code=RIVA_ASR_LANGUAGE, max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    streaming_config = riva.client.StreamingRecognitionConfig(config=config, interim_results=False)

    def _one_chunk():
        yield utterance_or_audio

    transcript = ""
    try:
        for response in asr_service.streaming_response_generator(
            audio_chunks=_one_chunk(), streaming_config=streaming_config
        ):
            for result in response.results:
                if result.is_final and result.alternatives:
                    transcript = result.alternatives[0].transcript.strip()
    except Exception as e:  # noqa: BLE001 — an ASR failure (auth/network/Riva outage) must
        # NEVER crash the audio loop and silence the whole call. Log and return
        # empty; the caller then skips this utterance and keeps the call alive.
        import logging
        logging.getLogger("asr").warning("Riva ASR failed (%s) — skipping this utterance", e)
        return ""
    return transcript


class MicStreamASR:
    """Continuous mic capture -> Riva streaming ASR, with interim results.

    Ported from Neha's implementation. Threading model: `sounddevice`'s
    input stream calls `_on_audio_block` on its own callback thread, which
    just enqueues raw PCM bytes; a second thread drains the queue through
    Riva's streaming generator and updates `self.final_transcript`. `listen()`
    itself just polls until SILENCE_TIMEOUT_S has passed since the last
    speech activity, then tears both down. Kept as a class (not a function)
    because it owns a live gRPC service handle worth reusing across turns
    instead of reconnecting every call.
    """

    def __init__(self, on_interim: Optional[Callable[[str], None]] = None):
        """on_interim: optional callback fired with each interim transcript
        chunk — this is the barge-in *signal*; wire it to cancel TTS
        playback in whatever's driving the call loop."""
        import riva.client  # deferred import: not needed in mock mode
        import sounddevice as sd  # deferred import: only needed live

        self._sd = sd
        self.on_interim = on_interim
        self.sample_rate = SAMPLE_RATE
        self.block_size = 1600
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._stop = threading.Event()
        self.final_transcript = ""
        self._last_speech_ts = time.time()

        auth = riva.client.Auth(
            uri=RIVA_SERVER_URI, use_ssl=True,
            metadata_args=[["authorization", f"Bearer {RIVA_ASR_API_KEY}"],
                           ["function-id", RIVA_ASR_FUNCTION_ID]],
        )
        self._asr = riva.client.ASRService(auth)
        recognition_config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=self.sample_rate, language_code=RIVA_ASR_LANGUAGE,
            enable_automatic_punctuation=True, max_alternatives=1,
        )
        self._streaming_config = riva.client.StreamingRecognitionConfig(
            config=recognition_config, interim_results=True,
        )

    def _on_audio_block(self, indata, frames, time_info, status):
        if not self._stop.is_set():
            self._audio_q.put(bytes(indata))

    def _audio_chunks(self):
        while not self._stop.is_set():
            yield self._audio_q.get()

    def _drain_responses(self, responses):
        for response in responses:
            if self._stop.is_set():
                break
            if not response.results or not response.results[0].alternatives:
                continue
            result = response.results[0]
            text = result.alternatives[0].transcript.strip()
            if not text:
                continue
            self._last_speech_ts = time.time()
            if result.is_final:
                self.final_transcript = text
            elif self.on_interim:
                self.on_interim(text)

    def listen(self, max_seconds: float = 20.0) -> str:
        """Block until the customer finishes speaking (silence for
        SILENCE_TIMEOUT_S) or max_seconds is hit. Returns the final transcript."""
        self._stop.clear()
        self.final_transcript = ""
        self._last_speech_ts = time.time()
        while not self._audio_q.empty():
            self._audio_q.get()

        print(f"\n  [listening... speak now, {max_seconds:.0f}s max]")

        with self._sd.RawInputStream(
            samplerate=self.sample_rate, blocksize=self.block_size, dtype="int16",
            channels=1, callback=self._on_audio_block,
        ):
            responses = self._asr.streaming_response_generator(
                audio_chunks=self._audio_chunks(), streaming_config=self._streaming_config,
            )
            worker = threading.Thread(target=self._drain_responses, args=(responses,), daemon=True)
            worker.start()

            deadline = time.time() + max_seconds
            while time.time() < deadline:
                time.sleep(0.1)
                if self.final_transcript and (time.time() - self._last_speech_ts) > SILENCE_TIMEOUT_S:
                    break
            self._stop.set()
            worker.join(timeout=1)

        return self.final_transcript

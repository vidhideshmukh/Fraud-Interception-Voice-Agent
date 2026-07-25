"""Health check for every hosted model the app depends on — run this to see
which ones are UP and which are DOWN, without placing a call.

    python tools/healthcheck.py

Reads the same .env the app uses. Each service is independent:
  - LLM   : HTTP  to NIM_BASE_URL           (dialog + guardrails)
  - Riva TTS : gRPC to grpc.nvcf.nvidia.com  (agent voice)   <- the usual suspect
  - Riva ASR : gRPC to grpc.nvcf.nvidia.com  (hearing the customer)

A Riva "failed to establish link to worker" / DEADLINE_EXCEEDED means the hosted
NVCF function has no GPU worker (scaled to zero / degraded) — the model is down.
"""
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def _ok(msg):   print(f"  \033[32mUP\033[0m   {msg}")
def _bad(msg):  print(f"  \033[31mDOWN\033[0m {msg}")
def _hdr(msg):  print(f"\n=== {msg} ===")


def check_llm():
    _hdr("LLM (NIM / Build API)")
    base = os.getenv("NIM_BASE_URL", "")
    key = os.getenv("NVIDIA_API_KEY", "")
    model = os.getenv("NIM_MODEL_REALTIME", "")
    print(f"  base_url={base} model={model}")
    if not base:
        _bad("NIM_BASE_URL not set"); return
    try:
        import httpx
        t0 = time.time()
        r = httpx.post(base.rstrip("/") + "/chat/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                             "max_tokens": 1}, timeout=15)
        dt = time.time() - t0
        if r.status_code == 200:
            _ok(f"chat/completions 200 in {dt:.1f}s")
        else:
            _bad(f"HTTP {r.status_code}: {r.text[:160]}")
    except Exception as e:
        _bad(f"{type(e).__name__}: {str(e)[:160]}")


def _riva_auth():
    import riva.client
    uri = os.getenv("RIVA_SERVER_URI", "grpc.nvcf.nvidia.com:443")
    return riva.client, uri


def check_tts():
    _hdr("Riva TTS (agent voice)")
    key = os.getenv("RIVA_TTS_API_KEY") or os.getenv("NVIDIA_API_KEY", "")
    fn = os.getenv("RIVA_TTS_FUNCTION_ID", "")
    voice = os.getenv("RIVA_TTS_VOICE", "Magpie-Multilingual.EN-US.Aria")
    rate = int(os.getenv("RIVA_TTS_SAMPLE_RATE", "22050"))
    try:
        rc, uri = _riva_auth()
        print(f"  server={uri} function={fn} voice={voice}")
        auth = rc.Auth(uri=uri, use_ssl=True,
                       metadata_args=[["authorization", f"Bearer {key}"], ["function-id", fn]])
        svc = rc.SpeechSynthesisService(auth)
        t0 = time.time(); n = 0
        for resp in svc.synthesize_online(text="Health check.", voice_name=voice,
                                          language_code="en-US", sample_rate_hz=rate):
            n += len(resp.audio) if resp.audio else 0
        dt = time.time() - t0
        if n > 0:
            _ok(f"synthesized {n} bytes in {dt:.1f}s")
        else:
            _bad(f"reachable but produced 0 audio bytes in {dt:.1f}s (function degraded)")
    except Exception as e:
        _bad(f"{type(e).__name__}: {str(e)[:200]}")


def check_asr():
    _hdr("Riva ASR (hearing the customer)")
    key = os.getenv("RIVA_ASR_API_KEY") or os.getenv("NVIDIA_API_KEY", "")
    fn = os.getenv("RIVA_ASR_FUNCTION_ID", "")
    rate = int(os.getenv("RIVA_SAMPLE_RATE", "16000"))
    try:
        rc, uri = _riva_auth()
        print(f"  server={uri} function={fn}")
        auth = rc.Auth(uri=uri, use_ssl=True,
                       metadata_args=[["authorization", f"Bearer {key}"], ["function-id", fn]])
        svc = rc.ASRService(auth)
        cfg = rc.RecognitionConfig(sample_rate_hertz=rate, language_code="en-US", max_alternatives=1)
        # a short silent buffer just to exercise the endpoint (empty transcript is fine)
        t0 = time.time()
        svc.offline_recognize(b"\x00\x00" * rate, cfg)  # 1s of silence
        _ok(f"endpoint responded in {time.time()-t0:.1f}s")
    except Exception as e:
        _bad(f"{type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["llm", "tts", "asr"]
    if "llm" in which: check_llm()
    if "tts" in which: check_tts()
    if "asr" in which: check_asr()
    print()

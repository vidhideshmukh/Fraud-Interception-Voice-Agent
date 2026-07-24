"""Anti-vishing handshake. The bank pushes a one-time verification CODE (a
numeric OTP) to the customer's app BEFORE dialing. On the call, the customer
reads the code back. Trust flows bank -> customer; the agent never asks for
credentials. In the demo, the 'app' is this module plus demo/barclays_app/ (the
phone-styled web page that displays what this module generates).

Why a numeric OTP (aligning with Jennifer's latest auth): a short code pushed to
the in-app notification is the same channel a real bank OTP uses, and it's the
strongest anti-vishing signal in the pitch (it appears ONLY inside the
authenticated bank app; a scammer can't write to that channel, and the agent
never speaks it first).

The one thing that bit us last time we used a code — and is fixed here — is how
a spoken code is TRANSCRIBED. People read digits in ways ASR renders literally:
"double one" (=11), "triple seven" (=777), "oh" (=0), and sometimes the digit
words themselves ("six eight zero"). `_extract_code()` normalises all of those
back to a digit string BEFORE matching, so a correct read never fails on how it
was spoken. (The other past issue — the NeMo Guardrails self-check flagging a
bare digit read-back as "PIN/OTP disclosure" — is handled in
security/guardrails.py: the LLM self-check is skipped during the verification
step; see check_input()'s docstring.)
"""
from __future__ import annotations

import os
import re
import secrets
import time

# 6-digit OTP by default (standard length; long enough to be unguessable, short
# enough to read aloud). Configurable — set to 5 to match Jennifer's exactly.
CODE_LENGTH = int(os.getenv("VERIFICATION_CODE_LENGTH", "6"))
CODE_TTL_SECONDS = int(os.getenv("VERIFICATION_CODE_TTL_SECONDS", str(3 * 60)))  # NFR: 10 min TTL

# Spoken number-words -> digit. Includes the common ASR renderings of "0"
# ("oh", "o", "nought"). Deliberately EXCLUDES ambiguous everyday homophones
# ("to", "for", "won") — those appear as ordinary filler in a read-back and
# would inject spurious digits; Riva transcribes actual spoken digits AS digits
# anyway, so this word map is a fallback, not the primary path.
_WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
# "double X" -> XX, "triple X" -> XXX. This is the exact case that failed live
# before: a repeated digit spoken as "double one" collapsed to a single "1".
_REPEATERS = {"double": 2, "triple": 3, "treble": 3}

# session_id -> {"code": str, "sent_at": float}
_active: dict[str, dict] = {}


def _extract_code(text: str) -> str:
    """Pull the digit string out of a spoken read-back, tolerant of how people
    say codes. Handles: raw digit runs ("680863" / "68 08 63"), digit words
    ("six eight zero..."), zero-as-"oh", and repeats ("double one" -> "11").
    Non-numeric filler ("okay, the code is ...") is simply ignored."""
    tokens = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # "double"/"triple" + a following digit or digit-word -> repeat it.
        if tok in _REPEATERS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            digit = nxt if (nxt.isdigit() and len(nxt) == 1) else _WORD_TO_DIGIT.get(nxt)
            if digit:
                out.append(digit * _REPEATERS[tok])
                i += 2
                continue
        if tok.isdigit():
            out.append(tok)                      # a whole digit run, e.g. "680863"
        elif tok in _WORD_TO_DIGIT:
            out.append(_WORD_TO_DIGIT[tok])      # a single spoken digit word
        i += 1
    return "".join(out)


def send_push(session_id: str) -> str:
    """Generate and 'push' a fresh one-time numeric code, replacing any prior
    one for this session (also used to re-push mid-call on expiry — see
    resend()). secrets.randbelow keeps it unguessable and allows a leading
    zero (a real OTP can start with 0)."""
    code = "".join(str(secrets.randbelow(10)) for _ in range(CODE_LENGTH))
    _active[session_id] = {"code": code, "sent_at": time.time()}
    return code


def resend(session_id: str) -> str:
    """Explicit re-push on expiry — same mechanics as send_push, named
    separately so call sites read as intent ('the code went stale mid-call')
    rather than looking like the initial handshake."""
    return send_push(session_id)


def is_expired(session_id: str) -> bool:
    entry = _active.get(session_id)
    if not entry:
        return True
    return (time.time() - entry["sent_at"]) > CODE_TTL_SECONDS


def verify_code(session_id: str, spoken: str) -> bool:
    """True if the digits read back exactly match the pushed code. Matching is
    on the NORMALISED digit string (see _extract_code), so a code spoken as
    'double six eight oh eight six' verifies against '668086' — the read was
    correct, only the transcription differed. Fails closed on an expired code
    or if no digits were spoken at all."""
    entry = _active.get(session_id)
    if not entry or is_expired(session_id):
        return False
    spoken_digits = _extract_code(spoken)
    return bool(spoken_digits) and spoken_digits == entry["code"]

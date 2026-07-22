"""Multi-channel delivery of the verification code (in-app push + SMS + WhatsApp).

SECURITY NOTE — read before enabling SMS/WhatsApp. The IN-APP PUSH is the secure
channel, and it's the whole anti-vishing guarantee: trust flows bank -> customer
through a channel a scammer CANNOT write to. SMS is SIM-swappable and
sender-spoofable; WhatsApp is better but still not the authenticated app. So the
SMS/WhatsApp senders here are explicitly the WEAKER FALLBACK for customers who
aren't on the app — exactly what the dashboard's channel badges have always
said ("Roadmap — for customers not on the app"). Enabling them widens reach at
the cost of the strongest security property, so the app stays the primary; SMS/
WhatsApp are additive, never a replacement.

Safety of this module itself:
- OFF BY DEFAULT. Nothing is sent unless NOTIFY_SMS_WHATSAPP=true AND the Twilio
  credentials are set — so tests, offline runs, and an un-configured demo send
  NOTHING (they just log what they WOULD send). No accidental messages or cost.
- Best-effort, NEVER raises: a failed SMS/WhatsApp send is logged and skipped,
  it can't break a fraud call.
"""
from __future__ import annotations

import os

from app.observability.logging_setup import get_logger

log = get_logger("notify")

# Master switch — must be explicitly on to send anything real.
NOTIFY_ENABLED = os.getenv("NOTIFY_SMS_WHATSAPP", "false").lower() == "true"

# Twilio credentials (the de-facto provider for both SMS and WhatsApp; works
# in the UK and India). All read from the environment — never hardcode these.
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_SMS_FROM = os.getenv("TWILIO_SMS_FROM", "")            # e.g. +447700900000
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")  # e.g. whatsapp:+14155238886 (Twilio sandbox)

_MESSAGE = ("Barclays fraud check: your one-time verification code is {code}. "
            "Read it to the agent to verify this call. We will never ask for your "
            "PIN, full card number or password.")

_client = None


def _twilio():
    """Lazy Twilio client — the `twilio` package is only imported (and only
    needs installing) when sending is actually enabled and configured."""
    global _client
    if _client is None:
        from twilio.rest import Client  # deferred: pip install twilio, only if you enable this
        _client = Client(TWILIO_SID, TWILIO_TOKEN)
    return _client


def _send_sms(phone: str, code: str) -> None:
    _twilio().messages.create(to=phone, from_=TWILIO_SMS_FROM, body=_MESSAGE.format(code=code))


def _send_whatsapp(phone: str, code: str) -> None:
    to = phone if phone.startswith("whatsapp:") else f"whatsapp:{phone}"
    _twilio().messages.create(to=to, from_=TWILIO_WHATSAPP_FROM, body=_MESSAGE.format(code=code))


def deliver_code(name: str, phone: str, code: str) -> list[str]:
    """Deliver `code` to the customer across every ENABLED + CONFIGURED channel.

    Returns the list of channels the code actually went to — always includes
    'app' (the secure in-app push, which demo/barclays_app renders). SMS and
    WhatsApp are added only when NOTIFY_SMS_WHATSAPP=true and the matching
    Twilio 'from' is set. Never raises."""
    delivered = ["app"]  # the in-app push always happens — the secure primary channel
    if not (NOTIFY_ENABLED and phone):
        if phone:
            log.info("SMS/WhatsApp off (NOTIFY_SMS_WHATSAPP!=true) — would send code to %s at %s", name, phone)
        return delivered

    sms_ready = bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_SMS_FROM)
    wa_ready = bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_WHATSAPP_FROM)
    for channel, ready, send in (("sms", sms_ready, _send_sms),
                                 ("whatsapp", wa_ready, _send_whatsapp)):
        if not ready:
            log.warning("%s enabled but not configured (set the Twilio 'from') — skipping", channel)
            continue
        try:
            send(phone, code)
            delivered.append(channel)
            log.info("sent verification code to %s at %s via %s", name, phone, channel)
        except Exception as e:  # noqa: BLE001 — a send failure must never break a fraud call
            log.warning("%s send to %s failed (%s: %s) — continuing", channel, phone, type(e).__name__, e)
    return delivered

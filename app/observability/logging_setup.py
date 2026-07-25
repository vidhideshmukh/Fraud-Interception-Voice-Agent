"""Production-shaped logging: structured, levelled, and PARTITIONED to disk.

Replaces scattered print() calls with the standard `logging` module so the app
emits a real operational log stream that:
  - goes to BOTH the console (so you still see it live) AND a file,
  - the file rotates DAILY (logs/app/app.log -> app.log.YYYY-MM-DD), keeping
    14 days — i.e. partitioned by date, the standard way ops logs are kept,
  - has levels (INFO/WARNING/ERROR) so noise can be tuned per environment.

The tamper-evident AUDIT trail is a separate, stricter stream — see audit.py,
which writes one JSON record per line to logs/audit/audit-YYYY-MM-DD.jsonl
(also date-partitioned). Ops log = "what the process did"; audit log =
"the compliance record of every call event".

Configure once per process via setup_logging() (idempotent); then any module
does `log = get_logger(__name__)`. Controlled by env: LOG_DIR (default 'logs'),
LOG_LEVEL (default 'INFO').
"""
from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_configured = False


def setup_logging() -> None:
    """Idempotent: wire a daily-rotated file handler + a console handler onto
    the root logger. Safe to call from every entry point (server, voice_loop,
    tests) — only the first call takes effect."""
    global _configured
    if _configured:
        return
    app_dir = LOG_DIR / "app"
    app_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    # Daily rotation at midnight; the rolled files get a .YYYY-MM-DD suffix, so
    # the log is partitioned by date and old partitions are pruned after 14 days.
    file_handler = TimedRotatingFileHandler(
        app_dir / "app.log", when="midnight", backupCount=14, encoding="utf-8", utc=True)
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.handlers[:] = [file_handler, console]  # replace defaults so we don't double-log

    # Quiet noisy third-party loggers. NeMo Guardrails logs EVERY internal Colang
    # event at INFO (~40 lines per input+output check), and httpx logs every HTTP
    # request — together that's ~90% of the per-turn console noise. Drop them to
    # WARNING so only meaningful lines show. Set NOISY_LOG_LEVEL to override
    # (e.g. INFO to bring the detail back for debugging).
    noisy_level = os.getenv("NOISY_LOG_LEVEL", "WARNING").upper()
    for noisy in ("nemoguardrails", "nemoguardrails.colang", "httpx", "openai",
                  "uvicorn.access"):
        logging.getLogger(noisy).setLevel(noisy_level)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger; ensures logging is configured first."""
    setup_logging()
    return logging.getLogger(name)

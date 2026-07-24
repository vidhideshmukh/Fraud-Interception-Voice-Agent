# 5-minute demo script + talk track

Panel layout on screen: one browser window, `demo/barclays_app/index.html`
(served at `/app/` by the FastAPI server) — LEFT: "customer phone" panel
showing the app push and live call transcript · MIDDLE: live transaction
ticker · RIGHT: live audit-log panel. Click "Start live transaction stream"
before the hook so the flagged case fires on its own mid-sentence; the
manual trigger buttons on the same page are the fallback if the ticker
misbehaves on stage (flow is identical either way — see the risk register).

## 0:00–0:40 — the hook

> "Last year Indians lost thousands of crores to payment fraud. Banks detect much
> of it in seconds — but ACTING on it takes minutes to hours: a call-centre queue,
> or a blunt auto-block that infuriates a legitimate customer. We close that gap.
> When the fraud model fires, our agent calls the customer within seconds and
> resolves it live, before the money settles. And because a bank call is
> indistinguishable from a vishing scam, we never ask the customer for anything —
> the bank proves ITSELF to the customer first. Watch."

## 0:40–2:10 — Path B first (fraud caught — the dramatic one)

1. Trigger `txn_lagos_01`. Point at RCA reason: *"card-present in Lagos, 8 minutes
   after online in Pune — the model explains itself."*
2. Point at phone panel: *"Before dialing, the app receives a one-time code.
   Trust flows bank → customer. We never ask for PIN or OTP — say it twice."*
3. Play the call. Customer reads the code back → verified. Customer: "That wasn't me!"
4. Card blocked, reissue ordered, chargeback opened — point at audit panel filling
   in real time. *"Money never moved. Thirty seconds, end to end."*

## 2:10–3:20 — Path A (false positive recovered — the business one)

5. Trigger `txn_gadget_02`. *"Same pipeline, opposite outcome: the customer DID buy
   the laptop. Today this customer gets a blocked card and a support queue. Here:
   thirty seconds, hold released, customer keeps spending. False positives are the
   silent cost of fraud ops — we recover them."*

## 3:20–4:10 — the compliance spine (why a bank would actually deploy this)

6. Scroll the audit chain: *"Every turn is hash-chained — tamper anywhere and the
   chain breaks. Guardrails block credential requests and financial advice at the
   rail level, not just the prompt. In production, PII-bearing turns route to a
   LOCAL Nemotron via NemoClaw's privacy router — transaction data never leaves
   bank infrastructure. NemoClaw is alpha today, so we present it as the hardening
   layer; the seams for it are already in our code."*

## 4:10–4:40 — architecture in one breath

7. Show the workflow diagram: *"One design insight: the live loop is single-hop —
   Riva ASR → Nemotron Nano → Riva TTS, under a second, or the customer hears dead
   air. All heavier multi-agent reasoning — RCA, resolution planning — runs async,
   off the critical path, on Nemotron Super via the NeMo Agent Toolkit."*

## 4:40–5:00 — close

> "Quantifiable both ways: fraud stopped before settlement, and false positives
> recovered. Built entirely on the NVIDIA stack: Riva, Nemotron NIMs, NeMo Agent
> Toolkit, Guardrails, NemoClaw-ready. Thank you."

## Anticipated judge questions (rehearse answers)

- **"What if the customer doesn't answer?"** → Path C exists: hold stays, retry
  scheduled, app/SMS fallback. Fail-safe, never fail-open. (Show it if asked.)
- **"What about latency with real Riva?"** → Single-hop loop + streaming ASR/TTS +
  Nano-class model; we measured X ms/turn locally (fill in real number at event).
- **"Deepfake/voice-spoofing of the customer?"** → The code authenticates the
  BANK to the customer; customer-side auth can add voice biometrics later — the
  code is possession-based (their phone), which survives voice cloning.
- **"Isn't a code just as slow as the old word-phrase?"** → No — the earlier
  word-triplet phrase was the actual latency risk (ASR mishears, forces
  retries); a 6-digit code is what ASR is most reliable on. The verification
  check itself is a plain string compare either way, never the bottleneck.
- **"Why not run NemoClaw live?"** → Alpha, Ubuntu-only, GPU-hungry; we designed
  the seams (privacy-router adapter, audit chain) and can demo those today.
- **"Regulatory?"** → Mandatory disclosure on first turn, recorded-call notice,
  immutable audit trail, human escalation path — mapped to RBI digital-lending
  and outbound-call norms (verify exact circular refs before finals).

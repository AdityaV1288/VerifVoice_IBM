# VerifVoice — Context Primer (trimmed)

> Single-file briefing. Load THIS instead of the full master context + playbook OCR + per-person plans.
> It captures every locked decision in ~200 lines. Read the named deep-docs only when you need detail.

## 1. What we are building (one breath)

**VerifVoice** — a consented, privacy-preserving, real-time **coercion-risk signal** for payment
providers, with AI running on **IBM LinuxONE**.

- One-liner that wins: *"Banks authenticate **who** is paying. VerifVoice helps verify whether the
  **decision to pay** is being coerced."*
- Track: **Real-Time AI for Critical Decisions** (detect → decide → respond).
- We are **not** an "AI deepfake detector" and **not** a wiretap. We are an **opt-in,
  accessibility-aware transactional safety layer**.

## 2. The problem & our novelty

A scam victim often enters their **own correct** PIN/OTP — every bank check passes while the
*decision* is compromised by a concurrent coercive call. Existing transaction fraud engines can't
see the call; the bank's auth can succeed while the payment is coerced.

**Timing thesis (our actual novelty):** *a call and a payment are individually ambiguous, but
jointly and temporally they can indicate coercion.* Synthetic speech alone is not fraud (AAC/TTS
users exist); "hospital" is normal in a real emergency — but it becomes suspicious when an
unknown caller uses **authority + urgency + secrecy + payment instruction** AND the user
**immediately** attempts an unusual transfer to a **new/risky payee**.

## 3. Authoritative data to cite

- **PwC:** UPI fraud losses INR 573 Cr (FY22–23) → INR 1,087 Cr (FY23–24); social engineering /
  AI-voice calls are a key path.
- **McAfee survey** (7,054 adults, 7 countries incl. 1,010 India): ~1 in 4 experienced/knew
  someone hit by an AI voice scam — frame as *evidence of exposure*, not official crime rate.
- **RBI guidance:** regulated entities must already identify suspicious transactional behavior
  and manage false positives. → We supply a new **time-sensitive coercion signal**, not a
  replacement fraud engine.

## 4. Non-negotiable pivots (the "behaviours" — never violate)

1. **Opt-in selective analysis**, never universal interception.
2. **Honest VoIP testbed** (or just an uploaded-sample demo), never "passive packet mirroring."
3. **Multi-signal decisioning** — synthetic speech is a *supporting* feature, never a gate alone.
4. **Transaction-specific, reversible friction** — never an account freeze / phone blacklist.
5. **Real LinuxONE inference evidence** — validate, don't assert (ONNX → checker → zDLC attempt → record CPU/accelerated).
6. **No Kafka/C++/Go micro-optimization** before the golden path works; Python is the orchestration language.
7. **Accessibility, language uncertainty, replay, emergency payments are demonstrated tests** (AAC user must not trigger a pause).

## 5. Team roles (6 people)

| # | Role | Deliverable |
|---|---|---|
| P1 | AI & audio authenticity | `audio_model.py` → spoof score + quality/status field |
| P2 | Speech & coercion detection | `semantic_model.py` → coercion score + reason codes |
| P3 | IBM Z deployment & backend | FastAPI service on LinuxONE; defines API contract first |
| P4 | Risk fusion, privacy, bank simulation | `risk_engine.py` + `bank_simulator.py` + tests |
| P5 | Frontend & live demo | real dashboard wired to the real backend |
| P6 | Testing, evidence, presentation | results table, screenshots, slides, backup video |

> **Critical rule:** P3 defines the API contract IMMEDIATELY; P1/P2/P4/P5 build against it.

## 6. Architecture (simplified)

```
Sample call audio (uploaded file, in-memory only)
  → audio processing + speech analysis (real inference + quality gate)
  → Risk fusion service on LinuxONE (scores, thresholds, reason codes, uncertainty)
  → Expiring risk-token API (pseudonymous id, level, expiry, reasons)  [no raw audio/transcript leaves]
  → Bank policy simulator (amount + beneficiary history → warn/verify/pause)
  → Dashboard + audit evidence (explain decision; show deletion + latency)
```

## 7. Three-signal model + fusion formula

1. **Audio authenticity** (S) — synthetic / replayed / converted / manipulated?
2. **Coercion intent** (C) — urgency, authority, secrecy, verification-refusal, financial instruction, threat.
3. **Transaction context** (N) — payment to a new/risky beneficiary during the danger window?

```
C  = 0.6 · C_rules  +  0.4 · C_semantic     (renormalize onto rules if the learned model is absent)
R  = 0.5 · C  +  0.3 · (S·C)  +  0.2 · N    (S amplifies C; S alone ≈ 0 risk)
```
- If `S` is `None` (model unavailable): **renormalize onto C** and flag `SPOOF_UNAVAILABLE` — never invent S.
- If `status != analyzed`: treat C as weak/0 — **abstain, never claim fraud**.

## 8. Reason-code vocabulary (shared contract)

`COERCIVE_URGENCY` · `AUTHORITY_CLAIM` · `SECRECY_PRESSURE` · `VERIFICATION_REFUSAL` ·
`FINANCIAL_INSTRUCTION` · `THREAT` · `EMERGENCY_NARRATIVE` (context only, never scored) ·
`NEW_PAYEE` · `SPOOF_UNAVAILABLE`.

## 9. API contract (P3 owns this)

| Endpoint | Purpose |
|---|---|
| `GET /health` | status + model readiness |
| `POST /analyze` | sample (text or audio path) → coercion + spoof signals → returns a token |
| `POST /analyze/upload` | multipart audio upload (for P5) |
| `POST /payment/decision` | unexpired token + `{amount, new_beneficiary}` → policy decision |
| `GET /metrics` | latency + request counts |

`/analyze` returns: `request_id`, `token_id`, `token_ttl_seconds`, `audio_status`,
`coercion_score`, `reason_codes`, `signals`, `lang`, `semantic_score/status/model`,
`spoof_score/status/classification`, `quality_score`, `model_name`.

## 10. Risk token (JSON — the interface between call-risk and bank)

```json
{ "risk_level": "high", "reason_codes": ["COERCIVE_URGENCY","NEW_PAYEE_REQUEST","SYNTHETIC_AUDIO_SUPPORT"],
  "confidence": 0.91, "model_quality": 0.84, "issued_at": "...", "expires_at": "...",
  "audio_retained": false }
```
Never transmit phone numbers or transcript/audio — only a pseudonymous subject reference.

## 11. Policy ladder (proportionate, never a permanent block)

| Risk R | Decision | Action |
|---|---|---|
| < 0.30 | observe | no friction |
| 0.30–0.55 | warn | in-call notice: "…verify using a known number" |
| 0.55–0.75 | verify | show reasons, end call, out-of-band verification |
| > 0.75 (+financial intent) | pause | short **reversible** hold; biometric re-auth; optional trusted contact |

Safe override = end/pause call + device biometric + acknowledge named beneficiary + amount.
Message style: **"This call shows signs of urgent payment pressure. The recipient is new…"** (never "Deepfake detected. Payment blocked").

## 12. Safety & abstention (non-negotiable)

- Synthetic score alone never blocks; low-quality / unsupported-language → **abstain** (`insufficient_text` / `unsupported_language`), never auto-restrict.
- Emergency narrative alone never raises risk; all high-risk actions need ≥2 independent signals.
- No raw audio / full transcript on disk; token + buffer deleted after scoring.

## 13. Evaluation scenarios (must demonstrate)

| Test | Expected safe behaviour |
|---|---|
| Synthetic coercive call + new payee | high risk → warn/hold/step-up |
| Human scammer + coercive intent | true positive WITHOUT deepfake dependency |
| AAC/TTS user paying known merchant | no restriction |
| Genuine emergency + known payee | warning at most, minimal friction |
| Genuine emergency + new personal QR | strong warning + reversible step-up + safe override |
| Replayed deepfake | confidence drops, uncertainty explained |
| Hindi/English code-switching | detect cues or fail safe |
| Bad network / low quality | "insufficient signal", not "fraud" |
| Benign synthetic bot | synthetic score alone → no bank action |
| Scam after call ends | short-lived token still correlates |

## 14. LinuxONE requirements (what must REALLY run there)

Risk fusion model · ≥1 genuine AI inference component · FastAPI decision API · risk-token
generation · payment policy simulator (or its endpoint) · metrics endpoint. Media capture + UI
may run elsewhere — **state the boundary explicitly in the video.**

## 15. Data lifecycle (demo must visibly show)

1. ring buffer grows during the call → 2. inference consumes it → 3. no `.wav`/transcript file
written → 4. termination clears the buffer → 5. bank receives only the token → 6. audit record has
model/action metadata but **not** call content.

## 16. MVP scope

- **Must:** consented sample call · rolling audio windows · one real acoustic signal · one
  semantic/coercion signal · fusion score + reason codes + uncertainty · expiring token ·
  simulated bank flow (known/new payee) · graduated warning + step-up · safe override ·
  LinuxONE-hosted AI/decision API · measured latency · ≥6 scenario tests.
- **Should:** Hindi/English code-switching · AAC scenario · replay scenario · dashboard ·
  ONNX/zDLC attempt · containerization.
- **Do NOT:** real UPI/NPCI · real interception · national blacklist · police reporting ·
  production Kafka · large LLM in critical path · permanent audio storage · claims of nationwide
  capacity / perfect accuracy / zero false positives / legal-compliance certification.

## 17. Acceptance thresholds (internal goals, not market claims)

Warning within **8s** of decisive coercion evidence · policy response within **500ms** · no
permanent block from call score alone · no raw audio/transcript on disk · all high-risk actions
carry ≥2 signals · AAC benign does not pause · low-quality can abstain · golden path succeeds 3×
before recording.

## 18. What we never claim

First at live-call analysis / deepfake detection / scam warnings. No "99% accuracy." No "saves
lives" / guaranteed prevention. No Telum/Secure-Execution claims unless actually verified on the
assigned VM. Real latency + honest limitations beat a perfect mismatched number.

## 19. Current code state (what's already built)

- `semantic_model.py` + `semantic_config.py` — P2 rule engine (7 signals, EN+Hindi, negation mask, abstention gates). **DONE.**
- `semantic_classifier.py`, `export_semantic_model.py` — P2 learned coercion layer.
- `audio_authenticity.py` — P1 adapter (spoof score + quality).
- `pipeline.py` — combines P1+P2 into one `analyze_call()` result.
- `main.py` — P3 FastAPI backend (endpoints above).
- `token_store.py` — in-memory expiring tokens (TTL 120s).
- `risk_engine.py` — **fallback** fusion `decide(coercion_score, spoof_score, new_beneficiary, amount)` — P4 replaces this.
- Deep docs for detail: `PERSON1_AUDIO_MODEL_PLAN.md`, `PERSON2_ARCHITECTURE.md`,
  `PERSON2_TASKS.md`, `PERSON3_ARCHITECTURE.md`, `DEPLOYMENT.md`, `SEMANTIC_LAYER_PLAN.md`.

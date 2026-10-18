# verifvoice_risk: risk fusion, privacy and bank simulation

This is the VerifVoice component that turns **call signals and transaction context into a
proportionate, reversible payment action**. It also makes sure that only an expiring,
pseudonymous risk token leaves the protected zone. Section numbers (§) refer to the
VerifVoice design document.

| File | What it does |
|---|---|
| `risk_engine.py` | Call-risk fusion (§6.2–6.5), transaction fusion and proportionate actions (§5.2 steps 9–10), expiring signed tokens (§5.2 step 8), privacy guards and deletion checks (§7.2–7.3), and the `RiskEngine` facade for the API. |
| `bank_simulator.py` | Simulated Payment Decision API: a payee registry with new-beneficiary detection, history-based amount checks, token consumption, holds and overrides, and a pseudonymous audit trail. |
| `demo.py` | Scenario walkthrough for the video: `python -m verifvoice_risk.demo`. |
| `tests/` | 237 tests: safe and suspicious transactions, human-scammer, AAC and real-emergency cases, tokens, Redis concurrency, privacy, deletion. 9 of them need NumPy and are skipped without it. |

The core needs **only the Python ≥ 3.9 standard library**. There are no compiled wheels, so
it should run on Windows, macOS and LinuxONE (s390x) alike.

Out of scope: audio capture, the ring buffer, the anti-spoofing model, ASR, the coercion
model and the FastAPI app itself. Those belong to teammates. This module consumes their
outputs.

---

## Quick start

macOS / Linux / LinuxONE:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r verifvoice_risk/requirements-dev.txt
python -m pytest verifvoice_risk
python -m verifvoice_risk.demo
```

Windows (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r verifvoice_risk\requirements-dev.txt
python -m pytest verifvoice_risk
python -m verifvoice_risk.demo
```

Run the commands from the folder that **contains** `verifvoice_risk/`.

---

## 1. Input contract

This is the team's agreed payload. `RiskEngine.evaluate()` accepts it as is:

```json
{
  "audio_analysis":    {"status": "ok", "prediction": "likely_spoof", "spoof_score": 0.93},
  "coercion_analysis": {"status": "ok", "coercion_level": "high"},
  "transaction":       {"amount": 500, "currency": "USD", "new_beneficiary": true}
}
```

| Field | Rule |
|---|---|
| `*.status` | Required. `"ok"` means the signal is used. **Any other value** (`"error"`, `"unsupported_language"`, …) means the signal is unavailable. |
| `audio_analysis.spoof_score` | Required when status is ok. A number in [0, 1], where higher means more likely synthetic or replayed. |
| `audio_analysis.prediction` | Optional string. Recorded only. The score drives the risk. |
| `coercion_analysis.coercion_level` | Required when status is ok. One of `none`, `low`, `medium`, `high` (case-insensitive). |
| `transaction.amount` | Positive number. |
| `transaction.currency` | 3-letter code (normalised to upper case). |
| `transaction.new_beneficiary` | `true`/`false`. Required by `evaluate()`. The bank simulator ignores it and detects novelty itself. |

**Optional extensions.** These unlock the rest of the §6 design. Unknown keys are ignored,
so teammates can add them whenever they're ready.

| Field | Effect |
|---|---|
| `audio_analysis.speaker_match` (0–1) | Replaces the neutral 0.5 prior in the speaker term. Raises confidence. |
| `audio_analysis.quality` (0–1) | Quality-gate output. Lowers confidence, never the score (§5.2 step 4). |
| `audio_analysis.model_version`, `coercion_analysis.model_version` | Written to the audit log (§7.2 item 9). |
| `coercion_analysis.coercion_score` (0–1) | Used in the formula instead of the level→score mapping. |
| `coercion_analysis.verification_refused` (bool) | Adds the **+0.25 refusal boost** and lifts the phase to P4. |
| `coercion_analysis.cues` (list) | Any of `COERCIVE_URGENCY`, `SECRECY_REQUEST`, `AUTHORITY_CLAIM`, `FINANCIAL_INSTRUCTION`, `VERIFICATION_REFUSED`, `SPEAKER_HANDOFF`. These become reason codes. |
| `coercion_analysis.phase` (`P0`–`P4`) | Applies the §6.3 phase ceiling. P3/P4 count as financial intent. |
| `coercion_analysis.financial_intent` (bool) | Enables the CRITICAL band. |
| `transaction.beneficiary_id`, `transaction.channel` | Parsed for the bank side. |

Malformed input raises `InputValidationError` (a `ValueError`). Map it to HTTP 422.
Malformed values are never silently dropped.

---

## 2. Call-risk fusion

The formula is from §6.2:

```
call_risk = 0.40·spoof + 0.35·semantic + 0.25·(1 − speaker_match) + 0.25·[verification refused]
score     = clip(min(max(call_risk, intent_floor), intent_ceiling, phase_ceiling), 0, 1)
```

* **semantic** comes from `coercion_level`: none 0.0, low 0.25, medium 0.60, high 0.90.
  It comes from `coercion_score` instead when that is sent.
* **Missing measurements use neutral priors** (0.5) and lower the confidence. Unknown is
  not evidence either way.
* **Intent gate (the AAC safeguard, §8).** Each coercion level caps the score: none/low cap
  at 0.50, medium at 0.75, high at 1.00. So a synthetic voice with no coercion **can never
  reach token level**, however high `spoof_score` is. Refusal counts as the pressure phase
  (P4), so it lifts the cap to 1.00.
* **Intent floor (catches human scammers).** `coercion_level: high` guarantees a score of
  at least 0.55, the token threshold.
  * Without the floor, a scammer using their own real voice scores
    0.40·0.05 + 0.315 + 0.125 = 0.46, and the bank never hears about the call.
  * The floor gives a *medium* token, not high. Even with every §6 signal (low speaker
    match, refusal), the doc's formula only puts a human scammer at medium.
  * Medium coercion has no floor, so a real emergency in a genuine voice (0.355) is still
    not escalated.
  * An explicit early `phase` (P0) still overrides the floor.
* **Phase ceilings (§6.3)** apply only when a `phase` is sent. They can only lower the
  score.
* **Abstention.** If the coercion signal is unavailable, the engine abstains. The level
  stays at elevated or below, and no token is issued, whatever the config says.
* **Confidence** is the share of fusion weight backed by real measurements, scaled by
  `quality`. For your current format it is 0.75, because there's no speaker_match.

Bands follow §6.4. Intervals are `[low, high)`.

| Score | Band | Level | Token? |
|---|---|---|---|
| < 0.30 | PASSIVE_MONITORING | low | no |
| 0.30–0.55 | ELEVATED_WATCH | elevated | no |
| 0.55–0.75 | SOFT_PROMPT | medium | **yes** |
| 0.75–0.90 | STRONG_PROMPT | high | **yes** |
| > 0.90 **and** financial intent | CRITICAL_ALERT | critical | **yes** |

Without financial-intent evidence, a score above 0.90 stays at STRONG_PROMPT (high).

**The team example works out as:** 0.40·0.93 + 0.35·0.90 + 0.25·0.5 = **0.812**. That is
STRONG_PROMPT, a high token with confidence 0.75 and reason codes `SYNTHETIC_AUDIO_HIGH`
and `COERCIVE_LANGUAGE`.

---

## 3. Risk tokens

A token is issued once the call reaches **SOFT_PROMPT (≥ 0.55) or higher** (team decision).
It contains exactly what §5.2 step 8 allows, and nothing else (no score, no call ID, no
audio, no transcript):

```json
{"schema_version": "1", "token_id": "tok_…", "subject_ref": "sub_…", "level": "high",
 "confidence": 0.75, "reason_codes": ["SYNTHETIC_AUDIO_HIGH", "COERCIVE_LANGUAGE"],
 "issued_at": 1800000000.0, "expires_at": 1800000600.0,
 "retention": "Risk signal only: …", "signature": "…"}
```

* **Pseudonymous.** `subject_ref` is an HMAC-SHA256 of the customer ID under the
  pseudonym key. The bank recomputes it from its own customer ID to look the token up.
* **Signed.** HMAC-SHA256 under the signing key. Anything tampered with or forged is
  treated as *no signal*, logged, and audited as `token_status: "invalid"`.
* **Expiring.** The default TTL is 600 s from the last qualifying assessment. Expired
  tokens are deleted on access and by `purge_expired()`. The delete is
  compare-and-delete, so it never removes a token that was refreshed in the meantime.
* **High-water mark.** During the TTL, a weaker assessment refreshes the expiry but never
  downgrades the level. This holds across API workers too: `issue()` is an atomic
  read-modify-write (a Redis WATCH/MULTI transaction, retried on conflict).
* **A calm moment doesn't erase earlier risk.** At payment time, both
  `evaluate(payload, subject_id=…)` and the bank simulator use the customer's *active*
  token, even if the latest call window scored low.
* **Revocable.** `revoke(customer_id)` is the opt-out / "easy off switch".
* **Stores.**
  * `InMemoryTokenStore` (default) is thread-safe and works for one process.
  * `RedisTokenStore(redis.Redis(...))` uses Redis's native `PX` TTL. Use it when several
    API workers must share tokens.
  * A custom store implements `put`, `get`, `update` (atomic), `discard` (atomic
    compare-and-delete), `delete`, `purge_expired` and `tokens`. See `TokenStore` in
    `risk_engine.py`.
* **Keys.** `VERIFVOICE_SIGNING_KEY` and `VERIFVOICE_PSEUDONYM_KEY` environment variables,
  at least 16 bytes each, never stored in the repo. If they're unset, each process
  generates a random key and logs a warning. That's fine for local dev, but **multi-worker
  deployments must set them** or workers won't recognise each other's tokens.

---

## 4. Bank simulator and proportionate actions

**New-beneficiary detection** uses the bank's own registry and never trusts the client
flag. A payee is new if it is:

* `not_registered`: never added or paid;
* `added_recently`: added (or first paid) less than 24 h ago. This stops the "send ₹1
  first, then ₹50,000" trick;
* `never_paid`: added a while ago but never paid.

Otherwise it is `established`.

**Amount tier**, checked in this order:

1. `small` if below a per-currency floor (INR 1,000 / USD 50).
2. `large` if at or above a per-currency bar (INR 25,000 / USD 1,000).
3. `large` if more than 3× the median of the customer's last 50 payments (needs at least 5
   payments of history).
4. `normal` otherwise.

**Action ladder:** `ALLOW → WARN → VERIFY_OUT_OF_BAND → HOLD`.

1. Start from the token level: medium starts at ALLOW, high at WARN, critical at
   VERIFY_OUT_OF_BAND.
2. Move up one rung for a new beneficiary, and one for a large amount.
3. Move down one rung for a small amount, and one for a token with confidence below 0.5.
4. Clamp the result. Critical never drops below WARN.
5. **No valid token means ALLOW.** VerifVoice adds no friction, and the bank's own fraud
   engine still applies.

| Token | Known payee, normal amount | New payee, normal amount | New payee, large amount |
|---|---|---|---|
| none | ALLOW | ALLOW | ALLOW |
| medium | ALLOW | WARN | VERIFY_OUT_OF_BAND |
| high | WARN | **VERIFY_OUT_OF_BAND** (team example) | HOLD |
| critical | VERIFY_OUT_OF_BAND | HOLD | HOLD |

There is **no block or freeze state**. Every pending payment can be completed or
cancelled by the customer:

| Action | Status | How it completes |
|---|---|---|
| ALLOW | APPROVED → EXECUTED | immediately |
| WARN | PENDING_CONFIRMATION | `acknowledge_warning()` |
| VERIFY_OUT_OF_BAND | PENDING_VERIFICATION | `verify_out_of_band(confirmed=True)`. `False` cancels. |
| HOLD (30 min) | ON_HOLD | `verify_out_of_band(True)` releases it early (the safe override), or `confirm_after_hold()` once the hold ends |

Any open payment can also be cancelled with `cancel_payment()`. Other payments on the
account keep flowing while one is held. Step-up auth (OTP/PIN) is left off the ladder on
purpose: coercion victims authenticate voluntarily (§2.1).

---

## 5. Privacy and deletion checks

* **`AuditLog` rejects, at write time,** any record that contains:
  * audio, transcript, phone or ID fields (`transcript`, `audio`, `customer_id`, …);
  * binary payloads;
  * phone- or account-like numbers (10 or more digits);
  * the raw IDs involved in that record (customer, payee, call), passed per record as
    `append(record, raw_identifiers)`. IDs of 6 or more characters are also caught
    inside longer text; shorter ones only as an exact value.

  Checking per record keeps each write O(record size), however many customers the
  service has seen. The engine and bank validate each record *before* changing any
  state, so a rejected record never leaves a half-issued token or half-stored payment.

  Audit entries use `sub_…`, `ben_…` and `call_…` pseudonyms, model versions, thresholds
  and actions only (§7.2 item 9). The log is in memory: in a long-running deployment, ship
  the records (already content-free) to durable storage.
* **Extra payload keys are ignored and never stored.** For example, a `transcript` sent by
  mistake is dropped.
* **`run_deletion_checks(token_service=…, audit_logs=[…], buffers=[…], artifact_dirs=[…], raw_identifiers=[…])`**
  inspects without modifying anything, and returns a JSON-ready report covering:
  * `audio_buffers_zeroized`: every byte of each ring buffer is zero. This works on
    `bytearray`, `array.array` and NumPy arrays;
  * `no_expired_tokens_retained`;
  * `tokens_minimal_and_signed`;
  * `audit_log[i]_content_free`: also searches for any `raw_identifiers` you pass;
  * `no_media_or_transcript_files`: no `.wav`, `.mp3`, `.flac`, … files and no
    `*transcript*` files under the given folders.
* **`zeroize(buffer)`** is a helper the audio module can call when a call ends. The buffer
  must be writable and contiguous, such as a `bytearray` or NumPy array. This was tested
  with float32, int16, big-endian `>i2`, and 1-D and 2-D layouts. A strided view raises a
  clear `TypeError`: zeroize the underlying ring buffer instead.

---

## 6. API at a glance

```python
from verifvoice_risk import RiskEngine, TokenService, BankSimulator, RedisTokenStore

tokens = TokenService(ttl_seconds=600)            # or TokenService(store=RedisTokenStore(redis.Redis()))
engine = RiskEngine(tokens)
bank   = BankSimulator(tokens)

# a) One-shot, on the team payload: returns call_risk, token, transaction and decision
result = engine.evaluate(payload, subject_id="cust-001")

# b) Streaming: each call update can issue or refresh the token...
assessment, token = engine.assess_call(audio_dict, coercion_dict, subject_id="cust-001", call_id="c-42")
# ...and the bank consumes it at payment time
bank.register_customer("cust-001", currency="USD", established_beneficiaries=["acct-landlord"])
payment = bank.submit_payment("cust-001", "acct-stranger", 500, "USD")
payment.status, payment.decision.action, payment.decision.user_message
bank.verify_out_of_band(payment.payment_id, confirmed=True)
```

Every result has a `to_dict()` that is JSON-serialisable.

## 7. FastAPI integration

This snippet was run against FastAPI 0.143 with `TestClient`. It returned 200 for the team
payload, 422 for bad input and 409 for an invalid transition.

```python
from fastapi import FastAPI, HTTPException, Header
from verifvoice_risk import BankSimulator, InputValidationError, InvalidTransition, RiskEngine, TokenService

tokens = TokenService(ttl_seconds=600)   # keys from VERIFVOICE_* env vars
engine = RiskEngine(tokens)
bank = BankSimulator(tokens)
app = FastAPI()

@app.post("/v1/evaluate")
def evaluate(payload: dict, x_customer_id: str = Header(...)):
    # In production, take the subject from the authenticated session, not a header.
    try:
        return engine.evaluate(payload, subject_id=x_customer_id)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@app.post("/v1/payments/{payment_id}/verify")
def verify(payment_id: str, confirmed: bool = True):
    try:
        return bank.verify_out_of_band(payment_id, confirmed).to_dict()
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
```

Use plain `def` endpoints, not `async def`. FastAPI then runs them in its threadpool, and
the stores and simulator are lock-protected for that. In production, return only `token`
and `decision` to the bank. `call_risk` is there for the demo and for debugging.

---

## 8. Deployment notes (LinuxONE / s390x, Windows)

* Standard library only, with no byte-order-dependent code. The buffer checks work byte
  by byte, and HMAC and JSON don't depend on endianness. So nothing should change on
  big-endian s390x. **This has not been run on s390x yet** (see below).
* `redis`, `fakeredis` and `pytest` are pure Python and install from PyPI on s390x.
* Use `RedisTokenStore` and set both key environment variables whenever more than one
  uvicorn worker or process serves requests.
* `BankSimulator` state is in memory and resets on restart. It's a simulator.

## 9. What has been verified

* All tests pass on **CPython 3.14.7** and **CPython 3.9.25** on macOS arm64: 228, plus
  the 9 NumPy tests, which pass with NumPy 2.5.4 installed (237 in total).
* A mutation check: 34 deliberate bugs were introduced one at a time and every one was
  caught. Examples:
  * wrong weight, AAC gate removed, signature check disabled, expiry off-by-one;
  * raw ID in the audit log, intent floor removed or misplaced;
  * each review fix reverted (calm window ignoring the active token, token-store races,
    partial state after a rejected audit record, global ID set growth).
* Two fixes were also reproduced with scripts, before and after:
  * a lookup racing a fresh `issue()` used to delete the new token;
  * audit cost used to grow with customer count. It now stays flat: 0.21 ms per
    assessment plus payment at both 1k and 20k customers.
* The FastAPI snippet above was run end to end with `TestClient`.
* **Not yet verified:** a run on IBM LinuxONE (s390x) and on Windows. Neither was
  available. First thing on the LinuxONE VM:
  `python3 -m pytest verifvoice_risk && python3 -m verifvoice_risk.demo`.

## 10. Assumptions to confirm with the team

1. **coercion_level → score mapping** (0 / 0.25 / 0.60 / 0.90). If the coercion model has
   a numeric score internally, send it as `coercion_score`.
2. **`spoof_score` is a calibrated probability** where higher means more likely spoofed.
   `prediction` is recorded, but it doesn't drive the score.
3. **Any status other than `ok` means the signal is unavailable.**
4. **The policy numbers are prototype defaults**, all set in `PolicyConfig`. Calibrate them
   against the §6.6 target of ≤ 1 strong prompt per 1,000 benign calls:

   | Setting | Default |
   |---|---|
   | Token TTL | 10 min |
   | Hold | 30 min |
   | New-payee window | 24 h |
   | Amount bars | INR 1k / 25k, USD 50 / 1k |
   | History multiplier | 3× median |
   | Low-confidence threshold | 0.5 |
5. **The high-coercion floor is 0.55 (a medium token), on purpose.**
   * Even with every §6 signal, the doc's formula puts a human scammer at medium.
   * The typical scam ask is a large transfer, which already gets out-of-band
     verification.
   * A 0.75 floor would let one classifier's `high` label trigger strong prompts on its
     own, and its false-positive rate isn't known yet.
   * Once the coercion model's precision has been measured against the §6.6 target, it's
     one setting to change: `coercion_level_floors["high"]`.
6. **`evaluate()` trusts the client's `new_beneficiary` flag.** Add `beneficiary_id` to the
   transaction object so the bank side can detect novelty itself.

## 11. Known limitations with the current upstream format

* **Human (non-cloned) scammers are now caught, but only through the coercion signal.**
  `coercion_level: high` is floored to a medium token. Friction depends on the payee:

  | Payee and amount | Action |
  |---|---|
  | New payee, normal amount | WARN |
  | New payee, unusual amount | VERIFY_OUT_OF_BAND |
  | Established payee, normal amount | ALLOW |

  If the coercion model rates a human scam as only `medium`, it still isn't caught.
  `verification_refused` and `speaker_match` remain the most valuable additions upstream.
* **CRITICAL is unreachable** until the coercion module sends financial-intent evidence
  (the `FINANCIAL_INSTRUCTION` cue, `financial_intent`, or phase P3/P4).
* `UNUSUAL_TIME` is in the vocabulary but not computed yet.
* The fusion is the hand-set §6.2 formula, not a trained model. There's no labelled data
  to fit a logistic regression or XGBoost model honestly.

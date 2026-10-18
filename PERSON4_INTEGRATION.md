# Person 4 — Integration Plan & Expectation List

> Owner of integration: team lead / P3 (backend), with P2 (semantic) as the upstream signal.
> Person 4's repo: `https://github.com/aryanjain5/VerifVoice` → package **`verifvoice_risk/`**.

## 1. What Person 4 actually shipped (verified from GitHub)

Package `verifvoice_risk/` contains:
- `risk_engine.py` (**63 KB** — large, needs review) and `bank_simulator.py` (15 KB)
- `__init__.py` — public API, `demo.py`, `tests/`, `requirements.txt`, `requirements-dev.txt`
- `__pycache__/` committed (should be gitignored) and `.DS_Store` at repo root

Public API exported from `__init__.py`:
- **Engine:** `RiskEngine`, `score_call`, `decide_action`, `classify_amount`
- **Bank sim:** `BankSimulator`, `Payment`, `InvalidTransition`, states `APPROVED / CANCELLED /
  EXECUTED / ON_HOLD / PENDING_CONFIRMATION / PENDING_VERIFICATION`
- **Tokens:** `TokenService`, `InMemoryTokenStore`, `RedisTokenStore`, `RiskToken`
- **Data classes:** `CallSignal`, `CoercionAnalysis`, `AudioAnalysis`, `Transaction`,
  `TransactionContext`, `CallRiskAssessment`, `ActionDecision`, `AuditLog`, `DeletionReport`
- **Config:** `FusionConfig`, `PolicyConfig`; **errors:** `InputValidationError`, `PrivacyViolation`
- **Decisions:** `ALLOW`, `WARN`, `VERIFY_OUT_OF_BAND`, `HOLD`; `POLICY_VERSION`, `REASON_CODES`
- **Privacy/deletion:** `find_media_artifacts`, `find_privacy_violations`, `is_zeroized`,
  `run_deletion_checks`, `zeroize`

→ This is a **richer, object-oriented design** than our minimal fallback. Good breadth, but it
must be adapted to the flat, minimal backend we already have running.

## 2. Contract mismatches to resolve BEFORE wiring (important)

| # | Ours (current backend) | Person 4's package | What to do |
|---|---|---|---|
| 1 | Flat function `decide(coercion_score, spoof_score, new_beneficiary, amount) -> dict` | Class-based `RiskEngine` + `score_call()` + `decide_action()` | Write a **thin adapter** exposing the flat `decide()` signature (see §4) |
| 2 | Decision vocab: `observe / warn / verify / pause` | `ALLOW / WARN / VERIFY_OUT_OF_BAND / HOLD` | **Pick one vocabulary** and map it; P5 dashboard + P6 tests depend on it |
| 3 | `token_store.py` (flat, TTL 120s) | `TokenService` + `InMemoryTokenStore` + `RedisTokenStore` | **Reconcile** — keep ONE token path; do not run two stores |
| 4 | Top-level flat modules imported by `main.py` | Package `verifvoice_risk/` | Keep as a vendored package **or** flatten; update imports accordingly |
| 5 | `requirements.txt` in our venv | Their `requirements.txt` (163 B) + `-dev` (107 B) | Check for new/conflicting deps (e.g. `redis`) before installing on s390x |
| 6 | 63 KB single file | — | Likely over-engineered for a 24h demo; review for dead code / Redis-only paths |

## 3. Expectation list — what we need from Person 4

**Must deliver (functional):**
1. A **single entry point** matching our contract, or the adapter itself:
   `decide(coercion_score: float, spoof_score: float|None, new_beneficiary: bool, amount: float) -> {"decision", "risk_score", "reason_codes", "message"}`.
2. **Fusion formula** `R = 0.5·C + 0.3·(S·C) + 0.2·N`, with `S=None` → renormalize onto C and flag `SPOOF_UNAVAILABLE` (never invent S).
3. **Decision vocabulary mapped** to our 4 levels (`observe/warn/verify/pause`) — or a written note for P5 to adopt `ALLOW/WARN/VERIFY_OUT_OF_BAND/HOLD`.
4. **`bank_simulator.py`**: a simple call to evaluate a transaction (amount + known/new beneficiary) and return a decision + human message; no global account block; a **safe override path**.
5. **Expiring risk token**: reuse our `token_store.py` (TTL 120s) or provide their `TokenService` with TTL + expiry deletion. Pseudonymous id only — **no transcript/phone/audio**.
6. **New-beneficiary detection**: clear input (bool or a beneficiary-history check).
7. **Proportionate actions**: observe → warn → verify → pause; pause must be **reversible** and transaction-specific.
8. **Deletion checks**: `run_deletion_checks()` / `zeroize()` proving no raw audio/transcript is persisted (audit metadata only).
9. **Test cases** (min 4): benign + known payee → no friction; urgent demand + new payee → verify/pause + reason codes; synthetic voice + benign → no high-risk; genuine emergency → no block, override preserved.
10. **No new heavy deps** (Redis optional; in-memory for demo), no secrets, no hardcoded paths.

**Must confirm / clean up:**
- Remove committed `__pycache__/` and `.DS_Store`; add `.gitignore`.
- State `POLICY_VERSION` and threshold config so P6 can record it.

## 4. How we integrate it (step-by-step)

1. **Vendor the code.** Copy `verifvoice_risk/` into our repo as a subpackage (recommended), or
   flatten to top-level `risk_engine.py` + `bank_simulator.py`. Keep their `tests/` separate.
2. **Write an adapter** (if they didn't) that maps our flat contract onto their classes:
   ```python
   # risk_adapter.py
   from verifvoice_risk import RiskEngine, FusionConfig, PolicyConfig
   _engine = RiskEngine(FusionConfig(), PolicyConfig())
   def decide(coercion_score, spoof_score, new_beneficiary, amount=0.0):
       r = _engine.score_call(coercion=coercion_score, spoof=spoof_score,
                              new_beneficiary=new_beneficiary, amount=amount)
       d = _engine.decide_action(r)     # -> ActionDecision
       return {"decision": _map(d.action), "risk_score": round(r.risk, 4),
               "reason_codes": d.reason_codes, "message": d.message}
   ```
3. **Map decisions:** `ALLOW→observe`, `WARN→warn`, `VERIFY_OUT_OF_BAND→verify`, `HOLD→pause`
   (or update P5 to Person 4's labels — decide once, document it).
4. **Reconcile tokens:** keep `token_store.py` in `main.py` (it already stores scores on
   `/analyze` and expires on `/payment/decision`), and **do not** also instantiate their
   `TokenService` unless we replace ours. One source of truth.
5. **Wire `/payment/decision`** in `main.py`: replace the fallback `risk_engine.decide(...)` call
   with the adapter; keep the reason-code merge:
   `reason_codes = dedupe(token["reason_codes"] + result["reason_codes"])`.
6. **Add bank-simulator context** for P5: expose a helper that, given beneficiary history,
   returns `new_beneficiary` + a transaction message (or accept it as a request field as today).
7. **Run the shared scenario matrix** (§13 of the primer) end-to-end and record real numbers:
   verify observe/warn/verify/pause boundaries, safe override, AAC/emergency cases.
8. **Privacy gate:** run their `run_deletion_checks()` + our "no `.wav`/transcript on disk" check;
   confirm the token/buffer is deleted after scoring; capture the evidence for P6.

## 5. Definition of done for this integration

- [ ] `/payment/decision` returns the 4-level decision with `risk_score`, merged `reason_codes`, and a safe message.
- [ ] `S=None` path degrades gracefully (`SPOOF_UNAVAILABLE`), never invents a spoof score.
- [ ] Single token store; token expires and is removed (TTL verified).
- [ ] No permanent block; pause is reversible; override works.
- [ ] 4+ scenario tests pass and are recorded; no raw audio/transcript persisted.
- [ ] New deps confirmed on the LinuxONE VM (s390x) — Redis optional, in-memory default.

---

## 6. Integration status — DONE (day-one formula wired in)

What was actually wired:

| # | Decision | Value |
|---|---|---|
| 1 | **Fusion formula** | Day-one `R = 0.5·C + 0.3·(S·C) + 0.2·N`; `S=None` → `R = 0.8·C + 0.2·N` + `SPOOF_UNAVAILABLE`. Owned by `verifvoice_risk.risk_engine.decide()`. |
| 2 | **Decision vocabulary** | Lowercase `observe / warn / verify / pause` (canonical for P5 + P6). |
| 3 | **coercion_score → coercion_level** | `coercion_level_from_score()`: `<0.125` none, `<0.425` low, `<0.75` medium, else high. |
| 4 | **Reason-code translation** | `risk_adapter.py` maps team vocab ↔ P4 internal (`SECRECY_PRESSURE↔SECRECY_REQUEST`, `VERIFICATION_REFUSAL↔VERIFICATION_REFUSED`, `NEW_PAYEE↔NEW_BENEFICIARY`). `THREAT` / `EMERGENCY_NARRATIVE` carry as reason codes, not P4 cues. |
| 5 | **Tokens** | ONE store: `token_store.py` (TTL 120s, scores-only). P4's `TokenService` is the optional bank-facing layer and is **not** instantiated in the demo path. |
| 6 | **POLICY_VERSION** | **`vv-risk-1.0.0`** — record this for P6. |
| 7 | **Files** | `verifvoice_risk/` (vendored) · `risk_adapter.py` (wiring) · `main.py` / `demo.py` call `risk_adapter.decide` · `risk_engine.py` is now a backward-compat shim. |

Verification: `python smoke_test.py` (scenario matrix + end-to-end API) passes; P4's suite
232 passed / 5 skipped (Redis); P2's semantic tests 6 passed.

Still to do on LinuxONE (s390x): run `pytest verifvoice_risk && python smoke_test.py` on the
VM. The package is pure stdlib so it should be unchanged there, but it has not been run on
s390x yet.

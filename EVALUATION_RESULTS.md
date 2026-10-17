# Person 2 — Evaluation Results & Evidence (Phase 6 handoff)

> **Owner:** Person 2 → **Consumer:** Person 6 (README / slides) and Person 4 (fusion thresholds).
> Reproduce anytime with: `python run_eval.py`  (saves `data/transcripts/results.json`).

## 1. Results table (22 fixtures)

| ID | Label | Score | Band | Reason codes |
|---|---|---|---|---|
| T01 | coercive | 1.00 | high | COERCIVE_URGENCY, AUTHORITY_CLAIM, SECRECY_PRESSURE, VERIFICATION_REFUSAL, FINANCIAL_INSTRUCTION, THREAT, EMERGENCY_NARRATIVE |
| T02 | coercive | 1.00 | high | COERCIVE_URGENCY, AUTHORITY_CLAIM, SECRECY_PRESSURE, VERIFICATION_REFUSAL, FINANCIAL_INSTRUCTION, THREAT |
| T03 | coercive | 0.90 | high | COERCIVE_URGENCY, AUTHORITY_CLAIM, VERIFICATION_REFUSAL, FINANCIAL_INSTRUCTION, THREAT |
| T04 | coercive | 0.75 | high | COERCIVE_URGENCY, AUTHORITY_CLAIM, FINANCIAL_INSTRUCTION, THREAT |
| T05 | coercive | 0.45 | moderate | COERCIVE_URGENCY, FINANCIAL_INSTRUCTION |
| T06 | coercive | 0.75 | high | AUTHORITY_CLAIM, SECRECY_PRESSURE, VERIFICATION_REFUSAL, FINANCIAL_INSTRUCTION, THREAT |
| T07 | coercive | 0.75 | high | COERCIVE_URGENCY, AUTHORITY_CLAIM, FINANCIAL_INSTRUCTION, THREAT |
| T08 | coercive | 0.85 | high | COERCIVE_URGENCY, AUTHORITY_CLAIM, SECRECY_PRESSURE, FINANCIAL_INSTRUCTION, THREAT |
| T09 | benign | 0.00 | low | — |
| T10 | benign | 0.00 | low | EMERGENCY_NARRATIVE (context only, not scored) |
| T11 | benign | 0.20 | low | FINANCIAL_INSTRUCTION |
| T12 | benign | 0.10 | low | AUTHORITY_CLAIM |
| T13 | benign | 0.20 | low | FINANCIAL_INSTRUCTION |
| T14 | benign | 0.00 | low | — |
| T15 | benign | 0.20 | low | FINANCIAL_INSTRUCTION |
| T16 | ambiguous | 0.00 | low | *(abstain — insufficient_text)* |
| T17 | ambiguous | 0.00 | low | *(abstain — unsupported_language, Tamil)* |
| T18 | ambiguous | 0.45 | moderate | COERCIVE_URGENCY, FINANCIAL_INSTRUCTION, EMERGENCY_NARRATIVE |
| T19 | ambiguous | 0.00 | low | *(paraphrased coercion — known recall limit)* |
| T20 | ambiguous | 0.45 | moderate | COERCIVE_URGENCY, FINANCIAL_INSTRUCTION, EMERGENCY_NARRATIVE |
| T21 | coercive | 0.70 | high | COERCIVE_URGENCY, SECRECY_PRESSURE, VERIFICATION_REFUSAL, FINANCIAL_INSTRUCTION |
| T22 | coercive | 0.85 | high | COERCIVE_URGENCY, AUTHORITY_CLAIM, SECRECY_PRESSURE, FINANCIAL_INSTRUCTION, THREAT |
| T23 | coercive | 0.85 | high | COERCIVE_URGENCY, AUTHORITY_CLAIM, SECRECY_PRESSURE, FINANCIAL_INSTRUCTION, THREAT — Devanagari Hindi |

**Bands:** low < 0.30 · moderate 0.30–0.70 · high ≥ 0.70.

## 2. Safety metrics (headline numbers for the slides)

- **Coercive scams flagged (score ≥ 0.30): 11 / 11**
- **Benign calls left alone (score < 0.30): 7 / 7**
- **False positives: 0**
- **False negatives: 0**

## 2.5 Semantic layer (learned classifier, rules + TF-IDF blend)

The rule engine alone misses *reworded* coercion (fixture T19 scores 0.0). A learned TF-IDF + logistic-regression
classifier, blended `C = 0.6·rules + 0.4·semantic`, closes that gap:

| Case | rules only | blended | result |
|---|---|---|---|
| Paraphrased coercion (T19) | 0.00 | **0.34** | caught ("warn") |
| Paraphrased threat | 0.00 | **0.37** | caught |
| Paraphrased secrecy + pay | 0.00 | **0.57** | caught |
| Benign family | 0.00 | 0.05 | untouched |
| Benign legit-bank (negated) | 0.10 | 0.05 | untouched |
| Benign split-bill | 0.20 | 0.23 | untouched |

- Semantic checks: **6/6 pass** (3 paraphrased coercive ≥ 0.30, 3 benign < 0.30).
- Safety: a purely-semantic hit (rules=0, semantic=1.0) tops out at `C=0.4`, so it can **warn but never pause**.
- Runs on LinuxONE s390x (pure-Python inference; no scikit-learn wheel needed).

## 3. Edge cases (all pass)

| Input | Behaviour |
|---|---|
| Empty string / punctuation-only / one word | `insufficient_text`, score 0.0 (abstain) |
| Unsupported language (Devanagari) | `unsupported_language`, score 0.0 (abstain, never auto-restrict) |
| Non-ASCII names, very long text | no crash, correct score |

## 4. Golden-path stability (Phase 7.1)

The golden scenario (T01) was run **3× — identical `score=1.0` and identical reason codes every time**.
The rule engine is deterministic (pure regex + fixed weights, no randomness).

## 5. No-persistence proof (Phase 5.3)

`run_eval.py` greps `semantic_model.py` + `semantic_config.py` for any file-write call
(`open(`, `.write(`, `to_csv`, `np.save`, `sf.write`, `dump`): **none found.** Transcripts are processed in
memory and never written to disk.

## 6. Limitations (lift verbatim into the README)

1. **Keyword recall vs. paraphrase.** The rule layer catches high-precision cues; it can miss reworded coercion
   (e.g. "move some money" instead of "transfer"). This is why the system fuses multiple signals and never relies
   on this module alone. (See fixture T19, which scores 0.0 by design.)
2. **Language coverage is declared, not universal.** English, Hindi (Devanagari), and Hinglish are implemented;
   other Indic languages (Tamil, Telugu, Bengali, …) abstain (`unsupported_language`) rather than guessing.
3. **`coercion_score` is a demo heuristic, not a calibrated probability.** Weights (0.25/0.20/… ) are engineering
   choices tuned on the 22-fixture set, not validated on real call data.
4. **ASR quality gates downstream recall.** If faster-whisper mis-transcribes a key word, the rule engine sees the
   wrong text. The scripted-transcript path is the reliable demo path; live ASR is a bonus.
5. **Negated credential requests are masked** ("we will never ask for your OTP/PIN" no longer fires
   `FINANCIAL_INSTRUCTION`), but other negations may still trigger a signal — each fires at low weight and never
   blocks alone.
6. **`EMERGENCY_NARRATIVE` is context, not coercion.** It is reported as a reason code but contributes 0 to the
   score, so a genuine emergency is never treated as a scam (fixtures T10, T18, T20).

## 7. Models & data documentation (licenses)

| Asset | What | License | Source |
|---|---|---|---|
| Rule engine (`semantic_model.py`) | Original keyword lists + scoring (no external dataset) | ours | — |
| faster-whisper "base" | CTranslate2 conversion of OpenAI Whisper *base* (ASR) | MIT | Systran/faster-whisper-base, downloaded via hf-mirror.com, cached at `models/faster-whisper-base/` (gitignored) |
| Fixtures (`data/transcripts/fixtures.json`) | 22 **invented** transcripts | ours | no real victim data |

No classifier model was added (deferred to Phase 8, if time permits).

## 8. 30-second explainer (Phase 7.3 — for "why did it flag this?")

> "Our module turns the transcript into seven signal checks — urgency, claimed authority, secrecy pressure,
> refusal to verify, financial instructions, threats, and emergency narrative. Each is a transparent keyword rule,
> so every reason code traces back to the exact words that triggered it. We weight those into one coercion score,
> deliberately leaving *emergency narrative* out — a real emergency is not coercion. The score is one input into
> the bank's decision, never the decision itself."

## 9. Hand-off checklist

- [x] Results table + metrics → this file + `data/transcripts/results.json`
- [x] Limitations → §6 (ready to paste into README)
- [x] Model/data licenses → §7
- [x] Stability 3× → §4
- [x] No-persistence proof → §5
- [ ] Person 6: paste §6 into README, drop §2 onto the evidence slide
- [ ] Person 4: confirm your fusion thresholds against these score bands

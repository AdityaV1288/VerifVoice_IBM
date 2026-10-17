# Semantic Layer Plan — add "understanding" to Person 2's coercion detector

> **Status:** PLAN ONLY (no code changed). Owner: Person 2.
> **Goal:** close the paraphrase/negation gap (e.g. fixture T19 scores 0.0 today) by adding a *learned*
> semantic signal **alongside** the rule engine — not replacing it.

---

## 1. The gap we're closing

The rule engine catches high-precision cues (`transfer`, `don't tell anyone`) but scores **0.0** on reworded
coercion like *"I need you to move some money… it has to happen fast"*. A learned classifier generalizes
across paraphrases, negation, and near-synonyms — the thing keyword matching fundamentally can't do.

## 2. Design principle (non-negotiable)

**The semantic layer is a SECOND signal, fused with the rules — never a replacement.**

| | Rules (keep) | Semantic classifier (add) |
|---|---|---|
| Role | explainable backbone | recall boost on paraphrase/negation |
| Determinism | ✅ | ✅ (inference is deterministic) |
| Explainability | ✅ traceable reason codes | 🟡 outputs a score + a reason code, but not per-word trace |
| Portability | pure Python → LinuxONE | Tier 1 sklearn → LinuxONE; Tier 2 ONNX → laptop |
| Never blocks alone | ✅ | ✅ (weighted into C at 0.3, and C is only 0.5 of R → max 0.15 of R) |

## 3. Two tiers (build Tier 1 first)

| | Tier 1 — TF-IDF + logistic regression | Tier 2 — small transformer → ONNX |
|---|---|---|
| Model | `TfidfVectorizer(1-3 grams)` + `LogisticRegression` | fine-tune `prajjwal1/bert-tiny` (4.4M) or `distilbert-base-uncased` (66M) |
| Deps | scikit-learn (already installed) | transformers + torch (already installed) + onnx |
| Effort | ~30–60 min | ~2–3 h (GPU nice-to-have; CPU fine for tiny) |
| Runs on | LinuxONE ✅ | laptop (ONNX Runtime; s390x wheels uncertain — same as AASIST) |
| Catches | common paraphrases, simple negation | deep paraphrase, negation, context |

**Ship Tier 1 no matter what; Tier 2 only if time remains.**

## 4. Interface contract (`semantic_classifier.py`)

Mirror `semantic_model.py` so the pipeline treats both the same way:

```python
def classify(text: str, lang: str = "en") -> dict:
    """Return a learned coercion signal (complement to the rule engine)."""
    # {
    #   "semantic_score": float 0..1,          # learned P(coercive)
    #   "reason_codes": ["SEMANTIC_COERCION"] if score >= threshold else [],
    #   "status": "analyzed" | "insufficient_text" | "unsupported_language" | "model_unavailable",
    #   "model_name": "tfidf_logreg" | "bert_tiny_onnx",
    #   "threshold": 0.5,                       # configurable
    # }
```

- If the model isn't loaded/downloaded → `status: "model_unavailable"`, `semantic_score: None` (honest, like
  P1's default). The rule engine still works.
- `SEMANTIC_COERCION` is the single new reason code added to the shared vocabulary.

## 5. Fusion (how it enters the pipeline)

Keep the outer fusion untouched; blend the two coercion signals inside `pipeline.analyze_call()`:

```
coercion_score = 0.7 * rules_score  +  0.3 * semantic_score       # configurable weights
R = 0.5 * coercion_score + 0.3 * spoof_score + 0.2 * new_beneficiary   # unchanged
```

- If `semantic_score` is `None` (model unavailable) → renormalize onto the rules (like we do for spoof).
- Merge reason codes: rules codes + `SEMANTIC_COERCION` (when it fires).
- **Safety:** semantic alone contributes at most `0.3 × 0.5 = 0.15` to `R`, so it can never trigger a block
  by itself — it only nudges borderline cases across a threshold.

## 6. Dataset strategy (where the data comes from)

**Reality:** there is no large public "coercive phone-scam transcript" corpus. So the data is built, not found.

### 6.1 Primary — LLM-generated synthetic (best ROI)
- Generate **~1000–2000** labelled transcripts via an LLM, covering:
  - **coercive:** urgency + authority, secrecy + financial, threat + financial, romance, tech-support, OTP,
    lottery, kidnapping, code-switched Hinglish, **paraphrased** versions (no trigger words), **negated**
    versions ("don't transfer" vs "transfer").
  - **benign:** family chat, genuine emergency (no money), AAC/TTS user, legit bank call, splitting a bill,
    appointment reminders, known-merchant payments, **negated** ("we never ask for your OTP").
- Prompt pattern: *"write a short phone-call transcript where a scammer pressures someone to pay, phrased so it
  contains none of these words: transfer, UPI, OTP, urgent…"* — that forces paraphrase.
- Use our 23 fixtures as the seed; ask the LLM to produce variants.

### 6.2 Secondary — proxy datasets (HuggingFace Hub)
Search `scam`, `fraud`, `phishing`, `threat`, `urgent` on the Hub. Useful proxies:
- **Jigsaw Toxic Comment / `civil_comments`** — threat/abuse language → boosts the *threat* signal.
- **SMS Spam Collection (UCI)** — cheap spam proxy.
- **Phishing email datasets** (e.g. `ealvaradob/phishing-dataset`) — financial-request language.
- ⚠️ These are *starters*, not phone-call coercion; use them for the threat/financial classes only.

### 6.3 Audio (already covered by P1)
- ASVspoof 2019 LA / 2021 DF, MLAAD — these are *acoustic*, not transcript; not needed for this layer.

### 6.4 Format + honesty
- `data/semantic/train.jsonl`, `val.jsonl`, `test.jsonl`: `{"text": "...", "label": 0|1}`.
- All synthetic + proxy → disclose in README: *"trained on generated + proxy data; scores are a demo
  heuristic, not calibrated probability."*

## 7. Training (`train_semantic_classifier.py`)

- **Tier 1:** TF-IDF → LogisticRegression; grid-search `C`; report **precision/recall/AUC** on the held-out
  test split. Save via `joblib` (or `skl2onnx` for ONNX).
- **Tier 2 (stretch):** fine-tune `bert-tiny` (sequence classification, 2 labels) on the same data, export to
  ONNX with `transformers` + `onnx`. Runs on laptop; keep Tier 1 as the LinuxONE fallback.
- Both: log model name/version/threshold in `semantic_classifier.py`, like P1's `ModelInfo`.

## 8. Integration steps (ordered)

1. Generate + save the dataset (Phase A).
2. Train Tier 1 + save model + threshold (Phase B).
3. Write `semantic_classifier.py` with the §4 contract (Phase C).
4. Blend in `pipeline.py` (§5) + add `SEMANTIC_COERCION` to the reason-code config (Phase C).
5. Extend `fixtures.json` with ~10 paraphrase/negation cases; run `run_eval.py` (Phase D).
6. Re-run the full API + payment loop; confirm T19-style cases now score > 0 (Phase D).
7. (Stretch) Tier 2 → ONNX (Phase E).
8. Document limitations + calibration caveat in `EVALUATION_RESULTS.md` (Phase F).

## 9. Where it runs

| Component | Location |
|---|---|
| TF-IDF + logreg | **LinuxONE** (sklearn is portable) — keeps "learned AI on LinuxONE" true |
| bert-tiny ONNX | laptop (onnxruntime; s390x wheels uncertain, same as AASIST) |

## 10. Acceptance criteria

- [ ] `semantic_classifier.classify()` returns the §4 schema; `model_unavailable` when the model is absent.
- [ ] Paraphrased-coercion fixtures (e.g. T19) score **> 0.30** after blending.
- [ ] Benign fixtures still score **< 0.30** (no new false positives).
- [ ] `SEMANTIC_COERCION` appears in reason codes only when the classifier fires.
- [ ] Semantic signal can't block alone (max 0.15 contribution to R).
- [ ] Reproducible: `train_semantic_classifier.py` regenerates the same model from the same data.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Synthetic data biases the model | keep the 23-fixture hold-out set as the honest eval; disclose synthetic origin |
| Classifier fires false positives on benign | tune threshold on val; weight it low (0.3) in the blend |
| Hindi/Hinglish not covered by bert-tiny | Tier 1 TF-IDF handles Hinglish via char n-grams; add generated Hinglish data |
| Explaining a black-box signal to judges | emit `SEMANTIC_COERCION` + say "a learned model supports the rule signal; rules stay the audit trail" |

---

## Task checklist (with time budget)

- [ ] **A. Dataset** (60–90 min): LLM-generate ~1–2k labelled transcripts + paraphrases/negations; save train/val/test.
- [ ] **B. Tier 1 train** (30–45 min): TF-IDF + logreg; tune; save model + threshold + metrics.
- [ ] **C. Wire in** (30 min): `semantic_classifier.py` + pipeline blend + reason code.
- [ ] **D. Evaluate** (30 min): extend fixtures, run `run_eval.py`, confirm T19 fixed + no false positives.
- [ ] **E. (Stretch) Tier 2** (2–3 h): bert-tiny fine-tune → ONNX.
- [ ] **F. Document** (15 min): limitations + provenance in README/EVALUATION_RESULTS.

# VerifVoice

**Privacy-preserving real-time AI that helps banks stop voice-coercion fraud.**

> *Banks authenticate who is paying. VerifVoice helps verify whether the decision to pay is being coerced.*

VerifVoice detects coercive voice scams while they happen and passes a privacy-preserving, expiring risk
signal to a bank so it can step up authentication before money leaves. Raw audio is never persisted, synthetic
speech is never treated as proof of fraud, and the user retains a safe override.

## Modules (by owner)

| Module | Owner | Purpose |
|---|---|---|
| `main.py` | Person 3 | FastAPI backend (`/health`, `/analyze`, `/analyze/upload`, `/payment/decision`, `/metrics`) — runs on LinuxONE |
| `pipeline.py` | Person 3 | Integration: runs Person 1 + Person 2 together (`analyze_call`) |
| `token_store.py` | Person 3 | In-memory risk tokens (TTL 120 s) |
| `risk_engine.py` | P4 (fallback by P3) | Risk fusion `R = 0.5C + 0.3(S·C) + 0.2N` (spoof amplifies coercion, never acts alone) + policy ladder |
| `semantic_model.py` | Person 2 | Speech & coercion detection (transcript → coercion score + reason codes) |
| `semantic_config.py` | Person 2 | Signals, weights, reason codes, gates |
| `semantic_classifier.py` | Person 2 | Learned coercion signal (TF-IDF + logistic regression, pure-Python inference) |
| `train_semantic_classifier.py` / `export_semantic_model.py` | Person 2 | Train + export the semantic classifier (portable JSON) |
| `scripts/generate_semantic_dataset.py` | Person 2 | Synthetic labelled-transcript generator |
| `audio_authenticity/` | Person 1 | Audio authenticity package (`analyze_audio` → AASIST spoof score) |
| `semantic_tests.py` | Person 2 | Smoke tests (6 core behaviours) |
| `run_eval.py` | Person 2 | Formal evaluation (23 fixtures + semantic checks + stability) |
| `vm_test.py` / `vm_evidence.py` / `vm_semantic_test.py` | Person 3 | LinuxONE deployment smoke / evidence scripts |

## Setup

```powershell
. .\activate.ps1           # activates venv + ffmpeg on PATH
python -c "import semantic_model; print('ok')"
```

See `ENVIRONMENT.md` for the full setup guide.

## Quick checks

```powershell
python semantic_tests.py    # 6 smoke tests
python run_eval.py          # 23-fixture eval + semantic-layer checks + stability
python demo.py              # end-to-end: golden scam -> PAUSE, AAC user -> OBSERVE
```

## Status

**Full stack integrated + deployed** (tag `v1.0-integration` + semantic layer). The complete
`analyze → token → payment decision` loop runs on **LinuxONE s390x**, with three AI signals:

| Signal | Where it runs |
|---|---|
| Rule engine (coercion) | LinuxONE s390x |
| Semantic classifier (learned TF-IDF+logreg) | LinuxONE s390x (pure Python) |
| AASIST anti-spoof (ONNX) | laptop (no ONNX Runtime s390x wheel) |

- Person 2 rule engine: 23/23 fixtures pass (English + Hindi + Hinglish), 0 false positives.
- Semantic layer: paraphrased coercion now caught (was a known keyword gap).
- Open for **P4's real risk engine** — drop-in, no rearchitecture.

## Models & limitations (honest disclosure)

- **Semantic classifier** (`models/semantic_classifier.json`, 1481 features): trained on **synthetic + generated**
  transcripts (not real call data), so its score is a *demo heuristic, not a calibrated probability*.
- **AASIST** (`models/aasist.onnx`): MIT-licensed pretrained ONNX (ASVspoof 2019 LA); score is an uncalibrated
  softmax; laptop-only.
- **Rule engine**: transparent keyword rules; misses reworded coercion (that gap is why the semantic layer exists).
- **Simulated**: the bank action is a *proposed* policy decision — no real transfer is touched.

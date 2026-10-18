"""pipeline.py — combined VerifVoice analysis (Person 1 + Person 2 + semantic layer).

Signal modules:
  * `semantic_model`         -> rule-based coercion score + reason codes (C_rules)
  * `semantic_classifier`    -> learned coercion score (C_semantic, TF-IDF+logreg)
  * `audio_authenticity`     -> spoof score (S)

Coercion is a blend:  C = 0.6 * C_rules + 0.4 * C_semantic   (renormalizes onto
the rules when the learned model is unavailable). P4's fusion then does:
R = 0.5*C + 0.3*S + 0.2*N. A purely-semantic hit (rules=0, semantic=1.0) reaches
C=0.4 at most, so it can "warn" but never "pause" a payment on its own.
"""

import uuid

import semantic_classifier
import semantic_model
from audio_authenticity import analyze_audio

RULES_WEIGHT = 0.6
SEMANTIC_WEIGHT = 0.4


def analyze_call(source: str, input_type: str = "text", lang: str = "en") -> dict:
    """Run all signal modules and return a unified, JSON-ready result."""
    result = {
        "request_id": str(uuid.uuid4()),
        "input_type": input_type,
        "lang": lang,
        # --- Person 2: rule-based coercion ---
        "coercion_score": None,
        "coercion_status": None,
        "reason_codes": [],
        "signals": {},
        # --- semantic layer: learned coercion ---
        "semantic_score": None,
        "semantic_status": None,
        "semantic_model": None,
        # --- Person 1: audio authenticity ---
        "spoof_score": None,
        "spoof_status": None,
        "spoof_classification": None,
        "quality_score": None,
        "model_name": None,
        "spoof_reasons": [],
        "spoof_warnings": [],
        # --- demo-only echo (never persisted, never in the token) ---
        "transcript": None,
    }

    # 1) Resolve the transcript text (transcribe if audio).
    text = semantic_model.transcribe(source) if input_type == "audio" else source
    result["transcript"] = text

    # 2) Run the two coercion signals on the text.
    if text:
        rules = semantic_model.analyze_transcript(text, lang=lang)
        sem = semantic_classifier.classify(text, lang=lang)
    else:
        # Transcription failed -> abstain (weak signal, never "fraud").
        rules = {"coercion_score": 0.0, "reason_codes": [], "signals": {},
                 "status": "insufficient_text", "lang": lang}
        sem = {"semantic_score": None, "reason_codes": [], "status": "insufficient_text"}

    result["coercion_status"] = rules["status"]
    result["signals"] = rules["signals"]
    result["semantic_score"] = sem["semantic_score"]
    result["semantic_status"] = sem["status"]
    result["semantic_model"] = sem.get("model_name")

    # 3) Blend coercion: 0.7 rules + 0.3 semantic (renormalize if semantic absent).
    if rules["status"] == "analyzed":
        if sem["semantic_score"] is not None:
            c = RULES_WEIGHT * rules["coercion_score"] + SEMANTIC_WEIGHT * sem["semantic_score"]
        else:
            c = rules["coercion_score"]  # learned model unavailable -> rules alone
    else:
        c = rules["coercion_score"]  # abstention -> keep 0.0
    result["coercion_score"] = round(min(1.0, c), 4)

    # 4) Merge reason codes (dedupe, preserve order).
    result["reason_codes"] = list(dict.fromkeys(
        rules.get("reason_codes", []) + sem.get("reason_codes", [])
    ))

    # 5) Person 1 — audio authenticity (audio only; independent of transcription).
    if input_type == "audio":
        try:
            audio = analyze_audio(source)
            d = audio.to_dict()
            result["spoof_score"] = d["authenticity_score"]
            result["spoof_status"] = d["status"]
            result["spoof_classification"] = d["classification"]
            result["quality_score"] = d["quality_score"]
            result["model_name"] = d["model_name"]
            result["spoof_reasons"] = list(d["reasons"])
            result["spoof_warnings"] = list(d["warnings"])
        except Exception as exc:  # never let P1 break the coercion path
            result["spoof_status"] = "error"
            result["spoof_warnings"] = [str(exc)]

    return result


# --------------------------------------------------------------------------- #
# Reference only — Person 4 owns fusion + policy.
#   R = 0.5 * coercion_score + 0.3 * spoof_score + 0.2 * new_beneficiary
# When spoof_score is None (model unavailable), P4 degrades gracefully
# (renormalize onto coercion + flag SPOOF_UNAVAILABLE).
# --------------------------------------------------------------------------- #

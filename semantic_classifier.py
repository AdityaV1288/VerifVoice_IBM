"""semantic_classifier.py — learned coercion signal (Tier 1: TF-IDF + logistic regression).

Complements (does NOT replace) semantic_model's rule engine. Returns a learned
P(coercive) plus the reason code SEMANTIC_COERCION when it fires. If the model
file is missing, returns model_unavailable and the rule engine still works.

Interface mirrors semantic_model so pipeline.py can treat both the same way.
"""

import os

import semantic_config as config

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "semantic_classifier.joblib",
)
MODEL_NAME = "tfidf_logreg"
DEFAULT_THRESHOLD = 0.5

_model = None  # lazy-loaded {"vectorizer", "clf", "threshold"}


def _get_model():
    global _model
    if _model is None:
        import joblib  # local import: sklearn/joblib are optional at module load
        _model = joblib.load(MODEL_PATH)
    return _model


def classify(text: str, lang: str = "en") -> dict:
    """Return the learned coercion signal (§4 schema of SEMANTIC_LAYER_PLAN.md)."""
    if not isinstance(text, str):
        text = str(text)

    # Length gate — too little text can't be classified meaningfully.
    if len(text.split()) < config.MIN_TOKENS:
        return _result(None, [], "insufficient_text")

    try:
        model = _get_model()
    except Exception:
        # Model not trained/downloaded -> honest unavailable, never fabricate.
        return _result(None, [], "model_unavailable")

    vectorizer, clf, threshold = model["vectorizer"], model["clf"], model["threshold"]
    score = float(clf.predict_proba(vectorizer.transform([text]))[0][1])
    reason_codes = ["SEMANTIC_COERCION"] if score >= threshold else []

    return {
        "semantic_score": round(score, 4),
        "reason_codes": reason_codes,
        "status": "analyzed",
        "model_name": MODEL_NAME,
        "threshold": threshold,
    }


def _result(score, codes, status):
    return {
        "semantic_score": score,
        "reason_codes": codes,
        "status": status,
        "model_name": MODEL_NAME,
        "threshold": DEFAULT_THRESHOLD,
    }

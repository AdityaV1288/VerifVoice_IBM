"""semantic_classifier.py — learned coercion signal (TF-IDF + logistic regression).

Portable inference: the trained model's parameters live in
models/semantic_classifier.json (exported by export_semantic_model.py), and the
TF-IDF transform + logistic regression are re-implemented here in pure Python
(stdlib only) so the classifier runs on LinuxONE s390x where scikit-learn has no
wheel. Returns a learned P(coercive) plus SEMANTIC_COERCION when it fires.

If the JSON is missing, returns model_unavailable — the rule engine still works.
"""

import json
import math
import os
import re

import semantic_config as config

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "semantic_classifier.json",
)
MODEL_NAME = "tfidf_logreg"
DEFAULT_THRESHOLD = 0.5

# sklearn's default word tokenizer: lowercase, then 2+ alphanumeric-word tokens.
_TOKEN_RE = re.compile(r"\b\w\w+\b")

_model = None  # lazy-loaded dict


def _get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, encoding="utf-8") as fh:
            _model = json.load(fh)
    return _model


def _tfidf_transform(text: str) -> dict:
    """Replicate sklearn TfidfVectorizer (sublinear_tf, l2 norm) for one text.

    Returns {feature_index: tfidf_weight} (already L2-normalized).
    """
    model = _get_model()
    vocab = model["vocabulary"]
    idf = model["idf"]
    ngram_max = model["ngram_max"]

    tokens = _TOKEN_RE.findall(text.lower())

    # Count n-grams (1..ngram_max) that are in the vocabulary.
    counts = {}
    for n in range(1, ngram_max + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i + n])
            if gram in vocab:
                counts[gram] = counts.get(gram, 0) + 1

    vec = {}
    norm_sq = 0.0
    for gram, count in counts.items():
        idx = vocab[gram]
        tf = (1.0 + math.log(count)) if model["sublinear_tf"] else float(count)
        weight = tf * idf[idx]
        vec[idx] = weight
        norm_sq += weight * weight

    norm = math.sqrt(norm_sq) if norm_sq > 0.0 else 1.0
    return {idx: w / norm for idx, w in vec.items()}


def _predict_proba(text: str) -> float:
    """sigmoid(x . coef + intercept) — the binary logistic-regression decision."""
    model = _get_model()
    coef = model["coef"]
    z = model["intercept"]
    for idx, weight in _tfidf_transform(text).items():
        z += weight * coef[idx]
    # stable sigmoid
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def classify(text: str, lang: str = "en") -> dict:
    """Return the learned coercion signal (SEMANTIC_LAYER_PLAN.md §4 schema)."""
    if not isinstance(text, str):
        text = str(text)

    if len(text.split()) < config.MIN_TOKENS:
        return _result(None, [], "insufficient_text")

    try:
        model = _get_model()
        score = _predict_proba(text)
    except Exception:
        return _result(None, [], "model_unavailable")

    score = round(score, 4)
    threshold = model.get("threshold", DEFAULT_THRESHOLD)
    reason_codes = ["SEMANTIC_COERCION"] if score >= threshold else []

    return {
        "semantic_score": score,
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

"""export_semantic_model.py — extract trained TF-IDF+logreg params to a portable JSON.

sklearn has no s390x wheel, so inference on LinuxONE uses a pure-Python
re-implementation that loads these exported parameters. This script runs on the
laptop (which has sklearn) and writes models/semantic_classifier.json.

Run:  python export_semantic_model.py
"""

import json
import os

import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "models", "semantic_classifier.joblib")
DST = os.path.join(HERE, "models", "semantic_classifier.json")


def main() -> int:
    model = joblib.load(SRC)
    vectorizer, clf, threshold = model["vectorizer"], model["clf"], model["threshold"]

    data = {
        "vocabulary": dict(vectorizer.vocabulary_),   # {ngram: index}
        "idf": [float(x) for x in vectorizer.idf_],   # idf per index
        "coef": [float(x) for x in clf.coef_[0]],     # coefficient per index
        "intercept": float(clf.intercept_[0]),
        "threshold": float(threshold),
        "ngram_max": int(vectorizer.ngram_range[1]),
        "sublinear_tf": bool(vectorizer.sublinear_tf),
    }
    with open(DST, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    print(f"Exported {len(data['vocabulary'])} features -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

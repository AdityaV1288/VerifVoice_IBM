"""train_semantic_classifier.py — Tier 1: TF-IDF + logistic regression.

Trains a binary coercive/benign classifier on the synthetic dataset, reports
metrics, and saves the model + vectorizer + threshold for semantic_classifier.py.

Run:  python train_semantic_classifier.py
"""

import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "semantic")
MODEL_PATH = os.path.join(HERE, "models", "semantic_classifier.joblib")
THRESHOLD = 0.5


def load(split: str):
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    texts, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(r["label"])
    return texts, labels


def main() -> int:
    train_x, train_y = load("train")
    val_x, val_y = load("val")
    test_x, test_y = load("test")

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_x)
    X_val = vectorizer.transform(val_x)
    X_test = vectorizer.transform(test_x)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, train_y)

    # Probabilities for class 1 (coercive).
    val_p = clf.predict_proba(X_val)[:, 1]
    test_p = clf.predict_proba(X_test)[:, 1]

    val_pred = (val_p >= THRESHOLD).astype(int)
    test_pred = (test_p >= THRESHOLD).astype(int)

    print("=== validation (threshold %.2f) ===" % THRESHOLD)
    print(f"accuracy  {accuracy_score(val_y, val_pred):.3f}")
    print(f"precision {precision_score(val_y, val_pred):.3f}")
    print(f"recall    {recall_score(val_y, val_pred):.3f}")
    print(f"f1        {f1_score(val_y, val_pred):.3f}")
    print(f"auc       {roc_auc_score(val_y, val_p):.3f}")
    print("=== test ===")
    print(f"accuracy  {accuracy_score(test_y, test_pred):.3f}")
    print(f"precision {precision_score(test_y, test_pred):.3f}")
    print(f"recall    {recall_score(test_y, test_pred):.3f}")
    print(f"f1        {f1_score(test_y, test_pred):.3f}")
    print(f"auc       {roc_auc_score(test_y, test_p):.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "clf": clf, "threshold": THRESHOLD}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

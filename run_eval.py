"""run_eval.py — Person 2 formal evaluation (Phase 5).

Runs every labelled fixture through semantic_model.analyze() and prints a results
table for Person 6, plus edge-case and no-persistence checks.

Run:  python run_eval.py
"""

import json
import os
import re

import semantic_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "data", "transcripts", "fixtures.json")

BANDS = {"low": (0.0, 0.30), "moderate": (0.30, 0.70), "high": (0.70, 1.01)}


def classify(score: float) -> str:
    for band, (lo, hi) in BANDS.items():
        if lo <= score < hi:
            return band
    return "?"


def run_fixtures(fixtures):
    print("=" * 108)
    print(f"{'ID':<5}{'Label':<10}{'Score':<7}{'Band':<9}{'Exp':<9}{'Result':<7}Reason codes")
    print("-" * 108)
    passed = failed = deferred = 0
    rows = []
    for f in fixtures:
        r = sm.analyze(f["text"])
        score = r["coercion_score"]
        band = classify(score)
        exp = f["expected_band"]
        if exp == "deferred":
            res = "DEFER"
            deferred += 1
        else:
            ok = band == exp
            res = "PASS" if ok else "FAIL"
            passed += ok
            failed += (not ok)
        codes = ", ".join(r["reason_codes"]) or "-"
        print(f"{f['id']:<5}{f['label']:<10}{score:<7.2f}{band:<9}{exp:<9}{res:<7}{codes}")
        rows.append({"id": f["id"], "name": f["name"], "label": f["label"],
                     "score": score, "band": band, "expected": exp, "status": r["status"]})
    print("-" * 108)
    print(f"RESULT: {passed} passed, {failed} failed, {deferred} deferred (Phase 8)")
    return rows


def summarize(rows):
    coercive = [r for r in rows if r["label"] == "coercive"]
    benign = [r for r in rows if r["label"] == "benign"]

    # Safety metrics (the ones that matter for the README/slides).
    false_neg = [r for r in coercive if r["expected"] != "deferred" and r["score"] < 0.30]
    false_pos = [r for r in benign if r["score"] >= 0.30]
    print("\n--- Safety metrics ---")
    print(f"coercive caught (score >= 0.30): "
          f"{sum(1 for r in coercive if r['score'] >= 0.30)}/{len(coercive)}")
    print(f"benign left alone (score < 0.30):  "
          f"{sum(1 for r in benign if r['score'] < 0.30)}/{len(benign)}")
    if false_neg:
        print("false negatives (scam scored low):", [r["id"] for r in false_neg])
    if false_pos:
        print("false positives (benign scored >= 0.30):", [r["id"] for r in false_pos])
    else:
        print("false positives: none")


def run_edge_cases():
    print("\n--- Edge cases ---")
    cases = [
        ("empty string", ""),
        ("punctuation only", "!!! ??? ..."),
        ("one word", "Transfer!"),
        ("non-ASCII name", "Rahul — call me back."),
        ("very long text", ("Please stay calm and listen carefully. " * 60)),
    ]
    for name, text in cases:
        r = sm.analyze(text)
        print(f"{name:<18} score={r['coercion_score']:<5} status={r['status']}")


def check_no_persistence():
    print("\n--- No-persistence check ---")
    # The module must never write audio/transcripts to disk.
    write_patterns = re.compile(r"\b(open\(|\.write\(|to_csv|to_json|np\.save|sf\.write|dump)")
    offenders = []
    for fname in ("semantic_model.py", "semantic_config.py"):
        with open(os.path.join(HERE, fname), encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if write_patterns.search(line):
                    offenders.append(f"{fname}:{i}: {line.strip()}")
    if offenders:
        print("WRITE CALLS FOUND:")
        for o in offenders:
            print("  ", o)
    else:
        print("PASS - no file-write calls in semantic_model.py / semantic_config.py.")


def check_stability(fixtures):
    """Golden path must produce identical results across 3 runs (determinism)."""
    print("\n--- Golden-path stability (3 identical runs) ---")
    golden = next(f for f in fixtures if f["id"] == "T01")["text"]
    outputs = [sm.analyze(golden) for _ in range(3)]
    scores = [o["coercion_score"] for o in outputs]
    codes = [tuple(o["reason_codes"]) for o in outputs]
    stable = len(set(scores)) == 1 and len(set(codes)) == 1
    print(f"score={scores[0]} codes={list(codes[0])} | stable={stable}")
    return stable


def save_results(rows):
    """Write machine-readable results for Person 6 (and the README)."""
    out = os.path.join(HERE, "data", "transcripts", "results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved results -> {out}")


def run_semantic_checks():
    """Verify the rule + learned-classifier blend on paraphrase/negation cases.

    Paraphrased coercion (no trigger words) must now reach >= 0.30, while
    benign/negated text must stay < 0.30.
    """
    import pipeline
    print("\n--- Semantic layer checks (rules + TF-IDF classifier blend) ---")
    cases = [
        ("paraphrased coercion (T19)", "I need you to move some money to a different account for me. It has to happen fast, and I would rather you did not mention it to anyone just yet.", True),
        ("paraphrased threat", "Your service will be suspended unless you clear the outstanding amount this minute.", True),
        ("paraphrased secrecy+pay", "I would like you to keep this private and settle the balance right away.", True),
        ("benign family", "Hi beta, how are you? How was school today? Did you have lunch?", False),
        ("benign legit bank", "This is your banks security team. We will never ask for your OTP or PIN. Call the number on your card.", False),
        ("benign split bill", "That meal came to twelve hundred, send me your half whenever, no rush.", False),
    ]
    passed = failed = 0
    for name, text, is_coercive in cases:
        r = pipeline.analyze_call(text)
        score = r["coercion_score"]
        ok = (score >= 0.30) if is_coercive else (score < 0.30)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: coercion={score} semantic={r['semantic_score']}")
        passed += ok
        failed += (not ok)
    print(f"semantic checks: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    data = json.load(open(FIXTURES, encoding="utf-8"))
    rows = run_fixtures(data["fixtures"])
    summarize(rows)
    run_edge_cases()
    check_no_persistence()
    check_stability(data["fixtures"])
    run_semantic_checks()
    save_results(rows)

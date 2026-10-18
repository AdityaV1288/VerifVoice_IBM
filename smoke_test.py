"""smoke_test.py — integration smoke test for Person 4's risk engine (day-one formula).

Two parts:
  A) Pure-function scenario matrix through ``risk_adapter.decide`` (deterministic).
  B) End-to-end API run through ``main.app`` (analyze -> token -> payment/decision).

Run:  python smoke_test.py
Exit code is non-zero if any assertion fails.
"""

from __future__ import annotations

import sys

import risk_adapter
import token_store

DECISION_VOCAB = {"observe", "warn", "verify", "pause"}


def check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise SystemExit(f"FAILED: {label}")


# ---------------------------------------------------------------------------
# A) Pure-function scenario matrix (primer §13 mapped onto day-one formula)
# ---------------------------------------------------------------------------

def part_a() -> None:
    print("== A) scenario matrix (risk_adapter.decide) ==")

    cases = [
        # (title, coercion, spoof, new_beneficiary, expected_decision)
        ("benign + known payee", 0.05, 0.05, False, "observe"),
        ("AAC/TTS synthetic, benign, known payee", 0.0, 0.97, False, "observe"),
        ("AAC/TTS synthetic, benign, NEW payee", 0.0, 0.97, True, "observe"),
        ("human scammer, new payee", 0.90, 0.05, True, "verify"),
        ("human scammer, known payee", 0.90, 0.05, False, "warn"),
        ("synthetic coercive call, new payee", 0.90, 0.93, True, "pause"),
        ("genuine emergency, known payee", 0.60, 0.05, False, "warn"),
        ("spoof unavailable, high coercion, new payee", 0.90, None, True, "pause"),
        ("spoof unavailable, benign", 0.0, None, False, "observe"),
    ]

    for title, c, s, n, expected in cases:
        d = risk_adapter.decide(coercion_score=c, spoof_score=s, new_beneficiary=n)
        assert d["decision"] in DECISION_VOCAB, d
        assert set(d) == {"decision", "risk_score", "reason_codes", "message"}, set(d)
        check(d["decision"] == expected, f"{title}: got {d['decision']} (R={d['risk_score']})")
        if n:
            check("NEW_PAYEE" in d["reason_codes"], f"{title}: NEW_PAYEE code present")
        if s is None:
            check("SPOOF_UNAVAILABLE" in d["reason_codes"], f"{title}: SPOOF_UNAVAILABLE code present")

    # Safety invariants (non-negotiable, primer §12)
    aac = risk_adapter.decide(0.0, 0.97, False)
    check(aac["decision"] == "observe", "synthetic voice alone never blocks")
    human = risk_adapter.decide(0.9, 0.05, True)
    check(human["decision"] != "observe", "human scammer caught WITHOUT deepfake")


# ---------------------------------------------------------------------------
# B) End-to-end API run
# ---------------------------------------------------------------------------

def part_b() -> None:
    print("== B) end-to-end API (analyze -> token -> payment/decision) ==")
    from fastapi.testclient import TestClient

    import main

    client = TestClient(main.app)

    r = client.get("/health")
    check(r.status_code == 200, f"/health 200 (got {r.status_code})")
    check(r.json()["risk_engine"] == "ready", "/health reports risk_engine ready")

    body = {
        "source": "Transfer 50000 right now, do not tell anyone, I am calling from the bank",
        "input_type": "text",
        "lang": "en",
    }
    r = client.post("/analyze", json=body)
    check(r.status_code == 200, f"/analyze 200 (got {r.status_code})")
    data = r.json()
    token_id = data["token_id"]
    check(bool(token_id), "analyze returned a token_id")
    check(data["token_ttl_seconds"] == 120, f"token TTL is 120s (got {data['token_ttl_seconds']})")

    r = client.post(
        "/payment/decision",
        json={"token_id": token_id, "amount": 50000, "new_beneficiary": True},
    )
    check(r.status_code == 200, f"/payment/decision 200 (got {r.status_code})")
    d = r.json()
    check(d["decision"] in DECISION_VOCAB, f"decision vocab lowercase (got {d['decision']!r})")
    check(isinstance(d["risk_score"], float) and 0.0 <= d["risk_score"] <= 1.0, "risk_score in [0,1]")
    check("reason_codes" in d and "message" in d, "response has reason_codes + message")
    # text input -> no spoof model -> SPOOF_UNAVAILABLE must be flagged, never invented
    check("SPOOF_UNAVAILABLE" in d["reason_codes"], "text path flags SPOOF_UNAVAILABLE")
    check("NEW_PAYEE" in d["reason_codes"], "new_beneficiary adds NEW_PAYEE")

    # Reusing an unknown/expired token must be rejected.
    r = client.post(
        "/payment/decision",
        json={"token_id": "does-not-exist", "amount": 100, "new_beneficiary": False},
    )
    check(r.status_code == 400, f"unknown token -> 400 (got {r.status_code})")


def part_c() -> None:
    print("== C) token reconciliation ==")
    check(token_store.TOKEN_TTL_SECONDS == 120, "single token store TTL == 120s")

    # The store must hold ONLY scores (no audio / transcript / phone fields).
    allowed = {"coercion_score", "spoof_score", "reason_codes", "created_at", "expires_at"}
    for tok in token_store._tokens.values():
        check(set(tok) == allowed, f"token fields are scores-only (got {sorted(tok)})")
    check(len(token_store._tokens) == 1, "exactly one live token after part B (no duplicate store)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    print("\nSMOKE TEST: ALL PASS")

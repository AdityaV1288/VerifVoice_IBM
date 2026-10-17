"""risk_engine.py — risk fusion + graduated policy (Person 4's domain).

This is a FALLBACK implementation shipped by Person 3 so the demo works
end-to-end before Person 4's real `risk_engine`/`bank_simulator` lands.
Person 4 replaces the body of `decide()` (keeping the same signature and
return shape) without touching `main.py`.

Fusion:  R = 0.5*C + 0.3*(S*C) + 0.2*N
  C = coercion_score (0..1), S = spoof_score (0..1 or None), N = new beneficiary (0|1)

Spoof is SUPPORTING evidence — it AMPLIFIES coercion but never acts alone, so a
synthetic voice with benign intent (C ~ 0) adds ~0 risk (an AAC/TTS user paying a
known merchant is not flagged). When S is None (model unavailable) we renormalize
onto C and flag the degraded path in the reason codes.
"""


def decide(coercion_score: float, spoof_score: float | None,
           new_beneficiary: bool, amount: float = 0.0) -> dict:
    """Return {decision, risk_score, reason_codes, message} for one payment."""
    n = 1.0 if new_beneficiary else 0.0

    if spoof_score is None:
        # Spoof model unavailable -> renormalize onto coercion. Never invent S.
        risk_score = 0.8 * coercion_score + 0.2 * n
        degraded = True
    else:
        risk_score = 0.5 * coercion_score + 0.3 * (spoof_score * coercion_score) + 0.2 * n
        degraded = False

    risk_score = round(min(1.0, max(0.0, risk_score)), 4)

    # Graduated policy ladder (from PERSON3_ARCHITECTURE.md §6).
    if risk_score < 0.30:
        decision = "observe"
        message = "No additional friction. Payment proceeds normally."
    elif risk_score < 0.55:
        decision = "warn"
        message = ("This caller is requesting an urgent financial action. "
                   "Verify using a known number.")
    elif risk_score < 0.75:
        decision = "verify"
        message = ("This call shows signs of urgent payment pressure. The recipient "
                   "is new. End the call and verify using a number you already trust.")
    else:
        decision = "pause"
        message = ("This call shows signs of urgent payment pressure. The recipient "
                   "is new. Your payment is paused briefly; you can review or "
                   "continue securely.")

    reason_codes = ["NEW_PAYEE"] if new_beneficiary else []
    if degraded:
        reason_codes.append("SPOOF_UNAVAILABLE")

    return {
        "decision": decision,
        "risk_score": risk_score,
        "reason_codes": reason_codes,
        "message": message,
    }

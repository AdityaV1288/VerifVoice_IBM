"""Scenario walkthrough for the demo video: ``python -m verifvoice_risk.demo``.

Uses fixed demo keys and a simulated clock so the output is reproducible.
"""

from __future__ import annotations

import json

from .bank_simulator import BankSimulator
from .risk_engine import RiskEngine, TokenService, run_deletion_checks, zeroize

DEMO_SIGNING_KEY = "demo-only-signing-key-not-for-production"
DEMO_PSEUDONYM_KEY = "demo-only-pseudonym-key-not-for-production"


class _Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def _ok(score: float) -> dict:
    return {"status": "ok", "prediction": "likely_spoof" if score >= 0.5 else "likely_bonafide", "spoof_score": score}


def _coercion(level: str, **extra) -> dict:
    return {"status": "ok", "coercion_level": level, **extra}


# (title, audio, coercion, payee is new?, amount, currency)
SCENARIOS = [
    ("Safe: genuine voice, known payee", _ok(0.04), _coercion("low"), False, 2000, "INR"),
    ("AAC user: synthetic voice, no pressure", _ok(0.97), _coercion("none"), True, 30000, "INR"),
    ("Real emergency: genuine voice, distress", _ok(0.05), _coercion("medium"), True, 20000, "INR"),
    ("Team example: clone + high coercion", _ok(0.93), _coercion("high"), True, 500, "USD"),
    ("Clone + high coercion, large transfer", _ok(0.93), _coercion("high"), True, 50000, "INR"),
    (
        "Clone + refusal + money ask, known payee",
        _ok(0.95),
        _coercion("high", verification_refused=True, cues=["FINANCIAL_INSTRUCTION"]),
        False,
        2000,
        "INR",
    ),
    ("Human scammer, high coercion", _ok(0.05), _coercion("high"), True, 20000, "INR"),
    ("Human scammer, everyday amount", _ok(0.05), _coercion("high"), True, 2000, "INR"),
    ("Human scammer + refusal detected", _ok(0.05), _coercion("high", verification_refused=True), True, 20000, "INR"),
]


def main() -> None:
    clock = _Clock()
    tokens = TokenService(signing_key=DEMO_SIGNING_KEY, pseudonym_key=DEMO_PSEUDONYM_KEY, ttl_seconds=600, clock=clock)
    engine = RiskEngine(tokens)
    bank = BankSimulator(tokens)

    header = f"{'Scenario':<42} {'score':>5}  {'level':<9} {'token':<8} {'payee':<5} {'tier':<6} action"
    print(header)
    print("-" * len(header))
    for i, (title, audio, coercion, new_payee, amount, currency) in enumerate(SCENARIOS):
        customer = f"cust-{i}"
        bank.register_customer(
            customer, currency=currency, established_beneficiaries=["payee-known"], history=[1500] * 10
        )
        assessment, token = engine.assess_call(audio, coercion, subject_id=customer, call_id=f"call-{i}")
        payment = bank.submit_payment(customer, "payee-new" if new_payee else "payee-known", amount, currency)
        print(
            f"{title:<42} {assessment.score:>5.3f}  {assessment.level:<9} "
            f"{token.level if token else '-':<8} {'new' if new_payee else 'known':<5} "
            f"{payment.amount_tier:<6} {payment.decision.action}"
        )

    team = bank.payments()[3]
    print("\nWhat the customer sees (team example):")
    print("  " + team.decision.user_message)
    print("  rationale: " + " | ".join(team.decision.rationale))

    print("\nWhat the bank receives (the token, nothing else):")
    print(json.dumps(tokens.get_active("cust-3").to_dict(), indent=2))

    print("\nData lifecycle (doc 7.3):")
    ring_buffer = bytearray(b"\x5a" * 32000)  # stand-in for 1 s of 16 kHz int16 audio
    live = run_deletion_checks(buffers=[ring_buffer])
    print(f"  during call  : buffer zeroized = {live.passed}")
    zeroize(ring_buffer)
    clock.now += 600
    purged = tokens.purge_expired()
    print(f"  call ended   : buffer zeroized, {purged} expired token(s) purged")
    raw_ids = ["payee-known", "payee-new"] + [f"{kind}-{i}" for i in range(len(SCENARIOS)) for kind in ("cust", "call")]
    report = run_deletion_checks(
        token_service=tokens,
        audit_logs=[engine.audit_log, bank.audit_log],
        buffers=[ring_buffer],
        raw_identifiers=raw_ids,
    )
    for check in report.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print(f"  deletion checks passed: {report.passed}")


if __name__ == "__main__":
    main()

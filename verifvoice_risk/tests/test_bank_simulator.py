import json
import threading
from dataclasses import replace

import pytest

from verifvoice_risk.bank_simulator import (
    APPROVED,
    CANCELLED,
    EXECUTED,
    ON_HOLD,
    PENDING_CONFIRMATION,
    PENDING_VERIFICATION,
    BankSimulator,
    InvalidTransition,
)
from verifvoice_risk.risk_engine import (
    ALLOW,
    AuditLog,
    PrivacyViolation,
    HOLD,
    VERIFY_OUT_OF_BAND,
    WARN,
    InputValidationError,
    find_privacy_violations,
    run_deletion_checks,
    zeroize,
)

from .conftest import TEAM_EXAMPLE, TTL, audio, coercion

DAY = 24 * 60 * 60
HOLD_SECONDS = 30 * 60


def call(engine, customer, kind):
    """Feed one call update through the risk engine (issues a token if >= 0.55)."""
    signals = {
        "benign": (audio(0.04), coercion("none")),
        "aac": (audio(0.97), coercion("none")),  # synthetic voice, no coercion
        "medium": (audio(0.90), coercion("medium")),  # 0.695
        "human": (audio(0.05, prediction="likely_bonafide"), coercion("high")),  # genuine voice, floored to 0.55
        "high": (TEAM_EXAMPLE["audio_analysis"], TEAM_EXAMPLE["coercion_analysis"]),  # 0.812
        "critical": (
            audio(0.95),
            coercion("high", verification_refused=True, cues=["FINANCIAL_INSTRUCTION", "SECRECY_REQUEST"]),
        ),
    }[kind]
    assessment, token = engine.assess_call(*signals, subject_id=customer, call_id=f"call-{customer}")
    return assessment, token


@pytest.fixture
def asha(bank):
    """Established INR customer with a regular payee and a payment history."""
    return bank.register_customer(
        "cust-asha", currency="INR", established_beneficiaries=["upi:grocer@okbank"], history=[1200, 800, 1500, 900, 1100]
    )


@pytest.fixture
def usd_customer(bank):
    return bank.register_customer("cust-001", currency="USD", established_beneficiaries=["acct-landlord"])


# ---------------------------------------------------------------------------
# Safe transactions
# ---------------------------------------------------------------------------


class TestSafe:
    def test_known_payee_no_call(self, bank, asha):
        p = bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000)
        assert p.status == EXECUTED and p.decision.action == ALLOW
        assert p.token_status == "none" and not p.new_beneficiary and p.decision.user_message == ""

    def test_known_payee_during_medium_call(self, bank, engine, asha):
        call(engine, "cust-asha", "medium")
        p = bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000)
        assert p.token_status == "active" and p.decision.call_level == "medium"
        assert p.status == EXECUTED and p.decision.action == ALLOW

    def test_new_payee_without_call_risk(self, bank, asha):
        p = bank.submit_payment("cust-asha", "upi:plumber@okbank", 2000)
        assert p.new_beneficiary and p.decision.action == ALLOW and p.status == EXECUTED
        assert p.decision.reason_codes == ("NEW_BENEFICIARY",)  # informational only

    def test_benign_call_issues_no_token(self, bank, engine, asha):
        assessment, token = call(engine, "cust-asha", "benign")
        assert token is None and assessment.level == "low"
        assert bank.submit_payment("cust-asha", "upi:new@okbank", 5000).decision.action == ALLOW

    def test_aac_user_synthetic_voice_is_not_penalised(self, bank, engine, asha):
        assessment, token = call(engine, "cust-asha", "aac")
        assert "SYNTHETIC_AUDIO_HIGH" in assessment.reason_codes and token is None
        p = bank.submit_payment("cust-asha", "upi:new@okbank", 30000)
        assert p.decision.action == ALLOW and p.status == EXECUTED

    def test_after_token_expires_no_friction(self, bank, engine, clock, asha):
        call(engine, "cust-asha", "high")
        clock.advance(TTL)
        p = bank.submit_payment("cust-asha", "upi:new@okbank", 5000)
        assert p.token_status == "expired" and p.decision.action == ALLOW

    def test_small_amount_to_new_payee_during_medium_call(self, bank, engine, asha):
        call(engine, "cust-asha", "medium")
        p = bank.submit_payment("cust-asha", "upi:chai@okbank", 200)  # medium +1 new -1 small
        assert p.amount_tier == "small" and p.decision.action == ALLOW


# ---------------------------------------------------------------------------
# Suspicious transactions
# ---------------------------------------------------------------------------


class TestSuspicious:
    def test_team_example_end_to_end(self, bank, engine, usd_customer):
        assessment, token = call(engine, "cust-001", "high")
        assert assessment.score == pytest.approx(0.812) and token.level == "high"
        p = bank.submit_payment("cust-001", "acct-stranger", 500, "USD")
        assert p.new_beneficiary and p.amount_tier == "normal"
        assert p.decision.action == VERIFY_OUT_OF_BAND and p.status == PENDING_VERIFICATION
        assert p.decision.token_id == token.token_id
        assert p.decision.user_message.startswith("This call shows signs of urgent payment pressure.")
        assert "verify using a number you already trust" in p.decision.user_message
        p = bank.verify_out_of_band(p.payment_id, confirmed=True)
        assert p.status == EXECUTED and p.executed_at is not None

    def test_out_of_band_declined_cancels(self, bank, engine, usd_customer):
        call(engine, "cust-001", "high")
        p = bank.submit_payment("cust-001", "acct-stranger", 500)
        assert bank.verify_out_of_band(p.payment_id, confirmed=False).status == CANCELLED

    def test_medium_call_new_payee_warns(self, bank, engine, asha):
        call(engine, "cust-asha", "medium")
        p = bank.submit_payment("cust-asha", "upi:new@okbank", 2000)  # normal for her (< 3x median 1100)
        assert p.decision.action == WARN and p.status == PENDING_CONFIRMATION
        assert bank.acknowledge_warning(p.payment_id).status == EXECUTED

    def test_high_call_new_payee_large_amount_is_held_briefly(self, bank, engine, clock, asha):
        call(engine, "cust-asha", "high")
        p = bank.submit_payment("cust-asha", "upi:stranger@okbank", 50000)
        assert p.decision.action == HOLD and p.status == ON_HOLD
        assert p.hold_until == clock.now + HOLD_SECONDS
        assert set(p.decision.reason_codes) >= {"NEW_BENEFICIARY", "UNUSUAL_AMOUNT", "COERCIVE_LANGUAGE"}
        with pytest.raises(InvalidTransition, match="hold active"):
            bank.confirm_after_hold(p.payment_id)
        clock.advance(HOLD_SECONDS)
        assert bank.confirm_after_hold(p.payment_id).status == EXECUTED

    def test_hold_released_early_by_out_of_band_verification(self, bank, engine, asha):
        call(engine, "cust-asha", "high")
        p = bank.submit_payment("cust-asha", "upi:stranger@okbank", 50000)
        assert bank.verify_out_of_band(p.payment_id, confirmed=True).status == EXECUTED

    def test_critical_call_new_payee_is_held(self, bank, engine, asha):
        assessment, token = call(engine, "cust-asha", "critical")
        assert assessment.band == "CRITICAL_ALERT" and token.level == "critical"
        p = bank.submit_payment("cust-asha", "upi:stranger@okbank", 5000)
        assert p.decision.action == HOLD
        assert {"SECRECY_REQUEST", "VERIFICATION_REFUSED", "FINANCIAL_INSTRUCTION"} <= set(p.decision.reason_codes)

    def test_critical_call_known_payee_still_verified(self, bank, engine, asha):
        call(engine, "cust-asha", "critical")
        p = bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000)
        assert p.decision.action == VERIFY_OUT_OF_BAND

    def test_history_relative_unusual_amount(self, bank, engine, asha):
        call(engine, "cust-asha", "high")
        # median history 1100 -> 3x = 3300; 4000 is unusual for her, below the 25k absolute bar
        p = bank.submit_payment("cust-asha", "upi:grocer@okbank", 4000)
        assert p.amount_tier == "large" and p.decision.action == VERIFY_OUT_OF_BAND

    def test_one_rupee_probe_does_not_make_payee_trusted(self, bank, engine, asha):
        probe = bank.submit_payment("cust-asha", "upi:scammer@okbank", 1)
        assert probe.status == EXECUTED
        call(engine, "cust-asha", "high")
        p = bank.submit_payment("cust-asha", "upi:scammer@okbank", 50000)
        assert p.new_beneficiary and p.beneficiary_reason == "added_recently"
        assert p.decision.action == HOLD

    def test_human_scammer_new_payee_is_warned(self, bank, engine, asha):
        assessment, token = call(engine, "cust-asha", "human")
        assert assessment.raw_score < 0.55 and assessment.score == pytest.approx(0.55)
        assert token.level == "medium"
        p = bank.submit_payment("cust-asha", "upi:stranger@okbank", 2000)
        assert p.decision.action == WARN and p.status == PENDING_CONFIRMATION
        assert p.decision.user_message == (
            "This call shows signs of urgent payment pressure. The recipient is new. "
            "Take a moment before you continue."
        )

    def test_human_scammer_unusual_amount_needs_out_of_band_check(self, bank, engine, asha):
        call(engine, "cust-asha", "human")
        p = bank.submit_payment("cust-asha", "upi:stranger@okbank", 20000)  # > 3x her median
        assert p.amount_tier == "large" and p.decision.action == VERIFY_OUT_OF_BAND

    def test_human_scammer_one_rupee_probe_then_large_transfer(self, bank, engine, asha):
        bank.submit_payment("cust-asha", "upi:scammer@okbank", 1)
        call(engine, "cust-asha", "human")
        p = bank.submit_payment("cust-asha", "upi:scammer@okbank", 50000)
        assert p.new_beneficiary and p.decision.action == VERIFY_OUT_OF_BAND

    def test_human_scammer_known_payee_normal_amount_is_allowed(self, bank, engine, asha):
        # Proportionality: a medium token alone adds no friction for an
        # established payee at a normal amount.
        call(engine, "cust-asha", "human")
        assert bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000).decision.action == ALLOW

    def test_token_for_another_customer_is_not_applied(self, bank, engine, asha, usd_customer):
        call(engine, "cust-001", "critical")
        assert bank.submit_payment("cust-asha", "upi:new@okbank", 50000).decision.action == ALLOW

    def test_tampered_token_is_ignored_and_audited(self, bank, engine, token_service, asha):
        _, token = call(engine, "cust-asha", "medium")
        token_service.store.put(replace(token, level="critical"), TTL)
        p = bank.submit_payment("cust-asha", "upi:new@okbank", 5000)
        assert p.token_status == "invalid" and p.decision.action == ALLOW
        assert bank.audit_log.records()[0]["token_status"] == "invalid"


# ---------------------------------------------------------------------------
# New-beneficiary detection
# ---------------------------------------------------------------------------


class TestNewBeneficiary:
    def test_lifecycle(self, bank, clock, asha):
        assert bank.beneficiary_status("cust-asha", "upi:x@okbank") == {"new": True, "reason": "not_registered"}
        bank.add_beneficiary("cust-asha", "upi:x@okbank")
        assert bank.beneficiary_status("cust-asha", "upi:x@okbank")["reason"] == "added_recently"
        clock.advance(DAY)
        assert bank.beneficiary_status("cust-asha", "upi:x@okbank")["reason"] == "never_paid"
        bank.submit_payment("cust-asha", "upi:x@okbank", 1000)
        assert bank.beneficiary_status("cust-asha", "upi:x@okbank") == {"new": False, "reason": "established"}

    def test_registered_payees_are_established(self, bank, asha):
        assert bank.beneficiary_status("cust-asha", "upi:grocer@okbank") == {"new": False, "reason": "established"}

    def test_add_beneficiary_is_idempotent(self, bank, clock, asha):
        first = bank.add_beneficiary("cust-asha", "upi:x@okbank")
        clock.advance(DAY)
        assert bank.add_beneficiary("cust-asha", "upi:x@okbank").added_at == first.added_at

    def test_payees_are_per_customer(self, bank, asha, usd_customer):
        assert bank.beneficiary_status("cust-001", "upi:grocer@okbank")["new"] is True


# ---------------------------------------------------------------------------
# Agency: no freeze, always an override, valid transitions only
# ---------------------------------------------------------------------------


class TestAgency:
    def test_account_not_frozen_while_a_payment_is_held(self, bank, engine, asha):
        call(engine, "cust-asha", "high")
        held = bank.submit_payment("cust-asha", "upi:stranger@okbank", 50000)
        assert held.status == ON_HOLD
        other = bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000)
        assert other.status == PENDING_CONFIRMATION  # proportionate WARN, not blocked
        assert bank.acknowledge_warning(other.payment_id).status == EXECUTED
        assert bank.cancel_payment(held.payment_id).status == CANCELLED

    def test_every_open_payment_can_be_completed(self, bank, engine, clock, asha):
        call(engine, "cust-asha", "critical")
        payments = [
            bank.submit_payment("cust-asha", "upi:grocer@okbank", 200),  # WARN
            bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000),  # VERIFY
            bank.submit_payment("cust-asha", "upi:new@okbank", 5000),  # HOLD
        ]
        assert [p.status for p in payments] == [PENDING_CONFIRMATION, PENDING_VERIFICATION, ON_HOLD]
        bank.acknowledge_warning(payments[0].payment_id)
        bank.verify_out_of_band(payments[1].payment_id, confirmed=True)
        clock.advance(HOLD_SECONDS)
        bank.confirm_after_hold(payments[2].payment_id)
        assert all(p.status == EXECUTED for p in bank.payments())

    def test_invalid_transitions(self, bank, engine, asha):
        done = bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000)
        with pytest.raises(InvalidTransition):
            bank.acknowledge_warning(done.payment_id)
        with pytest.raises(InvalidTransition):
            bank.cancel_payment(done.payment_id)
        call(engine, "cust-asha", "high")
        pending = bank.submit_payment("cust-asha", "upi:new@okbank", 2000)
        assert pending.status == PENDING_VERIFICATION
        with pytest.raises(InvalidTransition):
            bank.confirm_after_hold(pending.payment_id)
        with pytest.raises(InvalidTransition):
            bank.acknowledge_warning(pending.payment_id)

    def test_status_trail_is_audited(self, bank, engine, asha):
        call(engine, "cust-asha", "high")
        p = bank.submit_payment("cust-asha", "upi:new@okbank", 2000)
        bank.verify_out_of_band(p.payment_id, confirmed=True)
        trail = [r for r in bank.audit_log.records() if r["event"] == "payment_status"]
        assert trail == [
            {
                "event": "payment_status",
                "at": trail[0]["at"],
                "payment_ref": p.payment_id,
                "from": PENDING_VERIFICATION,
                "to": EXECUTED,
                "via": "out_of_band_verified",
            }
        ]

    def test_allowed_payment_audit_trail(self, bank, asha):
        p = bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000)
        records = bank.audit_log.records()
        assert records[0]["status"] == APPROVED and records[1]["from"] == APPROVED and records[1]["to"] == EXECUTED
        assert p.status == EXECUTED


# ---------------------------------------------------------------------------
# Privacy and deletion
# ---------------------------------------------------------------------------


class TestPrivacyAndDeletion:
    def test_bank_audit_is_pseudonymous(self, bank, engine, asha):
        call(engine, "cust-asha", "high")
        bank.submit_payment("cust-asha", "upi:stranger@okbank", 50000)
        bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000)
        dumped = json.dumps(bank.audit_log.records())
        for raw in ("cust-asha", "upi:stranger@okbank", "upi:grocer@okbank", "50000"):
            assert raw not in dumped
        decision = bank.audit_log.records()[0]
        assert decision["subject_ref"].startswith("sub_") and decision["beneficiary_ref"].startswith("ben_")
        assert decision["subject_ref"] == engine.token_service.subject_ref("cust-asha")

    def test_rejected_decision_record_leaves_no_payment(self, token_service):
        strict = BankSimulator(token_service, audit_log=AuditLog(forbidden_values=["payment_decision"]))
        customer = strict.register_customer("cust-x", currency="USD", established_beneficiaries=["acct-a"])
        with pytest.raises(PrivacyViolation):
            strict.submit_payment("cust-x", "acct-a", 100)
        assert strict.payments() == []
        assert customer.beneficiaries["acct-a"].completed_payments == 1 and customer.history["USD"] == []

    def test_rejected_status_record_leaves_payment_unchanged(self, bank, engine, asha):
        call(engine, "cust-asha", "high")
        p = bank.submit_payment("cust-asha", "upi:stranger@okbank", 2000)
        bank.audit_log.forbid("out_of_band_verified")
        with pytest.raises(PrivacyViolation):
            bank.verify_out_of_band(p.payment_id, confirmed=True)
        assert p.status == PENDING_VERIFICATION and p.executed_at is None
        assert "upi:stranger@okbank" not in asha.beneficiaries

    def test_full_call_lifecycle_deletion_checks(self, bank, engine, clock, tmp_path, asha):
        ring_buffer = bytearray(b"\x11\x22" * 16000)  # audio held only in memory during the call
        call(engine, "cust-asha", "high")
        p = bank.submit_payment("cust-asha", "upi:stranger@okbank", 5000)
        bank.verify_out_of_band(p.payment_id, confirmed=True)

        live = run_deletion_checks(token_service=engine.token_service, buffers=[ring_buffer])
        assert not live.passed  # buffer still holds audio while the call is live

        zeroize(ring_buffer)  # call ends
        clock.advance(TTL)
        assert engine.token_service.purge_expired() == 1
        report = run_deletion_checks(
            token_service=engine.token_service,
            audit_logs=[engine.audit_log, bank.audit_log],
            buffers=[ring_buffer],
            artifact_dirs=[tmp_path],
            raw_identifiers=["cust-asha", "upi:stranger@okbank", "call-cust-asha"],
        )
        assert report.passed, report.to_dict()
        for log in (engine.audit_log, bank.audit_log):
            for record in log.records():
                assert find_privacy_violations(record, ["cust-asha", "upi:stranger@okbank"]) == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_customer(self, bank):
        with pytest.raises(KeyError):
            bank.submit_payment("nobody", "upi:x@okbank", 100)

    @pytest.mark.parametrize("amount", [0, -10, "100", float("nan"), True])
    def test_bad_amount(self, bank, asha, amount):
        with pytest.raises(InputValidationError):
            bank.submit_payment("cust-asha", "upi:x@okbank", amount)

    def test_bad_inputs(self, bank, asha):
        with pytest.raises(InputValidationError):
            bank.submit_payment("cust-asha", "", 100)
        with pytest.raises(InputValidationError):
            bank.submit_payment("cust-asha", "upi:x@okbank", 100, currency="RUPEES")
        with pytest.raises(InputValidationError):
            bank.register_customer("cust-asha")
        with pytest.raises(InputValidationError):
            bank.register_customer("")
        with pytest.raises(KeyError):
            bank.get_payment("pay_missing")

    def test_concurrent_payments_keep_state_consistent(self, bank, asha):
        def worker():
            for _ in range(25):
                bank.submit_payment("cust-asha", "upi:grocer@okbank", 1000)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(bank.payments()) == 200 and all(p.status == EXECUTED for p in bank.payments())
        assert asha.beneficiaries["upi:grocer@okbank"].completed_payments == 1 + 200
        assert len(bank.audit_log) == 400  # one decision + one status change each

    def test_invalid_payee_ids_rejected_at_registration(self, bank):
        for payees in ([""], [123], ["ok", None]):
            with pytest.raises(InputValidationError):
                bank.register_customer("cust-bad", established_beneficiaries=payees)
        assert bank.register_customer("cust-bad").customer_id == "cust-bad"  # nothing half-registered

    def test_payment_to_dict_is_json_ready(self, bank, engine, asha):
        call(engine, "cust-asha", "high")
        p = bank.submit_payment("cust-asha", "upi:stranger@okbank", 50000)
        body = json.loads(json.dumps(p.to_dict()))
        assert body["status"] == ON_HOLD and body["action"] == HOLD and body["hold_until"] is not None

import array
import copy
import itertools
import json
import logging
import math
import threading
import time
from dataclasses import replace

import pytest

from verifvoice_risk.risk_engine import (
    ACTION_LADDER,
    ALLOW,
    AMOUNT_TIERS,
    CRITICAL_ALERT,
    ELEVATED_WATCH,
    HOLD,
    PASSIVE_MONITORING,
    SOFT_PROMPT,
    STRONG_PROMPT,
    TOKEN_FIELDS,
    TOKEN_LEVELS,
    VERIFY_OUT_OF_BAND,
    WARN,
    AudioAnalysis,
    AuditLog,
    CallSignal,
    CoercionAnalysis,
    FusionConfig,
    InMemoryTokenStore,
    InputValidationError,
    PolicyConfig,
    PrivacyViolation,
    RedisTokenStore,
    RiskEngine,
    RiskToken,
    TokenService,
    Transaction,
    TransactionContext,
    band_for_score,
    classify_amount,
    decide_action,
    find_media_artifacts,
    find_privacy_violations,
    is_zeroized,
    run_deletion_checks,
    score_call,
    zeroize,
)

from .conftest import PSEUDONYM_KEY, SIGNING_KEY, TEAM_EXAMPLE, TTL, audio, coercion


def assess(audio_dict, coercion_dict, config=FusionConfig()):
    return score_call(AudioAnalysis.from_dict(audio_dict), CoercionAnalysis.from_dict(coercion_dict), config, now=0.0)


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


class TestInputContract:
    def test_team_example_parses(self):
        a = AudioAnalysis.from_dict(TEAM_EXAMPLE["audio_analysis"])
        c = CoercionAnalysis.from_dict(TEAM_EXAMPLE["coercion_analysis"])
        t = Transaction.from_dict(TEAM_EXAMPLE["transaction"])
        assert (a.available, a.spoof_score, a.prediction) == (True, 0.93, "likely_spoof")
        assert (c.available, c.level) == (True, "high")
        assert (t.amount, t.currency, t.new_beneficiary) == (500.0, "USD", True)

    @pytest.mark.parametrize("bad", [1.5, -0.01, "0.9", True, None, float("nan"), float("inf"), [0.9]])
    def test_spoof_score_must_be_unit_number(self, bad):
        with pytest.raises(InputValidationError):
            AudioAnalysis.from_dict({"status": "ok", "spoof_score": bad})

    def test_spoof_score_required_when_ok(self):
        with pytest.raises(InputValidationError, match="spoof_score is required"):
            AudioAnalysis.from_dict({"status": "ok", "prediction": "likely_spoof"})

    @pytest.mark.parametrize("data", [{"spoof_score": 0.5}, {"status": "", "spoof_score": 0.5}, {"status": 1}])
    def test_status_required(self, data):
        with pytest.raises(InputValidationError, match="status"):
            AudioAnalysis.from_dict(data)

    def test_non_ok_status_means_signal_unavailable(self):
        a = AudioAnalysis.from_dict({"status": "error", "spoof_score": "garbage"})
        c = CoercionAnalysis.from_dict({"status": "unsupported_language"})
        assert not a.available and a.status == "error"
        assert not c.available and c.status == "unsupported_language"
        assert not AudioAnalysis.from_dict(None).available

    def test_sections_must_be_objects(self):
        with pytest.raises(InputValidationError):
            AudioAnalysis.from_dict([0.9])
        with pytest.raises(InputValidationError):
            CoercionAnalysis.from_dict("high")

    @pytest.mark.parametrize("level", ["extreme", "", None, 3])
    def test_unknown_coercion_level_rejected(self, level):
        with pytest.raises(InputValidationError, match="coercion_level"):
            CoercionAnalysis.from_dict({"status": "ok", "coercion_level": level})

    def test_coercion_level_and_status_are_normalised(self):
        c = CoercionAnalysis.from_dict({"status": " OK ", "coercion_level": " High "})
        assert c.available and c.level == "high"

    def test_cues_are_normalised_and_validated(self):
        c = CoercionAnalysis.from_dict(coercion("high", cues=["secrecy_request", "VERIFICATION_REFUSED"]))
        assert c.cues == ("SECRECY_REQUEST", "VERIFICATION_REFUSED")
        assert c.verification_refused is True
        with pytest.raises(InputValidationError, match="unknown coercion cue"):
            CoercionAnalysis.from_dict(coercion("high", cues=["BAD_VIBES"]))
        with pytest.raises(InputValidationError, match="list"):
            CoercionAnalysis.from_dict(coercion("high", cues="SECRECY_REQUEST"))

    def test_phase_validated(self):
        assert CoercionAnalysis.from_dict(coercion("low", phase="p2")).phase == "P2"
        with pytest.raises(InputValidationError, match="phase"):
            CoercionAnalysis.from_dict(coercion("low", phase="P9"))

    @pytest.mark.parametrize(
        "txn",
        [
            {"amount": 0, "currency": "USD", "new_beneficiary": True},
            {"amount": -5, "currency": "USD", "new_beneficiary": True},
            {"amount": "500", "currency": "USD", "new_beneficiary": True},
            {"amount": True, "currency": "USD", "new_beneficiary": True},
            {"amount": float("nan"), "currency": "USD", "new_beneficiary": True},
            {"amount": 500, "currency": "US", "new_beneficiary": True},
            {"amount": 500, "currency": "US1", "new_beneficiary": True},
            {"amount": 500, "currency": 840, "new_beneficiary": True},
            {"amount": 500, "currency": "USD", "new_beneficiary": "yes"},
            {"amount": 500, "currency": "USD"},
        ],
    )
    def test_transaction_validation(self, txn):
        with pytest.raises(InputValidationError):
            Transaction.from_dict(txn)

    def test_currency_normalised(self):
        assert Transaction.from_dict({"amount": 1, "currency": "inr", "new_beneficiary": False}).currency == "INR"


# ---------------------------------------------------------------------------
# Fusion formula (doc 6.2-6.4)
# ---------------------------------------------------------------------------


class TestFusion:
    def test_team_example_matches_doc_formula(self):
        a = assess(TEAM_EXAMPLE["audio_analysis"], TEAM_EXAMPLE["coercion_analysis"])
        # 0.40*0.93 + 0.35*0.90 + 0.25*(1 - 0.5 neutral speaker prior)
        assert a.score == pytest.approx(0.812)
        assert a.band == STRONG_PROMPT and a.level == "high"
        assert a.confidence == pytest.approx(0.75)  # speaker_match not measured
        assert a.reason_codes == ("SYNTHETIC_AUDIO_HIGH", "COERCIVE_LANGUAGE")
        assert a.defaults_used == ("speaker_match",)
        assert a.token_eligible and not a.abstained

    def test_speaker_match_extension_used_when_present(self):
        a = assess(audio(0.93, speaker_match=0.2), coercion("high"))
        assert a.score == pytest.approx(0.372 + 0.315 + 0.25 * 0.8)
        assert a.confidence == pytest.approx(1.0)

    def test_refusal_boost_is_added_and_score_clipped(self):
        a = assess(audio(0.93), coercion("high", verification_refused=True))
        assert a.raw_score == pytest.approx(0.812 + 0.25)
        assert a.score == 1.0
        assert "VERIFICATION_REFUSED" in a.reason_codes

    @pytest.mark.parametrize("spoof", [0.0, 0.05, 0.1, 0.2])
    def test_human_scammer_with_high_coercion_reaches_the_bank(self, spoof):
        # Genuine (non-cloned) voice, team format only: the formula alone gives
        # 0.40*spoof + 0.315 + 0.125 < 0.55, so the high-coercion floor lifts it.
        a = assess(audio(spoof), coercion("high"))
        assert a.raw_score < 0.55
        assert a.score == pytest.approx(0.55) and a.floor == 0.55
        assert a.band == SOFT_PROMPT and a.level == "medium" and a.token_eligible
        assert a.reason_codes == ("COERCIVE_LANGUAGE",)

    def test_floor_never_lowers_a_higher_score(self):
        a = assess(TEAM_EXAMPLE["audio_analysis"], TEAM_EXAMPLE["coercion_analysis"])
        assert a.score == pytest.approx(0.812) and a.floor == 0.55

    def test_refusal_still_adds_on_top_of_the_floor(self):
        a = assess(audio(0.05), coercion("high", verification_refused=True))
        # 0.40*0.05 + 0.35*0.90 + 0.125 + 0.25 = 0.71
        assert a.score == pytest.approx(0.71) and a.level == "medium"
        assert "VERIFICATION_REFUSED" in a.reason_codes

    @pytest.mark.parametrize("level", ["none", "low", "medium"])
    def test_floor_only_applies_to_high_coercion(self, level):
        a = assess(audio(0.05), coercion(level))
        assert a.floor == 0.0 and not a.token_eligible

    def test_explicit_early_phase_overrides_the_floor(self):
        a = assess(audio(0.05), coercion("high", phase="P0"))
        assert a.score == 0.50 and not a.token_eligible
        b = assess(audio(0.05), coercion("high", phase="P1"))
        assert b.score == pytest.approx(0.55) and b.token_eligible

    def test_critical_requires_financial_intent(self):
        no_fin = assess(audio(0.95), coercion("high", verification_refused=True))
        fin = assess(audio(0.95), coercion("high", verification_refused=True, cues=["FINANCIAL_INSTRUCTION"]))
        assert no_fin.score > 0.90 and no_fin.band == STRONG_PROMPT
        assert fin.band == CRITICAL_ALERT and fin.level == "critical" and fin.financial_intent

    @pytest.mark.parametrize("flag", [{"financial_intent": True}, {"phase": "P3"}, {"phase": "P4"}])
    def test_financial_intent_sources(self, flag):
        a = assess(audio(0.95), coercion("high", verification_refused=True, **flag))
        assert a.financial_intent and "FINANCIAL_INSTRUCTION" in a.reason_codes

    @pytest.mark.parametrize(
        "score,fin,band",
        [
            (0.0, False, PASSIVE_MONITORING),
            (0.2999, False, PASSIVE_MONITORING),
            (0.30, False, ELEVATED_WATCH),
            (0.5499, False, ELEVATED_WATCH),
            (0.55, False, SOFT_PROMPT),
            (0.7499, False, SOFT_PROMPT),
            (0.75, False, STRONG_PROMPT),
            (0.90, True, STRONG_PROMPT),  # "> 0.90" is strict
            (0.9001, True, CRITICAL_ALERT),
            (0.99, False, STRONG_PROMPT),
            (1.0, True, CRITICAL_ALERT),
        ],
    )
    def test_band_boundaries(self, score, fin, band):
        assert band_for_score(score, fin) == band

    def test_token_threshold_is_exactly_055(self):
        # 0.40*s + 0.35*0.60 + 0.125 == 0.55  ->  s = 0.5375
        at = assess(audio(0.5375), coercion("medium"))
        below = assess(audio(0.5370), coercion("medium"))
        assert at.score == pytest.approx(0.55) and at.band == SOFT_PROMPT and at.token_eligible
        assert below.score < 0.55 and below.band == ELEVATED_WATCH and not below.token_eligible

    def test_explicit_phase_ceiling(self):
        a = assess(audio(0.99), coercion("high", phase="P0"))
        assert a.ceiling == 0.50 and a.score == 0.50 and not a.token_eligible
        b = assess(audio(0.99), coercion("high", phase="P2"))
        assert b.score == pytest.approx(0.75) and b.ceiling == 0.75

    def test_numeric_coercion_score_overrides_level_mapping(self):
        a = assess(audio(0.5), coercion("medium", coercion_score=0.4))
        assert a.score == pytest.approx(0.2 + 0.35 * 0.4 + 0.125)
        # the high-coercion floor follows the module's level, not the number
        assert assess(audio(0.5), coercion("high", coercion_score=0.4)).score == pytest.approx(0.55)

    def test_low_quality_lowers_confidence_not_score(self):
        clean = assess(audio(0.93), coercion("high"))
        noisy = assess(audio(0.93, quality=0.25), coercion("high"))
        assert noisy.score == clean.score
        assert noisy.confidence == pytest.approx((0.40 * 0.25 + 0.35) / 1.0)

    def test_audio_unavailable_uses_neutral_prior(self):
        a = assess({"status": "error"}, coercion("high"))
        assert a.score == pytest.approx(0.20 + 0.315 + 0.125)
        assert a.confidence == pytest.approx(0.35)
        assert a.defaults_used == ("spoof_score", "speaker_match")
        assert "SYNTHETIC_AUDIO_HIGH" not in a.reason_codes

    @pytest.mark.parametrize("level", ["none", "low", "medium", "high"])
    def test_score_is_monotonic_in_spoof(self, level):
        scores = [assess(audio(s / 20), coercion(level)).score for s in range(21)]
        assert scores == sorted(scores)

    def test_model_versions_recorded(self):
        a = assess(audio(0.9, model_version="aasist-onnx-0.3"), coercion("high", model_version="rules-1"))
        assert a.model_versions == {"acoustic": "aasist-onnx-0.3", "coercion": "rules-1"}

    def test_config_validation(self):
        with pytest.raises(ValueError):
            FusionConfig(soft_prompt_threshold=0.2)  # below elevated
        with pytest.raises(ValueError):
            FusionConfig(coercion_level_scores={"high": 0.9})
        with pytest.raises(ValueError, match="exceeds its ceiling"):
            FusionConfig(coercion_level_floors={"none": 0.0, "low": 0.0, "medium": 0.8, "high": 0.55})
        with pytest.raises(ValueError):
            PolicyConfig(base_steps={"medium": 0, "high": 9, "critical": 2})
        with pytest.raises(ValueError):
            PolicyConfig(small_amount={"USD": 2000.0})  # not < large


# ---------------------------------------------------------------------------
# Safety cases: AAC users, real emergencies, abstention (doc 8)
# ---------------------------------------------------------------------------


class TestSafetyCases:
    @pytest.mark.parametrize("spoof", [0.8, 0.95, 1.0])
    @pytest.mark.parametrize("level", ["none", "low"])
    def test_aac_synthetic_voice_without_coercion_never_reaches_bank(self, spoof, level):
        a = assess(audio(spoof), coercion(level))
        assert a.level in ("low", "elevated")
        assert not a.token_eligible
        assert "SYNTHETIC_AUDIO_HIGH" in a.reason_codes  # recorded for audit, not acted on

    def test_real_emergency_genuine_voice_is_not_escalated(self):
        a = assess(audio(0.05), coercion("medium"))
        assert a.level == "elevated" and not a.token_eligible

    def test_benign_call(self):
        a = assess(audio(0.05), coercion("none"))
        assert a.band == PASSIVE_MONITORING and a.reason_codes == ()

    @pytest.mark.parametrize("coercion_dict", [None, {"status": "unsupported_language"}, {"status": "error"}])
    def test_missing_coercion_signal_abstains(self, coercion_dict):
        a = assess(audio(1.0), coercion_dict)
        assert a.abstained and a.abstain_reason
        assert a.floor == 0.0  # no intent evidence, so no intent floor in the audit trail
        assert a.level in ("low", "elevated") and not a.token_eligible

    def test_abstention_holds_even_under_a_permissive_config(self):
        loose = FusionConfig(coercion_level_ceilings={"none": 1.0, "low": 1.0, "medium": 1.0, "high": 1.0})
        a = assess(audio(1.0, speaker_match=0.0), None, loose)
        assert a.score > 0.55 and a.band == ELEVATED_WATCH and not a.token_eligible


# ---------------------------------------------------------------------------
# Proportionate actions (transaction fusion)
# ---------------------------------------------------------------------------


def ctx(new_ben, tier, amount=100.0, currency="INR"):
    return TransactionContext(amount, currency, new_ben, tier)


class TestPolicy:
    @pytest.mark.parametrize(
        "level,new_ben,tier,conf,expected",
        [
            ("medium", False, "normal", 0.75, ALLOW),  # known payee: low friction
            ("medium", True, "normal", 0.75, WARN),
            ("medium", True, "large", 0.75, VERIFY_OUT_OF_BAND),
            ("high", False, "normal", 0.75, WARN),
            ("high", True, "normal", 0.75, VERIFY_OUT_OF_BAND),  # team example
            ("high", True, "large", 0.75, HOLD),
            ("high", True, "small", 0.75, WARN),
            ("high", True, "normal", 0.35, WARN),  # low-confidence token
            ("critical", False, "normal", 0.9, VERIFY_OUT_OF_BAND),
            ("critical", True, "normal", 0.9, HOLD),
            ("critical", True, "large", 0.9, HOLD),
            ("critical", False, "small", 0.3, WARN),  # critical floor
        ],
    )
    def test_action_matrix(self, level, new_ben, tier, conf, expected):
        call = CallSignal(level, conf, ("COERCIVE_LANGUAGE",), "tok_x")
        assert decide_action(call, ctx(new_ben, tier)).action == expected

    @pytest.mark.parametrize("new_ben,tier", list(itertools.product([True, False], AMOUNT_TIERS)))
    def test_no_call_signal_means_no_friction(self, new_ben, tier):
        d = decide_action(None, ctx(new_ben, tier))
        assert d.action == ALLOW and d.user_message == "" and d.call_level is None

    def test_sub_threshold_call_signal_is_ignored(self):
        assert decide_action(CallSignal("elevated", 1.0, ()), ctx(True, "large")).action == ALLOW

    def test_never_blocks_and_always_reversible(self):
        for level, new_ben, tier, conf in itertools.product(
            TOKEN_LEVELS, [True, False], AMOUNT_TIERS, [0.1, 0.5, 1.0]
        ):
            d = decide_action(CallSignal(level, conf, ()), ctx(new_ben, tier))
            assert d.action in ACTION_LADDER and d.reversible

    def test_user_messages(self):
        call = CallSignal("high", 0.75, ("SYNTHETIC_AUDIO_HIGH", "COERCIVE_LANGUAGE"))
        verify = decide_action(call, ctx(True, "normal"))
        assert verify.user_message == (
            "This call shows signs of urgent payment pressure. The caller's voice may be synthetic or "
            "recorded. The recipient is new. End the call and verify using a number you already trust."
        )
        hold = decide_action(call, ctx(True, "large"))
        assert "paused this transfer for 30 minutes" in hold.user_message
        assert "not frozen" in hold.user_message and hold.hold_seconds == 1800
        assert decide_action(call, ctx(False, "normal")).user_message.endswith("Take a moment before you continue.")

    def test_rationale_is_traceable(self):
        d = decide_action(CallSignal("high", 0.75, ()), ctx(True, "large"))
        assert d.rationale == ("call risk high: start at WARN", "+1 new beneficiary", "+1 unusual amount", "=> HOLD")
        assert d.reason_codes == ("NEW_BENEFICIARY", "UNUSUAL_AMOUNT")

    def test_bad_amount_tier(self):
        with pytest.raises(ValueError):
            decide_action(None, ctx(True, "huge"))

    @pytest.mark.parametrize(
        "amount,currency,history,tier",
        [
            (500, "INR", (), "small"),
            (5000, "INR", (), "normal"),
            (25000, "INR", (), "large"),
            (40, "USD", (), "small"),
            (500, "USD", (), "normal"),
            (1000, "USD", (), "large"),
            (999999, "EUR", (), "normal"),  # no thresholds configured for EUR
            (4000, "INR", [1000] * 10, "large"),  # > 3x median
            (3000, "INR", [1000] * 10, "normal"),  # exactly 3x is not "more than"
            (4000, "INR", [1000] * 4, "normal"),  # too little history
            (900, "INR", [100] * 10, "small"),  # small floor wins over history
        ],
    )
    def test_classify_amount(self, amount, currency, history, tier):
        assert classify_amount(amount, currency, PolicyConfig(), history) == tier


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


class TestTokens:
    def high(self):
        return assess(audio(0.93), coercion("high"))

    def medium(self):
        return assess(audio(0.9), coercion("medium"))

    def test_no_token_below_soft_prompt(self, token_service):
        elevated = assess(audio(0.05), coercion("medium"))
        assert elevated.level == "elevated"
        assert token_service.issue("cust-001", elevated) is None
        assert token_service.store.tokens() == []

    def test_token_is_minimal_and_pseudonymous(self, token_service):
        token = token_service.issue("cust-001", self.high())
        body = token.to_dict()
        assert set(body) == TOKEN_FIELDS
        assert body["subject_ref"].startswith("sub_") and "cust-001" not in json.dumps(body)
        assert body["level"] == "high" and body["expires_at"] - body["issued_at"] == TTL
        assert "score" not in body  # level only, per doc 5.2 step 8
        assert find_privacy_violations(body) == []

    def test_signature_and_tamper_detection(self, token_service):
        token = token_service.issue("cust-001", self.high())
        assert token_service.verify(token)
        assert not token_service.verify(replace(token, level="medium"))
        assert not token_service.verify(replace(token, expires_at=token.expires_at + 3600))
        token_service.store.put(replace(token, level="critical"), TTL)
        assert token_service.lookup("cust-001").status == "invalid"

    def test_other_signing_key_cannot_forge(self, token_service, clock):
        forger = TokenService(signing_key="x" * 32, pseudonym_key=PSEUDONYM_KEY, clock=clock)
        forged = forger.issue("cust-001", self.high())
        assert forged.subject_ref == token_service.subject_ref("cust-001")
        assert not token_service.verify(forged)

    def test_expiry(self, token_service, clock):
        token_service.issue("cust-001", self.high())
        clock.advance(TTL - 1)
        assert token_service.lookup("cust-001").status == "active"
        clock.advance(1)
        assert token_service.lookup("cust-001").status == "expired"
        assert token_service.store.tokens() == []  # expired token deleted on access
        assert token_service.lookup("cust-001").status == "none"

    def test_purge_expired(self, token_service, clock):
        token_service.issue("cust-001", self.high())
        token_service.issue("cust-002", self.medium())
        clock.advance(TTL)
        assert token_service.purge_expired() == 2
        assert token_service.store.tokens() == []

    def test_high_water_mark_refreshes_without_downgrading(self, token_service, clock):
        first = token_service.issue("cust-001", self.high())
        clock.advance(300)
        second = token_service.issue("cust-001", self.medium())
        assert second.level == "high" and second.reason_codes == first.reason_codes
        assert second.expires_at == clock.now + TTL > first.expires_at
        assert token_service.get_active("cust-001").token_id == second.token_id

    def test_upgrade_and_reset_after_expiry(self, token_service, clock):
        assert token_service.issue("cust-001", self.medium()).level == "medium"
        assert token_service.issue("cust-001", self.high()).level == "high"
        clock.advance(TTL)
        assert token_service.issue("cust-001", self.medium()).level == "medium"

    def test_revoke(self, token_service):
        token_service.issue("cust-001", self.high())
        assert token_service.revoke("cust-001") is True
        assert token_service.lookup("cust-001").status == "none"
        assert token_service.revoke("cust-001") is False

    def test_subject_refs_are_keyed(self, token_service, clock):
        same = TokenService(signing_key=SIGNING_KEY, pseudonym_key=PSEUDONYM_KEY, clock=clock)
        other = TokenService(signing_key=SIGNING_KEY, pseudonym_key="another-pseudonym-key-xyz", clock=clock)
        ref = token_service.subject_ref("cust-001")
        assert ref == same.subject_ref("cust-001") != other.subject_ref("cust-001")
        assert ref != token_service.subject_ref("cust-002")
        with pytest.raises(InputValidationError):
            token_service.subject_ref("")

    def test_keys_from_environment(self, monkeypatch, clock):
        monkeypatch.setenv("VERIFVOICE_SIGNING_KEY", SIGNING_KEY)
        monkeypatch.setenv("VERIFVOICE_PSEUDONYM_KEY", PSEUDONYM_KEY)
        a, b = TokenService(clock=clock), TokenService(clock=clock)
        assert b.verify(a.issue("cust-001", self.high()))

    def test_ephemeral_key_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("VERIFVOICE_SIGNING_KEY", raising=False)
        monkeypatch.delenv("VERIFVOICE_PSEUDONYM_KEY", raising=False)
        with caplog.at_level(logging.WARNING, logger="verifvoice.risk"):
            TokenService()
        assert "ephemeral random key" in caplog.text

    def test_short_key_rejected(self):
        with pytest.raises(ValueError):
            TokenService(signing_key="short", pseudonym_key=PSEUDONYM_KEY)

    def test_concurrent_issue_and_lookup(self, token_service):
        errors = []

        def worker(i):
            try:
                for _ in range(50):
                    token_service.issue(f"cust-{i}", self.high())
                    assert token_service.lookup(f"cust-{i}").status == "active"
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        tokens = token_service.store.tokens()
        assert len(tokens) == 16 and all(token_service.verify(t) for t in tokens)

    def test_discard_only_deletes_the_named_token(self, token_service):
        token = token_service.issue("cust-001", self.high())
        assert token_service.store.discard(token.subject_ref, "tok_someone_else") is False
        assert token_service.store.get(token.subject_ref) == token
        assert token_service.store.discard(token.subject_ref, token.token_id) is True
        assert token_service.store.get(token.subject_ref) is None

    def test_lookup_never_deletes_a_token_refreshed_meanwhile(self, clock):
        class RacingStore(InMemoryTokenStore):
            on_get = None

            def get(self, subject_ref):
                token = super().get(subject_ref)
                hook, self.on_get = self.on_get, None
                if hook:
                    hook()  # another request issues a fresh token right after our read
                return token

        store = RacingStore()
        service = TokenService(store=store, signing_key=SIGNING_KEY, pseudonym_key=PSEUDONYM_KEY, clock=clock)
        service.issue("cust-001", self.high())
        clock.advance(TTL)  # the stored token is now expired
        store.on_get = lambda: service.issue("cust-001", self.high())
        found = service.lookup("cust-001")
        assert found.status == "active" and found.token.expires_at == clock.now + TTL

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1800000000"])
    def test_from_dict_rejects_non_finite_timestamps(self, token_service, bad):
        body = token_service.issue("cust-001", self.high()).to_dict()
        with pytest.raises(InputValidationError, match="expires_at"):
            RiskToken.from_dict({**body, "expires_at": bad})

    def test_from_dict_rejects_extra_fields(self, token_service):
        body = token_service.issue("cust-001", self.high()).to_dict()
        assert RiskToken.from_dict(body).to_dict() == body
        with pytest.raises(InputValidationError, match="extra"):
            RiskToken.from_dict({**body, "transcript": "send the money"})
        with pytest.raises(InputValidationError):
            RiskToken.from_dict({**body, "level": "elevated"})


class TestRedisStore:
    @pytest.fixture
    def redis_service(self, clock):
        fakeredis = pytest.importorskip("fakeredis")
        self.client = fakeredis.FakeRedis()
        store = RedisTokenStore(self.client)
        return TokenService(store=store, signing_key=SIGNING_KEY, pseudonym_key=PSEUDONYM_KEY, ttl_seconds=TTL, clock=clock)

    def test_round_trip_with_native_ttl(self, redis_service):
        token = redis_service.issue("cust-001", assess(audio(0.93), coercion("high")))
        found = redis_service.lookup("cust-001")
        assert found.status == "active" and found.token == token
        key = "verifvoice:token:" + token.subject_ref
        assert 0 < self.client.pttl(key) <= TTL * 1000
        assert [t.token_id for t in redis_service.store.tokens()] == [token.token_id]
        assert redis_service.revoke("cust-001") and redis_service.lookup("cust-001").status == "none"

    def test_redis_evicts_on_ttl(self, clock):
        fakeredis = pytest.importorskip("fakeredis")
        store = RedisTokenStore(fakeredis.FakeRedis(decode_responses=True))
        service = TokenService(
            store=store, signing_key=SIGNING_KEY, pseudonym_key=PSEUDONYM_KEY, ttl_seconds=0.05, clock=time.time
        )
        service.issue("cust-001", assess(audio(0.93), coercion("high")))
        assert service.lookup("cust-001").status == "active"
        time.sleep(0.2)
        assert service.lookup("cust-001").status == "none"

    def test_discard_only_deletes_the_named_token(self, redis_service):
        token = redis_service.issue("cust-001", assess(audio(0.93), coercion("high")))
        assert redis_service.store.discard(token.subject_ref, "tok_someone_else") is False
        assert redis_service.lookup("cust-001").status == "active"
        assert redis_service.store.discard(token.subject_ref, token.token_id) is True
        assert redis_service.lookup("cust-001").status == "none"

    def test_concurrent_worker_cannot_downgrade_a_token(self, clock):
        fakeredis = pytest.importorskip("fakeredis")
        server = fakeredis.FakeServer()
        keys = dict(signing_key=SIGNING_KEY, pseudonym_key=PSEUDONYM_KEY, ttl_seconds=TTL)
        worker_a = TokenService(store=RedisTokenStore(fakeredis.FakeRedis(server=server)), clock=clock, **keys)
        worker_b = TokenService(store=RedisTokenStore(fakeredis.FakeRedis(server=server)), clock=clock, **keys)
        critical = assess(audio(0.95), coercion("high", verification_refused=True, cues=["FINANCIAL_INSTRUCTION"]))
        medium = assess(audio(0.9), coercion("medium"))
        reads = []

        def racing_clock():
            # Worker A has WATCHed and read the key (empty); worker B commits a
            # critical token before A writes. A's EXEC must fail and retry.
            if not reads:
                worker_b.issue("cust-001", critical)
            reads.append(clock.now)
            return clock.now

        worker_a.clock = racing_clock
        result = worker_a.issue("cust-001", medium)
        assert len(reads) == 2  # first attempt aborted by WATCH, second saw B's token
        assert result.level == "critical"
        assert worker_b.lookup("cust-001").token.level == "critical"

    def test_corrupt_entry_is_invalid_not_a_crash(self, redis_service):
        ref = redis_service.subject_ref("cust-001")
        self.client.set("verifvoice:token:" + ref, "{not json")
        assert redis_service.lookup("cust-001").status == "invalid"
        assert redis_service.issue("cust-001", assess(audio(0.93), coercion("high"))) is not None


# ---------------------------------------------------------------------------
# Privacy guards
# ---------------------------------------------------------------------------


class TestPrivacy:
    @pytest.mark.parametrize(
        "record",
        [
            {"event": "x", "transcript": "send money"},
            {"event": "x", "nested": {"audio": "AAAA"}},
            {"event": "x", "blob": b"\x00\x01"},
            {"event": "x", "note": "+91 98765 43210"},
            {"event": "x", "items": ["cust-001 paid"]},
            {"event": "x", "customer_id": "c1"},
        ],
    )
    def test_audit_log_rejects_content(self, record):
        log = AuditLog(forbidden_values=["cust-001"])
        with pytest.raises(PrivacyViolation):
            log.append(record)
        assert len(log) == 0

    @pytest.mark.parametrize(
        "value", ["2026-10-17", "vv-risk-1.0.0", "tok_0123456789abcdef0123456789abcdef", "12345"]
    )
    def test_no_false_positives(self, value):
        assert find_privacy_violations({"v": value}) == []

    def test_short_raw_ids_do_not_collide_with_minted_ids(self, engine):
        for subject in ("1", "7", "a", "ab1"):
            engine.assess_call(audio(0.93), coercion("high"), subject_id=subject)  # must not raise
        log = AuditLog(forbidden_values=["1"])
        with pytest.raises(PrivacyViolation):
            log.append({"who": "1"})

    def test_rejected_record_leaves_no_token_behind(self, token_service):
        strict = RiskEngine(token_service, audit_log=AuditLog(forbidden_values=["STRONG_PROMPT"]))
        with pytest.raises(PrivacyViolation):
            strict.assess_call(audio(0.93), coercion("high"), subject_id="cust-001")
        assert token_service.store.tokens() == [] and len(strict.audit_log) == 0

    def test_raw_ids_are_checked_per_record(self):
        log = AuditLog()
        with pytest.raises(PrivacyViolation):
            log.append({"event": "x", "note": "paid by cust-001234"}, ("cust-001234",))
        log.append({"event": "x", "note": "nothing personal"}, ("cust-001234",))
        assert log.forbidden_values() == () and len(log) == 1

    def test_short_raw_ids_match_exactly_not_inside_vocabulary(self):
        log = AuditLog()
        log.append({"event": "call_assessed", "band": "STRONG_PROMPT"}, ("call", "high"))
        with pytest.raises(PrivacyViolation):
            log.append({"event": "x", "who": "call"}, ("call",))

    def test_audit_guard_does_not_grow_with_customers(self, engine):
        for i in range(300):
            engine.assess_call(audio(0.93), coercion("high"), subject_id=f"cust-{i:06d}", call_id=f"call-{i:06d}")
        assert engine.audit_log.forbidden_values() == ()
        dumped = json.dumps(engine.audit_log.records())
        assert "cust-000123" not in dumped and "call-000123" not in dumped

    def test_engine_audit_is_content_free(self, engine):
        payload = copy.deepcopy(TEAM_EXAMPLE)
        payload["coercion_analysis"]["transcript"] = "mom I had an accident, send money now"
        result = engine.evaluate(payload, subject_id="cust-001")
        dumped = json.dumps(engine.audit_log.records()) + json.dumps(result)
        assert "send money now" not in dumped and "cust-001" not in dumped
        for record in engine.audit_log.records():
            assert find_privacy_violations(record, ["cust-001"]) == []

    def test_call_id_is_pseudonymised(self, engine):
        engine.assess_call(audio(0.93), coercion("high"), subject_id="cust-001", call_id="call-2026-abc")
        record = engine.audit_log.records()[-1]
        assert record["call_ref"].startswith("call_")
        assert "call-2026-abc" not in json.dumps(record)

    def test_audit_records_are_copies(self):
        log = AuditLog()
        rec = {"event": "x", "codes": ["A"]}
        log.append(rec)
        rec["codes"].append("transcript text")
        log.records()[0]["codes"].append("y")
        assert log.records() == [{"event": "x", "codes": ["A"]}]


# ---------------------------------------------------------------------------
# Deletion checks (doc 7.3)
# ---------------------------------------------------------------------------


class TestDeletionChecks:
    def test_zeroize_bytearray_and_pcm_array(self):
        buf = bytearray(b"\x01\x02" * 8000)
        pcm = array.array("h", [123, -456, 789] * 1000)  # int16 PCM ring buffer
        assert not is_zeroized(buf) and not is_zeroized(pcm)
        zeroize(buf)
        zeroize(pcm)
        assert is_zeroized(buf) and is_zeroized(pcm) and len(pcm) == 3000
        assert is_zeroized(bytearray()) and is_zeroized(memoryview(bytes(10)))

    def test_zeroize_rejects_immutable(self):
        with pytest.raises(TypeError, match="read-only"):
            zeroize(b"\x01\x02")

    def test_non_contiguous_view(self):
        data = bytearray(b"\x00\x01" * 10)
        assert is_zeroized(memoryview(data)[::2]) and not is_zeroized(memoryview(data)[1::2])

    def test_find_media_artifacts(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "call.WAV").write_bytes(b"RIFF")
        (tmp_path / "call_transcript.txt").write_text("hello")
        (tmp_path / "metrics.json").write_text("{}")
        found = find_media_artifacts([tmp_path])
        assert sorted(p.name for p in found) == ["call.WAV", "call_transcript.txt"]

    def test_full_lifecycle_passes(self, engine, clock, tmp_path):
        ring = bytearray(b"\x7f" * 32000)  # stand-in for the audio module's ring buffer
        engine.evaluate(TEAM_EXAMPLE, subject_id="cust-001")
        zeroize(ring)  # call ended
        clock.advance(TTL)
        engine.token_service.purge_expired()
        report = run_deletion_checks(
            token_service=engine.token_service,
            audit_logs=[engine.audit_log],
            buffers=[ring],
            artifact_dirs=[tmp_path],
        )
        assert report.passed, report.to_dict()
        assert [c.name for c in report.checks] == [
            "audio_buffers_zeroized",
            "no_expired_tokens_retained",
            "tokens_minimal_and_signed",
            "audit_log[0]_content_free",
            "no_media_or_transcript_files",
        ]
        json.dumps(report.to_dict())

    def test_detects_each_failure(self, engine, clock, tmp_path):
        engine.evaluate(TEAM_EXAMPLE, subject_id="cust-001")
        clock.advance(TTL + 1)  # expired but not purged
        (tmp_path / "leak.wav").write_bytes(b"RIFF")
        report = run_deletion_checks(
            token_service=engine.token_service,
            buffers=[bytearray(b"\x01")],
            artifact_dirs=[tmp_path],
        )
        failed = {c.name for c in report.checks if not c.passed}
        assert failed == {"audio_buffers_zeroized", "no_expired_tokens_retained", "no_media_or_transcript_files"}
        assert not report.passed

    def test_deletion_check_searches_for_given_raw_ids(self):
        log = AuditLog()
        log.append({"event": "x", "note": "ok"})
        log._records.append({"event": "imported", "note": "legacy row for cust-001234"})  # bypassed the guard
        report = run_deletion_checks(audit_logs=[log], raw_identifiers=["cust-001234"])
        assert not report.passed
        assert run_deletion_checks(audit_logs=[log]).passed  # the ID is only known when passed in

    @pytest.mark.parametrize("dtype", ["float32", "int16", ">i2", "<f4"])
    @pytest.mark.parametrize("shape", [(16000,), (50, 320)])
    def test_numpy_ring_buffers(self, dtype, shape):
        np = pytest.importorskip("numpy")
        ring = np.ones(shape, dtype=dtype)
        assert not is_zeroized(ring)
        zeroize(ring)
        assert is_zeroized(ring) and not ring.any()

    def test_numpy_strided_view(self):
        np = pytest.importorskip("numpy")
        view = np.ones(32000, dtype=np.int16)[::2]
        assert not is_zeroized(view)
        with pytest.raises(TypeError, match="C-contiguous"):
            zeroize(view)

    def test_detects_tampered_token(self, engine):
        _, token = engine.assess_call(audio(0.93), coercion("high"), subject_id="cust-001")
        engine.token_service.store.put(replace(token, confidence=1.0), TTL)
        report = run_deletion_checks(token_service=engine.token_service)
        assert {c.name: c.passed for c in report.checks}["tokens_minimal_and_signed"] is False


# ---------------------------------------------------------------------------
# One-shot evaluate() on the team payload
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_team_example_suspicious(self, engine):
        result = engine.evaluate(TEAM_EXAMPLE, subject_id="cust-001")
        json.dumps(result)  # JSON-ready for FastAPI
        assert result["call_risk"]["level"] == "high"
        assert result["token"]["level"] == "high"
        assert result["transaction"]["amount_tier"] == "normal"
        assert result["decision"]["action"] == VERIFY_OUT_OF_BAND
        assert result["decision"]["token_id"] == result["token"]["token_id"]
        assert result["decision"]["reason_codes"] == ["SYNTHETIC_AUDIO_HIGH", "COERCIVE_LANGUAGE", "NEW_BENEFICIARY"]

    def test_safe_transaction(self, engine):
        payload = {
            "audio_analysis": {"status": "ok", "prediction": "likely_bonafide", "spoof_score": 0.04},
            "coercion_analysis": {"status": "ok", "coercion_level": "low"},
            "transaction": {"amount": 60, "currency": "USD", "new_beneficiary": False},
        }
        result = engine.evaluate(payload, subject_id="cust-001")
        assert result["call_risk"]["level"] == "low"
        assert result["token"] is None
        assert result["decision"]["action"] == ALLOW and result["decision"]["user_message"] == ""

    def test_suspicious_large_amount_is_held(self, engine):
        payload = copy.deepcopy(TEAM_EXAMPLE)
        payload["transaction"]["amount"] = 5000
        result = engine.evaluate(payload, subject_id="cust-001")
        assert result["decision"]["action"] == HOLD and result["decision"]["hold_seconds"] == 1800

    @pytest.mark.parametrize("amount,action", [(500, WARN), (5000, VERIFY_OUT_OF_BAND)])
    def test_human_scammer_team_payload_gets_friction(self, engine, amount, action):
        payload = {
            "audio_analysis": {"status": "ok", "prediction": "likely_bonafide", "spoof_score": 0.05},
            "coercion_analysis": {"status": "ok", "coercion_level": "high"},
            "transaction": {"amount": amount, "currency": "USD", "new_beneficiary": True},
        }
        result = engine.evaluate(payload, subject_id="cust-007")
        assert result["call_risk"]["score"] == pytest.approx(0.55)
        assert result["token"]["level"] == "medium"
        assert result["decision"]["action"] == action
        assert result["decision"]["action"] != ALLOW

    def test_aac_user_paying_new_payee_gets_no_friction(self, engine):
        payload = copy.deepcopy(TEAM_EXAMPLE)
        payload["coercion_analysis"]["coercion_level"] = "none"
        result = engine.evaluate(payload, subject_id="cust-aac")
        assert result["token"] is None and result["decision"]["action"] == ALLOW

    def test_calm_window_does_not_erase_earlier_risk(self, engine):
        engine.assess_call(audio(0.93), coercion("high"), subject_id="cust-001")  # earlier in the call
        calm = copy.deepcopy(TEAM_EXAMPLE)
        calm["audio_analysis"] = audio(0.05)
        calm["coercion_analysis"] = coercion("none")
        result = engine.evaluate(calm, subject_id="cust-001")
        assert result["call_risk"]["level"] == "low"
        assert result["token"]["level"] == "high"
        assert result["decision"]["action"] == VERIFY_OUT_OF_BAND

    def test_calm_window_after_token_expiry_is_allowed(self, engine, clock):
        engine.assess_call(audio(0.93), coercion("high"), subject_id="cust-001")
        clock.advance(TTL)
        calm = copy.deepcopy(TEAM_EXAMPLE)
        calm["coercion_analysis"] = coercion("none")
        result = engine.evaluate(calm, subject_id="cust-001")
        assert result["token"] is None and result["decision"]["action"] == ALLOW

    def test_without_subject_no_token_but_same_decision(self, engine):
        result = engine.evaluate(TEAM_EXAMPLE)
        assert result["token"] is None
        assert result["decision"]["action"] == VERIFY_OUT_OF_BAND
        assert engine.token_service.store.tokens() == []

    def test_existing_stronger_token_is_used(self, engine):
        engine.assess_call(audio(0.95), coercion("high", verification_refused=True, cues=["FINANCIAL_INSTRUCTION"]),
                           subject_id="cust-001")
        result = engine.evaluate(TEAM_EXAMPLE, subject_id="cust-001")
        assert result["call_risk"]["level"] == "high"
        assert result["token"]["level"] == "critical"
        assert result["decision"]["action"] == HOLD

    @pytest.mark.parametrize("bad", [None, [], {"audio_analysis": {}}, {**TEAM_EXAMPLE, "transaction": None}])
    def test_malformed_payload(self, engine, bad):
        with pytest.raises(InputValidationError):
            engine.evaluate(bad)

"""VerifVoice risk fusion, expiring risk tokens, and privacy / deletion checks.

This module owns four things (design doc sections in brackets):

1. Call-risk fusion [6.2-6.5]: acoustic + coercion signals -> score, band,
   level, confidence and reason codes.
2. Transaction fusion [5.2 steps 9-10]: call-risk signal + payment context ->
   a proportionate, always-reversible action (ALLOW / WARN /
   VERIFY_OUT_OF_BAND / HOLD). Nothing here ever blocks or freezes.
3. Expiring risk tokens [5.2 step 8, 7.2]: signed, pseudonymous, reason-coded,
   short-lived. Only issued at the soft-prompt band (>= 0.55) and above.
4. Privacy guards and deletion checks [7.2-7.3]: content-free audit log,
   buffer zeroisation check, token-store expiry check, media-file scan.

It also exposes :func:`decide`, the flat **day-one** fusion formula the team
locks to (``R = 0.5*C + 0.3*(S*C) + 0.2*N``), which is what the FastAPI
``/payment/decision`` endpoint calls. The richer two-stage
:class:`RiskEngine` / :class:`BankSimulator` machinery above implements the
"original design" formula and remains available for the fuller path.

Standard library only (Python >= 3.9), so it runs unchanged on Windows, macOS
and IBM LinuxONE (s390x). ``RedisTokenStore`` needs the pure-Python ``redis``
package, but only if you use it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import statistics
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger("verifvoice.risk")

SCHEMA_VERSION = "1"
POLICY_VERSION = "vv-risk-1.0.0"

# --------------------------------------------------------------------------
# Fixed, auditable vocabulary [6.5]
# --------------------------------------------------------------------------

SYNTHETIC_AUDIO_HIGH = "SYNTHETIC_AUDIO_HIGH"
COERCIVE_URGENCY = "COERCIVE_URGENCY"
SECRECY_REQUEST = "SECRECY_REQUEST"
AUTHORITY_CLAIM = "AUTHORITY_CLAIM"
FINANCIAL_INSTRUCTION = "FINANCIAL_INSTRUCTION"
VERIFICATION_REFUSED = "VERIFICATION_REFUSED"
SPEAKER_HANDOFF = "SPEAKER_HANDOFF"
NEW_BENEFICIARY = "NEW_BENEFICIARY"
UNUSUAL_TIME = "UNUSUAL_TIME"  # in the doc vocabulary; not emitted yet
# Extensions to the doc vocabulary:
COERCIVE_LANGUAGE = "COERCIVE_LANGUAGE"  # coercion module only sends a level
UNUSUAL_AMOUNT = "UNUSUAL_AMOUNT"  # explains amount-driven escalation

# Cues the coercion module may send in the optional ``cues`` list.
CALL_CUE_CODES = frozenset(
    {
        COERCIVE_URGENCY,
        SECRECY_REQUEST,
        AUTHORITY_CLAIM,
        FINANCIAL_INSTRUCTION,
        VERIFICATION_REFUSED,
        SPEAKER_HANDOFF,
    }
)
CALL_REASON_CODES = CALL_CUE_CODES | {SYNTHETIC_AUDIO_HIGH, COERCIVE_LANGUAGE}
TRANSACTION_REASON_CODES = frozenset({NEW_BENEFICIARY, UNUSUAL_AMOUNT, UNUSUAL_TIME})
REASON_CODES = CALL_REASON_CODES | TRANSACTION_REASON_CODES

# --------------------------------------------------------------------------
# Team (P1/P2/P3) reason-code vocabulary <-> this package's internal codes.
#
# The rest of the team uses: COERCIVE_URGENCY, AUTHORITY_CLAIM, SECRECY_PRESSURE,
# VERIFICATION_REFUSAL, FINANCIAL_INSTRUCTION, THREAT, EMERGENCY_NARRATIVE,
# NEW_PAYEE, SPOOF_UNAVAILABLE. This package historically used SECRECY_REQUEST /
# VERIFICATION_REFUSED / NEW_BENEFICIARY. Translate at the boundary so the two
# vocabularies never leak into each other.
# --------------------------------------------------------------------------

TEAM_REASON_CODES = frozenset(
    {
        COERCIVE_URGENCY,
        AUTHORITY_CLAIM,
        "SECRECY_PRESSURE",
        "VERIFICATION_REFUSAL",
        FINANCIAL_INSTRUCTION,
        "THREAT",
        "EMERGENCY_NARRATIVE",
        "NEW_PAYEE",
        "SPOOF_UNAVAILABLE",
    }
)

# Team cue -> this package's internal cue (None = the team cue has no internal
# equivalent; it is carried as a reason code only and never scored on its own).
TEAM_CUE_TO_INTERNAL: Mapping[str, Optional[str]] = {
    COERCIVE_URGENCY: COERCIVE_URGENCY,
    AUTHORITY_CLAIM: AUTHORITY_CLAIM,
    "SECRECY_PRESSURE": SECRECY_REQUEST,
    "VERIFICATION_REFUSAL": VERIFICATION_REFUSED,
    FINANCIAL_INSTRUCTION: FINANCIAL_INSTRUCTION,
    "THREAT": None,               # already folded into coercion_score by P2's rules
    "EMERGENCY_NARRATIVE": None,  # context only, never scored (primer §8)
}

# This package's internal code -> team reason code (for responses).
INTERNAL_TO_TEAM_CODE: Mapping[str, str] = {
    COERCIVE_URGENCY: COERCIVE_URGENCY,
    SECRECY_REQUEST: "SECRECY_PRESSURE",
    AUTHORITY_CLAIM: AUTHORITY_CLAIM,
    FINANCIAL_INSTRUCTION: FINANCIAL_INSTRUCTION,
    VERIFICATION_REFUSED: "VERIFICATION_REFUSAL",
    NEW_BENEFICIARY: "NEW_PAYEE",
}


def translate_codes_to_team(codes: Iterable[str]) -> Tuple[str, ...]:
    """Map internal reason codes to the team's vocabulary (for API responses)."""
    return tuple(dict.fromkeys(INTERNAL_TO_TEAM_CODE.get(c, c) for c in codes))


COERCION_LEVELS = ("none", "low", "medium", "high")
PHASES = ("P0", "P1", "P2", "P3", "P4")

# Call-side bands [6.4]
PASSIVE_MONITORING = "PASSIVE_MONITORING"
ELEVATED_WATCH = "ELEVATED_WATCH"
SOFT_PROMPT = "SOFT_PROMPT"
STRONG_PROMPT = "STRONG_PROMPT"
CRITICAL_ALERT = "CRITICAL_ALERT"

LEVEL_BY_BAND = {
    PASSIVE_MONITORING: "low",
    ELEVATED_WATCH: "elevated",
    SOFT_PROMPT: "medium",
    STRONG_PROMPT: "high",
    CRITICAL_ALERT: "critical",
}
LEVEL_RANK = {"low": 0, "elevated": 1, "medium": 2, "high": 3, "critical": 4}
# Levels that produce a token for the bank (soft-prompt band and above).
TOKEN_LEVELS = ("medium", "high", "critical")

# Bank-side actions, in order of friction. There is deliberately no BLOCK or
# FREEZE. Step-up auth (OTP/PIN) is left out on purpose: the victim of a
# coercion scam authenticates voluntarily [2.1], so it adds friction without
# breaking the scammer's hold.
ALLOW = "ALLOW"
WARN = "WARN"
VERIFY_OUT_OF_BAND = "VERIFY_OUT_OF_BAND"
HOLD = "HOLD"
ACTION_LADDER = (ALLOW, WARN, VERIFY_OUT_OF_BAND, HOLD)

AMOUNT_TIERS = ("small", "normal", "large")


class InputValidationError(ValueError):
    """Malformed input from an upstream module. Map to HTTP 422."""


class PrivacyViolation(RuntimeError):
    """A record would have stored call content or a raw identifier."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _check_unit(value: float, name: str) -> None:
    if not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True)
class FusionConfig:
    """Call-risk fusion parameters. Defaults follow design doc 6.2-6.4."""

    w_spoof: float = 0.40
    w_semantic: float = 0.35
    w_speaker: float = 0.25
    refusal_boost: float = 0.25
    # Neutral priors used when a measurement is missing: unknown is neither
    # evidence of a scam nor evidence of safety.
    spoof_prior: float = 0.5
    speaker_match_prior: float = 0.5
    # coercion_level -> semantic score fed into the formula. Ignored when the
    # coercion module sends a numeric ``coercion_score``.
    coercion_level_scores: Mapping[str, float] = field(
        default_factory=lambda: {"none": 0.0, "low": 0.25, "medium": 0.60, "high": 0.90}
    )
    # Intent gate (doc 8, AAC safety): highest call risk each coercion level
    # may reach. none/low stay below the token threshold, so synthetic audio on
    # its own can never put a token in front of the bank.
    coercion_level_ceilings: Mapping[str, float] = field(
        default_factory=lambda: {"none": 0.50, "low": 0.50, "medium": 0.75, "high": 1.00}
    )
    # Intent floor: lowest call risk each coercion level guarantees. High
    # coercion alone reaches the soft-prompt band, so a human (non-cloned)
    # scammer still puts a token in front of the bank. Medium stays at 0 so
    # genuine-voice real emergencies are not escalated. Phase ceilings still
    # apply on top.
    coercion_level_floors: Mapping[str, float] = field(
        default_factory=lambda: {"none": 0.0, "low": 0.0, "medium": 0.0, "high": 0.55}
    )
    # Phase-aware ceilings [6.3], applied only when a phase is known.
    phase_ceilings: Mapping[str, float] = field(
        default_factory=lambda: {"P0": 0.50, "P1": 0.65, "P2": 0.75, "P3": 0.90, "P4": 1.00}
    )
    elevated_threshold: float = 0.30
    soft_prompt_threshold: float = 0.55
    strong_prompt_threshold: float = 0.75
    critical_threshold: float = 0.90  # strictly greater, plus financial intent
    synthetic_high_threshold: float = 0.80

    def __post_init__(self) -> None:
        for name in (
            "w_spoof",
            "w_semantic",
            "w_speaker",
            "refusal_boost",
            "spoof_prior",
            "speaker_match_prior",
            "synthetic_high_threshold",
        ):
            _check_unit(getattr(self, name), name)
        if self.w_spoof + self.w_semantic + self.w_speaker <= 0:
            raise ValueError("fusion weights must not all be zero")
        thresholds = (
            self.elevated_threshold,
            self.soft_prompt_threshold,
            self.strong_prompt_threshold,
            self.critical_threshold,
        )
        if not all(0.0 < t <= 1.0 for t in thresholds) or list(thresholds) != sorted(set(thresholds)):
            raise ValueError("band thresholds must be strictly increasing within (0, 1]")
        for table, keys in (
            ("coercion_level_scores", COERCION_LEVELS),
            ("coercion_level_ceilings", COERCION_LEVELS),
            ("coercion_level_floors", COERCION_LEVELS),
            ("phase_ceilings", PHASES),
        ):
            mapping = getattr(self, table)
            missing = [k for k in keys if k not in mapping]
            if missing:
                raise ValueError(f"{table} is missing {missing}")
            for k in keys:
                _check_unit(mapping[k], f"{table}[{k!r}]")
        for level in COERCION_LEVELS:
            if self.coercion_level_floors[level] > self.coercion_level_ceilings[level]:
                raise ValueError(f"coercion_level_floors[{level!r}] exceeds its ceiling")


@dataclass(frozen=True)
class PolicyConfig:
    """Bank-side proportionate-action policy. All values are prototype defaults."""

    # Starting rung on ACTION_LADDER for each token level.
    base_steps: Mapping[str, int] = field(
        default_factory=lambda: {"medium": 0, "high": 1, "critical": 2}
    )
    # Absolute amount bands per currency. Unknown currency -> "normal" unless
    # the customer's history says otherwise.
    small_amount: Mapping[str, float] = field(
        default_factory=lambda: {"INR": 1000.0, "USD": 50.0}
    )
    large_amount: Mapping[str, float] = field(
        default_factory=lambda: {"INR": 25000.0, "USD": 1000.0}
    )
    # History-relative "large": amount > multiplier x median of past payments.
    history_multiplier: float = 3.0
    min_history: int = 5
    history_window: int = 50
    # Tokens below this confidence earn one rung less friction.
    min_confidence: float = 0.5
    # Brief, reversible pause [5.2 step 10]. Released by out-of-band
    # verification or by the customer confirming after it ends.
    hold_seconds: int = 30 * 60
    # A payee stays "new" this long after being added, even once paid
    # (stops the "send 1 rupee first, then 50,000" trick).
    new_beneficiary_window_seconds: int = 24 * 60 * 60

    def __post_init__(self) -> None:
        if set(self.base_steps) != set(TOKEN_LEVELS):
            raise ValueError(f"base_steps must have exactly the keys {TOKEN_LEVELS}")
        for level, step in self.base_steps.items():
            if not (isinstance(step, int) and 0 <= step < len(ACTION_LADDER)):
                raise ValueError(f"base_steps[{level!r}] must be 0..{len(ACTION_LADDER) - 1}")
        for currency, small in self.small_amount.items():
            large = self.large_amount.get(currency)
            if large is not None and not small < large:
                raise ValueError(f"small_amount[{currency}] must be < large_amount[{currency}]")
        _check_unit(self.min_confidence, "min_confidence")
        if self.history_multiplier <= 1 or self.min_history < 1 or self.history_window < self.min_history:
            raise ValueError("invalid history settings")
        if self.hold_seconds <= 0 or self.new_beneficiary_window_seconds < 0:
            raise ValueError("hold_seconds must be > 0 and the beneficiary window >= 0")


DEFAULT_FUSION = FusionConfig()
DEFAULT_POLICY = PolicyConfig()


# --------------------------------------------------------------------------
# Input contract (team format + optional extensions)
# --------------------------------------------------------------------------


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValidationError(f"{name} must be an object, got {type(value).__name__}")
    return value


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{name} must be a number in [0, 1], got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InputValidationError(f"{name} must be in [0, 1], got {value!r}")
    return value


def _opt_unit(value: Any, name: str) -> Optional[float]:
    return None if value is None else _unit(value, name)


def _opt_str(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InputValidationError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _opt_bool(value: Any, name: str) -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise InputValidationError(f"{name} must be true/false, got {type(value).__name__}")
    return value


def _status(data: Mapping[str, Any], name: str) -> str:
    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        raise InputValidationError(f"{name}.status is required (e.g. 'ok')")
    return status.strip().lower()


@dataclass(frozen=True)
class AudioAnalysis:
    """Output of the acoustic / anti-spoofing module.

    Team format: ``{"status": "ok", "prediction": "likely_spoof",
    "spoof_score": 0.93}``. Optional extensions: ``speaker_match`` (0..1),
    ``quality`` (0..1, from the quality gate), ``model_version``.
    Any status other than "ok" means the signal is unavailable.
    """

    available: bool
    status: str
    spoof_score: Optional[float] = None
    prediction: Optional[str] = None
    speaker_match: Optional[float] = None
    quality: Optional[float] = None
    model_version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "AudioAnalysis":
        if data is None:
            return cls(available=False, status="missing")
        data = _mapping(data, "audio_analysis")
        status = _status(data, "audio_analysis")
        if status != "ok":
            return cls(available=False, status=status)
        if "spoof_score" not in data:
            raise InputValidationError("audio_analysis.spoof_score is required when status is 'ok'")
        return cls(
            available=True,
            status=status,
            spoof_score=_unit(data["spoof_score"], "audio_analysis.spoof_score"),
            prediction=_opt_str(data.get("prediction"), "audio_analysis.prediction"),
            speaker_match=_opt_unit(data.get("speaker_match"), "audio_analysis.speaker_match"),
            quality=_opt_unit(data.get("quality"), "audio_analysis.quality"),
            model_version=_opt_str(data.get("model_version"), "audio_analysis.model_version"),
        )


def coercion_level_from_score(score: float) -> str:
    """Map a numeric coercion score (0..1) to the nearest coercion level.

    This is the inverse of ``FusionConfig.coercion_level_scores``
    (none 0.0, low 0.25, medium 0.60, high 0.90), using midpoint thresholds.
    The team's pipeline emits a continuous ``coercion_score`` rather than a
    discrete level, so this lets ``CoercionAnalysis`` consume that format.
    """
    _unit(score, "coercion_score")
    if score < 0.125:
        return "none"
    if score < 0.425:
        return "low"
    if score < 0.75:
        return "medium"
    return "high"


@dataclass(frozen=True)
class CoercionAnalysis:
    """Output of the coercive-language module.

    Team format: ``{"status": "ok", "coercion_level": "high"}`` where the level
    is one of none/low/medium/high. Optional extensions: ``coercion_score``
    (0..1), ``cues`` (list of reason codes), ``verification_refused`` (bool),
    ``phase`` ("P0".."P4"), ``financial_intent`` (bool), ``model_version``.
    A status other than "ok" (e.g. "unsupported_language") means abstain.

    ``coercion_level`` may be omitted when ``coercion_score`` is present; the
    level is then derived with :func:`coercion_level_from_score`. ``cues`` may
    use either the team's vocabulary or this package's internal cue codes.
    """

    available: bool
    status: str
    level: Optional[str] = None
    score: Optional[float] = None
    cues: Tuple[str, ...] = ()
    verification_refused: bool = False
    phase: Optional[str] = None
    financial_intent: Optional[bool] = None
    model_version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "CoercionAnalysis":
        if data is None:
            return cls(available=False, status="missing")
        data = _mapping(data, "coercion_analysis")
        status = _status(data, "coercion_analysis")
        if status != "ok":
            return cls(available=False, status=status)
        level = data.get("coercion_level")
        if level is None and "coercion_score" in data:
            # Team pipeline sends a numeric score, not a level. Derive the level.
            level = coercion_level_from_score(data["coercion_score"])
        if not isinstance(level, str) or level.strip().lower() not in COERCION_LEVELS:
            raise InputValidationError(
                f"coercion_analysis.coercion_level must be one of {COERCION_LEVELS}, got {level!r}"
            )
        raw_cues = data.get("cues", ())
        if isinstance(raw_cues, (str, bytes)) or not isinstance(raw_cues, Sequence):
            raise InputValidationError("coercion_analysis.cues must be a list of strings")
        cues: List[str] = []
        for cue in raw_cues:
            code = cue.strip().upper() if isinstance(cue, str) else None
            if code not in CALL_CUE_CODES and code not in TEAM_CUE_TO_INTERNAL:
                raise InputValidationError(
                    f"unknown coercion cue {cue!r}; allowed: "
                    f"{sorted(CALL_CUE_CODES | set(TEAM_CUE_TO_INTERNAL))}"
                )
            internal = TEAM_CUE_TO_INTERNAL.get(code, code)
            if internal is not None and internal not in cues:
                cues.append(internal)
        phase = _opt_str(data.get("phase"), "coercion_analysis.phase")
        if phase is not None:
            phase = phase.strip().upper()
            if phase not in PHASES:
                raise InputValidationError(f"coercion_analysis.phase must be one of {PHASES}")
        refused = bool(_opt_bool(data.get("verification_refused"), "coercion_analysis.verification_refused"))
        return cls(
            available=True,
            status=status,
            level=level.strip().lower(),
            score=_opt_unit(data.get("coercion_score"), "coercion_analysis.coercion_score"),
            cues=tuple(cues),
            verification_refused=refused or VERIFICATION_REFUSED in cues,
            phase=phase,
            financial_intent=_opt_bool(data.get("financial_intent"), "coercion_analysis.financial_intent"),
            model_version=_opt_str(data.get("model_version"), "coercion_analysis.model_version"),
        )


def _currency(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value.strip()) != 3 or not value.strip().isalpha():
        raise InputValidationError(f"{name} must be a 3-letter currency code, got {value!r}")
    return value.strip().upper()


def _amount(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise InputValidationError(f"{name} must be a positive finite number, got {value!r}")
    return value


@dataclass(frozen=True)
class Transaction:
    """Team format: ``{"amount": 500, "currency": "USD", "new_beneficiary": true}``.

    Optional: ``beneficiary_id`` (lets the bank detect novelty itself instead
    of trusting the client's flag) and ``channel`` ("UPI", "QR", ...).
    """

    amount: float
    currency: str
    new_beneficiary: Optional[bool] = None
    beneficiary_id: Optional[str] = None
    channel: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any, require_new_beneficiary: bool = True) -> "Transaction":
        data = _mapping(data, "transaction")
        new_beneficiary = _opt_bool(data.get("new_beneficiary"), "transaction.new_beneficiary")
        if require_new_beneficiary and new_beneficiary is None:
            raise InputValidationError("transaction.new_beneficiary (true/false) is required")
        return cls(
            amount=_amount(data.get("amount"), "transaction.amount"),
            currency=_currency(data.get("currency"), "transaction.currency"),
            new_beneficiary=new_beneficiary,
            beneficiary_id=_opt_str(data.get("beneficiary_id"), "transaction.beneficiary_id"),
            channel=_opt_str(data.get("channel"), "transaction.channel"),
        )


# --------------------------------------------------------------------------
# Call-risk fusion [6.2-6.4]
# --------------------------------------------------------------------------


def _dedupe(codes: Iterable[str]) -> Tuple[str, ...]:
    seen: List[str] = []
    for code in codes:
        if code not in seen:
            seen.append(code)
    return tuple(seen)


def band_for_score(score: float, financial_intent: bool, config: FusionConfig = DEFAULT_FUSION) -> str:
    """Map a call-risk score to its 6.4 band (intervals are [low, high))."""
    if score > config.critical_threshold and financial_intent:
        return CRITICAL_ALERT
    if score >= config.strong_prompt_threshold:
        return STRONG_PROMPT
    if score >= config.soft_prompt_threshold:
        return SOFT_PROMPT
    if score >= config.elevated_threshold:
        return ELEVATED_WATCH
    return PASSIVE_MONITORING


@dataclass(frozen=True)
class CallRiskAssessment:
    score: float  # after floor and ceilings, clipped to [0, 1], rounded to 4 dp
    raw_score: float  # formula output before floor / ceilings / clipping
    band: str
    level: str
    confidence: float  # share of fusion weight backed by real measurements
    reason_codes: Tuple[str, ...]
    financial_intent: bool
    abstained: bool
    abstain_reason: Optional[str]
    ceiling: float
    floor: float
    signals_used: Tuple[str, ...]
    defaults_used: Tuple[str, ...]
    model_versions: Mapping[str, str]
    assessed_at: float
    policy_version: str = POLICY_VERSION

    @property
    def token_eligible(self) -> bool:
        return not self.abstained and self.level in TOKEN_LEVELS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "raw_score": self.raw_score,
            "band": self.band,
            "level": self.level,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "financial_intent": self.financial_intent,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "ceiling": self.ceiling,
            "floor": self.floor,
            "signals_used": list(self.signals_used),
            "defaults_used": list(self.defaults_used),
            "model_versions": dict(self.model_versions),
            "token_eligible": self.token_eligible,
            "assessed_at": self.assessed_at,
            "policy_version": self.policy_version,
        }


def score_call(
    audio: AudioAnalysis,
    coercion: CoercionAnalysis,
    config: FusionConfig = DEFAULT_FUSION,
    now: Optional[float] = None,
) -> CallRiskAssessment:
    """Fuse acoustic and coercion signals into a call-risk assessment.

    call_risk = w_spoof*spoof + w_semantic*semantic + w_speaker*(1 - speaker_match)
                + refusal_boost (if verification refused)
    raised to the coercion-level floor (high coercion alone reaches the
    soft-prompt band, so human scammers are caught), then capped by
    min(intent ceiling, phase ceiling) and clipped to [0, 1].
    Missing coercion -> abstain (no token). Missing acoustic measurements ->
    neutral priors and lower confidence.
    """
    reasons: List[str] = []
    signals_used: List[str] = []
    defaults_used: List[str] = []
    evidence = 0.0

    quality = audio.quality if audio.quality is not None else 1.0
    if audio.available:
        spoof = audio.spoof_score
        signals_used.append("spoof_score")
        evidence += config.w_spoof * quality
        if spoof >= config.synthetic_high_threshold:
            reasons.append(SYNTHETIC_AUDIO_HIGH)
    else:
        spoof = config.spoof_prior
        defaults_used.append("spoof_score")

    if audio.available and audio.speaker_match is not None:
        speaker_match = audio.speaker_match
        signals_used.append("speaker_match")
        evidence += config.w_speaker * quality
    else:
        speaker_match = config.speaker_match_prior
        defaults_used.append("speaker_match")

    abstained = not coercion.available
    abstain_reason = f"coercion signal unavailable (status={coercion.status})" if abstained else None
    refused = False
    explicit_phase = None
    if abstained:
        semantic = 0.0
        level_ceiling = config.coercion_level_ceilings["none"]
        floor = 0.0
        financial_intent = False
    else:
        signals_used.append("coercion")
        evidence += config.w_semantic
        semantic = coercion.score if coercion.score is not None else config.coercion_level_scores[coercion.level]
        refused = coercion.verification_refused
        explicit_phase = coercion.phase
        if coercion.level in ("medium", "high"):
            reasons.append(COERCIVE_LANGUAGE)
        reasons.extend(coercion.cues)
        if refused:
            reasons.append(VERIFICATION_REFUSED)
        financial_intent = bool(
            FINANCIAL_INSTRUCTION in coercion.cues
            or coercion.financial_intent
            or (explicit_phase is not None and explicit_phase in ("P3", "P4"))
        )
        if financial_intent:
            reasons.append(FINANCIAL_INSTRUCTION)
        # Refusal to verify is itself the pressure phase (P4, ceiling 1.0).
        level_ceiling = 1.0 if refused else config.coercion_level_ceilings[coercion.level]
        floor = config.coercion_level_floors[coercion.level]

    raw = (
        config.w_spoof * spoof
        + config.w_semantic * semantic
        + config.w_speaker * (1.0 - speaker_match)
        + (config.refusal_boost if refused else 0.0)
    )
    phase = "P4" if refused else explicit_phase
    phase_ceiling = config.phase_ceilings[phase] if phase else 1.0
    ceiling = min(level_ceiling, phase_ceiling)
    # Floor first, then ceilings: an explicit early phase (P0) still wins.
    score = round(min(max(raw, floor, 0.0), 1.0, ceiling), 4)

    band = band_for_score(score, financial_intent, config)
    if abstained and LEVEL_RANK[LEVEL_BY_BAND[band]] > LEVEL_RANK["elevated"]:
        band = ELEVATED_WATCH  # abstention never accuses, even under a custom config

    model_versions = {}
    if audio.model_version:
        model_versions["acoustic"] = audio.model_version
    if coercion.model_version:
        model_versions["coercion"] = coercion.model_version

    total_weight = config.w_spoof + config.w_semantic + config.w_speaker
    return CallRiskAssessment(
        score=score,
        raw_score=round(raw, 4),
        band=band,
        level=LEVEL_BY_BAND[band],
        confidence=round(evidence / total_weight, 4),
        reason_codes=_dedupe(reasons),
        financial_intent=financial_intent,
        abstained=abstained,
        abstain_reason=abstain_reason,
        ceiling=ceiling,
        floor=floor,
        signals_used=tuple(signals_used),
        defaults_used=tuple(defaults_used),
        model_versions=model_versions,
        assessed_at=time.time() if now is None else now,
    )


# --------------------------------------------------------------------------
# Transaction fusion and proportionate actions [5.2 steps 9-10, 8]
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CallSignal:
    """The part of the call risk the bank is allowed to see."""

    level: str
    confidence: float
    reason_codes: Tuple[str, ...]
    token_id: Optional[str] = None

    @classmethod
    def from_token(cls, token: "RiskToken") -> "CallSignal":
        return cls(token.level, token.confidence, tuple(token.reason_codes), token.token_id)

    @classmethod
    def from_assessment(cls, assessment: CallRiskAssessment) -> Optional["CallSignal"]:
        if not assessment.token_eligible:
            return None
        return cls(assessment.level, assessment.confidence, assessment.reason_codes)


@dataclass(frozen=True)
class TransactionContext:
    amount: float
    currency: str
    new_beneficiary: bool
    amount_tier: str  # "small" | "normal" | "large"


def classify_amount(
    amount: float,
    currency: str,
    policy: PolicyConfig = DEFAULT_POLICY,
    history: Sequence[float] = (),
) -> str:
    """Small (absolute floor) wins first, then absolute or history-relative large."""
    currency = currency.upper()
    small = policy.small_amount.get(currency)
    large = policy.large_amount.get(currency)
    if small is not None and amount < small:
        return "small"
    if large is not None and amount >= large:
        return "large"
    recent = list(history)[-policy.history_window:]
    if len(recent) >= policy.min_history:
        median = statistics.median(recent)
        if median > 0 and amount > policy.history_multiplier * median:
            return "large"
    return "normal"


@dataclass(frozen=True)
class ActionDecision:
    action: str
    reason_codes: Tuple[str, ...]
    user_message: str
    rationale: Tuple[str, ...]
    call_level: Optional[str]
    token_id: Optional[str]
    hold_seconds: Optional[int]
    policy_version: str = POLICY_VERSION
    reversible: bool = True  # every action can be overridden by the customer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "user_message": self.user_message,
            "rationale": list(self.rationale),
            "call_level": self.call_level,
            "token_id": self.token_id,
            "hold_seconds": self.hold_seconds,
            "reversible": self.reversible,
            "policy_version": self.policy_version,
        }


def build_user_message(action: str, reason_codes: Sequence[str], hold_seconds: int) -> str:
    """Customer-facing text, following the wording in design doc 9."""
    if action == ALLOW:
        return ""
    parts: List[str] = []
    if set(reason_codes) & CALL_REASON_CODES:
        parts.append("This call shows signs of urgent payment pressure.")
    if SYNTHETIC_AUDIO_HIGH in reason_codes:
        parts.append("The caller's voice may be synthetic or recorded.")
    if NEW_BENEFICIARY in reason_codes:
        parts.append("The recipient is new.")
    if UNUSUAL_AMOUNT in reason_codes:
        parts.append("This amount is larger than your usual payments.")
    if action == WARN:
        parts.append("Take a moment before you continue.")
    else:
        parts.append("End the call and verify using a number you already trust.")
    if action == HOLD:
        minutes = max(1, round(hold_seconds / 60))
        parts.append(
            f"We have paused this transfer for {minutes} minutes. Your account is not frozen, "
            "and you can release the transfer once you have verified."
        )
    return " ".join(parts)


def decide_action(
    call: Optional[CallSignal],
    txn: TransactionContext,
    policy: PolicyConfig = DEFAULT_POLICY,
) -> ActionDecision:
    """Combine an (unexpired) call-risk signal with payment context.

    Start on the rung for the token level (medium -> ALLOW, high -> WARN,
    critical -> VERIFY_OUT_OF_BAND), +1 for a new beneficiary, +1 for an
    unusual amount, -1 for a small amount, -1 for a low-confidence token.
    Clamp to [ALLOW, HOLD]; critical never drops below WARN. With no call
    signal VerifVoice adds no friction: the bank's own fraud engine still
    applies.
    """
    if txn.amount_tier not in AMOUNT_TIERS:
        raise ValueError(f"amount_tier must be one of {AMOUNT_TIERS}")
    txn_reasons: List[str] = []
    if txn.new_beneficiary:
        txn_reasons.append(NEW_BENEFICIARY)
    if txn.amount_tier == "large":
        txn_reasons.append(UNUSUAL_AMOUNT)

    if call is None or call.level not in TOKEN_LEVELS:
        return ActionDecision(
            action=ALLOW,
            reason_codes=tuple(txn_reasons),
            user_message="",
            rationale=("no active call-risk token: VerifVoice adds no friction",),
            call_level=None,
            token_id=None,
            hold_seconds=None,
        )

    step = policy.base_steps[call.level]
    rationale = [f"call risk {call.level}: start at {ACTION_LADDER[step]}"]
    if txn.new_beneficiary:
        step += 1
        rationale.append("+1 new beneficiary")
    if txn.amount_tier == "large":
        step += 1
        rationale.append("+1 unusual amount")
    elif txn.amount_tier == "small":
        step -= 1
        rationale.append("-1 small amount")
    if call.confidence < policy.min_confidence:
        step -= 1
        rationale.append(f"-1 low confidence ({call.confidence:.2f} < {policy.min_confidence:.2f})")
    floor = ACTION_LADDER.index(WARN) if call.level == "critical" else 0
    step = max(floor, min(step, len(ACTION_LADDER) - 1))
    action = ACTION_LADDER[step]
    rationale.append(f"=> {action}")

    reasons = _dedupe(list(call.reason_codes) + txn_reasons)
    return ActionDecision(
        action=action,
        reason_codes=reasons,
        user_message=build_user_message(action, reasons, policy.hold_seconds),
        rationale=tuple(rationale),
        call_level=call.level,
        token_id=call.token_id,
        hold_seconds=policy.hold_seconds if action == HOLD else None,
    )


# --------------------------------------------------------------------------
# Expiring risk tokens [5.2 step 8, 7.2 items 5-6]
# --------------------------------------------------------------------------

TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "token_id",
        "subject_ref",
        "level",
        "confidence",
        "reason_codes",
        "issued_at",
        "expires_at",
        "retention",
        "signature",
    }
)


@dataclass(frozen=True)
class RiskToken:
    """The only thing that leaves the protected zone: pseudonymous subject
    reference, level, confidence, reason codes, timestamps, retention."""

    token_id: str
    subject_ref: str
    level: str
    confidence: float
    reason_codes: Tuple[str, ...]
    issued_at: float
    expires_at: float
    retention: str
    signature: str = ""
    schema_version: str = SCHEMA_VERSION

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def signing_payload(self) -> bytes:
        body = self.to_dict()
        body.pop("signature")
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "token_id": self.token_id,
            "subject_ref": self.subject_ref,
            "level": self.level,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "retention": self.retention,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RiskToken":
        data = _mapping(data, "token")
        if set(data) != TOKEN_FIELDS:
            extra = sorted(set(data) - TOKEN_FIELDS)
            missing = sorted(TOKEN_FIELDS - set(data))
            raise InputValidationError(f"token fields mismatch (extra={extra}, missing={missing})")
        if data["level"] not in TOKEN_LEVELS:
            raise InputValidationError(f"token level must be one of {TOKEN_LEVELS}")
        codes = data["reason_codes"]
        if not isinstance(codes, list) or not all(c in REASON_CODES for c in codes):
            raise InputValidationError("token reason_codes must be known reason codes")
        for name in ("token_id", "subject_ref", "retention", "signature", "schema_version"):
            if not isinstance(data[name], str):
                raise InputValidationError(f"token {name} must be a string")
        for name in ("issued_at", "expires_at"):
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InputValidationError(f"token {name} must be a finite number")
        return cls(
            token_id=data["token_id"],
            subject_ref=data["subject_ref"],
            level=data["level"],
            confidence=_unit(data["confidence"], "token.confidence"),
            reason_codes=tuple(codes),
            issued_at=data["issued_at"],
            expires_at=data["expires_at"],
            retention=data["retention"],
            signature=data["signature"],
            schema_version=data["schema_version"],
        )


TokenUpdate = Callable[[Optional[RiskToken]], RiskToken]


class TokenStore(Protocol):
    """One active token per pseudonymous subject.

    ``update`` and ``discard`` must be atomic. They stop concurrent API
    workers from overwriting a stronger token or deleting a fresh one.
    """

    def put(self, token: RiskToken, ttl_seconds: float) -> None: ...

    def get(self, subject_ref: str) -> Optional[RiskToken]: ...

    def update(self, subject_ref: str, fn: TokenUpdate, ttl_seconds: float) -> RiskToken:
        """Atomically replace the stored token with ``fn(current_or_None)``."""
        ...

    def discard(self, subject_ref: str, token_id: str) -> bool:
        """Delete the stored token only if it is still ``token_id``."""
        ...

    def delete(self, subject_ref: str) -> bool: ...

    def purge_expired(self, now: float) -> int: ...

    def tokens(self) -> List[RiskToken]: ...


class InMemoryTokenStore:
    """Thread-safe, process-local store. Fine for one API worker and for tests."""

    def __init__(self) -> None:
        self._tokens: Dict[str, RiskToken] = {}
        self._lock = threading.Lock()

    def put(self, token: RiskToken, ttl_seconds: float) -> None:
        with self._lock:
            self._tokens[token.subject_ref] = token

    def get(self, subject_ref: str) -> Optional[RiskToken]:
        with self._lock:
            return self._tokens.get(subject_ref)

    def update(self, subject_ref: str, fn: TokenUpdate, ttl_seconds: float) -> RiskToken:
        with self._lock:
            token = fn(self._tokens.get(subject_ref))
            self._tokens[subject_ref] = token
            return token

    def discard(self, subject_ref: str, token_id: str) -> bool:
        with self._lock:
            current = self._tokens.get(subject_ref)
            if current is None or current.token_id != token_id:
                return False
            del self._tokens[subject_ref]
            return True

    def delete(self, subject_ref: str) -> bool:
        with self._lock:
            return self._tokens.pop(subject_ref, None) is not None

    def purge_expired(self, now: float) -> int:
        with self._lock:
            expired = [ref for ref, tok in self._tokens.items() if tok.is_expired(now)]
            for ref in expired:
                del self._tokens[ref]
            return len(expired)

    def tokens(self) -> List[RiskToken]:
        with self._lock:
            return list(self._tokens.values())


class RedisTokenStore:
    """Redis-backed store; Redis deletes each key itself when its TTL (PX) ends.

    ``client`` is a ``redis.Redis`` (or ``fakeredis.FakeRedis``) instance.
    Use this when several API workers must share tokens.
    """

    max_retries = 16

    def __init__(self, client: Any, prefix: str = "verifvoice:token:") -> None:
        self._client = client
        self._prefix = prefix

    def _key(self, subject_ref: str) -> str:
        return self._prefix + subject_ref

    @staticmethod
    def _encode(token: RiskToken) -> str:
        return json.dumps(token.to_dict(), sort_keys=True)

    def put(self, token: RiskToken, ttl_seconds: float) -> None:
        ttl_ms = int(math.ceil(ttl_seconds * 1000))
        if ttl_ms <= 0:
            self.delete(token.subject_ref)
            return
        self._client.set(self._key(token.subject_ref), self._encode(token), px=ttl_ms)

    def update(self, subject_ref: str, fn: TokenUpdate, ttl_seconds: float) -> RiskToken:
        """Optimistic transaction (WATCH/MULTI): retried if another worker
        writes the same key between our read and our write."""
        from redis.exceptions import WatchError

        key = self._key(subject_ref)
        ttl_ms = max(1, int(math.ceil(ttl_seconds * 1000)))
        for _ in range(self.max_retries):
            with self._client.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    current = None
                    if raw is not None:
                        try:
                            current = self._decode(raw)
                        except InputValidationError as exc:
                            logger.warning("unreadable token in store: %s", exc)
                    token = fn(current)
                    pipe.multi()
                    pipe.set(key, self._encode(token), px=ttl_ms)
                    pipe.execute()
                    return token
                except WatchError:
                    continue
        raise RuntimeError(f"token update kept conflicting; gave up after {self.max_retries} attempts")

    def discard(self, subject_ref: str, token_id: str) -> bool:
        from redis.exceptions import WatchError

        key = self._key(subject_ref)
        with self._client.pipeline() as pipe:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                if raw is None:
                    return False
                try:
                    current = self._decode(raw)
                except InputValidationError:
                    return False
                if current.token_id != token_id:
                    return False
                pipe.multi()
                pipe.delete(key)
                pipe.execute()
                return True
            except WatchError:
                return False  # replaced meanwhile: keep the newer token

    def _decode(self, raw: Any) -> RiskToken:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return RiskToken.from_dict(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise InputValidationError(f"stored token is not valid JSON: {exc}") from exc

    def get(self, subject_ref: str) -> Optional[RiskToken]:
        raw = self._client.get(self._key(subject_ref))
        return None if raw is None else self._decode(raw)

    def delete(self, subject_ref: str) -> bool:
        return bool(self._client.delete(self._key(subject_ref)))

    def purge_expired(self, now: float) -> int:
        return 0  # Redis evicts expired keys on its own

    def tokens(self) -> List[RiskToken]:
        out = []
        for key in self._client.scan_iter(match=self._prefix + "*"):
            raw = self._client.get(key)
            if raw is not None:
                out.append(self._decode(raw))
        return out


def _load_key(explicit: Union[str, bytes, None], env_var: str) -> bytes:
    if explicit is not None:
        key = explicit.encode("utf-8") if isinstance(explicit, str) else bytes(explicit)
    elif os.environ.get(env_var):
        key = os.environ[env_var].encode("utf-8")
    else:
        logger.warning(
            "%s not set: using an ephemeral random key. Tokens will not verify across "
            "processes or restarts. Set it for any multi-worker deployment.",
            env_var,
        )
        key = secrets.token_bytes(32)
    if len(key) < 16:
        raise ValueError(f"{env_var} must be at least 16 bytes")
    return key


@dataclass(frozen=True)
class TokenLookup:
    status: str  # "active" | "none" | "expired" | "invalid"
    token: Optional[RiskToken] = None


class TokenService:
    """Issues, verifies, expires and revokes risk tokens.

    Keys come from arguments or the VERIFVOICE_SIGNING_KEY /
    VERIFVOICE_PSEUDONYM_KEY environment variables (never from the repo).
    """

    def __init__(
        self,
        store: Optional[TokenStore] = None,
        signing_key: Union[str, bytes, None] = None,
        pseudonym_key: Union[str, bytes, None] = None,
        ttl_seconds: float = 600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not ttl_seconds > 0:
            raise ValueError("ttl_seconds must be > 0")
        self.store: TokenStore = store if store is not None else InMemoryTokenStore()
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._signing_key = _load_key(signing_key, "VERIFVOICE_SIGNING_KEY")
        self._pseudonym_key = _load_key(pseudonym_key, "VERIFVOICE_PSEUDONYM_KEY")
        self._lock = threading.Lock()

    # -- pseudonymisation --------------------------------------------------

    def pseudonymize(self, value: str, prefix: str) -> str:
        """Keyed hash: stable for a given key, not reversible without it."""
        if not isinstance(value, str) or not value:
            raise InputValidationError(f"{prefix} identifier must be a non-empty string")
        digest = hmac.new(self._pseudonym_key, f"{prefix}:{value}".encode("utf-8"), hashlib.sha256)
        return f"{prefix}_{digest.hexdigest()[:32]}"

    def subject_ref(self, subject_id: str) -> str:
        return self.pseudonymize(subject_id, "sub")

    # -- signing -----------------------------------------------------------

    def _sign(self, token: RiskToken) -> str:
        return hmac.new(self._signing_key, token.signing_payload(), hashlib.sha256).hexdigest()

    def verify(self, token: RiskToken) -> bool:
        return hmac.compare_digest(token.signature, self._sign(token))

    # -- lifecycle -----------------------------------------------------------

    def _retention(self) -> str:
        return (
            "Risk signal only: contains no audio, transcript or phone number. "
            f"Expires {int(self.ttl_seconds)}s after issue and is deleted on expiry."
        )

    def issue(self, subject_id: str, assessment: CallRiskAssessment) -> Optional[RiskToken]:
        """Issue (or refresh) the subject's token if the call reached SOFT_PROMPT+.

        High-water mark: while a stronger unexpired token exists, a weaker
        assessment refreshes its expiry but does not downgrade it.
        """
        if not assessment.token_eligible:
            return None
        ref = self.subject_ref(subject_id)

        def build(existing: Optional[RiskToken]) -> RiskToken:
            now = self.clock()
            if existing is not None and (
                existing.subject_ref != ref or existing.is_expired(now) or not self.verify(existing)
            ):
                existing = None
            level, confidence, reasons = assessment.level, assessment.confidence, assessment.reason_codes
            if existing is not None and LEVEL_RANK[existing.level] > LEVEL_RANK[level]:
                level, confidence, reasons = existing.level, existing.confidence, existing.reason_codes
            unsigned = RiskToken(
                token_id="tok_" + secrets.token_hex(16),
                subject_ref=ref,
                level=level,
                confidence=confidence,
                reason_codes=tuple(reasons),
                issued_at=now,
                expires_at=now + self.ttl_seconds,
                retention=self._retention(),
            )
            return replace(unsigned, signature=self._sign(unsigned))

        with self._lock:  # in-process; the store's update is atomic across workers
            return self.store.update(ref, build, self.ttl_seconds)

    def lookup(self, subject_id: str) -> TokenLookup:
        """What the bank calls at payment time."""
        ref = self.subject_ref(subject_id)
        for _ in range(2):
            try:
                token = self.store.get(ref)
            except InputValidationError as exc:
                logger.warning("unreadable token in store: %s", exc)
                return TokenLookup("invalid")
            if token is None:
                return TokenLookup("none")
            if not token.is_expired(self.clock()):
                break
            if self.store.discard(ref, token.token_id):
                return TokenLookup("expired")
            # A fresh token replaced the expired one while we looked: re-read it.
        else:
            return TokenLookup("expired")
        if token.subject_ref != ref or not self.verify(token):
            logger.warning("token %s failed signature/subject verification", token.token_id)
            return TokenLookup("invalid")
        return TokenLookup("active", token)

    def get_active(self, subject_id: str) -> Optional[RiskToken]:
        return self.lookup(subject_id).token

    def revoke(self, subject_id: str) -> bool:
        """Delete a subject's token (opt-out / 'easy off switch')."""
        return self.store.delete(self.subject_ref(subject_id))

    def purge_expired(self) -> int:
        """Run on a schedule (e.g. every minute) with the in-memory store."""
        return self.store.purge_expired(self.clock())


# --------------------------------------------------------------------------
# Privacy guards [7.2 items 4, 6, 9]
# --------------------------------------------------------------------------

FORBIDDEN_RECORD_KEYS = frozenset(
    {
        "audio",
        "audio_bytes",
        "pcm",
        "wav",
        "waveform",
        "samples",
        "transcript",
        "transcription",
        "text",
        "utterance",
        "utterances",
        "phone",
        "phone_number",
        "msisdn",
        "caller_id",
        "caller_number",
        "account_number",
        "customer_id",
        "subject_id",
        "beneficiary_id",
        "call_id",
    }
)
_PHONE_CHARS = re.compile(r"^\+?[\d\s\-().]+$")
# Opaque identifiers this module mints (sub_/call_/ben_/tok_/pay_ + hex).
_MINTED_ID = re.compile(r"^[a-z]{2,5}_[0-9a-f]{16,64}$")
# Raw identifiers shorter than this are only matched exactly, not as substrings
# (so an ID like "call" or "high" cannot collide with fixed vocabulary).
_MIN_SUBSTRING_MATCH = 6


def _looks_like_phone_or_account(value: str) -> bool:
    return bool(_PHONE_CHARS.match(value)) and sum(ch.isdigit() for ch in value) >= 10


def _contains_raw_identifier(text: str, forbidden: Sequence[str]) -> bool:
    if _MINTED_ID.match(text):
        return text in forbidden
    return any(
        text == value or (len(value) >= _MIN_SUBSTRING_MATCH and value in text) for value in forbidden
    )


def find_privacy_violations(record: Any, forbidden_values: Iterable[str] = (), _path: str = "$") -> List[str]:
    """Return every place a record holds call content or a raw identifier."""
    forbidden = [v for v in forbidden_values if v]
    problems: List[str] = []
    if isinstance(record, Mapping):
        for key, value in record.items():
            here = f"{_path}.{key}"
            if str(key).lower() in FORBIDDEN_RECORD_KEYS:
                problems.append(f"{here}: forbidden field")
            problems.extend(find_privacy_violations(value, forbidden, here))
    elif isinstance(record, (list, tuple, set, frozenset)):
        for i, value in enumerate(record):
            problems.extend(find_privacy_violations(value, forbidden, f"{_path}[{i}]"))
    elif isinstance(record, (bytes, bytearray, memoryview)):
        problems.append(f"{_path}: binary payload")
    elif isinstance(record, str):
        if _looks_like_phone_or_account(record):
            problems.append(f"{_path}: looks like a phone/account number")
        if _contains_raw_identifier(record, forbidden):
            problems.append(f"{_path}: contains a raw identifier")
    return problems


class AuditLog:
    """Append-only, content-free audit trail.

    Every record is checked at write time for audio, transcripts, binary
    payloads, phone/account numbers, forbidden fields and the raw identifiers
    it involves. Callers pass those identifiers per record, so the check stays
    O(record size) however many customers the service has seen. ``forbid()``
    registers a small set of global values on top.
    """

    def __init__(self, forbidden_values: Iterable[str] = ()) -> None:
        self._records: List[Dict[str, Any]] = []
        self._forbidden = set(forbidden_values)
        self._lock = threading.Lock()

    def forbid(self, *values: str) -> None:
        """Register raw identifiers that must never appear in the log."""
        with self._lock:
            self._forbidden.update(v for v in values if v)

    def forbidden_values(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._forbidden)

    def check(self, record: Mapping[str, Any], raw_identifiers: Iterable[Optional[str]] = ()) -> None:
        """Raise PrivacyViolation if ``record`` may not be written. Call it before
        changing any state, so a rejected record never leaves partial state."""
        with self._lock:
            forbidden = self._forbidden.union(v for v in raw_identifiers if v)
        problems = find_privacy_violations(record, forbidden)
        if problems:
            raise PrivacyViolation("audit record rejected: " + "; ".join(problems))

    def append(self, record: Mapping[str, Any], raw_identifiers: Iterable[Optional[str]] = ()) -> None:
        raw_identifiers = tuple(raw_identifiers)
        self.check(record, raw_identifiers)
        snapshot = json.loads(json.dumps(record))
        with self._lock:
            self._records.append(snapshot)

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._records))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# --------------------------------------------------------------------------
# Deletion checks [7.3]
# --------------------------------------------------------------------------

MEDIA_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".amr", ".pcm", ".raw", ".webm"}
)

BufferLike = Union[bytes, bytearray, memoryview, Any]  # anything with the buffer protocol


def is_zeroized(buffer: BufferLike) -> bool:
    """True if every byte of the buffer is zero (numpy arrays work too)."""
    view = memoryview(buffer)
    if not view.c_contiguous:
        view = memoryview(view.tobytes())
    return view.cast("B") == bytes(view.nbytes)


def zeroize(buffer: BufferLike) -> None:
    """Overwrite a writable buffer (bytearray, numpy array, ...) with zeros."""
    view = memoryview(buffer)
    if view.readonly:
        raise TypeError("buffer is read-only; keep audio in a bytearray or writable array")
    if not view.c_contiguous:
        raise TypeError("buffer must be C-contiguous to zeroize in place")
    view.cast("B")[:] = bytes(view.nbytes)


def find_media_artifacts(paths: Iterable[Union[str, Path]]) -> List[Path]:
    """Audio or transcript files under the given files/directories."""
    found: List[Path] = []
    for root in paths:
        root = Path(root)
        candidates = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        for p in candidates:
            if p.suffix.lower() in MEDIA_EXTENSIONS or "transcript" in p.name.lower():
                found.append(p)
    return sorted(found)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DeletionReport:
    checks: Tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
        }


def run_deletion_checks(
    *,
    token_service: Optional[TokenService] = None,
    audit_logs: Sequence[AuditLog] = (),
    buffers: Sequence[BufferLike] = (),
    artifact_dirs: Sequence[Union[str, Path]] = (),
    raw_identifiers: Sequence[str] = (),
) -> DeletionReport:
    """Inspect (never modify) the data lifecycle. Call after a call ends and
    after ``token_service.purge_expired()``. ``raw_identifiers`` are customer,
    payee or call IDs that must not appear anywhere in the audit logs."""
    checks: List[CheckResult] = []

    if buffers:
        dirty = [i for i, buf in enumerate(buffers) if not is_zeroized(buf)]
        checks.append(
            CheckResult(
                "audio_buffers_zeroized",
                not dirty,
                f"{len(buffers)} buffer(s) checked" + (f"; non-zero: {dirty}" if dirty else ""),
            )
        )

    if token_service is not None:
        now = token_service.clock()
        try:
            tokens = token_service.store.tokens()
        except InputValidationError as exc:
            checks.append(CheckResult("token_store_readable", False, str(exc)))
            tokens = []
        expired = [t.token_id for t in tokens if t.is_expired(now)]
        checks.append(
            CheckResult(
                "no_expired_tokens_retained",
                not expired,
                f"{len(tokens)} token(s) stored; {len(expired)} past expiry",
            )
        )
        bad = []
        for t in tokens:
            if set(t.to_dict()) != TOKEN_FIELDS or not token_service.verify(t) or find_privacy_violations(t.to_dict()):
                bad.append(t.token_id)
        checks.append(
            CheckResult(
                "tokens_minimal_and_signed",
                not bad,
                "all tokens carry only the allowed fields and a valid signature"
                if not bad
                else f"failing tokens: {bad}",
            )
        )

    for i, log in enumerate(audit_logs):
        problems: List[str] = []
        forbidden = set(log.forbidden_values()).union(v for v in raw_identifiers if v)
        for record in log.records():
            problems.extend(find_privacy_violations(record, forbidden))
        checks.append(
            CheckResult(
                f"audit_log[{i}]_content_free",
                not problems,
                f"{len(log)} record(s) checked" + (f"; {problems[:5]}" if problems else ""),
            )
        )

    if artifact_dirs:
        files = find_media_artifacts(artifact_dirs)
        checks.append(
            CheckResult(
                "no_media_or_transcript_files",
                not files,
                "none found" if not files else f"found: {[str(f) for f in files[:10]]}",
            )
        )

    return DeletionReport(tuple(checks))


# --------------------------------------------------------------------------
# Day-one flat fusion + graduated policy (the team's shared contract)
# --------------------------------------------------------------------------
#
# The rest of the team (P1/P2/P3, primer §7, master context §25.4 "use this")
# locks the decision formula to the simple day-one form:
#
#     R = 0.5*C + 0.3*(S*C) + 0.2*N
#
#   C = coercion_score (0..1), S = spoof_score (0..1 or None),
#   N = 1.0 if new_beneficiary else 0.0.
#
# Spoof AMPLIFIES coercion and never acts alone: an AAC/TTS user with a
# synthetic voice and benign intent scores ~0 from the S*C term. When S is
# None the weights renormalise onto C and SPOOF_UNAVAILABLE is flagged instead
# of inventing a spoof score. Decisions use the lowercase vocabulary the
# frontend and tests depend on.
# --------------------------------------------------------------------------

DECISIONS = ("observe", "warn", "verify", "pause")

OBSERVE_MESSAGE = "No additional friction. Payment proceeds normally."
WARN_MESSAGE = "This caller is requesting an urgent financial action. Verify using a known number."
VERIFY_MESSAGE = (
    "This call shows signs of urgent payment pressure. The recipient is new. "
    "End the call and verify using a number you already trust."
)
PAUSE_MESSAGE = (
    "This call shows signs of urgent payment pressure. The recipient is new. "
    "Your payment is paused briefly; you can review or continue securely."
)


def decide(
    coercion_score: float,
    spoof_score: Optional[float],
    new_beneficiary: bool,
    amount: float = 0.0,
) -> Dict[str, Any]:
    """Flat risk fusion + graduated policy (the team's day-one formula).

    Returns ``{"decision", "risk_score", "reason_codes", "message"}`` with the
    lowercase decision vocabulary the frontend and P6 tests expect. ``amount``
    is accepted for signature compatibility with the API layer but does not
    enter the day-one formula (amount-driven escalation lives in the richer
    :class:`BankSimulator` path).
    """
    _unit(coercion_score, "coercion_score")
    n = 1.0 if new_beneficiary else 0.0

    if spoof_score is None:
        # Spoof model unavailable -> renormalise onto coercion. Never invent S.
        risk_score = 0.8 * coercion_score + 0.2 * n
        degraded = True
    else:
        _unit(spoof_score, "spoof_score")
        risk_score = 0.5 * coercion_score + 0.3 * (spoof_score * coercion_score) + 0.2 * n
        degraded = False

    risk_score = round(min(1.0, max(0.0, risk_score)), 4)

    if risk_score < 0.30:
        decision, message = "observe", OBSERVE_MESSAGE
    elif risk_score < 0.55:
        decision, message = "warn", WARN_MESSAGE
    elif risk_score < 0.75:
        decision, message = "verify", VERIFY_MESSAGE
    else:
        decision, message = "pause", PAUSE_MESSAGE

    reason_codes = ["NEW_PAYEE"] if new_beneficiary else []
    if degraded:
        reason_codes.append("SPOOF_UNAVAILABLE")

    return {
        "decision": decision,
        "risk_score": risk_score,
        "reason_codes": reason_codes,
        "message": message,
    }


# --------------------------------------------------------------------------
# Facade used by the API layer
# --------------------------------------------------------------------------


class RiskEngine:
    """Entry point for the FastAPI backend.

    * ``assess_call`` -- call signals in, assessment + optional token out.
    * ``evaluate``    -- the team's combined payload in, JSON-ready decision out.
    """

    def __init__(
        self,
        token_service: Optional[TokenService] = None,
        fusion: FusionConfig = DEFAULT_FUSION,
        policy: PolicyConfig = DEFAULT_POLICY,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        self.token_service = token_service if token_service is not None else TokenService()
        self.fusion = fusion
        self.policy = policy
        self.audit_log = audit_log if audit_log is not None else AuditLog()

    @property
    def clock(self) -> Callable[[], float]:
        return self.token_service.clock

    def assess_call(
        self,
        audio_analysis: Union[Mapping[str, Any], AudioAnalysis, None],
        coercion_analysis: Union[Mapping[str, Any], CoercionAnalysis, None],
        subject_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> Tuple[CallRiskAssessment, Optional[RiskToken]]:
        """Score one window/update of a call. With ``subject_id`` (the enrolled
        customer, taken from the authenticated session, never from the payload),
        a token is issued once the call reaches the soft-prompt band."""
        audio = audio_analysis if isinstance(audio_analysis, AudioAnalysis) else AudioAnalysis.from_dict(audio_analysis)
        coercion = (
            coercion_analysis
            if isinstance(coercion_analysis, CoercionAnalysis)
            else CoercionAnalysis.from_dict(coercion_analysis)
        )
        assessment = score_call(audio, coercion, self.fusion, now=self.clock())
        raw_ids = (subject_id, call_id)
        record = {
            "event": "call_assessed",
            "at": assessment.assessed_at,
            "call_ref": self.token_service.pseudonymize(call_id, "call") if call_id is not None else None,
            "subject_ref": self.token_service.subject_ref(subject_id) if subject_id is not None else None,
            "score": assessment.score,
            "band": assessment.band,
            "level": assessment.level,
            "confidence": assessment.confidence,
            "reason_codes": list(assessment.reason_codes),
            "abstained": assessment.abstained,
            "ceiling": assessment.ceiling,
            "floor": assessment.floor,
            "defaults_used": list(assessment.defaults_used),
            "model_versions": dict(assessment.model_versions),
            "token_id": None,
            "policy_version": assessment.policy_version,
        }
        self.audit_log.check(record, raw_ids)  # reject before any state changes
        token = self.token_service.issue(subject_id, assessment) if subject_id is not None else None
        record["token_id"] = token.token_id if token else None
        self.audit_log.append(record, raw_ids)
        return assessment, token

    def evaluate(self, payload: Mapping[str, Any], subject_id: Optional[str] = None) -> Dict[str, Any]:
        """One-shot decision for the team payload::

            {"audio_analysis": {...}, "coercion_analysis": {...},
             "transaction": {"amount": 500, "currency": "USD", "new_beneficiary": true}}

        With ``subject_id``, the decision uses the customer's active token. That
        is either the one this window issued, or one issued earlier in the call
        that has not expired, so a calm moment in the call does not wipe out
        earlier risk. ``token`` in the result is that active token.

        Raises InputValidationError on malformed input (map to HTTP 422).
        In production only ``token`` and ``decision`` should go to the bank;
        ``call_risk`` is included for the demo and for debugging.
        """
        payload = _mapping(payload, "payload")
        txn = Transaction.from_dict(payload.get("transaction"), require_new_beneficiary=True)
        assessment, token = self.assess_call(
            payload.get("audio_analysis"), payload.get("coercion_analysis"), subject_id=subject_id
        )
        if token is None and subject_id is not None:
            token = self.token_service.get_active(subject_id)
        call = CallSignal.from_token(token) if token is not None else CallSignal.from_assessment(assessment)
        context = TransactionContext(
            amount=txn.amount,
            currency=txn.currency,
            new_beneficiary=bool(txn.new_beneficiary),
            amount_tier=classify_amount(txn.amount, txn.currency, self.policy),
        )
        decision = decide_action(call, context, self.policy)
        self.audit_log.append(
            {
                "event": "decision",
                "at": self.clock(),
                "subject_ref": self.token_service.subject_ref(subject_id) if subject_id is not None else None,
                "token_id": decision.token_id,
                "call_level": decision.call_level,
                "new_beneficiary": context.new_beneficiary,
                "amount_tier": context.amount_tier,
                "action": decision.action,
                "reason_codes": list(decision.reason_codes),
                "policy_version": decision.policy_version,
            },
            (subject_id,),
        )
        return {
            "call_risk": assessment.to_dict(),
            "token": token.to_dict() if token else None,
            "transaction": {
                "amount": context.amount,
                "currency": context.currency,
                "new_beneficiary": context.new_beneficiary,
                "amount_tier": context.amount_tier,
            },
            "decision": decision.to_dict(),
        }

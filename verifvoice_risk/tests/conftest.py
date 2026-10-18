from typing import Any, Dict

import pytest

from verifvoice_risk.bank_simulator import BankSimulator
from verifvoice_risk.risk_engine import RiskEngine, TokenService

# Test-only keys. Real deployments read VERIFVOICE_SIGNING_KEY /
# VERIFVOICE_PSEUDONYM_KEY from the environment.
SIGNING_KEY = "test-signing-key-0123456789abcdef"
PSEUDONYM_KEY = "test-pseudonym-key-0123456789abcd"
TTL = 600

# The exact payload format shared by the acoustic / coercion teammates.
TEAM_EXAMPLE: Dict[str, Any] = {
    "audio_analysis": {"status": "ok", "prediction": "likely_spoof", "spoof_score": 0.93},
    "coercion_analysis": {"status": "ok", "coercion_level": "high"},
    "transaction": {"amount": 500, "currency": "USD", "new_beneficiary": True},
}


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def audio(spoof: float, **extra: Any) -> Dict[str, Any]:
    prediction = "likely_spoof" if spoof >= 0.5 else "likely_bonafide"
    return {"status": "ok", "prediction": prediction, "spoof_score": spoof, **extra}


def coercion(level: str, **extra: Any) -> Dict[str, Any]:
    return {"status": "ok", "coercion_level": level, **extra}


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def token_service(clock: FakeClock) -> TokenService:
    return TokenService(signing_key=SIGNING_KEY, pseudonym_key=PSEUDONYM_KEY, ttl_seconds=TTL, clock=clock)


@pytest.fixture
def engine(token_service: TokenService) -> RiskEngine:
    return RiskEngine(token_service)


@pytest.fixture
def bank(token_service: TokenService) -> BankSimulator:
    return BankSimulator(token_service)

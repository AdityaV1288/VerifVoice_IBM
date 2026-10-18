"""main.py — VerifVoice FastAPI backend (Person 3).

Runs on the LinuxONE VM (s390x). Wires Person 2's `semantic_model` (coercion)
and Person 1's `audio_authenticity` (spoof) via `pipeline.py`.

Integration points marked "TODO":
  * P4 (risk fusion) plugs its `risk_engine` into /payment/decision and fills
    `risk_level` / `token_ttl_seconds`.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Try it:
    GET  http://<host>:8000/health
    POST http://<host>:8000/analyze   {"source": "Transfer 50000 now, don't tell anyone", "input_type": "text"}
    GET  http://<host>:8000/metrics
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import semantic_model  # Person 2's module
import pipeline        # Person 1 + Person 2 integration layer
import risk_adapter    # risk fusion (Person 4's verifvoice_risk, wired via adapter)
import token_store     # in-memory risk tokens (the single token store)
from audio_authenticity import AudioAnalysisService, DetectorConfig as AudioDetectorConfig

# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class AnalyzeRequest(BaseModel):
    source: str = Field(..., description="Transcript text, or a path to an audio file")
    input_type: str = Field("text", description="'text' or 'audio'")
    lang: str = Field("en", description="Semantic-analysis language (e.g. 'en')")


class AnalyzeResponse(BaseModel):
    request_id: str
    token_id: str
    token_ttl_seconds: int
    # --- Person 2 fields (live) ---
    audio_status: str            # "analyzed" | "insufficient_text" | "unsupported_language"
    coercion_score: float
    reason_codes: list[str]
    signals: dict
    lang: str
    # --- semantic layer (learned coercion) ---
    semantic_score: float | None = None
    semantic_status: str | None = None
    semantic_model: str | None = None
    # --- Person 1 fields (live, via pipeline) ---
    spoof_score: float | None = None
    spoof_status: str | None = None
    spoof_classification: str | None = None
    quality_score: float | None = None
    model_name: str | None = None


class PaymentDecisionRequest(BaseModel):
    token_id: str = Field(..., description="Token returned by /analyze")
    amount: float = Field(..., description="Proposed transfer amount (INR)")
    new_beneficiary: bool = Field(False, description="True if payee is new/unusual")


class PaymentDecisionResponse(BaseModel):
    decision: str                              # "observe" | "warn" | "verify" | "pause"
    risk_score: float
    reason_codes: list[str]
    message: str


# --------------------------------------------------------------------------- #
# App + lifecycle
# --------------------------------------------------------------------------- #

# Lightweight in-memory metrics (reset on restart).
_metrics: dict = {"analyze_requests": 0, "payment_requests": 0,
                  "last_analyze_ms": None, "last_payment_ms": None}

# Cached audio-authenticity service for /health (avoid reloading weights each call).
_audio_service = None


def _audio_health() -> str:
    """Report P1's adapter readiness: 'ready' | 'model_unavailable' | 'unavailable'."""
    global _audio_service
    try:
        if _audio_service is None:
            _audio_service = AudioAnalysisService(AudioDetectorConfig.from_env())
        return "ready" if _audio_service.detector.health_check() else "model_unavailable"
    except Exception:
        return "unavailable"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the ASR model so the first /analyze (audio) isn't slow.
    if semantic_model.transcription_ready():
        try:
            semantic_model._get_model()
        except Exception:
            pass  # ASR optional; the text path always works
    yield


app = FastAPI(title="VerifVoice API", version="0.1.0", lifespan=lifespan)

# Allow the dashboard (P5) to call this API from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health():
    """Server status + model readiness (per the shared contract)."""
    return {
        "status": "ok",
        "semantic_rules": "ready",                       # Person 2 core (always)
        "transcription": ("ready" if semantic_model.transcription_ready() else "unavailable"),
        "audio_authenticity": _audio_health(),           # Person 1 adapter
        "risk_engine": "ready",                          # fusion (fallback always available)
    }


def _run_analyze(source: str, input_type: str, lang: str) -> AnalyzeResponse:
    """Shared core: run the pipeline, store a token, return the response."""
    t0 = time.perf_counter()
    result = pipeline.analyze_call(source, input_type=input_type, lang=lang)

    _metrics["analyze_requests"] += 1
    _metrics["last_analyze_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Store the scores under a short-lived token for /payment/decision.
    token_id, ttl = token_store.create_token({
        "coercion_score": result["coercion_score"],
        "spoof_score": result["spoof_score"],
        "reason_codes": result["reason_codes"],
    })

    return AnalyzeResponse(
        request_id=result["request_id"],
        token_id=token_id,
        token_ttl_seconds=ttl,
        audio_status=result["coercion_status"],
        coercion_score=result["coercion_score"],
        reason_codes=result["reason_codes"],
        signals=result["signals"],
        lang=result["lang"],
        semantic_score=result["semantic_score"],
        semantic_status=result["semantic_status"],
        semantic_model=result["semantic_model"],
        spoof_score=result["spoof_score"],
        spoof_status=result["spoof_status"],
        spoof_classification=result["spoof_classification"],
        quality_score=result["quality_score"],
        model_name=result["model_name"],
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Accept a sample (text or an audio path) and return coercion + spoof signals."""
    return _run_analyze(req.source, req.input_type, req.lang)


@app.post("/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(file: UploadFile = File(...), lang: str = "en"):
    """Accept an uploaded audio file (multipart) — for P5's frontend."""
    suffix = Path(file.filename or "call.wav").suffix or ".wav"
    with TemporaryDirectory() as directory:
        path = Path(directory) / f"call{suffix}"
        path.write_bytes(await file.read())
        return _run_analyze(str(path), input_type="audio", lang=lang)


@app.post("/payment/decision", response_model=PaymentDecisionResponse)
def payment_decision(req: PaymentDecisionRequest):
    """Combine an unexpired risk token with transaction context (R = 0.5C + 0.3S + 0.2N)."""
    t0 = time.perf_counter()
    token = token_store.get_token(req.token_id)
    if token is None:
        raise HTTPException(status_code=400, detail="token expired or unknown")

    result = risk_adapter.decide(
        coercion_score=token["coercion_score"],
        spoof_score=token["spoof_score"],
        new_beneficiary=req.new_beneficiary,
        amount=req.amount,
    )

    # Merge call-derived codes with transaction codes (NEW_PAYEE, SPOOF_UNAVAILABLE, ...).
    reason_codes = list(dict.fromkeys(token.get("reason_codes", []) + result["reason_codes"]))

    _metrics["payment_requests"] += 1
    _metrics["last_payment_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return PaymentDecisionResponse(
        decision=result["decision"],
        risk_score=result["risk_score"],
        reason_codes=reason_codes,
        message=result["message"],
    )


@app.get("/metrics")
def metrics():
    """Timing + request counts (per the shared contract)."""
    return {
        **_metrics,
        "transcription": ("ready" if semantic_model.transcription_ready() else "unavailable"),
    }

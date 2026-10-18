"""risk_adapter.py — thin wiring between the VerifVoice backend and Person 4's package.

This module is the single integration point for risk fusion. It exposes the flat
``decide()`` contract the FastAPI backend and ``demo.py`` already call, and it
delegates to Person 4's ``verifvoice_risk`` package (which now owns the day-one
formula). It also owns the reason-code translation table between the team's
vocabulary and Person 4's internal vocabulary.

Locked decisions (see PERSON4_INTEGRATION.md §6):

* **Fusion formula (day-one):** ``R = 0.5*C + 0.3*(S*C) + 0.2*N``, with
  ``S=None`` -> ``R = 0.8*C + 0.2*N`` and ``SPOOF_UNAVAILABLE`` flagged.
* **Decision vocabulary:** lowercase ``observe / warn / verify / pause``.
* **Tokens:** one store — ``token_store.py`` (TTL 120s, scores only). Person 4's
  ``TokenService`` is the optional *bank-facing* signed/pseudonymous token layer
  and is deliberately NOT instantiated in the demo's internal path, so there is
  exactly one token store in the deployed flow.
* **Reason codes:** team vocabulary is canonical on the wire; this module maps
  it to/from Person 4's internal codes when the object API is used.
* **POLICY_VERSION:** ``vv-risk-1.0.0`` (record this for P6).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

from verifvoice_risk import (
    POLICY_VERSION,
    TEAM_CUE_TO_INTERNAL,
    INTERNAL_TO_TEAM_CODE,
    TEAM_REASON_CODES,
    decide as _p4_decide,
    translate_codes_to_team,
)

# ---------------------------------------------------------------------------
# Reason-code translation table (P2/team <-> P4)
# ---------------------------------------------------------------------------

# Team (P2) reason code -> Person 4 internal cue code (None = context-only /
# no P4 equivalent; carried as a reason code on the wire but never a P4 cue).
TEAM_TO_P4_CUE: Dict[str, Any] = dict(TEAM_CUE_TO_INTERNAL)

# Person 4 internal reason code -> team reason code.
P4_TO_TEAM_CODE: Dict[str, str] = dict(INTERNAL_TO_TEAM_CODE)

# Full team vocabulary (primer §8), for validation/reference.
TEAM_REASON_CODES: Tuple[str, ...] = tuple(sorted(TEAM_REASON_CODES))


def to_p4_cues(team_codes: Iterable[str]) -> Tuple[str, ...]:
    """Translate P2 reason codes into Person 4's internal cue codes.

    ``THREAT`` and ``EMERGENCY_NARRATIVE`` have no P4 cue (they are already
    folded into ``coercion_score``, or are context-only), so they are dropped
    here — they still reach the response through the backend's own
    reason-code merge.
    """
    out = []
    for code in team_codes:
        code = str(code).strip().upper()
        internal = TEAM_TO_P4_CUE.get(code, code)
        if internal is not None and internal not in out:
            out.append(internal)
    return tuple(out)


def from_p4_codes(p4_codes: Iterable[str]) -> Tuple[str, ...]:
    """Translate Person 4's internal reason codes back to the team vocabulary."""
    return translate_codes_to_team(p4_codes)


# ---------------------------------------------------------------------------
# Flat decision entry point (the wiring)
# ---------------------------------------------------------------------------


def decide(
    coercion_score: float,
    spoof_score: float | None,
    new_beneficiary: bool,
    amount: float = 0.0,
) -> dict:
    """Risk fusion + graduated policy using the day-one formula.

    Returns ``{"decision", "risk_score", "reason_codes", "message"}`` with the
    lowercase decision vocabulary. Delegates to Person 4's ``verifvoice_risk``.
    """
    return _p4_decide(
        coercion_score=coercion_score,
        spoof_score=spoof_score,
        new_beneficiary=new_beneficiary,
        amount=amount,
    )

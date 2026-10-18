"""risk_engine.py — risk fusion + graduated policy (Person 4's domain).

Person 4's real implementation now lives in the ``verifvoice_risk`` package and
is wired through ``risk_adapter``. This module is kept as a thin backward-
compatible shim so any existing ``risk_engine.decide(...)`` call site keeps
working unchanged; prefer ``risk_adapter.decide`` (or ``verifvoice_risk.decide``)
in new code.

Fusion (day-one, locked):  R = 0.5*C + 0.3*(S*C) + 0.2*N
  C = coercion_score (0..1), S = spoof_score (0..1 or None), N = new beneficiary (0|1)
When S is None the weights renormalise onto C and SPOOF_UNAVAILABLE is flagged.
"""

from risk_adapter import decide  # noqa: F401  (re-export)

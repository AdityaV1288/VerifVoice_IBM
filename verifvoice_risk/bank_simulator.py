"""Simulated bank Payment Decision API (design doc 4 item 4, 5.2 steps 9-11).

Shows how a bank combines an unexpired VerifVoice risk token with its own
transaction context:

* new-beneficiary detection from the bank's own payee registry: never
  paid, added recently, or no completed payments. The client's
  ``new_beneficiary`` flag is never trusted.
* unusual-amount detection relative to the customer's own history.
* proportionate, reversible actions: ALLOW, WARN, VERIFY_OUT_OF_BAND and a
  brief HOLD. There is no blocked or frozen state, and every pending
  payment can be completed (after verification) or cancelled by the
  customer.
* a content-free audit trail that uses pseudonymous references only.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .risk_engine import (
    ALLOW,
    HOLD,
    VERIFY_OUT_OF_BAND,
    WARN,
    DEFAULT_POLICY,
    ActionDecision,
    AuditLog,
    CallSignal,
    InputValidationError,
    PolicyConfig,
    TokenService,
    TransactionContext,
    _amount,
    _currency,
    classify_amount,
    decide_action,
)

# Payment states. None of them freezes the account.
APPROVED = "APPROVED"  # ALLOW: executes immediately
EXECUTED = "EXECUTED"
PENDING_CONFIRMATION = "PENDING_CONFIRMATION"  # WARN: customer acknowledges
PENDING_VERIFICATION = "PENDING_VERIFICATION"  # out-of-band check first
ON_HOLD = "ON_HOLD"  # brief pause; OOB verification releases it early
CANCELLED = "CANCELLED"

STATUS_FOR_ACTION = {
    ALLOW: APPROVED,
    WARN: PENDING_CONFIRMATION,
    VERIFY_OUT_OF_BAND: PENDING_VERIFICATION,
    HOLD: ON_HOLD,
}
OPEN_STATUSES = frozenset({PENDING_CONFIRMATION, PENDING_VERIFICATION, ON_HOLD})

# Prior relationship assumed for payees passed to register_customer().
_ESTABLISHED_AGE_SECONDS = 365 * 24 * 60 * 60


class InvalidTransition(ValueError):
    """The requested step is not allowed from the payment's current status."""


@dataclass
class Beneficiary:
    beneficiary_id: str
    added_at: float
    completed_payments: int = 0
    last_paid_at: Optional[float] = None


@dataclass
class Customer:
    customer_id: str
    currency: str
    beneficiaries: Dict[str, Beneficiary] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)  # currency -> amounts


@dataclass
class Payment:
    payment_id: str
    customer_id: str
    beneficiary_id: str
    amount: float
    currency: str
    channel: str
    created_at: float
    status: str
    decision: ActionDecision
    new_beneficiary: bool
    beneficiary_reason: str
    amount_tier: str
    token_status: str
    hold_until: Optional[float] = None
    executed_at: Optional[float] = None
    cancelled_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """The bank's own response to its customer (raw IDs are fine here)."""
        return {
            "payment_id": self.payment_id,
            "customer_id": self.customer_id,
            "beneficiary_id": self.beneficiary_id,
            "amount": self.amount,
            "currency": self.currency,
            "channel": self.channel,
            "created_at": self.created_at,
            "status": self.status,
            "action": self.decision.action,
            "reason_codes": list(self.decision.reason_codes),
            "user_message": self.decision.user_message,
            "rationale": list(self.decision.rationale),
            "new_beneficiary": self.new_beneficiary,
            "beneficiary_reason": self.beneficiary_reason,
            "amount_tier": self.amount_tier,
            "token_status": self.token_status,
            "hold_until": self.hold_until,
            "executed_at": self.executed_at,
            "cancelled_at": self.cancelled_at,
        }


class BankSimulator:
    """In-memory bank that consumes VerifVoice tokens at payment time."""

    def __init__(
        self,
        token_service: TokenService,
        policy: PolicyConfig = DEFAULT_POLICY,
        audit_log: Optional[AuditLog] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.token_service = token_service
        self.policy = policy
        self.audit_log = audit_log if audit_log is not None else AuditLog()
        self.clock = clock if clock is not None else token_service.clock
        self._customers: Dict[str, Customer] = {}
        self._payments: Dict[str, Payment] = {}
        self._lock = threading.RLock()

    # -- set-up -------------------------------------------------------------

    def register_customer(
        self,
        customer_id: str,
        currency: str = "INR",
        established_beneficiaries: Iterable[str] = (),
        history: Iterable[float] = (),
    ) -> Customer:
        """Create a customer. ``established_beneficiaries`` are payees with a
        long prior relationship; ``history`` holds past amounts in ``currency``."""
        if not isinstance(customer_id, str) or not customer_id:
            raise InputValidationError("customer_id must be a non-empty string")
        currency = _currency(currency, "currency")
        payees = list(established_beneficiaries)
        if not all(isinstance(b, str) and b for b in payees):
            raise InputValidationError("established_beneficiaries must be non-empty strings")
        amounts = [_amount(a, "history amount") for a in history]
        now = self.clock()
        with self._lock:
            if customer_id in self._customers:
                raise InputValidationError(f"customer {customer_id!r} already registered")
            customer = Customer(customer_id, currency)
            for ben_id in payees:
                customer.beneficiaries[ben_id] = Beneficiary(
                    ben_id, added_at=now - _ESTABLISHED_AGE_SECONDS, completed_payments=1, last_paid_at=now
                )
            customer.history[currency] = amounts[-self.policy.history_window:]
            self._customers[customer_id] = customer
            return customer

    def add_beneficiary(self, customer_id: str, beneficiary_id: str) -> Beneficiary:
        """Customer adds a payee now. It counts as new for the configured window."""
        if not isinstance(beneficiary_id, str) or not beneficiary_id:
            raise InputValidationError("beneficiary_id must be a non-empty string")
        with self._lock:
            customer = self._customer(customer_id)
            existing = customer.beneficiaries.get(beneficiary_id)
            if existing is not None:
                return existing
            ben = Beneficiary(beneficiary_id, added_at=self.clock())
            customer.beneficiaries[beneficiary_id] = ben
            return ben

    # -- detection ------------------------------------------------------------

    def beneficiary_status(self, customer_id: str, beneficiary_id: str) -> Dict[str, Any]:
        """New-beneficiary detection from the bank's own registry."""
        with self._lock:
            customer = self._customer(customer_id)
            ben = customer.beneficiaries.get(beneficiary_id)
            now = self.clock()
            if ben is None:
                return {"new": True, "reason": "not_registered"}
            if now - ben.added_at < self.policy.new_beneficiary_window_seconds:
                return {"new": True, "reason": "added_recently"}
            if ben.completed_payments == 0:
                return {"new": True, "reason": "never_paid"}
            return {"new": False, "reason": "established"}

    # -- payment API ------------------------------------------------------------

    def submit_payment(
        self,
        customer_id: str,
        beneficiary_id: str,
        amount: float,
        currency: Optional[str] = None,
        channel: str = "UPI",
    ) -> Payment:
        """Decide on a payment attempt using the customer's unexpired token."""
        if not isinstance(beneficiary_id, str) or not beneficiary_id:
            raise InputValidationError("beneficiary_id must be a non-empty string")
        amount = _amount(amount, "amount")
        with self._lock:
            customer = self._customer(customer_id)
            currency = _currency(currency or customer.currency, "currency")
            novelty = self.beneficiary_status(customer_id, beneficiary_id)
            tier = classify_amount(amount, currency, self.policy, customer.history.get(currency, ()))
            lookup = self.token_service.lookup(customer_id)
            call = CallSignal.from_token(lookup.token) if lookup.token is not None else None
            context = TransactionContext(amount, currency, novelty["new"], tier)
            decision = decide_action(call, context, self.policy)

            now = self.clock()
            payment = Payment(
                payment_id="pay_" + secrets.token_hex(8),
                customer_id=customer_id,
                beneficiary_id=beneficiary_id,
                amount=amount,
                currency=currency,
                channel=channel,
                created_at=now,
                status=STATUS_FOR_ACTION[decision.action],
                decision=decision,
                new_beneficiary=novelty["new"],
                beneficiary_reason=novelty["reason"],
                amount_tier=tier,
                token_status=lookup.status,
                hold_until=now + self.policy.hold_seconds if decision.action == HOLD else None,
            )
            self.audit_log.append(
                {
                    "event": "payment_decision",
                    "at": now,
                    "payment_ref": payment.payment_id,
                    "subject_ref": self.token_service.subject_ref(customer_id),
                    "beneficiary_ref": self.token_service.pseudonymize(beneficiary_id, "ben"),
                    "token_status": lookup.status,
                    "token_id": decision.token_id,
                    "call_level": decision.call_level,
                    "new_beneficiary": novelty["new"],
                    "beneficiary_reason": novelty["reason"],
                    "amount_tier": tier,
                    "action": decision.action,
                    "status": payment.status,
                    "reason_codes": list(decision.reason_codes),
                    "rationale": list(decision.rationale),
                    "policy_version": decision.policy_version,
                },
                (customer_id, beneficiary_id),
            )  # raises before any state changes if the record is not content-free
            self._payments[payment.payment_id] = payment
            if payment.status == APPROVED:
                self._execute(payment, via="auto")
            return payment

    def acknowledge_warning(self, payment_id: str) -> Payment:
        """WARN path: the customer has read the warning and chooses to continue."""
        with self._lock:
            payment = self._open_payment(payment_id, {PENDING_CONFIRMATION})
            self._execute(payment, via="customer_acknowledged_warning")
            return payment

    def verify_out_of_band(self, payment_id: str, confirmed: bool) -> Payment:
        """The customer was reached on a trusted channel (call-back to a known
        number, in-branch, in-app with a fresh session). ``confirmed=True``
        executes the payment, which also releases a HOLD early (the safe
        override). ``confirmed=False`` cancels it."""
        with self._lock:
            payment = self._open_payment(payment_id, {PENDING_VERIFICATION, ON_HOLD})
            if confirmed:
                self._execute(payment, via="out_of_band_verified")
            else:
                self._cancel(payment, via="out_of_band_declined")
            return payment

    def confirm_after_hold(self, payment_id: str) -> Payment:
        """Once the hold has ended, the customer may confirm without OOB verification."""
        with self._lock:
            payment = self._open_payment(payment_id, {ON_HOLD})
            now = self.clock()
            if now < payment.hold_until:
                raise InvalidTransition(
                    f"hold active for another {int(payment.hold_until - now)}s; "
                    "verify out-of-band to release it early"
                )
            self._execute(payment, via="customer_confirmed_after_hold")
            return payment

    def cancel_payment(self, payment_id: str) -> Payment:
        with self._lock:
            payment = self._open_payment(payment_id, OPEN_STATUSES)
            self._cancel(payment, via="customer_cancelled")
            return payment

    def get_payment(self, payment_id: str) -> Payment:
        with self._lock:
            if payment_id not in self._payments:
                raise KeyError(f"unknown payment {payment_id!r}")
            return self._payments[payment_id]

    def payments(self) -> List[Payment]:
        with self._lock:
            return list(self._payments.values())

    # -- internals ------------------------------------------------------------

    def _customer(self, customer_id: str) -> Customer:
        if customer_id not in self._customers:
            raise KeyError(f"unknown customer {customer_id!r}")
        return self._customers[customer_id]

    def _open_payment(self, payment_id: str, allowed: Iterable[str]) -> Payment:
        payment = self.get_payment(payment_id)
        if payment.status not in allowed:
            raise InvalidTransition(f"payment is {payment.status}; expected one of {sorted(allowed)}")
        return payment

    def _transition(self, payment: Payment, new_status: str, via: str) -> None:
        """Audit first, then change state, so a rejected record changes nothing."""
        self.audit_log.append(
            {
                "event": "payment_status",
                "at": self.clock(),
                "payment_ref": payment.payment_id,
                "from": payment.status,
                "to": new_status,
                "via": via,
            },
            (payment.customer_id, payment.beneficiary_id),
        )
        payment.status = new_status

    def _execute(self, payment: Payment, via: str) -> None:
        self._transition(payment, EXECUTED, via)
        now = self.clock()
        customer = self._customer(payment.customer_id)
        ben = customer.beneficiaries.get(payment.beneficiary_id)
        if ben is None:
            # First payment registers the payee; it stays "new" for the window.
            ben = Beneficiary(payment.beneficiary_id, added_at=now)
            customer.beneficiaries[payment.beneficiary_id] = ben
        ben.completed_payments += 1
        ben.last_paid_at = now
        history = customer.history.setdefault(payment.currency, [])
        history.append(payment.amount)
        del history[: -self.policy.history_window]
        payment.executed_at = now

    def _cancel(self, payment: Payment, via: str) -> None:
        self._transition(payment, CANCELLED, via)
        payment.cancelled_at = self.clock()

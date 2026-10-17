"""generate_semantic_dataset.py — build the labelled transcript dataset (synthetic).

Deterministic (seeded). Produces train/val/test JSONL of {"text": str, "label": 0|1}
where label 1 = coercive, 0 = benign. Crucially, coercive examples are generated
in PARAPHRASED form (non-trigger wording) so the classifier learns semantics,
not keywords. Benign examples include negated financial mentions.

Run:  python scripts/generate_semantic_dataset.py
"""

import json
import os
import random

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "semantic")

# --------------------------------------------------------------------------- #
# Synonym sets — each generates a paraphrase variant of the same meaning.
# --------------------------------------------------------------------------- #
SYN = {
    "transfer": ["transfer", "move", "send", "wire", "forward", "hand over"],
    "pay": ["pay", "settle", "clear", "cover", "deposit"],
    "money": ["money", "funds", "the amount", "cash"],
    "now": ["now", "immediately", "right away", "without delay", "right now", "this minute"],
    "urgent": ["urgent", "time-sensitive", "an emergency", "critical"],
    "tell": ["tell anyone", "mention this to anyone", "share this", "say a word to anyone", "talk about this"],
    "secret": ["keep this secret", "keep this private", "keep this between us", "not say a word"],
    "blocked": ["blocked", "frozen", "suspended", "locked", "shut down", "deactivated"],
    "arrest": ["arrested", "taken into custody", "prosecuted"],
    "police": ["the police", "the authorities", "law enforcement", "the officials"],
    "bank": ["the bank", "the security team", "the fraud department"],
    "hospital": ["the hospital", "the emergency room", "the medical center"],
    "son": ["your son", "your daughter", "your husband", "your father"],
    "amount": ["fifty thousand rupees", "two lakh", "twenty five thousand", "one lakh", "seventy five thousand"],
    "verify": ["verify", "check", "confirm"],
    "code": ["the OTP", "the verification code", "the security code", "the code I just sent"],
}

# --------------------------------------------------------------------------- #
# Coercive templates — each {slot} is filled from SYN.
# --------------------------------------------------------------------------- #
COERCIVE = [
    "You need to {transfer} {money} {now}.",
    "I need you to {transfer} {money} {now}, it is {urgent}.",
    "{transfer} {money} {now} or there will be a problem.",
    "Don't {tell}, just {transfer} the {money} {now}.",
    "Please {secret}, and {transfer} {money} {now}.",
    "Your account will be {blocked} unless you {pay} {now}.",
    "Your account is {blocked}, {pay} {money} to restore it {now}.",
    "This is {police}, you will be {arrest} unless you {pay} {money} {now}.",
    "This is {bank}, {transfer} {money} to the safe account {now}.",
    "Share {code} so we can {verify} your account {now}.",
    "{transfer} {money} to {son} who is in {hospital} {now}.",
    "There has been an accident, {son} is in {hospital}, {transfer} {money} {now}.",
    "You have won a prize, {pay} {money} as a processing fee {now}.",
    "Invest {money} {now} and we will double it for you.",
    "This is {bank}, your account will be {blocked} in one hour unless you {pay} {now}.",
    "Do not {tell}, and do not {verify}, just {transfer} the {money}.",
]

# --------------------------------------------------------------------------- #
# Benign templates (label 0) — include negated financial mentions.
# --------------------------------------------------------------------------- #
BENIGN = [
    "Hi, how are you? Did you have lunch today?",
    "Just calling to say hello and catch up.",
    "We will never ask for your OTP or PIN, please call the number on your card.",
    "This is a reminder about your appointment tomorrow at eleven AM.",
    "Dad had a small accident, please come to the hospital when you can.",
    "Dinner was twelve hundred rupees, send me your share whenever you get a chance, no rush.",
    "Your grocery delivery is here, the total is eight hundred and fifty rupees, please pay as usual.",
    "This is my regular pharmacy order, please pay my usual amount for this month's medicine.",
    "We noticed a login from a new device, please review it in our official app.",
    "I deposited the money this morning, just letting you know.",
    "Please do not share your PIN with anyone, even if they say they are from the bank.",
    "The report is ready, you can pick it up whenever is convenient.",
]


def _fill(template: str) -> str:
    """Replace {slots} with a sampled synonym from SYN."""
    out = template
    for key, options in SYN.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", random.choice(options))
    return out


def generate(n: int) -> list[dict]:
    """Generate n coercive + n benign examples (with paraphrase variety)."""
    rows = []
    for _ in range(n):
        rows.append({"text": _fill(random.choice(COERCIVE)), "label": 1})
    for _ in range(n):
        rows.append({"text": _fill(random.choice(BENIGN)), "label": 0})
    random.shuffle(rows)
    return rows


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for split, n in (("train", 600), ("val", 150), ("test", 150)):
        rows = generate(n)
        path = os.path.join(OUT_DIR, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

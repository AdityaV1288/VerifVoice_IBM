"""vm_semantic_test.py — verify the semantic layer runs on the VM (s390x)."""
import json
import urllib.request

BASE = "http://localhost:8000"


def call(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read().decode())


print("=== /health ===")
with urllib.request.urlopen(BASE + "/health") as r:
    print(r.status, json.loads(r.read().decode()))

print("=== /analyze (paraphrased coercion) on s390x ===")
status, r = call("/analyze", {
    "source": "I need you to move some money to a different account for me. It has to happen fast, and I would rather you did not mention it to anyone just yet.",
    "input_type": "text",
})
print("status:", status)
print("coercion_score:", r["coercion_score"])
print("semantic_score:", r["semantic_score"], "| model:", r["semantic_model"])
print("reason_codes:", r["reason_codes"])

print("=== /payment/decision (new beneficiary) ===")
status2, d = call("/payment/decision", {
    "token_id": r["token_id"], "amount": 50000, "new_beneficiary": True,
})
print("decision:", d["decision"], "| risk:", d["risk_score"], "| codes:", d["reason_codes"])

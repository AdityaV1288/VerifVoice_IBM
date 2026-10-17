"""demo.py — full end-to-end demo walkthrough (laptop).

Runs the complete signal chain on real audio:
  audio -> ASR -> coercion (rules + semantic) + spoof (AASIST) -> fusion -> decision

Run:  python demo.py
"""

import pipeline
import risk_engine


def run_scenario(path: str, title: str, amount: float, new_beneficiary: bool) -> None:
    print("\n" + "=" * 72)
    print(f"SCENARIO: {title}")
    print("=" * 72)

    result = pipeline.analyze_call(path, input_type="audio")

    print(f"  audio status    : {result['coercion_status']}")
    print(f"  coercion_score  : {result['coercion_score']}   (0.6 rules + 0.4 semantic)")
    print(f"    - semantic    : {result['semantic_score']}   ({result['semantic_model']})")
    print(f"    - rule signals: {result['signals']}")
    print(f"  spoof_score     : {result['spoof_score']}   ({result['spoof_classification']})")
    print(f"  reason_codes    : {result['reason_codes']}")

    decision = risk_engine.decide(
        coercion_score=result["coercion_score"],
        spoof_score=result["spoof_score"],
        new_beneficiary=new_beneficiary,
        amount=amount,
    )
    print(f"  --> BANK DECISION: {decision['decision'].upper()}   (risk {decision['risk_score']})")
    print(f"      message: {decision['message']}")
    print(f"      codes:   {decision['reason_codes']}")


if __name__ == "__main__":
    print("VerifVoice - end-to-end demo")
    run_scenario(
        "samples/golden_scam.wav",
        "GOLDEN: AI-cloned 'son' + hospital admin pressures a transfer",
        amount=50000,
        new_beneficiary=True,
    )
    run_scenario(
        "samples/benign_pharmacy.wav",
        "ACCESSIBILITY: AAC/TTS user pays a KNOWN pharmacy (synthetic, but benign)",
        amount=850,
        new_beneficiary=False,
    )

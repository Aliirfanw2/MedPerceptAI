"""Manual LLM reasoning smoke test (patient on bed + staff nearby)."""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from monitor.ml_pipeline import PatientIntentPipeline

# Applied only by the test checker after LLM response — never sent to the model.
EXPECTED_DECISION = {
    "patient_status": "patient_on_bed",
    "staff_presence": "staff_nearby",
    "safety_label": "SAFE",
    "alert_type": "no_alert",
}


def _normalize_actual(structured: dict, key: str) -> str:
    raw = str(structured.get(key) or "").strip()
    if key == "safety_label":
        return raw.upper()
    return raw.lower()


def _check_llm_decision(structured: dict) -> tuple[dict[str, str], dict[str, bool], bool]:
    actual = {key: _normalize_actual(structured, key) for key in EXPECTED_DECISION}
    checks = {
        key: actual[key] == (
            EXPECTED_DECISION[key].upper() if key == "safety_label" else EXPECTED_DECISION[key]
        )
        for key in EXPECTED_DECISION
    }
    return actual, checks, all(checks.values())


class Command(BaseCommand):
    help = (
        "Run basic LLM reasoning test: Patient lying on bed, Nurse and Doctor standing nearby. "
        "Checks that the LLM independently returns SAFE / no_alert / patient_on_bed / staff_nearby."
    )

    def handle(self, *args, **options):
        pipeline = PatientIntentPipeline()
        self.stdout.write("Loading Llama and running basic reasoning test...")
        result = pipeline.run_basic_reasoning_test()

        self.stdout.write("\nScene payload sent to LLM:")
        self.stdout.write(json.dumps(result["scene_payload"], indent=2))

        structured = result["structured"]
        self.stdout.write("\nLLM structured output:")
        self.stdout.write(json.dumps(structured, indent=2, default=str))

        actual, checks, passed = _check_llm_decision(structured)
        self.stdout.write("\nExpected vs actual (checker only, not in LLM prompt):")
        for key, expected in EXPECTED_DECISION.items():
            ok = "OK" if checks[key] else "FAIL"
            self.stdout.write(f"  [{ok}] {key}: expected={expected!r} actual={actual[key]!r}")

        trace = result.get("trace") or {}
        if trace.get("fallback"):
            self.stdout.write(
                self.style.WARNING(
                    f"\nFallback used: {trace.get('fallback_reason')} "
                    f"(decision_source={structured.get('decision_source')})"
                )
            )

        latency = result.get("llama_latency_ms")
        if latency is not None:
            self.stdout.write(f"\nLlama latency: {latency} ms")

        if passed:
            self.stdout.write(self.style.SUCCESS("\nBasic reasoning test PASSED."))
        else:
            self.stdout.write(self.style.ERROR("\nBasic reasoning test FAILED (see mismatches above)."))

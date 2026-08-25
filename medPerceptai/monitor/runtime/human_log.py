"""Human-readable terminal output for reasoning decisions (demo / evaluation)."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from monitor.runtime.types import ReasoningContext, ReasoningResult

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 60


def human_readable_logs_enabled() -> bool:
    return os.environ.get("HUMAN_READABLE_LOGS", "0").strip().lower() in ("1", "true", "yes", "on")


def _field_line(label: str, field: Dict[str, Any]) -> str:
    value = field.get("value")
    source = field.get("source", "—")
    if isinstance(value, bool):
        value = "Yes" if value else "No"
    elif isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value) if value else "none"
    return f"- {label}: {value} [{source}]"


def format_reasoning_human(
    context: ReasoningContext,
    reasoning_result: ReasoningResult,
    risk_result: Optional[Dict[str, Any]] = None,
) -> str:
    _ = risk_result
    scores = reasoning_result.confidence_scores or {}
    scene = dict(context.scene or scores.get("scene") or {})
    structured = dict(scores.get("structured_reasoning") or {})
    sections = scores.get("display_sections")
    if not sections:
        from monitor.runtime.display_fields import build_display_sections

        sections = build_display_sections(
            scene=scene,
            structured=structured,
            scores=scores,
            intent=str(reasoning_result.intent or ""),
        )

    model = sections.get("model_output") or {}
    scene_section = sections.get("scene_sent_to_llm") or {}
    llm = sections.get("llm_final_decision") or sections.get("llm_decision") or {}
    system = sections.get("system_status") or {}

    capture_active = scores.get("capture_active")
    exit_reason = scores.get("capture_exit_reason")
    overlay_stale = bool(scores.get("overlay_stale"))
    if capture_active is False:
        overlay_stale = True
    now = datetime.now().strftime("%H:%M:%S")
    capture_fid = scores.get("capture_frame_id")

    scene_summary_field = scene_section.get("scene_summary") or scene_section.get("scene_json", {"value": "—", "source": "fusion"})
    scene_summary = str(scene_summary_field.get("value") or "—")
    if len(scene_summary) > 400:
        scene_summary = scene_summary[:400] + "…"

    lines = [
        _SEPARATOR,
        "MEDPERCEPTAI DECISION",
        f"Frame ID: {context.frame_id}",
        f"Capture frame ID: {capture_fid if capture_fid is not None else context.frame_id}",
        f"Capture active: {'Yes' if capture_active is True else 'No' if capture_active is False else 'unknown'}",
        f"Overlay stale: {'Yes' if overlay_stale else 'No'}",
        f"Capture exit: {exit_reason or 'none'}",
        f"Time: {now}",
        "",
        "MODEL OUTPUT",
        _field_line("Person detected", model.get("person_detected", {"value": "—", "source": "—"})),
        _field_line("Objects", model.get("objects", {"value": "—", "source": "yolo_object"})),
        _field_line("Role", model.get("role", {"value": "—", "source": "—"})),
        _field_line("Role confidence", model.get("role_confidence", {"value": "—", "source": "—"})),
        _field_line("Pose", model.get("pose", {"value": "—", "source": "—"})),
        _field_line("Pose confidence", model.get("pose_confidence", {"value": "—", "source": "—"})),
        _field_line("Bed relation", model.get("bed_relation", {"value": "—", "source": "—"})),
        _field_line("Missing signals", model.get("missing_signals", {"value": [], "source": "—"})),
        "",
        "SCENE SENT TO LLM",
        f"- llm_scene_frame_id: {scores.get('llm_scene_frame_id', scene.get('frame_id'))} [fusion]",
        f"- scene_summary: {scene_summary} [{scene_summary_field.get('source', 'fusion')}]",
        "",
        "LLM FINAL DECISION",
        _field_line("Patient status", llm.get("patient_status", {"value": "not provided", "source": "—"})),
        _field_line("Staff presence", llm.get("staff_presence", {"value": "not provided", "source": "—"})),
        _field_line("Staff activity", llm.get("staff_activity", {"value": "not provided", "source": "—"})),
        _field_line("Equipment context", llm.get("equipment_context", {"value": "not provided", "source": "—"})),
        _field_line("Safety label", llm.get("safety_label", {"value": "not provided", "source": "—"})),
        _field_line("Alert type", llm.get("alert_type", {"value": "not provided", "source": "—"})),
        _field_line("Risk level", llm.get("risk_level", {"value": "not provided", "source": "—"})),
        _field_line("Risk score", llm.get("risk_score", {"value": "not provided", "source": "—"})),
        _field_line("Reason", llm.get("reason", {"value": "not provided", "source": "—"})),
        _field_line("Summary", llm.get("summary", {"value": "not provided", "source": "—"})),
        "",
        "SYSTEM STATUS",
        _field_line("Mode", system.get("mode", {"value": scores.get("reasoning_mode"), "source": "runtime"})),
        _field_line("Decision source", system.get("decision_source", {"value": structured.get("decision_source"), "source": "runtime"})),
        _field_line("Capture active", system.get("capture_active", {"value": capture_active, "source": "capture_service"})),
        _field_line("Overlay stale", system.get("overlay_stale", {"value": overlay_stale, "source": "state_manager"})),
        _field_line("Capture exit reason", system.get("capture_exit_reason", {"value": exit_reason or "none", "source": "capture_service"})),
        _field_line("Fallback reason", system.get("fallback_reason", {"value": scores.get("fallback_reason") or "none", "source": "runtime"})),
        _field_line("Reasoning stale", system.get("reasoning_stale", {"value": scores.get("reasoning_stale"), "source": "runtime"})),
        _field_line("Consistency warning", system.get("reasoning_consistency_warning", {"value": scores.get("reasoning_consistency_warning"), "source": "runtime"})),
        _field_line("Warning text", system.get("warning_text", {"value": scores.get("warning_text") or "none", "source": "runtime"})),
        _SEPARATOR,
    ]
    return "\n".join(lines)


def format_decision_compact(
    context: ReasoningContext,
    reasoning_result: ReasoningResult,
    risk_result: Optional[Dict[str, Any]] = None,
) -> str:
    _ = risk_result
    scores = reasoning_result.confidence_scores or {}
    structured = dict(scores.get("structured_reasoning") or {})
    safety = str(structured.get("safety_label") or "not provided")
    risk = str(structured.get("risk_level") or "not provided")
    mode = str(scores.get("reasoning_mode") or "fallback")
    source = str(structured.get("decision_source") or scores.get("decision_source") or "fallback")
    alert = safety.upper() == "ALERT"
    raw_score = structured.get("risk_score") if structured.get("risk_score") is not None else scores.get("risk_score")
    score_text = "not provided"
    if raw_score is not None:
        try:
            score_text = str(int(round(float(raw_score))))
        except (TypeError, ValueError):
            score_text = "not provided"
    return (
        f"[decision] frame={context.frame_id} safety={safety} risk={risk} score={score_text} "
        f"alert={alert} mode={mode} source={source}"
    )


def emit_human_reasoning_log(
    context: ReasoningContext,
    reasoning_result: ReasoningResult,
    risk_result: Optional[Dict[str, Any]] = None,
) -> None:
    if not human_readable_logs_enabled():
        return
    try:
        print(format_reasoning_human(context, reasoning_result, risk_result), flush=True)
        logger.info(format_decision_compact(context, reasoning_result, risk_result))
    except Exception as exc:
        logger.warning("Human-readable reasoning log failed frame_id=%s: %s", context.frame_id, exc)

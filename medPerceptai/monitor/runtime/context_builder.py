from __future__ import annotations

from typing import Any, Dict

import numpy as np

from monitor.runtime.tensor_utils import coerce_landmarks
from monitor.runtime.types import ObjectResult, PoseResult, ReasoningContext, ReasoningResult, RoleResult

STAFF_OVERLAY_ROLES = frozenset({"nurse", "doctor", "staff", "relative", "relatives"})


def _primary_pose_from_scene(scene: Dict[str, Any]) -> str:
    primary = scene.get("primary_person") or {}
    pose = str(primary.get("pose") or scene.get("patient_status") or "").strip().lower()
    return pose if pose and pose != "unknown" else ""


def build_overlay_status_line(
    role_hint: str,
    intent: str,
    scene: Dict[str, Any] | None = None,
) -> str:
    """Second line on the video box — posture only for confirmed patients."""
    role = str(role_hint or "unknown").strip().lower()
    scene = scene or {}
    pose = _primary_pose_from_scene(scene)

    if role in STAFF_OVERLAY_ROLES:
        return ""

    if role == "patient":
        if pose:
            return pose
        intent_text = str(intent or "").strip()
        if intent_text.lower().startswith("patient "):
            return intent_text[8:].strip()[:30]
        return intent_text[:30] or "detected"

    return ""


def build_overlay_box_label(
    role_hint: str,
    confidence: float,
    intent: str,
    scene: Dict[str, Any] | None = None,
) -> str:
    role = str(role_hint or "unknown").strip().lower()
    head = f"{role} {confidence:.2f}"
    status = build_overlay_status_line(role_hint, intent, scene)
    if not status:
        return head
    return f"{head} | {status}"


def build_reasoning_context(
    *,
    frame_id: int,
    frame: np.ndarray,
    config: Dict[str, Any],
    scene: Dict[str, Any],
    object_result: ObjectResult,
    role_result: RoleResult,
    pose_result: PoseResult,
) -> ReasoningContext:
    primary = scene.get("primary_person") or {}
    bbox = primary.get("bbox") or {}
    primary_box = None
    if bbox:
        primary_box = (
            int(bbox.get("x1", 0)),
            int(bbox.get("y1", 0)),
            int(bbox.get("x2", 0)),
            int(bbox.get("y2", 0)),
        )

    role_hint = str(primary.get("role") or role_result.role_hint or "unknown")
    fused_keypoints = coerce_landmarks(primary.get("keypoints"))
    if fused_keypoints is not None:
        pose_summary = dict(primary.get("pose_summary") or {"available": False})
    else:
        pose_summary = {"available": False, "pose_detected": False}
    confidence = float(primary.get("object_conf") or object_result.confidence or 0.0)
    detection_hint = object_result.detection_hint
    if primary_box and confidence >= 0.25:
        detection_hint = "patient detected"
    elif primary_box:
        detection_hint = "patient under observation"

    crop = np.empty((0, 0, 3), dtype=getattr(frame, "dtype", np.uint8))
    if primary_box is not None:
        x1, y1, x2, y2 = primary_box
        if y2 > y1 and x2 > x1:
            crop = frame[y1:y2, x1:x2].copy()

    return ReasoningContext(
        frame_id=frame_id,
        frame=frame,
        config=config,
        scene=scene,
        object_result=object_result,
        role_result=role_result,
        pose_result=pose_result,
        crop=crop,
        detection_hint=detection_hint,
        role_hint=role_hint,
        pose_landmarks=fused_keypoints,
        pose_summary=pose_summary,
        primary_box=primary_box,
        confidence=confidence,
    )


def context_to_overlay_result(context: ReasoningContext) -> ReasoningResult:
    """Lightweight overlay for the video stream before slow Llama reasoning finishes."""
    bbox = None
    if context.primary_box is not None:
        x1, y1, x2, y2 = context.primary_box
        bbox = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": float(context.confidence or 0.0),
        }
    scene = context.scene or {}
    intent = build_overlay_status_line(
        context.role_hint or "unknown",
        context.detection_hint or "Scanning…",
        scene,
    )
    return ReasoningResult(
        frame_id=context.frame_id,
        intent=intent,
        alert_triggered=False,
        bbox=bbox,
        pose_summary=dict(context.pose_summary or {"available": False}),
        model_status={},
        use_reasoning=False,
        latency_ms=0,
        role_hint=context.role_hint or "unknown",
        detection_list=list(context.object_result.detections or []),
        primary_box=context.primary_box,
        pose_landmarks=context.pose_landmarks,
        confidence_scores={
            "scene": scene,
            "capture_frame_id": context.frame_id,
            "overlay_frame_id": context.frame_id,
            "overlay_stale": False,
            "capture_active": True,
        },
        monitor_status="monitor",
    )


def build_empty_context(job_frame_id: int, frame, config: dict, object_result: ObjectResult) -> ReasoningContext:
    empty_crop = np.empty((0, 0, 3), dtype=getattr(frame, "dtype", np.uint8))
    scene: Dict[str, Any] = {
        "frame_id": job_frame_id,
        "objects": list(object_result.detections or []),
        "persons": [],
        "roles": [],
        "poses": [],
        "relations": {},
        "staff_presence": "unknown_staff",
        "patient_status": "unknown",
        "bed_relation": "unknown",
        "missing_signals": ["object"],
        "confidence_scores": {},
        "primary_person": None,
    }
    return ReasoningContext(
        frame_id=job_frame_id,
        frame=frame,
        config=config,
        scene=scene,
        object_result=object_result,
        role_result=RoleResult(frame_id=job_frame_id),
        pose_result=PoseResult(frame_id=job_frame_id),
        crop=empty_crop,
        detection_hint=object_result.detection_hint,
        role_hint="unknown",
        pose_landmarks=None,
        pose_summary={"available": False},
        primary_box=None,
        confidence=0.0,
    )

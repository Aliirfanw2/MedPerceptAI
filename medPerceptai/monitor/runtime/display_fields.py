"""Dashboard/human-log display bundles — model output, scene, LLM decision, system status."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from monitor import presentation_config as pc
from monitor.runtime.fusion import MEDICAL_STAFF_ROLES, VISITOR_ROLES

logger = logging.getLogger(__name__)

LLM_UPDATING_MESSAGE = "Reasoning updating — waiting for current LLM decision"
_STALE_LLM_PLACEHOLDER = "updating"


def _missing_list(scene: Dict[str, Any], scores: Dict[str, Any]) -> List[str]:
    return list(scene.get("missing_signals") or scores.get("missing_signals") or [])


def _llm_value(structured: Dict[str, Any], key: str, default: str = "not provided") -> str:
    val = structured.get(key)
    if val is None or (isinstance(val, str) and not str(val).strip()):
        return default
    return str(val)


def _round_box(box: Any) -> Optional[List[int]]:
    if not box:
        return None
    try:
        return [int(round(float(v))) for v in box[:4]]
    except (TypeError, ValueError, IndexError):
        return None


def _normalize_role(role: Any) -> str:
    text = str(role or "unknown").strip().lower()
    if text in ("relatives", "relative"):
        return "relatives"
    return text


def _compact_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    box = _round_box(obj.get("box"))
    return {
        "label": str(obj.get("label") or ""),
        "confidence": round(float(obj.get("confidence") or 0.0), 3),
        "box": box,
    }


def _compact_person(person: Dict[str, Any]) -> Dict[str, Any]:
    bbox = person.get("bbox") or {}
    box = _round_box([bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")])
    nearest_bed = person.get("nearest_bed")
    bed_label = str(nearest_bed.get("label") or "") if isinstance(nearest_bed, dict) else None
    role = _normalize_role(person.get("role_for_llm") or person.get("role"))
    return {
        "id": person.get("id"),
        "label": person.get("label"),
        "role": role,
        "role_conf": person.get("role_conf"),
        "role_source": person.get("role_source"),
        "role_uncertain": bool(person.get("role_uncertain")),
        "pose": person.get("pose"),
        "pose_conf": person.get("pose_conf"),
        "pose_source": person.get("pose_source"),
        "relation_to_bed": person.get("relation_to_bed"),
        "staff_near": person.get("staff_near"),
        "bbox": box,
        "object_conf": round(float(person.get("object_conf") or 0.0), 3),
        "nearest_bed_label": bed_label,
    }


def _compact_role(role: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": _normalize_role(role.get("role")),
        "confidence": round(float(role.get("confidence") or 0.0), 3),
        "box": _round_box(role.get("box")),
    }


def _compact_pose(pose: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(pose.get("pose_summary") or {})
    return {
        "pose": summary.get("pose_class") or summary.get("pose"),
        "confidence": round(float(pose.get("confidence") or summary.get("pose_score") or 0.0), 3),
        "box": _round_box(pose.get("box")),
    }


def _primary_role_flags(primary: Dict[str, Any]) -> Dict[str, Any]:
    role = _normalize_role(primary.get("role_for_llm") or primary.get("role"))
    conf = float(primary.get("role_conf") or 0.0)
    uncertain = bool(primary.get("role_uncertain")) or conf < pc.ROLE_MIN_DISPLAY_CONF
    if uncertain and role not in ("unknown", "unknown_low_confidence"):
        role_for_llm = "unknown_low_confidence"
    else:
        role_for_llm = role
    return {
        "primary_person_role": role_for_llm,
        "primary_person_role_conf": round(conf, 3),
        "primary_person_role_source": str(primary.get("role_source") or "fusion"),
        "primary_person_is_patient_candidate": role_for_llm in ("patient", "unknown", "unknown_low_confidence"),
        "primary_person_is_staff_candidate": role_for_llm in MEDICAL_STAFF_ROLES,
        "role_uncertain": uncertain,
    }


def _llm_role_label(person: Dict[str, Any]) -> str:
    """Display role for LLM grounding (Patient, Doctor, Nurse, Relatives, Unknown)."""
    role = _normalize_role(person.get("role_for_llm") or person.get("role"))
    labels = {
        "patient": "Patient",
        "doctor": "Doctor",
        "nurse": "Nurse",
        "staff": "Staff",
        "relatives": "Relatives",
        "unknown": "Unknown",
        "unknown_low_confidence": "Unknown",
    }
    return labels.get(role, "Unknown")


def _is_person_object_label(label: str) -> bool:
    text = str(label or "").strip().lower()
    return any(k in text for k in ("person", "patient", "human")) and "bed" not in text


def _object_class_label(label: str) -> str:
    text = str(label or "").strip().lower().replace(" ", "_")
    if "bed" in text:
        return "bed"
    return text or "unknown"


def _is_patient_role_label(role: str) -> bool:
    return role in ("Patient", "Unknown") or _normalize_role(role) in (
        "patient",
        "unknown",
        "unknown_low_confidence",
    )


def _build_llm_objects(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects_out: List[Dict[str, Any]] = []
    obj_id = 0
    for obj in scene.get("objects") or []:
        label = str(obj.get("label") or "")
        if _is_person_object_label(label):
            continue
        objects_out.append(
            {
                "id": obj_id,
                "class": _object_class_label(label),
                "confidence": round(float(obj.get("confidence") or 0.0), 3),
            }
        )
        obj_id += 1
    return objects_out


def _build_llm_roles(persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    roles_out: List[Dict[str, Any]] = []
    for person in persons:
        pid = person.get("id")
        if pid is None:
            continue
        roles_out.append(
            {
                "id": int(pid),
                "role": _llm_role_label(person),
                "confidence": round(float(person.get("role_conf") or 0.0), 3),
            }
        )
    return roles_out


def _build_llm_poses(persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    poses_out: List[Dict[str, Any]] = []
    for person in persons:
        pid = person.get("id")
        if pid is None:
            continue
        pose = str(person.get("pose") or "unknown")
        poses_out.append(
            {
                "id": int(pid),
                "pose": pose,
                "confidence": round(float(person.get("pose_conf") or 0.0), 3),
            }
        )
    return poses_out


def _build_llm_relations(
    persons: List[Dict[str, Any]],
    objects: List[Dict[str, Any]],
    scene: Dict[str, Any],
) -> List[Dict[str, Any]]:
    relations: List[Dict[str, Any]] = []
    bed_ids = [o["id"] for o in objects if str(o.get("class") or "").lower() == "bed"]

    for person in persons:
        pid = person.get("id")
        if pid is None:
            continue
        rel_bed = str(person.get("relation_to_bed") or "unknown")
        if rel_bed not in ("unknown", "") and bed_ids:
            relations.append(
                {
                    "subject": int(pid),
                    "relation": rel_bed,
                    "object": int(bed_ids[0]),
                }
            )

    patients = [p for p in persons if _is_patient_role_label(_llm_role_label(p))]
    staff_persons = [
        p
        for p in persons
        if _normalize_role(p.get("role_for_llm") or p.get("role")) in MEDICAL_STAFF_ROLES
    ]

    for patient in patients:
        if not patient.get("staff_near"):
            continue
        patient_id = int(patient.get("id", 0))
        if staff_persons:
            for staff in staff_persons:
                relations.append(
                    {
                        "subject": int(staff.get("id", 0)),
                        "relation": "staff_nearby",
                        "object": patient_id,
                    }
                )
        else:
            for idx, det in enumerate(scene.get("roles") or []):
                if _normalize_role(det.get("role")) in MEDICAL_STAFF_ROLES:
                    relations.append(
                        {
                            "subject": f"staff_{idx}",
                            "relation": "staff_nearby",
                            "object": patient_id,
                        }
                    )
                    break
    return relations


def build_llm_scene_payload(scene: Dict[str, Any], *, frame_id: Optional[int] = None) -> Dict[str, Any]:
    """Structured scene for LLM — roles, poses, objects, relations only (no keypoints/bboxes)."""
    expected_fid = int(frame_id if frame_id is not None else scene.get("frame_id") or 0)
    persons = list(scene.get("persons") or [])
    if not persons and scene.get("primary_person"):
        persons = [dict(scene.get("primary_person") or {})]

    objects = _build_llm_objects(scene)
    if "expected_bed_present" in scene:
        expected_bed_present = bool(scene["expected_bed_present"])
    else:
        expected_bed_present = pc.expected_bed_present()
    relations = scene.get("llm_relations")
    if not isinstance(relations, list):
        relations = _build_llm_relations(persons, objects, scene)
    return {
        "frame_id": expected_fid,
        "roles": _build_llm_roles(persons),
        "poses": _build_llm_poses(persons),
        "objects": objects,
        "relations": relations,
        "environment": pc.scene_environment(),
        "expected_bed_present": expected_bed_present,
    }


def format_compact_scene_display(payload: Optional[Dict[str, Any]]) -> str:
    """One-line summary for dashboard (not full JSON)."""
    if not payload:
        return "—"
    roles = payload.get("roles") or []
    poses = payload.get("poses") or []
    objects = payload.get("objects") or []
    relations = payload.get("relations") or []
    role_text = ", ".join(f"{r.get('id')}:{r.get('role')}" for r in roles[:3]) or "none"
    return (
        f"env={payload.get('environment', 'hospital_room')} · "
        f"bed_expected={payload.get('expected_bed_present', '—')} · "
        f"roles=[{role_text}] · poses={len(poses)} · objects={len(objects)} · relations={len(relations)}"
    )


def build_scene_summary(
    scene: Dict[str, Any],
    *,
    llm_scene_payload: Optional[Dict[str, Any]] = None,
    frame_id: Optional[int] = None,
) -> str:
    """Compact scene summary sent to the LLM reasoner."""
    payload = llm_scene_payload or build_llm_scene_payload(scene, frame_id=frame_id)
    return format_compact_scene_display(payload)


def evaluate_llm_frame_sync(
    *,
    scene_frame_id: int,
    decision_frame_id: int,
    capture_frame_id: int = 0,
) -> Tuple[bool, int]:
    """True when LLM decision must not be shown as current (frame mismatch or capture lag)."""
    decision_fid = int(decision_frame_id or 0)
    scene_fid = int(scene_frame_id or 0)
    capture_fid = int(capture_frame_id or 0)
    lag = max(0, capture_fid - decision_fid) if capture_fid and decision_fid else 0
    frame_match = not scene_fid or not decision_fid or scene_fid == decision_fid
    stale = not frame_match or lag > pc.LLM_LAG_STALE_FRAMES
    return stale, lag


def log_llm_sync(
    *,
    scene_frame_id: int,
    decision_frame_id: int,
    capture_frame_id: int,
    stale: bool,
) -> None:
    logger.info(
        "[LLM_SYNC] scene_frame=%s decision_frame=%s capture_frame=%s stale=%s",
        scene_frame_id,
        decision_frame_id,
        capture_frame_id,
        stale,
    )


def evaluate_reasoning_consistency(
    scene: Dict[str, Any],
    structured: Dict[str, Any],
    scores: Dict[str, Any],
) -> Tuple[bool, str]:
    """Detect contradictions between YOLO/fusion model output and LLM narrative."""
    warnings: List[str] = []
    primary = dict(scene.get("primary_person") or {})
    model_role = _normalize_role(primary.get("role_display") or primary.get("role"))
    role_conf = float(primary.get("role_conf") or 0.0)
    staff_presence = str(structured.get("staff_presence") or "").lower()
    patient_status = str(structured.get("patient_status") or "").lower()

    medical_staff_detected = bool(
        int(scene.get("staff_role_count") or 0) > 0
        or any(
            _normalize_role(r.get("role")) in MEDICAL_STAFF_ROLES
            for r in (scene.get("roles") or [])
        )
    )

    staff_near_claim = any(
        token in staff_presence
        for token in (
            "staff_near",
            "staff_nearby",
            "staff nearby",
            "staff is nearby",
            "staff are nearby",
        )
    )
    llm_payload = scores.get("llm_scene_payload") or {}
    relations = llm_payload.get("relations") if isinstance(llm_payload.get("relations"), list) else []
    has_staff_nearby_relation = any(
        str(r.get("relation") or "") == "staff_nearby" for r in relations if isinstance(r, dict)
    )
    if staff_near_claim and not medical_staff_detected and not has_staff_nearby_relation:
        warnings.append("LLM says staff_nearby, but no staff role or relation detected.")

    if model_role in VISITOR_ROLES:
        if "patient" in patient_status and patient_status not in (
            "not detected",
            "not applicable",
            "not provided",
        ):
            warnings.append("LLM describes relatives as patient.")
        if staff_near_claim:
            warnings.append("LLM says staff_nearby, but primary role is Relatives (visitor).")

    if role_conf < pc.ROLE_MIN_DISPLAY_CONF and model_role not in ("unknown", "not detected"):
        warnings.append(f"Role confidence low ({role_conf:.2f} < {pc.ROLE_MIN_DISPLAY_CONF:.2f}).")

    text = " ".join(warnings)
    return bool(warnings), text


def build_display_sections(
    *,
    scene: Dict[str, Any],
    structured: Dict[str, Any],
    scores: Dict[str, Any],
    intent: str = "",
    decision_frame_id: Optional[int] = None,
    llm_scene_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    missing = _missing_list(scene, scores)
    primary = dict(scene.get("primary_person") or {})
    sources = dict(scene.get("display_sources") or {})

    decision_fid = int(
        decision_frame_id
        or scores.get("llm_output_frame_id")
        or scores.get("frame_id")
        or 0
    )
    payload_fid = int((llm_scene_payload or {}).get("frame_id") or 0) if isinstance(llm_scene_payload, dict) else 0
    scene_frame_id = int(
        scores.get("llm_scene_frame_id")
        or payload_fid
        or decision_fid
        or 0
    )
    capture_fid = int(scores.get("capture_frame_id") or 0)
    reasoning_stale, _ = evaluate_llm_frame_sync(
        scene_frame_id=scene_frame_id,
        decision_frame_id=decision_fid,
        capture_frame_id=capture_fid,
    )
    if bool(scores.get("reasoning_stale")):
        reasoning_stale = True
    scores["reasoning_stale"] = reasoning_stale

    if llm_scene_payload is None and not reasoning_stale:
        llm_scene_payload = build_llm_scene_payload(scene, frame_id=scene_frame_id or decision_fid)
    elif llm_scene_payload is None:
        llm_scene_payload = {"frame_id": scene_frame_id, "error": "scene frame mismatch"}

    person_detected = primary.get("bbox") is not None or scene.get("primary_person") is not None
    if "object" in missing:
        person_display = "not available"
        person_source = "default"
    elif not person_detected:
        person_display = "not detected"
        person_source = "default"
    else:
        person_display = "yes"
        person_source = sources.get("person_detected", "yolo_object")

    if "role" in missing:
        role_display = "not available"
        role_source = "default"
    else:
        role_display = str(primary.get("role_display") or primary.get("role") or "not detected")
        if role_display == "unknown":
            role_display = "not detected"
        role_source = str(primary.get("role_source") or sources.get("role", "fusion"))
        if primary.get("role_uncertain") or float(primary.get("role_conf") or 0.0) < pc.ROLE_MIN_DISPLAY_CONF:
            if role_display not in ("not available", "not detected"):
                role_display = f"{role_display} (uncertain)"

    if "pose" in missing:
        pose_display = "not available"
        pose_source = "default"
    elif not primary.get("pose_summary", {}).get("pose_detected") and str(primary.get("pose") or "unknown") == "unknown":
        pose_display = "not detected"
        pose_source = str(primary.get("pose_source") or sources.get("pose", "default"))
    else:
        pose_display = str(primary.get("pose") or scene.get("patient_status") or "not detected")
        if pose_display == "unknown":
            pose_display = "not detected"
        pose_source = str(primary.get("pose_source") or sources.get("pose", "yolo_pose+classifier"))

    objects: List[str] = []
    for obj in scene.get("objects") or []:
        label = str(obj.get("label") or "").strip()
        if label and label not in objects:
            objects.append(label)
    if "object" in missing and not objects:
        objects_display = "not available"
    elif not objects:
        objects_display = "not detected"
    else:
        objects_display = ", ".join(objects)

    if "bed" in missing or not scene.get("relations", {}).get("bed_count"):
        bed_display = "not detected"
        bed_source = "default_unknown"
    else:
        bed_display = str(scene.get("bed_relation") or primary.get("relation_to_bed") or "not detected")
        bed_source = str(sources.get("bed_relation", "yolo_object+geometry"))

    model_output = {
        "person_detected": {"value": person_display, "source": person_source},
        "objects": {"value": objects_display, "source": "yolo_object"},
        "role": {"value": role_display, "source": role_source},
        "role_confidence": {
            "value": primary.get("role_conf") if role_display not in ("not available", "not detected") else "—",
            "source": role_source,
        },
        "pose": {"value": pose_display, "source": pose_source},
        "pose_confidence": {
            "value": primary.get("pose_conf") or (primary.get("pose_summary") or {}).get("pose_score")
            if pose_display not in ("not available", "not detected")
            else "—",
            "source": pose_source,
        },
        "bed_relation": {"value": bed_display, "source": bed_source},
        "missing_signals": {"value": missing or [], "source": "aggregator+fusion"},
    }

    scene_sent_to_llm = {
        "llm_scene_frame_id": {"value": scene_frame_id or "—", "source": "fusion"},
        "scene_summary": {
            "value": build_scene_summary(scene, llm_scene_payload=llm_scene_payload),
            "source": "fusion",
        },
    }

    decision_source = str(
        structured.get("decision_source")
        or scores.get("decision_source")
        or ("llama" if str(scores.get("reasoning_mode", "")).startswith("llama") else "fallback")
    )
    is_fallback = decision_source == "fallback" or reasoning_stale
    stale_source = "stale"

    if reasoning_stale:
        consistency_warning, warning_text = False, ""
    else:
        consistency_warning, warning_text = evaluate_reasoning_consistency(scene, structured, scores)

    def _llm_field(key: str, default: str = "not provided") -> Dict[str, Any]:
        if reasoning_stale:
            return {"value": _STALE_LLM_PLACEHOLDER, "source": stale_source}
        return {
            "value": _llm_value(structured, key, default),
            "source": "fallback" if is_fallback else "llama",
        }

    llm_final_decision = {
        "patient_status": _llm_field("patient_status"),
        "staff_presence": _llm_field("staff_presence"),
        "staff_activity": _llm_field("staff_activity"),
        "equipment_context": _llm_field("equipment_context"),
        "safety_label": _llm_field("safety_label"),
        "alert_type": _llm_field("alert_type"),
        "risk_level": _llm_field("risk_level"),
        "risk_score": {
            "value": _STALE_LLM_PLACEHOLDER if reasoning_stale else (
                structured.get("risk_score") if structured.get("risk_score") is not None else "not provided"
            ),
            "source": stale_source if reasoning_stale else ("fallback" if is_fallback else "llama"),
        },
        "reason": {
            "value": LLM_UPDATING_MESSAGE if reasoning_stale else _llm_value(structured, "reason", intent or "not provided"),
            "source": stale_source if reasoning_stale else ("fallback" if is_fallback else "llama"),
        },
        "summary": {
            "value": LLM_UPDATING_MESSAGE if reasoning_stale else _llm_value(structured, "summary", intent or "not provided"),
            "source": stale_source if reasoning_stale else ("fallback" if is_fallback else "llama"),
        },
    }

    capture_active = scores.get("capture_active")
    overlay_stale = bool(scores.get("overlay_stale"))
    exit_reason = scores.get("capture_exit_reason")
    reasoning_mode = str(scores.get("reasoning_mode") or "fallback")
    fallback_reason = scores.get("llama_runtime_error") or scores.get("fallback_reason") or "none"

    system_status = {
        "mode": {"value": reasoning_mode, "source": "runtime"},
        "decision_source": {"value": decision_source, "source": "runtime"},
        "reasoning_stale": {"value": "Yes" if reasoning_stale else "No", "source": "runtime"},
        "capture_active": {
            "value": "Yes" if capture_active is True else ("No" if capture_active is False else "unknown"),
            "source": "capture_service",
        },
        "overlay_stale": {
            "value": "Yes" if overlay_stale else "No",
            "source": "state_manager",
        },
        "capture_exit_reason": {
            "value": exit_reason or "none",
            "source": "capture_service",
        },
        "fallback_reason": {
            "value": fallback_reason,
            "source": "runtime",
        },
        "reasoning_consistency_warning": {
            "value": "Yes" if consistency_warning else "No",
            "source": "runtime",
        },
        "warning_text": {
            "value": warning_text or "none",
            "source": "runtime",
        },
    }

    return {
        "model_output": model_output,
        "scene_sent_to_llm": scene_sent_to_llm,
        "llm_final_decision": llm_final_decision,
        "system_status": system_status,
    }


def patch_display_sections_capture(
    sections: Dict[str, Any],
    *,
    capture_active: bool,
    overlay_stale: bool,
    capture_exit_reason: Any = None,
) -> Dict[str, Any]:
    """Keep SYSTEM STATUS capture fields aligned with StateManager snapshot."""
    patched = dict(sections or {})
    if not patched:
        patched = build_display_sections(scene={}, structured={}, scores={})
    system = dict(patched.get("system_status") or {})
    system["capture_active"] = {
        "value": "Yes" if capture_active else "No",
        "source": "capture_service",
    }
    system["overlay_stale"] = {
        "value": "Yes" if overlay_stale else "No",
        "source": "state_manager",
    }
    system["capture_exit_reason"] = {
        "value": capture_exit_reason or "none",
        "source": "capture_service",
    }
    patched["system_status"] = system
    return patched


def enrich_scores_with_capture(
    scores: Dict[str, Any],
    structured: Dict[str, Any],
    scene: Dict[str, Any],
    snap: Dict[str, Any],
    intent: str = "",
    *,
    decision_frame_id: Optional[int] = None,
    live_scene: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rebuild display_sections after capture snapshot is known."""
    return reconcile_llm_display(
        scores,
        structured=structured,
        decision_scene=scene,
        snap=snap,
        intent=intent,
        decision_frame_id=decision_frame_id,
        live_scene=live_scene,
    )


def reconcile_llm_display(
    scores: Dict[str, Any],
    *,
    structured: Dict[str, Any],
    decision_scene: Dict[str, Any],
    snap: Dict[str, Any],
    intent: str = "",
    decision_frame_id: Optional[int] = None,
    live_scene: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Align model output with live perception; apply LLM fields only when frames are in sync."""
    scores = dict(scores or {})
    structured = dict(structured or {})
    decision_scene = dict(decision_scene or {})
    if "llm_scene_payload" in decision_scene:
        decision_scene = {k: v for k, v in decision_scene.items() if k != "llm_scene_payload"}

    decision_fid = int(
        decision_frame_id
        or scores.get("llm_output_frame_id")
        or scores.get("frame_id")
        or 0
    )
    has_llm_decision = bool(
        structured.get("decision_source") in ("llama", "fallback")
        or str(scores.get("reasoning_mode") or "").startswith("llama")
        or scores.get("reasoning_mode") == "fallback"
    )
    scene_fid = int(
        scores.get("llm_scene_frame_id")
        or decision_scene.get("frame_id")
        or decision_fid
        or 0
    )
    if has_llm_decision and decision_fid:
        scene_fid = decision_fid
    scores["llm_output_frame_id"] = decision_fid
    scores["llm_scene_frame_id"] = scene_fid
    scores["frame_id"] = decision_fid

    scores["capture_active"] = snap.get("capture_active")
    scores["capture_frame_id"] = snap.get("capture_frame_id")
    scores["capture_exit_reason"] = snap.get("capture_exit_reason")
    scores["overlay_stale"] = snap.get("overlay_stale")
    scores["overlay_frame_lag"] = snap.get("overlay_frame_lag")

    capture_fid = int(snap.get("capture_frame_id") or 0)
    reasoning_stale, lag = evaluate_llm_frame_sync(
        scene_frame_id=scene_fid,
        decision_frame_id=decision_fid,
        capture_frame_id=capture_fid,
    )

    live = dict(live_scene or {})
    live_fid = int(live.get("frame_id") or 0)
    if live_fid and decision_fid and live_fid > decision_fid:
        preview_lag = live_fid - decision_fid
        if preview_lag > pc.LLM_LAG_STALE_FRAMES:
            reasoning_stale = True

    scores["reasoning_stale"] = reasoning_stale
    scores["llm_lag_frames"] = lag
    log_llm_sync(
        scene_frame_id=scene_fid,
        decision_frame_id=decision_fid,
        capture_frame_id=capture_fid,
        stale=reasoning_stale,
    )

    model_scene = live if live_fid >= decision_fid else decision_scene
    if not model_scene:
        model_scene = decision_scene

    llm_payload = scores.get("llm_scene_payload")
    payload_fid = int(llm_payload.get("frame_id") or 0) if isinstance(llm_payload, dict) else 0
    if not isinstance(llm_payload, dict) or payload_fid != scene_fid:
        if scene_fid and not reasoning_stale:
            llm_payload = build_llm_scene_payload(decision_scene, frame_id=scene_fid)
            scores["llm_scene_payload"] = llm_payload
        elif scene_fid:
            llm_payload = {"frame_id": scene_fid}

    if reasoning_stale:
        scores["reasoning_consistency_warning"] = False
        scores["warning_text"] = ""
    else:
        consistency_warning, warning_text = evaluate_reasoning_consistency(decision_scene, structured, scores)
        scores["reasoning_consistency_warning"] = consistency_warning
        scores["warning_text"] = warning_text

    sections = build_display_sections(
        scene=model_scene,
        structured=structured,
        scores=scores,
        intent=intent,
        decision_frame_id=decision_fid,
        llm_scene_payload=llm_payload if isinstance(llm_payload, dict) else None,
    )
    scores["display_sections"] = sections
    return scores


def apply_stale_llm_mask_to_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Hide flat API LLM fields when reasoning is stale."""
    out = dict(state)
    scores = dict(out.get("confidence_scores") or {})
    reasoning_stale = bool(out.get("reasoning_stale") or scores.get("reasoning_stale"))
    if not reasoning_stale:
        return out

    out["reasoning_stale"] = True
    scores["reasoning_stale"] = True
    scores["reasoning_consistency_warning"] = False
    scores["warning_text"] = ""
    out["confidence_scores"] = scores
    out["alert"] = False
    out["intent"] = LLM_UPDATING_MESSAGE
    out["safety_label"] = "UPDATING"
    out["patient_status"] = _STALE_LLM_PLACEHOLDER
    out["staff_presence"] = _STALE_LLM_PLACEHOLDER
    out["alert_type"] = _STALE_LLM_PLACEHOLDER
    out["reason"] = LLM_UPDATING_MESSAGE
    out["summary"] = LLM_UPDATING_MESSAGE
    out["reasoning_consistency_warning"] = False
    out["warning_text"] = ""
    return out

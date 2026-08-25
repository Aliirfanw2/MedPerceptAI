"""Late fusion of full-frame object, role, and pose detections."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from monitor import presentation_config as pc
from monitor.runtime.tensor_utils import coerce_landmarks
from monitor.runtime.types import ObjectResult, PoseResult, RoleResult

logger = logging.getLogger(__name__)

MEDICAL_STAFF_ROLES = {"nurse", "doctor", "staff"}
VISITOR_ROLES = {"relative", "relatives"}
STAFF_ROLES = MEDICAL_STAFF_ROLES | VISITOR_ROLES
PATIENT_ROLES = {"patient"}
BED_LABEL_HINTS = ("bed", "patient bed", "hospital bed")

# Temporal smoothing for fused primary role (mirrors pipeline stable role).
_primary_role_stable: Dict[str, Any] = {"role": "unknown", "conf": 0.0, "frame_id": 0}


def reset_primary_role_stable() -> None:
    """Clear stable role state on new capture session."""
    global _primary_role_stable
    _primary_role_stable = {"role": "unknown", "conf": 0.0, "frame_id": 0}


_pose_hold_state: Dict[str, Any] = {
    "frame_id": 0,
    "bbox": None,
    "keypoints": None,
    "pose_summary": None,
    "pose": "unknown",
    "pose_conf": 0.0,
}


def reset_pose_hold() -> None:
    """Clear pose hold state on new capture session."""
    global _pose_hold_state
    _pose_hold_state = {
        "frame_id": 0,
        "bbox": None,
        "keypoints": None,
        "pose_summary": None,
        "pose": "unknown",
        "pose_conf": 0.0,
    }


def _stabilize_primary_pose(primary: Dict[str, Any], frame_id: int) -> None:
    """Hold pose keypoints briefly when YOLO/fusion drops them for a few frames."""
    global _pose_hold_state
    bbox = primary.get("bbox") or {}
    box = _box_tuple(bbox) if bbox else (0, 0, 0, 0)
    keypoints = primary.get("keypoints")

    if keypoints is not None:
        _pose_hold_state = {
            "frame_id": int(frame_id),
            "bbox": dict(bbox),
            "keypoints": keypoints,
            "pose_summary": dict(primary.get("pose_summary") or {}),
            "pose": str(primary.get("pose") or "unknown"),
            "pose_conf": float(primary.get("pose_conf") or 0.0),
        }
        return

    hold = _pose_hold_state
    if not hold.get("keypoints"):
        return
    age = int(frame_id) - int(hold.get("frame_id") or 0)
    if age > 6:
        return
    hold_box = _box_tuple(hold.get("bbox") or (0, 0, 0, 0))
    if box_iou(box, hold_box) < 0.2 and center_distance(box, hold_box) > 180:
        return

    primary["keypoints"] = hold["keypoints"]
    primary["pose_summary"] = dict(hold.get("pose_summary") or {"available": True, "pose_detected": True})
    primary["pose"] = str(hold.get("pose") or primary.get("pose") or "unknown")
    primary["pose_conf"] = float(hold.get("pose_conf") or primary.get("pose_conf") or 0.0)
    primary["pose_source"] = "pose_hold"
    logger.debug(
        "[POSE][HOLD] frame_id=%s reusing keypoints from frame_id=%s age=%s",
        frame_id,
        hold.get("frame_id"),
        age,
    )


def _normalize_role_name(role: str) -> str:
    text = str(role or "unknown").strip().lower()
    if text in ("relatives", "relative"):
        return "relatives"
    return text


def _apply_role_uncertainty(primary: Dict[str, Any]) -> None:
    """Mark low-confidence roles uncertain for display and LLM payload."""
    role = _normalize_role_name(str(primary.get("role") or "unknown"))
    conf = float(primary.get("role_conf") or 0.0)
    uncertain = conf < pc.ROLE_MIN_DISPLAY_CONF and role not in ("unknown",)
    primary["role_uncertain"] = uncertain
    if uncertain:
        primary["role_for_llm"] = "unknown_low_confidence"
        primary["role_display"] = f"{role} (uncertain)"
    else:
        primary["role_for_llm"] = role
        primary["role_display"] = role


def normalize_object_label(label: str) -> str:
    text = (label or "").strip().lower().replace("_", " ")
    if not text:
        return "unknown"
    if any(hint in text for hint in BED_LABEL_HINTS) or text == "bed":
        if "patient" in text:
            return "patient bed"
        if "hospital" in text:
            return "hospital bed"
        return "bed"
    if "person" in text or "patient" in text or "human" in text:
        return "person"
    return text


def _box_tuple(box: Any) -> Tuple[int, int, int, int]:
    if isinstance(box, dict):
        return (
            int(box.get("x1", box.get("x", 0))),
            int(box.get("y1", box.get("y", 0))),
            int(box.get("x2", box.get("x1", 0))),
            int(box.get("y2", box.get("y1", 0))),
        )
    return tuple(int(v) for v in box[:4])


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    acx, acy = box_center(a)
    bcx, bcy = box_center(b)
    return math.hypot(acx - bcx, acy - bcy)


POSE_MATCH_MIN_IOU = 0.08
POSE_MATCH_MAX_DIST = 220.0
POSE_MATCH_MIN_KP_IN_BOX = 0.35


def _landmark_point(pt: Any) -> tuple[float, float]:
    try:
        x = float(pt[0].item() if hasattr(pt[0], "item") else pt[0])
        y = float(pt[1].item() if hasattr(pt[1], "item") else pt[1])
        return x, y
    except Exception:
        return 0.0, 0.0


def _keypoints_in_box_ratio(
    landmarks: Any,
    person_box: Tuple[int, int, int, int],
) -> float:
    if not landmarks:
        return 0.0
    x1, y1, x2, y2 = person_box
    visible = 0
    inside = 0
    for pt in landmarks:
        px, py = _landmark_point(pt)
        if px <= 1.0 and py <= 1.0:
            continue
        visible += 1
        if x1 <= px <= x2 and y1 <= py <= y2:
            inside += 1
    if visible == 0:
        return 0.0
    return inside / visible


def _match_pose_detection(
    person_box: Tuple[int, int, int, int],
    pose_dets: List[Dict[str, Any]],
    *,
    frame_id: int,
    person_id: int,
) -> tuple[Optional[Dict[str, Any]], str]:
    if not pose_dets:
        return None, "no_pose_detections"

    scored: List[tuple[float, float, float, float, Dict[str, Any]]] = []
    for item in pose_dets:
        pose_box = _box_tuple(item.get("box", (0, 0, 0, 0)))
        iou = box_iou(person_box, pose_box)
        dist = center_distance(person_box, pose_box)
        kp_ratio = _keypoints_in_box_ratio(item.get("landmarks"), person_box)
        conf = float(item.get("confidence") or 0.0)
        scored.append((iou, -dist, kp_ratio, conf, item))

    scored.sort(key=lambda row: (-row[0], row[1], -row[2], -row[3]))
    best_iou, _neg_dist, best_kp_ratio, best_conf, best = scored[0]
    best_dist = -_neg_dist
    pose_box = _box_tuple(best.get("box", (0, 0, 0, 0)))

    overlap_ok = best_iou >= POSE_MATCH_MIN_IOU or best_dist <= POSE_MATCH_MAX_DIST
    keypoints_ok = best_kp_ratio >= POSE_MATCH_MIN_KP_IN_BOX
    if overlap_ok and keypoints_ok:
        status = "accepted"
    elif best_iou >= 0.15 and best_kp_ratio >= 0.25:
        status = "accepted"
    else:
        status = "rejected"

    logger.info(
        "[POSE][MATCH] frame_id=%s person_id=%s person_bbox=%s pose_bbox=%s iou=%.2f "
        "center_distance=%.0f kp_in_box=%.2f conf=%.3f %s",
        frame_id,
        person_id,
        person_box,
        pose_box,
        best_iou,
        best_dist,
        best_kp_ratio,
        best_conf,
        status,
    )

    if status == "rejected":
        return None, (
            f"rejected iou={best_iou:.2f} dist={best_dist:.0f} "
            f"kp_in_box={best_kp_ratio:.2f} (need iou>={POSE_MATCH_MIN_IOU} "
            f"or dist<={POSE_MATCH_MAX_DIST:.0f} and kp>={POSE_MATCH_MIN_KP_IN_BOX})"
        )

    return best, (
        f"matched iou={best_iou:.2f} dist={best_dist:.0f} kp_in_box={best_kp_ratio:.2f} conf={best_conf:.3f}"
    )


def _match_role_detection(
    target_box: Tuple[int, int, int, int],
    role_dets: List[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], str]:
    if not role_dets:
        return None, "no_role_detections"

    scored: List[tuple[float, float, float, Dict[str, Any]]] = []
    for item in role_dets:
        box = _box_tuple(item.get("box", (0, 0, 0, 0)))
        iou = box_iou(target_box, box)
        dist = center_distance(target_box, box)
        conf = float(item.get("confidence") or 0.0)
        scored.append((iou, dist, conf, item))

    scored.sort(key=lambda row: (-row[0], row[1], -row[2]))
    best_iou, best_dist, best_conf, best = scored[0]
    if best_iou >= 0.10 or best_dist < 200:
        return best, (
            f"matched role={best.get('role')} conf={best_conf:.3f} iou={best_iou:.2f} dist={best_dist:.0f}"
        )
    return None, f"no_overlap best_iou={best_iou:.2f} best_dist={best_dist:.0f} best_role={best.get('role')}"


def _is_bed_label(label: str) -> bool:
    text = normalize_object_label(label)
    return text in ("bed", "patient bed", "hospital bed")


def _is_person_object(label: str) -> bool:
    text = (label or "").lower()
    if _is_bed_label(text):
        return False
    return any(k in text for k in ("person", "patient", "human")) or text in ("", "unknown")


def _bed_relation(person_box: Tuple[int, int, int, int], bed_box: Optional[Tuple[int, int, int, int]]) -> str:
    if bed_box is None:
        return "unknown"
    iou = box_iou(person_box, bed_box)
    dist = center_distance(person_box, bed_box)
    if iou >= 0.12:
        return "on_bed"
    if dist <= 220:
        return "near_bed"
    if dist >= 360 and iou < 0.03:
        return "away_from_bed"
    if dist >= 280 and iou < 0.05:
        return "away_from_bed"
    return "near_bed"


def _patient_status_from_pose(pose_summary: Dict[str, Any]) -> str:
    pose_class = str(pose_summary.get("pose_class") or "").lower()
    if pose_class in ("standing", "lying", "sitting", "fall_risk"):
        return pose_class
    if pose_summary.get("fall_hint"):
        return "fall_risk"
    if pose_summary.get("standing_hint"):
        return "standing"
    return "unknown"


def _stabilize_primary_role(
    primary: Dict[str, Any],
    role_result: RoleResult,
    missing_signals: List[str],
    frame_id: int,
) -> None:
    """Apply frame-level stable role to primary_person; low-confidence → unknown."""
    global _primary_role_stable

    stable_frame = int(_primary_role_stable.get("frame_id") or 0)
    if frame_id - stable_frame > pc.ROLE_STABLE_TTL_FRAMES:
        _primary_role_stable = {"role": "unknown", "conf": 0.0, "frame_id": frame_id}

    if "role" in missing_signals:
        primary["role"] = "unknown"
        primary["role_conf"] = 0.0
        primary["role_source"] = "default"
        _apply_role_uncertainty(primary)
        return

    matched_role = _normalize_role_name(str(primary.get("role") or "unknown"))
    matched_conf = float(primary.get("role_conf") or 0.0)
    stable_hint = str(role_result.role_hint or "unknown").lower()
    stable_conf = float(role_result.role_confidence or 0.0)

    if matched_role == "unknown" and stable_hint != "unknown":
        accept_conf = pc.role_confidence_threshold(stable_hint)
        if stable_conf >= accept_conf:
            primary["role"] = stable_hint
            primary["role_conf"] = round(stable_conf, 3)
            primary["role_source"] = "yolo_role_stable"
            if stable_conf >= pc.ROLE_CONFIDENCE_SWITCH or _primary_role_stable["role"] == "unknown":
                _primary_role_stable = {"role": stable_hint, "conf": stable_conf, "frame_id": frame_id}
            _apply_role_uncertainty(primary)
            return

    if matched_role != "unknown":
        min_conf = pc.role_confidence_threshold(matched_role)
        if matched_conf < min_conf:
            if _primary_role_stable["role"] != "unknown":
                primary["role"] = _primary_role_stable["role"]
                primary["role_conf"] = _primary_role_stable["conf"]
                primary["role_source"] = "yolo_role_stable"
            else:
                primary["role"] = "unknown"
                primary["role_source"] = "default"
            _apply_role_uncertainty(primary)
            return

        prev_role = _normalize_role_name(str(_primary_role_stable.get("role") or "unknown"))
        if (
            prev_role != "unknown"
            and matched_role != prev_role
            and matched_conf < pc.ROLE_CONFIDENCE_SWITCH + 0.12
        ):
            primary["role"] = prev_role
            primary["role_conf"] = _primary_role_stable["conf"]
            primary["role_source"] = "yolo_role_stable"
            logger.info(
                "[ROLE][FUSION] primary switch blocked %s→%s conf=%.3f keeping stable=%s",
                prev_role,
                matched_role,
                matched_conf,
                prev_role,
            )
            _apply_role_uncertainty(primary)
            return

        if matched_conf >= pc.ROLE_CONFIDENCE_SWITCH or prev_role == "unknown":
            _primary_role_stable = {"role": matched_role, "conf": matched_conf, "frame_id": frame_id}
        primary["role_source"] = "fusion"
        _apply_role_uncertainty(primary)
        return

    if _primary_role_stable["role"] != "unknown":
        primary["role"] = _primary_role_stable["role"]
        primary["role_conf"] = _primary_role_stable["conf"]
        primary["role_source"] = "yolo_role_stable"
    else:
        primary["role_source"] = "default"
    _apply_role_uncertainty(primary)


def _select_primary_person(persons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not persons:
        return None
    patients = [p for p in persons if p.get("role") in PATIENT_ROLES]
    if patients:
        return max(patients, key=lambda p: float(p.get("object_conf") or 0.0))
    unknowns = [p for p in persons if p.get("role") == "unknown"]
    if unknowns:
        return max(unknowns, key=lambda p: float(p.get("object_conf") or 0.0))
    non_staff = [p for p in persons if str(p.get("role", "")).lower() not in STAFF_ROLES]
    if non_staff:
        return max(non_staff, key=lambda p: float(p.get("object_conf") or 0.0))
    return max(persons, key=lambda p: float(p.get("object_conf") or 0.0))


def fuse_scene(
    *,
    frame_id: int,
    object_result: ObjectResult,
    role_result: RoleResult,
    pose_result: PoseResult,
    missing_signals: List[str],
) -> Dict[str, Any]:
    objects = []
    for raw in object_result.detections or []:
        item = dict(raw)
        item["label"] = normalize_object_label(str(raw.get("label", "")))
        objects.append(item)
    role_dets = list(role_result.role_detections or [])
    pose_dets = list(pose_result.pose_detections or [])

    beds = [o for o in objects if _is_bed_label(str(o.get("label", "")))]
    if not beds:
        logger.info("[BED][FUSION] frame_id=%s no bed detected in objects=%s", frame_id, len(objects))
    else:
        logger.info(
            "[BED][FUSION] frame_id=%s beds=%s",
            frame_id,
            [(b.get("label"), round(float(b.get("confidence") or 0.0), 3), b.get("box")) for b in beds],
        )
    person_objects = [o for o in objects if _is_person_object(str(o.get("label", "person")))]
    if not person_objects and objects:
        person_objects = [o for o in objects if not _is_bed_label(str(o.get("label", "")))]

    persons: List[Dict[str, Any]] = []
    for idx, obj in enumerate(person_objects[:5]):
        box = _box_tuple(obj.get("box", (0, 0, 0, 0)))
        role_match, role_match_reason = _match_role_detection(box, role_dets)
        pose_match, pose_match_reason = _match_pose_detection(
            box,
            pose_dets,
            frame_id=frame_id,
            person_id=idx,
        )

        raw_role = str((role_match or {}).get("role") or "unknown").lower()
        role_conf = float((role_match or {}).get("confidence") or 0.0)
        known_roles = PATIENT_ROLES | STAFF_ROLES
        min_conf = pc.role_confidence_threshold(raw_role)
        role_source = "default"
        if role_match is None:
            role = "unknown"
            role_reason = role_match_reason
        elif raw_role not in known_roles:
            role = "unknown"
            role_reason = (
                f"unrecognized_class={raw_role} conf={role_conf:.3f} ({role_match_reason})"
            )
        elif role_conf < min_conf:
            role = "unknown"
            role_reason = (
                f"below_threshold role={raw_role} conf={role_conf:.3f} min={min_conf:.3f} "
                f"({role_match_reason})"
            )
        else:
            role = raw_role
            role_reason = role_match_reason
            role_source = "fusion"

        if role == "unknown":
            role_source = "default"
            logger.info(
                "[ROLE][FUSION] frame_id=%s person_id=%s role_unknown reason=%s object_box=%s",
                frame_id,
                idx,
                role_reason,
                box,
            )
        else:
            logger.info(
                "[ROLE][FUSION] frame_id=%s person_id=%s %s",
                frame_id,
                idx,
                role_reason,
            )
        pose_source = "default"
        if pose_match is None:
            pose_summary = {"available": False, "pose_detected": False}
            pose_conf = 0.0
            fused_pose = "unknown"
            fused_keypoints = None
        else:
            pose_summary = dict(pose_match.get("pose_summary") or {"available": False})
            landmarks = pose_match.get("landmarks")
            kp_ratio = _keypoints_in_box_ratio(landmarks, box)
            pose_summary["keypoints_inside_ratio"] = round(kp_ratio, 3)
            if kp_ratio < 0.18:
                pose_summary = {"available": False, "pose_detected": False, "pose_class": "unknown"}
                fused_pose = "unknown"
                fused_keypoints = None
                pose_conf = 0.0
                pose_source = "default"
            else:
                pose_conf = float(pose_match.get("confidence") or pose_summary.get("pose_score") or 0.0)
                fused_pose = _patient_status_from_pose(pose_summary)
                fused_keypoints = coerce_landmarks(landmarks)
                pose_source = "yolo_pose+classifier"

        nearest_bed_item = None
        nearest_bed_dist = float("inf")
        for bed in beds:
            bed_box = _box_tuple(bed.get("box", (0, 0, 0, 0)))
            dist = center_distance(box, bed_box)
            if dist < nearest_bed_dist:
                nearest_bed_dist = dist
                nearest_bed_item = bed

        nearest_bed_box = _box_tuple(nearest_bed_item["box"]) if nearest_bed_item else None
        relation_to_bed = _bed_relation(box, nearest_bed_box)
        if nearest_bed_item:
            logger.info(
                "[BED][FUSION] frame_id=%s person_id=%s relation=%s bed_label=%s dist=%.0f",
                frame_id,
                idx,
                relation_to_bed,
                nearest_bed_item.get("label"),
                nearest_bed_dist if nearest_bed_dist != float("inf") else -1,
            )
        staff_roles = [r for r in role_dets if _normalize_role_name(str(r.get("role", ""))) in MEDICAL_STAFF_ROLES]
        staff_near = False
        if staff_roles:
            for staff in staff_roles:
                staff_box = _box_tuple(staff.get("box", (0, 0, 0, 0)))
                if center_distance(box, staff_box) < 220 or box_iou(box, staff_box) > 0.05:
                    staff_near = True
                    break

        persons.append(
            {
                "id": idx,
                "bbox": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                "role": role,
                "role_conf": round(role_conf, 3),
                "role_source": role_source if role_match is not None else "default",
                "pose": fused_pose,
                "pose_conf": round(pose_conf, 3),
                "pose_source": pose_source,
                "keypoints": fused_keypoints,
                "pose_summary": pose_summary,
                "nearest_bed": nearest_bed_item,
                "relation_to_bed": relation_to_bed,
                "staff_near": staff_near,
                "object_conf": float(obj.get("confidence") or 0.0),
                "label": str(obj.get("label") or "person"),
            }
        )

    primary_person = _select_primary_person(persons)
    if primary_person is not None:
        _stabilize_primary_role(primary_person, role_result, missing_signals, frame_id)
        _stabilize_primary_pose(primary_person, frame_id)
    if primary_person and str(primary_person.get("role", "")).lower() in STAFF_ROLES:
        logger.info(
            "[ROLE][FUSION] frame_id=%s primary is staff role=%s — not used for patient fall risk",
            frame_id,
            primary_person.get("role"),
        )

    staff_role_dets = [
        r for r in role_dets if _normalize_role_name(str(r.get("role", ""))) in MEDICAL_STAFF_ROLES
    ]
    visitor_role_dets = [
        r for r in role_dets if _normalize_role_name(str(r.get("role", ""))) in VISITOR_ROLES
    ]
    if "role" in missing_signals:
        staff_presence = "unknown_staff"
        staff_presence_source = "not_detected"
    elif staff_role_dets:
        if primary_person and primary_person.get("staff_near"):
            staff_presence = "staff_near"
            staff_presence_source = "role_model+proximity"
        else:
            staff_presence = "staff_detected_not_near"
            staff_presence_source = "role_model+proximity"
    else:
        staff_presence = "staff_not_detected"
        staff_presence_source = "not_detected"

    if primary_person is not None:
        primary_person["staff_near"] = staff_presence == "staff_near"

    patient_status = "unknown"
    bed_relation = "unknown"
    if primary_person:
        patient_status = str(primary_person.get("pose") or "unknown")
        bed_relation = str(primary_person.get("relation_to_bed") or "unknown")

    confidence_scores = {
        "object": float(object_result.confidence or 0.0),
        "role": float(role_result.role_confidence or 0.0),
        "pose": float((pose_result.pose_summary or {}).get("pose_score") or 0.0),
    }
    if primary_person:
        confidence_scores["object"] = float(primary_person.get("object_conf") or confidence_scores["object"])
        confidence_scores["role"] = float(primary_person.get("role_conf") or confidence_scores["role"])
        confidence_scores["pose"] = float(primary_person.get("pose_conf") or confidence_scores["pose"])

    merged_missing = list(missing_signals)
    if not beds and "bed" not in merged_missing:
        merged_missing.append("bed")

    bed_relation_source = (
        "default_unknown" if not beds else "yolo_object+geometry"
    )
    display_sources = {
        "person_detected": "yolo_object" if primary_person else "default",
        "role": str((primary_person or {}).get("role_source") or "default"),
        "pose": str((primary_person or {}).get("pose_source") or "default"),
        "objects": "yolo_object",
        "bed_relation": bed_relation_source,
        "staff_presence": staff_presence_source,
    }

    return {
        "frame_id": frame_id,
        "objects": objects,
        "persons": persons,
        "roles": role_dets,
        "poses": [
            {
                "box": p.get("box"),
                "confidence": p.get("confidence"),
                "pose_summary": p.get("pose_summary"),
            }
            for p in pose_dets
        ],
        "relations": {
            "primary_person_id": primary_person.get("id") if primary_person else None,
            "bed_count": len(beds),
            "person_count": len(persons),
        },
        "staff_presence": staff_presence,
        "patient_status": patient_status,
        "bed_relation": bed_relation,
        "missing_signals": merged_missing,
        "confidence_scores": confidence_scores,
        "primary_person": primary_person,
        "display_sources": display_sources,
        "staff_role_count": len(staff_role_dets),
        "visitor_role_count": len(visitor_role_dets),
        "medical_staff_roles": [r.get("role") for r in staff_role_dets],
        "visitor_roles": [r.get("role") for r in visitor_role_dets],
    }

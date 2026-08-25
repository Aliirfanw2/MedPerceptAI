from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import DatabaseError
from django.http import JsonResponse
from django.http import StreamingHttpResponse

from monitor.event_store import (
    count_active_alerts,
    fetch_patient_history_rows,
    fetch_recent_alerts,
    fetch_recent_events,
    format_relative_time,
    persist_monitoring_event,
)
from monitor.ml_pipeline import DEFAULT_FALLBACK_VIDEO, PatientIntentPipeline
from monitor.models import Camera
from monitor.runtime.orchestrator import StreamOrchestrator
from monitor.runtime.state_manager import StateManager
from monitor.runtime.types import ReasoningResult
from monitor.presentation_config import (
    CAPTURE_MAX_FPS,
    FRAME_SLEEP_SECONDS,
    INFERENCE_EVERY_N_FRAMES,
    INFERENCE_EVERY_N_FRAMES_VIDEO,
    LLM_LAG_STALE_FRAMES,
    RUNTIME_QUEUE_SIZE,
    reasoning_enabled,
)

logger = logging.getLogger(__name__)

_PIPELINE: Optional[PatientIntentPipeline] = None
_PIPELINE_INIT_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
latest_inference_state: Dict[str, Any] = {
    "intent": "Waiting for live stream",
    "alert": False,
    "bbox": None,
    "frame_id": None,
    "capture_frame_id": None,
    "overlay_frame_id": None,
    "overlay_stale": True,
    "capture_active": False,
    "capture_exit_reason": None,
    "updated_at": None,
    "source": None,
    "status": "idle",
    "error": None,
    "building": None,
    "floor": None,
    "room_number": None,
    "camera_id": None,
    "recipient_count": 0,
    "latency_ms": None,
    "frame_width": None,
    "frame_height": None,
    "stream_fps": 0,
    "confidence_scores": {},
    "monitor_status": "idle",
    "role_hint": "unknown",
    "risk_score": None,
    "risk_level": "not provided",
    "reasoning_trace": [],
    "reasoning_history": [],
    "safety_label": "not provided",
    "alert_type": "not provided",
    "reason": None,
    "summary": None,
    "patient_status": "not provided",
    "staff_presence": "not provided",
    "decision_source": "fallback",
    "display_sections": {},
}

MJPEG_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
MJPEG_SUFFIX = b"\r\n"
_RUNTIME_STATE = StateManager()
_ORCHESTRATOR: Optional[StreamOrchestrator] = None
_ORCHESTRATOR_LOCK = threading.RLock()
_CAPTURE_CONFIG_SIG: Optional[str] = None
_STREAM_CLIENTS = 0
_STREAM_CLIENTS_LOCK = threading.Lock()
_REASONING_HISTORY: Deque[Dict[str, Any]] = deque(maxlen=40)
_REASONING_HISTORY_LOCK = threading.Lock()



def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return fallback



def _normalize_source(source: Any) -> str:
    normalized = str(source or "camera").strip().lower()
    if normalized in {"video", "file", "pre-recorded", "pre recorded"}:
        return "video"
    return "camera"


def _lookup_camera_record(camera_id: str):
    try:
        return Camera.objects.filter(camera_identifier=camera_id, is_active=True).first()
    except DatabaseError as exc:
        logger.warning("Camera lookup skipped (database unavailable): %s", exc)
        return None
    except Exception as exc:
        logger.warning("Camera lookup failed: %s", exc)
        return None


def _get_pipeline() -> PatientIntentPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_INIT_LOCK:
            if _PIPELINE is None:
                logger.info("Initializing PatientIntentPipeline (lazy load).")
                _PIPELINE = PatientIntentPipeline()
    return _PIPELINE


def _ai_reasoning_enabled(session) -> bool:
    """Rule-based by default; Llama only when ENABLE_REASONING=1 (session may opt out)."""
    if not reasoning_enabled():
        return False
    if session is not None and "enable_ai_reasoning" in session:
        return bool(session.get("enable_ai_reasoning"))
    return True


def _safe_process_frame(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy single-threaded path (tests/admin); runtime uses parallel workers instead."""
    use_reasoning = bool(config.get("enable_ai_reasoning", config.get("use_reasoning", False)))
    try:
        return _get_pipeline().process_frame(
            frame,
            stream_config=config,
            use_reasoning=use_reasoning,
        )
    except Exception as exc:
        logger.exception("Frame processing failed; returning passthrough frame: %s", exc)
        output = frame.copy()
        cv2.putText(
            output,
            "Inference recovering...",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return {
            "annotated_frame": output,
            "bbox": None,
            "intent": "Pipeline recovery mode",
            "alert_triggered": False,
            "pose_summary": {"available": False},
        }


def _session_or_env(session, session_key: str, env_key: str, default: Any) -> Any:
    if session_key in session and session.get(session_key) not in (None, ""):
        return session.get(session_key)
    env_value = os.environ.get(env_key)
    if env_value is not None and str(env_value).strip() != "":
        return env_value
    return default


def _resolve_path_candidate(candidate: Any) -> Path:
    path = Path(str(candidate).strip())
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path


def _resolve_video_path(session) -> Path:
    session_path = session.get("monitor_video_path") if session else None
    candidates = [
        session_path,
        os.environ.get("MONITOR_VIDEO_PATH"),
        str(Path(__file__).resolve().parent.parent / "media" / "demo.mp4"),
        str(Path(__file__).resolve().parent.parent / "media" / "test_video.mp4"),
        str(DEFAULT_FALLBACK_VIDEO),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = _resolve_path_candidate(candidate)
        if path.exists():
            return path
    return _resolve_path_candidate(candidates[1] or candidates[2])


def _monitor_runtime_config(session) -> Dict[str, Any]:
    video_path = _resolve_video_path(session)
    if session and session.get("monitor_source"):
        source = _normalize_source(session.get("monitor_source"))
    else:
        source = _normalize_source(os.environ.get("MONITOR_INPUT_SOURCE", "camera"))
    if source == "video" and not video_path.exists():
        logger.warning("Pre-recorded video configured but file missing: %s", video_path)
    camera_index = _safe_int(_session_or_env(session, "monitor_camera_index", "MONITOR_CAMERA_INDEX", "0"))
    floor_number = str(_session_or_env(session, "monitor_floor_number", "MONITOR_FLOOR_NUMBER", "3")).strip() or "3"
    room_number = str(_session_or_env(session, "monitor_room_number", "MONITOR_ROOM_NUMBER", "302")).strip() or "302"
    camera_id = str(_session_or_env(session, "monitor_camera_id", "MONITOR_CAMERA_ID", "Cam-1")).strip() or "Cam-1"
    building = str(_session_or_env(session, "monitor_building", "MONITOR_BUILDING", "Main Building")).strip() or "Main Building"

    camera_record = _lookup_camera_record(camera_id)
    if camera_record is not None:
        building = camera_record.building
        floor_number = camera_record.floor
        room_number = camera_record.room_number or room_number

    source_label = "Pre-recorded Video" if source == "video" else "Live Camera Feed"
    input_detail = str(video_path.name) if source == "video" else f"Device #{camera_index}"

    enable_ai_reasoning = _ai_reasoning_enabled(session)

    return {
        "source": source,
        "camera_index": camera_index,
        "video_path": video_path,
        "floor_number": floor_number,
        "room_number": room_number,
        "camera_id": camera_id,
        "building": building,
        "monitoring_display": f"Monitoring: Floor {floor_number}, Room {room_number}, {camera_id}",
        "source_label": source_label,
        "input_detail": input_detail,
        "enable_ai_reasoning": enable_ai_reasoning,
        "use_reasoning": enable_ai_reasoning,
    }


def _location_recipients(building: str, floor: str) -> list[str]:
    if not building or not floor:
        return []

    try:
        matching_users = User.objects.filter(
            is_active=True,
            staff_profile__assigned_building=building,
            staff_profile__assigned_floor=floor,
        ).distinct()
        return [user.get_username() for user in matching_users]
    except DatabaseError as exc:
        logger.warning("Recipient lookup skipped (database unavailable): %s", exc)
        return []
    except Exception as exc:
        logger.warning("Recipient lookup failed: %s", exc)
        return []


def _get_staff_profile(user: User):
    try:
        return user.staff_profile
    except Exception:
        return None


def _open_video_capture(video_path: Path) -> Tuple[Optional[cv2.VideoCapture], str]:
    if not video_path.exists():
        logger.warning("Configured video source is unavailable: %s", video_path)
        return None, "unavailable"

    resolved = str(video_path.resolve())
    attempts: List[Optional[int]] = [None, cv2.CAP_FFMPEG]
    if os.name == "nt":
        attempts.append(cv2.CAP_MSMF)

    for backend in attempts:
        try:
            capture = cv2.VideoCapture(resolved) if backend is None else cv2.VideoCapture(resolved, backend)
            if capture.isOpened():
                fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
                if fps <= 1.0:
                    fps = 24.0
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("Video opened (%s backend, %.1f fps): %s", backend, fps, video_path)
                return capture, f"video:{video_path}"
            _safe_release_capture(capture)
        except Exception as exc:
            logger.warning("Video open failed (backend=%s): %s", backend, exc)

    logger.warning("Video file could not be opened: %s", video_path)
    return None, "unavailable"


def _open_camera_capture(camera_index: int) -> Tuple[Optional[cv2.VideoCapture], str]:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        try:
            capture = cv2.VideoCapture(camera_index, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("Camera %s opened with backend %s", camera_index, backend)
                return capture, f"camera:{camera_index}"
            _safe_release_capture(capture)
        except cv2.error as exc:
            logger.warning("Camera backend %s failed for index %s: %s", backend, camera_index, exc)
        except Exception as exc:
            logger.warning("Camera open failed for index %s (backend %s): %s", camera_index, backend, exc)

    logger.warning("Camera index %s could not be opened.", camera_index)
    return None, "unavailable"


def _open_stream_source(config: Dict[str, Any]) -> Tuple[Optional[cv2.VideoCapture], str]:
    source = config["source"]
    camera_index = config["camera_index"]
    video_path = Path(config["video_path"])

    if source == "video":
        capture, source_name = _open_video_capture(video_path)
        if capture is not None:
            logger.info("Stream opened from pre-recorded video: %s", video_path)
            return capture, source_name
        logger.warning("Pre-recorded video could not be opened: %s", video_path)
        return None, "unavailable"

    capture, source_name = _open_camera_capture(camera_index)
    if capture is not None:
        return capture, source_name

    if video_path.exists():
        logger.warning("Camera unavailable; falling back to video file: %s", video_path)
        return _open_video_capture(video_path)

    return None, "unavailable"


def _build_status_frame(message: str, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (15, 23, 42)
    cv2.putText(frame, message, (48, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (59, 130, 246), 2, cv2.LINE_AA)
    if config:
        _apply_cctv_overlay(frame, config, offline=True)
    return frame


def _apply_cctv_overlay(frame: np.ndarray, config: Dict[str, Any], offline: bool = False) -> np.ndarray:
    """Light stream chrome: LIVE badge + detection timestamp (no large header blocks)."""
    _ = config
    height, width = frame.shape[:2]
    status = "OFFLINE" if offline else "LIVE"
    badge_color = (0, 0, 200) if offline else (0, 140, 0)
    x2 = width - 8
    x1 = max(8, x2 - 64)
    cv2.rectangle(frame, (x1, 8), (x2, 26), badge_color, -1)
    cv2.putText(
        frame,
        status,
        (x1 + 6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        timestamp,
        (8, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 220, 230),
        1,
        cv2.LINE_AA,
    )
    return frame


def _encode_jpeg_frame(frame: np.ndarray) -> Optional[bytes]:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return buffer.tobytes()


def _yield_mjpeg_chunk(jpeg_bytes: bytes, frame_kind: str = "frame") -> bytes:
    if frame_kind in {"live", "offline-status", "camera-offline"}:
        logger.debug("Yielding %s (%s bytes)", frame_kind, len(jpeg_bytes))
    return MJPEG_BOUNDARY + jpeg_bytes + MJPEG_SUFFIX


def _update_latest_inference_state(**updates: Any) -> None:
    with STATE_LOCK:
        latest_inference_state.update(updates)


def _snapshot_latest_inference_state() -> Dict[str, Any]:
    with STATE_LOCK:
        return dict(latest_inference_state)


def _structured_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    scores = state.get("confidence_scores") or {}
    return dict(scores.get("structured_reasoning") or {})


def _apply_llm_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    """Expose flat LLM decision fields on API state (no heuristic defaults)."""
    from monitor.runtime.display_fields import apply_stale_llm_mask_to_state

    out = apply_stale_llm_mask_to_state(dict(state))
    scores = out.get("confidence_scores") or {}
    reasoning_stale = bool(out.get("reasoning_stale") or scores.get("reasoning_stale"))
    stale = bool(out.get("overlay_stale")) or not bool(out.get("capture_active")) or reasoning_stale
    if stale:
        out["alert"] = False
        return out

    structured = _structured_from_state(state)
    for key in (
        "patient_status",
        "staff_presence",
        "alert_type",
        "reason",
        "summary",
        "safety_label",
        "risk_level",
        "decision_source",
    ):
        val = structured.get(key) if key in structured else out.get(key)
        if val is not None and (not isinstance(val, str) or val.strip()):
            out[key] = val
    risk_score = structured.get("risk_score")
    if risk_score is not None:
        try:
            out["risk_score"] = float(risk_score)
        except (TypeError, ValueError):
            out["risk_score"] = None
    elif out.get("risk_score") is None:
        out["risk_score"] = None
    label = str(out.get("safety_label") or "not provided").upper()
    out["alert"] = label == "ALERT"
    return out


def _apply_capture_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply authoritative capture/stale fields from StateManager (never stale exit_reason when live)."""
    from monitor.runtime.display_fields import patch_display_sections_capture, reconcile_llm_display

    out = dict(state)
    overlay_fid = int(out.get("overlay_frame_id") or out.get("frame_id") or 0)
    snap = _RUNTIME_STATE.live_snapshot(overlay_fid)

    out["capture_active"] = snap["capture_active"]
    out["capture_frame_id"] = snap["capture_frame_id"] or out.get("capture_frame_id")
    out["overlay_stale"] = snap["overlay_stale"]
    out["capture_exit_reason"] = snap["capture_exit_reason"]

    scores = dict(out.get("confidence_scores") or {})
    structured = dict(scores.get("structured_reasoning") or {})
    decision_scene = dict(scores.get("scene") or {})
    decision_fid_for_scene = int(out.get("llm_output_frame_id") or out.get("frame_id") or 0)
    live_scene = _RUNTIME_STATE.get_perception_scene(decision_fid_for_scene)
    scores = reconcile_llm_display(
        scores,
        structured=structured,
        decision_scene=decision_scene,
        snap=snap,
        intent=str(out.get("intent") or ""),
        decision_frame_id=int(out.get("llm_output_frame_id") or out.get("frame_id") or 0),
        live_scene=live_scene,
    )
    out["reasoning_stale"] = bool(scores.get("reasoning_stale"))
    out["llm_output_frame_id"] = scores.get("llm_output_frame_id")
    out["llm_scene_frame_id"] = scores.get("llm_scene_frame_id")
    out["llm_lag_frames"] = scores.get("llm_lag_frames")
    out["reasoning_consistency_warning"] = bool(scores.get("reasoning_consistency_warning"))
    out["warning_text"] = scores.get("warning_text") or ""

    display_sections = dict(scores.get("display_sections") or out.get("display_sections") or {})
    display_sections = patch_display_sections_capture(
        display_sections,
        capture_active=bool(snap["capture_active"]),
        overlay_stale=bool(snap["overlay_stale"]),
        capture_exit_reason=snap["capture_exit_reason"],
    )
    out["display_sections"] = display_sections
    scores["display_sections"] = display_sections
    out["confidence_scores"] = scores

    if snap["overlay_stale"]:
        reason = snap["capture_exit_reason"] or "no live stream"
        out["alert"] = False
        out["safety_label"] = "STALE"
        if reason and "stop requested" in str(reason).lower():
            out["intent"] = "No live stream — open the camera feed to resume monitoring"
        else:
            out["intent"] = f"Capture stopped ({reason})"
        out["status"] = "stale"
        scores["safety_label"] = "STALE"
    elif snap["capture_active"]:
        if out.get("status") in (None, "idle", "stale", "streaming", "ok"):
            out["status"] = "live"
        stale_intent = str(out.get("intent") or "")
        if not out.get("reasoning_stale") and (
            stale_intent.startswith("Capture stopped") or stale_intent.startswith("No live stream")
        ):
            structured = dict(scores.get("structured_reasoning") or {})
            out["intent"] = (
                structured.get("summary")
                or structured.get("reason")
                or "Live stream active — awaiting LLM decision"
            )
            if structured.get("safety_label"):
                out["safety_label"] = structured.get("safety_label")

    return _apply_llm_fields(out)


def _apply_capture_freshness(state: Dict[str, Any]) -> Dict[str, Any]:
    """Reconcile API state with live capture status so stale alerts are not shown as live."""
    return _apply_capture_snapshot(state)


def _on_capture_started(source_name: str, config: Dict[str, Any]) -> None:
    """Clear previous-session stale fields when a new capture session begins."""
    from monitor.runtime.fusion import reset_pose_hold, reset_primary_role_stable

    reset_primary_role_stable()
    reset_pose_hold()
    try:
        _get_pipeline().reset_capture_state()
    except Exception:
        pass
    logger.info("[runtime] Capture session started source=%s", source_name)
    _update_latest_inference_state(
        capture_active=True,
        overlay_stale=False,
        capture_exit_reason=None,
        status="live",
        source=source_name,
        intent="Live stream active",
        alert=False,
        safety_label="not provided",
        decision_source="fallback",
        building=config.get("building"),
        floor=config.get("floor_number"),
        room_number=config.get("room_number"),
        camera_id=config.get("camera_id"),
        connected_cameras=1,
        error=None,
        display_sections={},
        confidence_scores={},
        updated_at=time.time(),
    )


def _on_capture_ended(reason: str) -> None:
    logger.warning("[runtime] Capture ended — marking API state stale reason=%s", reason)
    _update_latest_inference_state(
        capture_active=False,
        overlay_stale=True,
        capture_exit_reason=reason,
        alert=False,
        safety_label="STALE",
        intent=f"Capture stopped — stale frame ({reason})",
        status="stale",
    )


def _append_reasoning_history(result: ReasoningResult, *, source: str, camera_id: str) -> None:
    scores = result.confidence_scores or {}
    entry = {
        "frame_id": result.frame_id,
        "timestamp": time.time(),
        "intent": result.intent,
        "risk_score": result.risk_score if result.risk_score is not None else scores.get("risk_score"),
        "risk_level": result.risk_level or scores.get("risk_level") or "not provided",
        "alert": bool(result.alert_triggered),
        "reasoning_mode": scores.get("reasoning_mode"),
        "object_pct": scores.get("object_pct"),
        "role_pct": scores.get("role_pct"),
        "pose_pct": scores.get("pose_pct"),
        "reasoning_pct": scores.get("reasoning_pct"),
        "reasoning_trace": list(result.reasoning_trace or scores.get("reasoning_trace") or []),
        "role_hint": result.role_hint,
        "source": source,
        "camera_id": camera_id,
        "latency_ms": result.latency_ms,
    }
    with _REASONING_HISTORY_LOCK:
        _REASONING_HISTORY.appendleft(entry)


def _snapshot_reasoning_history(limit: int = 12) -> List[Dict[str, Any]]:
    with _REASONING_HISTORY_LOCK:
        return list(_REASONING_HISTORY)[:limit]


def _build_live_log_feed(
    db_logs: List[Dict[str, Any]],
    memory_logs: List[Dict[str, Any]],
    limit: int = 35,
) -> List[Dict[str, Any]]:
    """Merge in-memory reasoning steps with DB events for the live terminal (newest first)."""
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for entry in memory_logs:
        key = f"mem:{entry.get('frame_id')}:{entry.get('timestamp')}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "id": key,
                "ts": entry.get("timestamp"),
                "event": "reasoning",
                "intent": entry.get("intent"),
                "alert": entry.get("alert"),
                "camera_id": entry.get("camera_id"),
                "source": entry.get("source"),
                "frame_id": entry.get("frame_id"),
                "latency_ms": entry.get("latency_ms"),
                "risk_score": entry.get("risk_score"),
                "risk_level": entry.get("risk_level"),
                "reasoning_mode": entry.get("reasoning_mode"),
                "role_hint": entry.get("role_hint"),
            }
        )

    for entry in db_logs:
        key = f"db:{entry.get('id')}:{entry.get('frame_id')}:{entry.get('ts')}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)

    merged.sort(key=lambda row: float(row.get("ts") or 0.0), reverse=True)
    return merged[:limit]


def _append_event_log(
    *,
    intent: str,
    alert: bool,
    camera_id: str,
    source: str,
    building: str,
    floor: str,
    room_number: str,
    bbox: Optional[Dict[str, Any]],
    latency_ms: Optional[int],
    frame_id: Optional[int] = None,
    role_hint: str = "",
    safety_label: Optional[str] = None,
    reasoning_stale: bool = False,
) -> None:
    llm_alert = str(safety_label or "").strip().upper() == "ALERT"
    if reasoning_stale and llm_alert:
        return
    persist_monitoring_event(
        intent=intent,
        alert=llm_alert and alert,
        safety_label=safety_label,
        camera_id=camera_id,
        source=source,
        building=building,
        floor=floor,
        room_number=room_number,
        bbox=bbox,
        latency_ms=latency_ms,
        frame_id=frame_id,
        role_hint=role_hint,
    )


def get_recent_event_logs(limit: int = 40, user: Optional[User] = None) -> List[Dict[str, Any]]:
    return fetch_recent_events(limit, user=user)


def get_recent_alerts(
    limit: int = 10,
    user: Optional[User] = None,
    *,
    current_safety_label: Optional[str] = None,
    current_capture_active: bool = True,
    current_reasoning_stale: bool = False,
) -> List[Dict[str, Any]]:
    return fetch_recent_alerts(
        limit,
        user=user,
        current_safety_label=current_safety_label,
        current_capture_active=current_capture_active,
        current_reasoning_stale=current_reasoning_stale,
    )


def get_patient_history_rows(limit: int = 25, user: Optional[User] = None) -> List[Dict[str, Any]]:
    return fetch_patient_history_rows(limit, user=user)


def _format_relative_time(timestamp: Optional[float]) -> str:
    return format_relative_time(timestamp)


def _read_fresh_frame(capture: cv2.VideoCapture, source_name: str) -> Tuple[bool, Optional[np.ndarray]]:
    try:
        if capture is None or not capture.isOpened():
            return False, None
        # Avoid capture.grab() on live cameras (CAP_DSHOW) — it can throw on Windows.
        success, frame = capture.read()
        if success and frame is not None:
            return True, frame

        if source_name.startswith("video:"):
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if total > 1 and pos >= max(0, total - 2):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif capture.grab():
                success, frame = capture.retrieve()
                if success and frame is not None:
                    return True, frame
        return False, None
    except cv2.error as exc:
        logger.warning("OpenCV frame read failed for %s: %s", source_name, exc)
        return False, None
    except Exception as exc:
        logger.warning("Frame read failed for %s: %s", source_name, exc)
        return False, None


def _video_frame_interval(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or CAPTURE_MAX_FPS)
    if fps < 1.0:
        fps = CAPTURE_MAX_FPS
    fps = min(fps, CAPTURE_MAX_FPS)
    return max(1.0 / fps, 0.02)


def _safe_release_capture(capture: Optional[cv2.VideoCapture]) -> None:
    if capture is None:
        return
    try:
        if capture.isOpened():
            capture.release()
    except cv2.error as exc:
        logger.debug("OpenCV release warning: %s", exc)
    except Exception as exc:
        logger.debug("Capture release warning: %s", exc)


def _apply_cached_inference_overlay(frame: np.ndarray, cached: Dict[str, Any]) -> np.ndarray:
    output = frame.copy()
    bbox = cached.get("bbox")
    intent = str(cached.get("intent") or "Live stream active")
    alert = bool(cached.get("alert_triggered", False))
    if not bbox:
        return output
    x1 = int(bbox.get("x1", 0))
    y1 = int(bbox.get("y1", 0))
    x2 = int(bbox.get("x2", 0))
    y2 = int(bbox.get("y2", 0))
    confidence = float(bbox.get("confidence") or 0.0)
    from monitor.ml_pipeline import PatientIntentPipeline
    from monitor.runtime.context_builder import build_overlay_box_label

    role_hint = str(cached.get("role_hint") or "person").strip().lower()
    label = build_overlay_box_label(role_hint, confidence, intent)

    PatientIntentPipeline.draw_detection_box(
        output,
        (x1, y1, x2, y2),
        label,
        alert_triggered=alert,
    )
    return output


def _config_signature(config: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(config.get("source")),
            str(config.get("camera_index")),
            str(config.get("video_path")),
            str(config.get("camera_id")),
            str(config.get("enable_ai_reasoning")),
        ]
    )


def _get_latest_display_chunk() -> Optional[bytes]:
    return _RUNTIME_STATE.get_display_chunk()


def _build_runtime_overlay_frame(
    frame: np.ndarray, config: Dict[str, Any], overlay: Optional[ReasoningResult]
) -> np.ndarray:
    if overlay is None:
        cached = _snapshot_latest_inference_state()
        return _apply_cached_inference_overlay(
            frame,
            {
                "intent": cached.get("intent") or "Live stream active",
                "alert_triggered": bool(cached.get("alert")),
                "bbox": cached.get("bbox"),
            },
        )
    return _get_pipeline().compose_overlay_frame(frame, overlay)


def _on_reasoning_result(result: ReasoningResult, config: Dict[str, Any], source_name: str) -> None:
    capture_active = _RUNTIME_STATE.is_capture_active()
    exit_reason = _RUNTIME_STATE.get_capture_exit_reason() or None

    stored = _snapshot_latest_inference_state()
    stored_fid = int(stored.get("llm_output_frame_id") or stored.get("frame_id") or 0)
    if stored_fid and int(result.frame_id or 0) < stored_fid:
        logger.debug(
            "Skipping out-of-order LLM result frame_id=%s (stored decision frame=%s)",
            result.frame_id,
            stored_fid,
        )
        return

    if not capture_active:
        logger.debug(
            "reasoning_result ignored for live API frame_id=%s — capture inactive reason=%s",
            result.frame_id,
            exit_reason,
        )
        stale_state = _apply_capture_freshness(_snapshot_latest_inference_state())
        _update_latest_inference_state(
            capture_active=False,
            overlay_stale=True,
            capture_exit_reason=exit_reason,
            alert=False,
            safety_label="STALE",
            intent=stale_state.get("intent")
            or f"Capture stopped — stale frame ({exit_reason or 'inactive'})",
            status="stale",
        )
        return

    intent = str(result.intent or "Live stream active")
    bbox = result.bbox
    pose_summary = result.pose_summary

    scores = dict(result.confidence_scores or {})
    scores.setdefault("overlay_frame_id", result.frame_id)
    snap = _RUNTIME_STATE.live_snapshot(int(result.frame_id or 0))
    scores["capture_active"] = snap["capture_active"]
    scores["capture_frame_id"] = snap["capture_frame_id"]
    scores["capture_exit_reason"] = snap["capture_exit_reason"]
    scores["overlay_stale"] = snap["overlay_stale"]
    scores["overlay_frame_lag"] = snap.get("overlay_frame_lag")

    structured = dict(scores.get("structured_reasoning") or {})
    scene = dict(scores.get("scene") or {})
    safety_label = str(structured.get("safety_label") or scores.get("safety_label") or "not provided")
    risk_level = str(structured.get("risk_level") or result.risk_level or scores.get("risk_level") or "not provided")
    scores["safety_label"] = safety_label
    from monitor.runtime.display_fields import enrich_scores_with_capture

    import copy

    scene_snapshot = copy.deepcopy(scene)
    if int(scene_snapshot.get("frame_id") or 0) != int(result.frame_id):
        scores["reasoning_stale"] = True
    scores["llm_output_frame_id"] = int(result.frame_id or 0)
    scores["llm_scene_frame_id"] = int(result.frame_id or 0)
    live_scene = _RUNTIME_STATE.get_perception_scene(int(result.frame_id or 0))
    scores = enrich_scores_with_capture(
        scores,
        structured,
        scene_snapshot,
        snap,
        intent=str(result.intent or ""),
        decision_frame_id=int(result.frame_id),
        live_scene=live_scene if int(live_scene.get("frame_id") or 0) <= int(result.frame_id or 0) else scene_snapshot,
    )
    reasoning_stale = bool(scores.get("reasoning_stale"))
    alert = (
        safety_label.upper() == "ALERT"
        and bool(snap["capture_active"])
        and not bool(snap["overlay_stale"])
        and not reasoning_stale
    )
    display_sections = scores.get("display_sections") or {}

    if alert:
        logger.info(
            "reasoning_result alert frame_id=%s intent=%r latency_ms=%s mode=%s",
            result.frame_id,
            intent,
            result.latency_ms,
            scores.get("reasoning_mode"),
        )

    persist_bbox = dict(bbox) if isinstance(bbox, dict) else None
    if persist_bbox is not None:
        rs = result.risk_score if result.risk_score is not None else scores.get("risk_score")
        if rs is not None:
            persist_bbox["risk_score"] = float(rs)
        persist_bbox["risk_level"] = str(result.risk_level or scores.get("risk_level") or "not provided")
        persist_bbox["reasoning_mode"] = scores.get("reasoning_mode")

    _append_event_log(
        intent=intent,
        alert=alert,
        camera_id=str(config.get("camera_id") or "Cam-1"),
        source=source_name,
        building=str(config.get("building") or ""),
        floor=str(config.get("floor_number") or ""),
        room_number=str(config.get("room_number") or ""),
        bbox=persist_bbox,
        latency_ms=result.latency_ms,
        frame_id=result.frame_id,
        role_hint=result.role_hint,
        safety_label=safety_label,
        reasoning_stale=reasoning_stale,
    )

    camera_id = str(config.get("camera_id") or "Cam-1")
    _append_reasoning_history(result, source=source_name, camera_id=camera_id)

    rs_log = result.risk_score if result.risk_score is not None else scores.get("risk_score")
    logger.info(
        "dashboard_api: frame_id=%s alert=%s intent=%r risk_score=%s risk_level=%s "
        "mode=%s enable_ai=%s",
        result.frame_id,
        alert,
        intent,
        rs_log if rs_log is not None else "not provided",
        result.risk_level or scores.get("risk_level"),
        scores.get("reasoning_mode"),
        bool(config.get("enable_ai_reasoning")),
    )

    stream_fps, (width, height) = _RUNTIME_STATE.get_stream_metrics()
    from monitor.runtime.display_fields import apply_stale_llm_mask_to_state

    publish_state = apply_stale_llm_mask_to_state(
        {
            "frame_id": result.frame_id,
            "llm_output_frame_id": scores.get("llm_output_frame_id", result.frame_id),
            "llm_scene_frame_id": scores.get("llm_scene_frame_id", result.frame_id),
            "capture_frame_id": scores.get("capture_frame_id", result.frame_id),
            "overlay_frame_id": scores.get("overlay_frame_id", result.frame_id),
            "overlay_stale": bool(snap["overlay_stale"]),
            "capture_active": bool(snap["capture_active"]),
            "capture_exit_reason": snap["capture_exit_reason"],
            "safety_label": safety_label,
            "decision_source": structured.get("decision_source") or scores.get("decision_source"),
            "patient_status": structured.get("patient_status"),
            "staff_presence": structured.get("staff_presence"),
            "alert_type": structured.get("alert_type"),
            "reason": structured.get("reason"),
            "summary": structured.get("summary"),
            "intent": intent,
            "alert": alert,
            "reasoning_stale": reasoning_stale,
            "reasoning_consistency_warning": bool(scores.get("reasoning_consistency_warning")),
            "warning_text": scores.get("warning_text") or "",
            "confidence_scores": scores,
        }
    )
    _update_latest_inference_state(
        frame_id=result.frame_id,
        llm_output_frame_id=publish_state.get("llm_output_frame_id", result.frame_id),
        llm_scene_frame_id=publish_state.get("llm_scene_frame_id", result.frame_id),
        capture_frame_id=scores.get("capture_frame_id", result.frame_id),
        overlay_frame_id=scores.get("overlay_frame_id", result.frame_id),
        overlay_stale=bool(snap["overlay_stale"]),
        capture_active=bool(snap["capture_active"]),
        capture_exit_reason=snap["capture_exit_reason"],
        safety_label=publish_state.get("safety_label", safety_label),
        decision_source=structured.get("decision_source") or scores.get("decision_source"),
        patient_status=publish_state.get("patient_status", structured.get("patient_status")),
        staff_presence=publish_state.get("staff_presence", structured.get("staff_presence")),
        alert_type=publish_state.get("alert_type", structured.get("alert_type")),
        reason=publish_state.get("reason", structured.get("reason")),
        summary=publish_state.get("summary", structured.get("summary")),
        intent=publish_state.get("intent", intent),
        alert=publish_state.get("alert", alert),
        bbox=bbox,
        updated_at=time.time(),
        source=source_name,
        status="live" if capture_active else "stale",
        error=None,
        pose_summary=pose_summary,
        building=config.get("building"),
        floor=config.get("floor_number"),
        room_number=config.get("room_number"),
        camera_id=camera_id,
        connected_cameras=1,
        registered_cameras=_count_registered_cameras(),
        latency_ms=result.latency_ms,
        frame_width=width,
        frame_height=height,
        stream_fps=stream_fps,
        model_status=result.model_status,
        confidence_scores=scores,
        reasoning_health=scores.get("reasoning_health"),
        monitor_status=result.monitor_status or "idle",
        role_hint=result.role_hint or "unknown",
        risk_score=result.risk_score if result.risk_score is not None else scores.get("risk_score"),
        risk_level=risk_level,
        display_sections=display_sections,
        reasoning_trace=list(result.reasoning_trace or scores.get("reasoning_trace") or []),
        reasoning_history=_snapshot_reasoning_history(12),
        reasoning_stale=bool(publish_state.get("reasoning_stale", scores.get("reasoning_stale"))),
        reasoning_consistency_warning=bool(publish_state.get("reasoning_consistency_warning")),
        warning_text=publish_state.get("warning_text", scores.get("warning_text")),
        llm_lag_frames=scores.get("llm_lag_frames"),
        pose_frame_id=scores.get("pose_frame_id"),
        pose_age_frames=scores.get("pose_age_frames"),
        pose_updated_at=scores.get("pose_updated_at"),
        reasoning_latency_ms=scores.get("reasoning_latency_ms") or result.latency_ms,
        yolo_latency_ms=scores.get("yolo_latency_ms"),
        llama_latency_ms=scores.get("llama_latency_ms"),
        recipient_count=len(
            _location_recipients(
                str(config.get("building") or ""),
                str(config.get("floor_number") or ""),
            )
        ),
    )


def _on_capture_offline(config: Dict[str, Any]) -> None:
    offline = _frame_to_jpeg_chunk(
        _build_status_frame("Camera Offline", config),
        config,
        offline=True,
        frame_kind="camera-offline",
    )
    _RUNTIME_STATE.set_display_chunk(offline)
    _update_latest_inference_state(
        intent="Camera Offline",
        alert=False,
        status="offline",
        error="Live camera frame could not be read.",
        connected_cameras=0,
        overlay_stale=True,
        capture_active=False,
        capture_exit_reason="frame read failed",
    )


def _on_stream_metrics(stream_fps: int, width: int, height: int, source_name: str) -> None:
    _update_latest_inference_state(
        stream_fps=stream_fps,
        frame_width=width,
        frame_height=height,
        connected_cameras=1,
        status="live",
        source=source_name,
    )


def _stop_capture_worker() -> None:
    global _ORCHESTRATOR
    with _ORCHESTRATOR_LOCK:
        if _ORCHESTRATOR is not None:
            _ORCHESTRATOR.stop()
            _ORCHESTRATOR = None


def _ensure_capture_worker(config: Dict[str, Any], capture: cv2.VideoCapture, source_name: str) -> None:
    global _ORCHESTRATOR, _CAPTURE_CONFIG_SIG
    signature = _config_signature(config)
    with _ORCHESTRATOR_LOCK:
        if _ORCHESTRATOR is not None and _CAPTURE_CONFIG_SIG == signature:
            _safe_release_capture(capture)
            return

        _stop_capture_worker()
        _CAPTURE_CONFIG_SIG = signature
        _RUNTIME_STATE.reset()

        video_mode = source_name.startswith("video:")
        inference_stride = INFERENCE_EVERY_N_FRAMES_VIDEO if video_mode else INFERENCE_EVERY_N_FRAMES

        def metrics_cb(fps: int, w: int, h: int) -> None:
            _on_stream_metrics(fps, w, h, source_name)

        _ORCHESTRATOR = StreamOrchestrator(
            state=_RUNTIME_STATE,
            pipeline_getter=_get_pipeline,
            read_frame_fn=_read_fresh_frame,
            video_interval_fn=_video_frame_interval,
            build_overlay_frame_fn=_build_runtime_overlay_frame,
            encode_chunk_fn=_frame_to_jpeg_chunk,
            on_reasoning_result=_on_reasoning_result,
            on_stream_metrics=metrics_cb,
            on_offline=lambda: _on_capture_offline(config),
            on_capture_ended=_on_capture_ended,
            inference_stride=inference_stride,
            queue_size=RUNTIME_QUEUE_SIZE,
        )
        logger.info("[runtime] Ensuring capture worker signature=%s source=%s", signature, source_name)
        _ORCHESTRATOR.start(capture, source_name, config)
        _on_capture_started(source_name, config)


def _stream_is_online(state: Dict[str, Any]) -> bool:
    """True when capture is running and the MJPEG pipeline is producing frames."""
    capture_live = _RUNTIME_STATE.is_capture_active() or bool(state.get("capture_active"))
    if not capture_live:
        return False
    status = str(state.get("status") or "").lower()
    if status in {"ok", "streaming", "live"}:
        return True
    if int(state.get("stream_fps") or 0) > 0:
        return True
    return int(state.get("connected_cameras") or 0) > 0


def _resolve_dashboard_latency(state: Dict[str, Any]) -> Optional[int]:
    """Best-effort end-to-end latency for dashboard stat cards."""
    scores = state.get("confidence_scores") or {}
    yolo_ms = scores.get("yolo_latency_ms")
    reasoning_ms = (
        scores.get("reasoning_latency_ms")
        or scores.get("llama_latency_ms")
        or state.get("reasoning_latency_ms")
    )
    pipeline_ms = state.get("latency_ms")
    if yolo_ms is not None and reasoning_ms is not None:
        total = int(yolo_ms) + int(reasoning_ms)
        if total > 0:
            return total
    for candidate in (pipeline_ms, reasoning_ms, yolo_ms):
        if candidate is not None:
            val = int(candidate)
            if val > 0:
                return val
    stream_fps = int(state.get("stream_fps") or 0)
    if stream_fps > 0:
        return max(1, int(round(1000 / stream_fps)))
    return None


def build_monitor_camera_context(session) -> Tuple[list[Dict[str, Any]], int, int]:
    """Build camera tiles from DB; fall back to the active session camera."""
    state = _apply_capture_freshness(_snapshot_latest_inference_state())
    stream_online = _stream_is_online(state)
    active_camera_id = str(
        state.get("camera_id")
        or _session_or_env(session, "monitor_camera_id", "MONITOR_CAMERA_ID", "Cam-1")
    ).strip()

    feeds: list[Dict[str, Any]] = []
    try:
        for cam in Camera.objects.filter(is_active=True).order_by("camera_identifier"):
            subtitle = f"Floor {cam.floor}"
            if cam.room_number:
                subtitle += f" • Room {cam.room_number}"
            feeds.append(
                {
                    "id": cam.camera_identifier,
                    "name": (cam.display_name.strip() or cam.camera_identifier),
                    "subtitle": subtitle,
                    "building": cam.building,
                    "floor": cam.floor,
                    "room_number": cam.room_number or "",
                    "is_online": False,
                    "show_live_stream": False,
                }
            )
    except DatabaseError:
        pass
    except Exception as exc:
        logger.warning("Camera feed list could not be loaded: %s", exc)

    if not feeds:
        floor = str(_session_or_env(session, "monitor_floor_number", "MONITOR_FLOOR_NUMBER", "3"))
        room = str(_session_or_env(session, "monitor_room_number", "MONITOR_ROOM_NUMBER", "302"))
        feeds.append(
            {
                "id": active_camera_id,
                "name": active_camera_id,
                "subtitle": f"Floor {floor} • Room {room}",
                "building": str(_session_or_env(session, "monitor_building", "MONITOR_BUILDING", "Unknown Building")),
                "floor": floor,
                "room_number": room,
                "is_online": False,
                "show_live_stream": False,
            }
        )

    if stream_online:
        matched = False
        for feed in feeds:
            if feed["id"] == active_camera_id:
                feed["is_online"] = True
                feed["show_live_stream"] = True
                matched = True
        if not matched:
            feeds[0]["is_online"] = True
            feeds[0]["show_live_stream"] = True
    else:
        for feed in feeds:
            feed["is_online"] = False
            feed["show_live_stream"] = False

    connected = sum(1 for feed in feeds if feed["is_online"])
    return feeds, connected, len(feeds)


def _count_registered_cameras(session=None) -> int:
    _, _, registered = build_monitor_camera_context(session or {})
    return registered


def _count_connected_cameras(session=None) -> int:
    _, connected, _ = build_monitor_camera_context(session or {})
    return connected


def _warm_pipeline_async() -> None:
    def _runner() -> None:
        try:
            _get_pipeline()
            logger.info("PatientIntentPipeline warmup complete (YOLO only; Llama loads on first reasoning).")
        except Exception as exc:
            logger.warning("Pipeline warmup failed: %s", exc)

    if _PIPELINE is None:
        threading.Thread(target=_runner, name="medpercept-pipeline-warmup", daemon=True).start()


def _frame_to_jpeg_chunk(frame: np.ndarray, config: Dict[str, Any], offline: bool, frame_kind: str) -> Optional[bytes]:
    display = _apply_cctv_overlay(frame, config, offline=offline)
    jpeg_bytes = _encode_jpeg_frame(display)
    if not jpeg_bytes:
        return None
    return _yield_mjpeg_chunk(jpeg_bytes, frame_kind=frame_kind)


def video_stream_gen(config: Dict[str, Any]) -> Iterator[bytes]:
    global _STREAM_CLIENTS
    _warm_pipeline_async()

    connecting = _frame_to_jpeg_chunk(
        _build_status_frame("Connecting to stream...", config),
        config,
        offline=True,
        frame_kind="connecting",
    )
    if connecting:
        yield connecting

    capture, source_name = _open_stream_source(config)
    _update_latest_inference_state(
        source=source_name,
        capture_exit_reason=None,
        overlay_stale=False,
        status="streaming" if capture is not None else "offline",
        monitoring_display=config.get("monitoring_display"),
        source_label=config.get("source_label"),
        building=config.get("building"),
        floor=config.get("floor_number"),
        room_number=config.get("room_number"),
        camera_id=config.get("camera_id"),
        connected_cameras=1 if capture is not None else 0,
        registered_cameras=_count_registered_cameras(),
    )

    if capture is None:
        missing_video = config.get("source") == "video"
        logger.error("No camera or fallback video source is available for streaming.")
        while True:
            error_message = (
                f"Video file not found: {config.get('video_path')}"
                if missing_video
                else "Video source unavailable. Check Settings or MONITOR_VIDEO_PATH in .env"
            )
            error_frame = _build_status_frame(error_message, config)
            chunk = _frame_to_jpeg_chunk(error_frame, config, offline=True, frame_kind="offline-status")
            if chunk:
                _update_latest_inference_state(
                    intent="Camera Offline",
                    alert=False,
                    status="offline",
                    error="No camera or fallback video source is available.",
                    connected_cameras=0,
                )
                yield chunk
            time.sleep(1.0)
        return

    with _STREAM_CLIENTS_LOCK:
        _STREAM_CLIENTS += 1
    logger.info("[runtime] MJPEG client connected; opening stream source=%s", source_name)
    _ensure_capture_worker(config, capture, source_name)

    try:
        while True:
            chunk = _get_latest_display_chunk()
            if chunk:
                yield chunk
            else:
                waiting = _frame_to_jpeg_chunk(
                    _build_status_frame("Starting AI pipeline...", config),
                    config,
                    offline=True,
                    frame_kind="waiting",
                )
                if waiting:
                    yield waiting
            time.sleep(0.04)
    finally:
        with _STREAM_CLIENTS_LOCK:
            _STREAM_CLIENTS -= 1
            remaining = _STREAM_CLIENTS
        if remaining <= 0:
            _stop_capture_worker()


@login_required
def live_stream_feed(request):
    try:
        stream_config = _monitor_runtime_config(request.session)
    except Exception as exc:
        logger.exception("Stream config resolution failed; using session defaults: %s", exc)
        stream_config = _monitor_runtime_config({})

    logger.info(
        "MJPEG stream started (source=%s, video=%s, camera_index=%s)",
        stream_config.get("source"),
        stream_config.get("video_path"),
        stream_config.get("camera_index"),
    )
    response = StreamingHttpResponse(
        video_stream_gen(stream_config),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@login_required
def get_latest_alert(request):
    from accounts.permissions import apply_location_filter_to_inference_state, user_can_view_location
    from monitor.runtime.pipeline_diagnostics import get_diagnostics

    api_started = time.time()
    state = _apply_capture_freshness(_snapshot_latest_inference_state())
    building = str(state.get("building") or "").strip()
    floor = str(state.get("floor") or state.get("floor_number") or "").strip()
    recipients = _location_recipients(building, floor)

    response_state = apply_location_filter_to_inference_state(request.user, dict(state))
    response_state["recipient_count"] = len(recipients)
    response_state["user_unit_access"] = user_can_view_location(request.user, building, floor)

    profile = _get_staff_profile(request.user)
    if profile:
        response_state["staff_role"] = profile.get_role_display()
        response_state["assigned_building"] = profile.assigned_building
        response_state["assigned_floor"] = profile.assigned_floor

    camera_feeds, connected, registered = build_monitor_camera_context(request.session)
    runtime_connected = int(state.get("connected_cameras") or 0)
    if _stream_is_online(state):
        runtime_connected = max(runtime_connected, 1)

    response_state["camera_feeds"] = camera_feeds
    response_state["connected_cameras"] = max(connected, runtime_connected)
    response_state["registered_cameras"] = registered
    response_state["active_alerts"] = count_active_alerts(request.user)

    reasoning_stale_api = bool(
        response_state.get("reasoning_stale")
        or (response_state.get("confidence_scores") or {}).get("reasoning_stale")
    )

    if (
        response_state.get("alert")
        and response_state.get("capture_active")
        and not response_state.get("overlay_stale")
        and not reasoning_stale_api
    ):
        response_state["active_alerts"] = max(response_state["active_alerts"], 1)
    else:
        response_state["active_alerts"] = 0

    response_state["patients_monitored"] = registered if registered else max(
        connected, response_state["connected_cameras"]
    )
    response_state["latency_ms"] = _resolve_dashboard_latency(state)
    response_state["stream_fps"] = state.get("stream_fps")
    response_state["frame_width"] = state.get("frame_width")
    response_state["frame_height"] = state.get("frame_height")
    response_state["enable_ai_reasoning"] = _ai_reasoning_enabled(request.session)
    response_state["recent_logs"] = get_recent_event_logs(30, user=request.user)
    response_state["live_logs"] = response_state["recent_logs"]
    response_state["logs_updated_at"] = time.time()
    response_state["patient_history"] = get_patient_history_rows(25, user=request.user)

    if not response_state.get("model_status"):
        try:
            response_state["model_status"] = dict(_get_pipeline().model_status)
        except Exception:
            response_state["model_status"] = {}

    try:
        pipeline = _get_pipeline()
        response_state["reasoning_health"] = dict(pipeline.reasoning_health)
        response_state["reasoning_display"] = pipeline.reasoning_health.get("display_mode", "")
    except Exception:
        response_state["reasoning_health"] = {}
        response_state["reasoning_display"] = "llama_unavailable"

    if not response_state.get("reasoning_trace") and state.get("reasoning_trace"):
        response_state["reasoning_trace"] = state.get("reasoning_trace")

    if not response_state.get("risk_level"):
        response_state["risk_level"] = state.get("risk_level", "not provided")

    if not response_state.get("display_sections") and state.get("confidence_scores", {}).get("display_sections"):
        response_state["display_sections"] = state["confidence_scores"]["display_sections"]

    response_state = _apply_llm_fields(response_state)

    scores = response_state.get("confidence_scores") or {}
    api_publish_ms = int((time.time() - api_started) * 1000)
    llm_frame = int(response_state.get("llm_output_frame_id") or response_state.get("frame_id") or 0)
    capture_frame = int(response_state.get("capture_frame_id") or 0)
    lag_frames = int(response_state.get("llm_lag_frames") or scores.get("llm_lag_frames") or 0)

    if not lag_frames and capture_frame and llm_frame:
        lag_frames = max(0, capture_frame - llm_frame)

    reasoning_stale_api = bool(response_state.get("reasoning_stale") or scores.get("reasoning_stale"))
    response_state["llm_lag_frames"] = lag_frames
    response_state["llm_output_frame_id"] = llm_frame

    get_diagnostics().on_api_publish(
        llm_frame_id=llm_frame,
        capture_frame_id=capture_frame,
        lag_frames=lag_frames,
        capture_active=bool(response_state.get("capture_active")),
        reasoning_stale=reasoning_stale_api,
        api_publish_ms=api_publish_ms,
    )

    response_state["recent_alerts"] = get_recent_alerts(
        12,
        user=request.user,
        current_safety_label=response_state.get("safety_label"),
        current_capture_active=bool(response_state.get("capture_active")),
        current_reasoning_stale=reasoning_stale_api,
    )

    # FORCE ALERT CONDITION:
    # Jab reasoning/dashboard output me "patient not in bed safely" ya off-bed status aaye,
    # response_state["alert"] true hoga. Frontend is alert par alarm play karega.
    scores = response_state.get("confidence_scores") or {}
    structured = scores.get("structured_reasoning") or {}
    display_sections = response_state.get("display_sections") or scores.get("display_sections") or {}

    alert_text_parts = [
        response_state.get("intent"),
        response_state.get("summary"),
        response_state.get("reason"),
        response_state.get("patient_status"),
        response_state.get("safety_label"),
        response_state.get("alert_type"),
        response_state.get("risk_level"),
        structured.get("summary") if isinstance(structured, dict) else "",
        structured.get("reason") if isinstance(structured, dict) else "",
        structured.get("patient_status") if isinstance(structured, dict) else "",
        structured.get("safety_label") if isinstance(structured, dict) else "",
        structured.get("alert_type") if isinstance(structured, dict) else "",
        structured.get("risk_level") if isinstance(structured, dict) else "",
        str(display_sections),
        str(scores),
    ]

    alert_text = " ".join(str(x or "").lower() for x in alert_text_parts)

    patient_not_bed_alert = (
        "patient not in bed safely" in alert_text
        or "not in bed safely" in alert_text
        or "patient_lying_off_bed" in alert_text
        or "lying_off_bed" in alert_text
        or "off_bed" in alert_text
        or "off bed" in alert_text
        or "not safely on bed" in alert_text
        or "not on bed safely" in alert_text
        or "patient not safely" in alert_text
    )

    if patient_not_bed_alert:
        response_state["alert"] = True
        response_state["safety_label"] = "ALERT"
        response_state["alert_type"] = "patient_not_in_bed_safely"
        response_state["risk_level"] = "critical"
        response_state["active_alerts"] = 1

        scores["safety_label"] = "ALERT"
        scores["alert_type"] = "patient_not_in_bed_safely"
        scores["risk_level"] = "critical"

        if isinstance(structured, dict):
            structured["safety_label"] = "ALERT"
            structured["alert_type"] = "patient_not_in_bed_safely"
            structured["risk_level"] = "critical"
            scores["structured_reasoning"] = structured

        response_state["confidence_scores"] = scores

    return JsonResponse(response_state)
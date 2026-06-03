from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from queue import Empty, Queue
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import DatabaseError
from django.http import JsonResponse
from django.http import StreamingHttpResponse

from monitor.ml_pipeline import DEFAULT_FALLBACK_VIDEO, PatientIntentPipeline
from monitor.models import Camera

logger = logging.getLogger(__name__)

_PIPELINE: Optional[PatientIntentPipeline] = None
_PIPELINE_INIT_LOCK = threading.Lock()
PIPELINE_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
latest_inference_state: Dict[str, Any] = {
    "intent": "System warming up",
    "alert": False,
    "bbox": None,
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
}

MJPEG_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
MJPEG_SUFFIX = b"\r\n"
FRAME_SLEEP_SECONDS = 0.03
EVENT_LOG: deque = deque(maxlen=120)
EVENT_LOG_LOCK = threading.Lock()

DISPLAY_FRAME_LOCK = threading.Lock()
_LATEST_DISPLAY_CHUNK: Optional[bytes] = None
_CAPTURE_STOP = threading.Event()
_CAPTURE_THREAD: Optional[threading.Thread] = None
_CAPTURE_CONFIG_SIG: Optional[str] = None
_INFERENCE_STOP = threading.Event()
_INFERENCE_THREAD: Optional[threading.Thread] = None
_INFERENCE_QUEUE: "Queue[Tuple[np.ndarray, int, Dict[str, Any]]]" = Queue(maxsize=1)
_WORKER_CACHED_INFERENCE: Optional[Dict[str, Any]] = None
_WORKER_CACHED_LOCK = threading.Lock()
_STREAM_CLIENTS = 0
_STREAM_CLIENTS_LOCK = threading.Lock()


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return fallback


INFERENCE_EVERY_N_FRAMES = max(1, _safe_int(os.environ.get("INFERENCE_EVERY_N_FRAMES", "2"), 2))


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
    if session is not None and "enable_ai_reasoning" in session:
        return bool(session.get("enable_ai_reasoning"))
    return os.environ.get("ENABLE_REASONING", "0").strip().lower() not in {"0", "false", "no", "off"}


def _safe_process_frame(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    use_reasoning = bool(config.get("enable_ai_reasoning", config.get("use_reasoning", False)))
    try:
        with PIPELINE_LOCK:
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
    height, width = frame.shape[:2]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    floor_number = config.get("floor_number", "--")
    room_number = config.get("room_number", "--")
    camera_id = config.get("camera_id", "--")
    building = config.get("building", "--")
    source_label = config.get("source_label", "Live Camera Feed")
    input_detail = config.get("input_detail", "")

    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (59, 130, 246), 2)

    header = f"MedPerceptAI | {building}"
    location = f"Floor {floor_number} | Room {room_number} | {camera_id}"
    source_line = f"{source_label} | {input_detail}"
    status = "OFFLINE" if offline else "LIVE"

    lines = [header, location, source_line, timestamp, status]
    y = 34
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (229, 231, 235), 2, cv2.LINE_AA)
        y += 30

    badge_color = (0, 0, 255) if offline else (0, 180, 0)
    cv2.rectangle(frame, (width - 150, 12), (width - 12, 48), badge_color, -1)
    cv2.putText(frame, status, (width - 132, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
) -> None:
    entry = {
        "ts": time.time(),
        "intent": intent,
        "alert": alert,
        "camera_id": camera_id,
        "source": source,
        "building": building,
        "floor": floor,
        "room_number": room_number,
        "bbox": bbox,
        "latency_ms": latency_ms,
        "event": "alert_emitted" if alert else "inference_update",
    }
    with EVENT_LOG_LOCK:
        if EVENT_LOG and not alert:
            last = EVENT_LOG[0]
            if (
                last.get("intent") == intent
                and last.get("camera_id") == camera_id
                and (time.time() - float(last.get("ts") or 0)) < 0.8
            ):
                last.update(entry)
                return
        EVENT_LOG.appendleft(entry)


def get_recent_event_logs(limit: int = 40, user: Optional[User] = None) -> List[Dict[str, Any]]:
    with EVENT_LOG_LOCK:
        events = list(EVENT_LOG)
    if user is not None:
        from accounts.permissions import filter_events_for_user

        events = filter_events_for_user(user, events)
    return events[:limit]


def get_recent_alerts(limit: int = 10, user: Optional[User] = None) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    for event in get_recent_event_logs(80, user=user):
        intent_text = str(event.get("intent") or "Alert detected")
        lowered = intent_text.lower()
        is_critical = bool(event.get("alert"))
        is_warning = any(token in lowered for token in ("fall", "stand", "distress", "agitation", "attempting"))
        if not is_critical and not is_warning:
            continue
        alerts.append(
            {
                "intent": intent_text,
                "camera_id": event.get("camera_id", "--"),
                "room_number": event.get("room_number", "--"),
                "floor": event.get("floor", "--"),
                "alert": is_critical,
                "updated_label": _format_relative_time(event.get("ts")),
                "severity": "Critical" if is_critical else "Warning",
                "severity_class": "badge-critical" if is_critical else "badge-warning",
            }
        )
        if len(alerts) >= limit:
            break
    return alerts


def get_patient_history_rows(limit: int = 25, user: Optional[User] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in get_recent_event_logs(limit, user=user):
        intent_text = str(event.get("intent") or "Under observation")
        rows.append(
            {
                "patient_label": f"Patient @ {event.get('camera_id', 'Cam')}",
                "unit": f"Floor {event.get('floor', '--')} • Room {event.get('room_number', '--')}",
                "camera_id": event.get("camera_id", "--"),
                "last_event": intent_text,
                "risk": "Critical" if event.get("alert") else ("Warning" if "stand" in intent_text.lower() else "OK"),
                "risk_class": "crit" if event.get("alert") else ("warn" if "stand" in intent_text.lower() else "ok"),
                "updated_label": _format_relative_time(event.get("ts")),
            }
        )
    return rows


def _format_relative_time(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "--"
    delta = max(0, int(time.time() - float(timestamp)))
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60} min ago"
    return f"{delta // 3600}h ago"


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
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    if fps < 1.0:
        fps = 24.0
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
    if bbox:
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        confidence = float(bbox.get("confidence") or 0.0)
        color = (0, 0, 255) if alert else (0, 200, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"patient {confidence:.2f} | {intent[:48]}",
            (x1 + 8, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            output,
            intent[:60],
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
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


def _set_latest_display_chunk(chunk: Optional[bytes]) -> None:
    global _LATEST_DISPLAY_CHUNK
    if not chunk:
        return
    with DISPLAY_FRAME_LOCK:
        _LATEST_DISPLAY_CHUNK = chunk


def _get_latest_display_chunk() -> Optional[bytes]:
    with DISPLAY_FRAME_LOCK:
        return _LATEST_DISPLAY_CHUNK


def _stop_inference_worker() -> None:
    global _INFERENCE_THREAD, _WORKER_CACHED_INFERENCE
    _INFERENCE_STOP.set()
    thread = _INFERENCE_THREAD
    if thread and thread.is_alive():
        thread.join(timeout=3.0)
    _INFERENCE_THREAD = None
    _INFERENCE_STOP.clear()
    while True:
        try:
            _INFERENCE_QUEUE.get_nowait()
        except Empty:
            break
    with _WORKER_CACHED_LOCK:
        _WORKER_CACHED_INFERENCE = None


def _start_inference_worker(config: Dict[str, Any], source_name: str) -> None:
    global _INFERENCE_THREAD
    if _INFERENCE_THREAD and _INFERENCE_THREAD.is_alive():
        return
    _INFERENCE_STOP.clear()
    _INFERENCE_THREAD = threading.Thread(
        target=_inference_worker_loop,
        args=(config, source_name),
        name="medpercept-inference-worker",
        daemon=True,
    )
    _INFERENCE_THREAD.start()


def _queue_inference_frame(frame: np.ndarray, frame_index: int, config: Dict[str, Any]) -> None:
    payload = (frame, frame_index, config)
    try:
        _INFERENCE_QUEUE.put_nowait(payload)
    except Exception:
        try:
            _INFERENCE_QUEUE.get_nowait()
        except Empty:
            pass
        try:
            _INFERENCE_QUEUE.put_nowait(payload)
        except Exception:
            pass


def _get_worker_cached_inference() -> Dict[str, Any]:
    with _WORKER_CACHED_LOCK:
        return dict(_WORKER_CACHED_INFERENCE or {})


def _inference_worker_loop(config: Dict[str, Any], source_name: str) -> None:
    global _WORKER_CACHED_INFERENCE
    while not _INFERENCE_STOP.is_set():
        try:
            frame, frame_index, job_config = _INFERENCE_QUEUE.get(timeout=0.25)
        except Empty:
            continue

        frame_started = time.time()
        try:
            result = _safe_process_frame(frame, job_config)
            cached = dict(result)
        except Exception as exc:
            logger.exception("Background inference failed: %s", exc)
            continue

        with _WORKER_CACHED_LOCK:
            _WORKER_CACHED_INFERENCE = cached

        intent = str(cached.get("intent") or "Live stream active")
        alert = bool(cached.get("alert_triggered", False))
        bbox = cached.get("bbox")
        pose_summary = cached.get("pose_summary")
        latency_ms = int((time.time() - frame_started) * 1000)
        display_frame = cached.get("annotated_frame")
        height, width = (0, 0)
        if display_frame is not None:
            height, width = display_frame.shape[:2]

        _append_event_log(
            intent=intent,
            alert=alert,
            camera_id=str(job_config.get("camera_id") or "Cam-1"),
            source=source_name,
            building=str(job_config.get("building") or ""),
            floor=str(job_config.get("floor_number") or ""),
            room_number=str(job_config.get("room_number") or ""),
            bbox=bbox,
            latency_ms=latency_ms,
        )

        pipeline_status = {}
        try:
            pipeline_status = dict(_get_pipeline().model_status)
        except Exception:
            pipeline_status = {}

        _update_latest_inference_state(
            intent=intent,
            alert=alert,
            bbox=bbox,
            updated_at=time.time(),
            source=source_name,
            status="ok",
            error=None,
            pose_summary=pose_summary,
            building=job_config.get("building"),
            floor=job_config.get("floor_number"),
            room_number=job_config.get("room_number"),
            camera_id=job_config.get("camera_id"),
            connected_cameras=1,
            registered_cameras=_count_registered_cameras(),
            latency_ms=latency_ms,
            frame_width=width,
            frame_height=height,
            model_status=pipeline_status,
            recipient_count=len(
                _location_recipients(
                    str(job_config.get("building") or ""),
                    str(job_config.get("floor_number") or ""),
                )
            ),
        )


def _stop_capture_worker() -> None:
    global _CAPTURE_THREAD
    _CAPTURE_STOP.set()
    _stop_inference_worker()
    thread = _CAPTURE_THREAD
    if thread and thread.is_alive():
        thread.join(timeout=3.0)
    _CAPTURE_THREAD = None
    _CAPTURE_STOP.clear()


def _capture_worker_loop(capture: cv2.VideoCapture, source_name: str, config: Dict[str, Any]) -> None:
    frame_index = 0
    fps_window_start = time.time()
    fps_frame_count = 0
    stream_fps = 0
    consecutive_failures = 0
    video_mode = source_name.startswith("video:")
    frame_interval = _video_frame_interval(capture) if video_mode else FRAME_SLEEP_SECONDS
    inference_stride = INFERENCE_EVERY_N_FRAMES
    if video_mode:
        inference_stride = max(INFERENCE_EVERY_N_FRAMES, 4)

    _start_inference_worker(config, source_name)

    try:
        while not _CAPTURE_STOP.is_set():
            success, frame = _read_fresh_frame(capture, source_name)
            if not success or frame is None:
                consecutive_failures += 1
                if video_mode:
                    try:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except cv2.error as exc:
                        logger.warning("Video rewind failed: %s", exc)
                    time.sleep(frame_interval)
                    continue

                offline = _frame_to_jpeg_chunk(
                    _build_status_frame("Camera Offline", config),
                    config,
                    offline=True,
                    frame_kind="camera-offline",
                )
                _set_latest_display_chunk(offline)
                _update_latest_inference_state(
                    intent="Camera Offline",
                    alert=False,
                    status="offline",
                    error="Live camera frame could not be read.",
                    connected_cameras=0,
                )
                if consecutive_failures >= 30:
                    logger.error("Capture worker stopping after repeated read failures for %s", source_name)
                    break
                time.sleep(0.2)
                continue

            consecutive_failures = 0
            frame_index += 1
            fps_frame_count += 1
            elapsed = time.time() - fps_window_start
            if elapsed >= 1.0:
                stream_fps = int(round(fps_frame_count / elapsed))
                fps_frame_count = 0
                fps_window_start = time.time()

            cached_inference = _get_worker_cached_inference()
            run_inference = frame_index % inference_stride == 0 or not cached_inference
            if run_inference:
                _queue_inference_frame(frame.copy(), frame_index, config)

            display_frame = _apply_cached_inference_overlay(frame, cached_inference)
            live_chunk = _frame_to_jpeg_chunk(display_frame, config, offline=False, frame_kind="live")
            _set_latest_display_chunk(live_chunk)

            height, width = frame.shape[:2]
            _update_latest_inference_state(
                stream_fps=stream_fps,
                frame_width=width,
                frame_height=height,
                connected_cameras=1,
                status="ok",
                source=source_name,
            )

            time.sleep(frame_interval)
    except cv2.error as exc:
        logger.exception("Capture worker OpenCV error for %s: %s", source_name, exc)
    except Exception as exc:
        logger.exception("Capture worker crashed for %s: %s", source_name, exc)
    finally:
        _stop_inference_worker()
        _safe_release_capture(capture)


def _ensure_capture_worker(config: Dict[str, Any], capture: cv2.VideoCapture, source_name: str) -> None:
    global _CAPTURE_THREAD, _CAPTURE_CONFIG_SIG
    signature = _config_signature(config)
    if _CAPTURE_THREAD and _CAPTURE_THREAD.is_alive() and _CAPTURE_CONFIG_SIG == signature:
        _safe_release_capture(capture)
        return

    _stop_capture_worker()
    _CAPTURE_CONFIG_SIG = signature
    _CAPTURE_STOP.clear()
    _CAPTURE_THREAD = threading.Thread(
        target=_capture_worker_loop,
        args=(capture, source_name, config),
        name="medpercept-capture-worker",
        daemon=True,
    )
    _CAPTURE_THREAD.start()


def build_monitor_camera_context(session) -> Tuple[list[Dict[str, Any]], int, int]:
    """Build camera tiles from DB; fall back to the active session camera."""
    state = _snapshot_latest_inference_state()
    stream_online = state.get("status") in {"ok", "streaming"}
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
            logger.info("PatientIntentPipeline warmup complete.")
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

    state = _snapshot_latest_inference_state()
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
    response_state["camera_feeds"] = camera_feeds
    response_state["connected_cameras"] = connected
    response_state["registered_cameras"] = registered
    if "active_alerts" not in response_state:
        response_state["active_alerts"] = 1 if response_state.get("alert") else 0
    response_state["patients_monitored"] = registered if registered else connected
    response_state["latency_ms"] = state.get("latency_ms")
    response_state["stream_fps"] = state.get("stream_fps")
    response_state["frame_width"] = state.get("frame_width")
    response_state["frame_height"] = state.get("frame_height")
    response_state["enable_ai_reasoning"] = _ai_reasoning_enabled(request.session)
    response_state["recent_logs"] = get_recent_event_logs(30, user=request.user)
    response_state["recent_alerts"] = get_recent_alerts(12, user=request.user)
    response_state["patient_history"] = get_patient_history_rows(25, user=request.user)
    if not response_state.get("model_status"):
        try:
            response_state["model_status"] = dict(_get_pipeline().model_status)
        except Exception:
            response_state["model_status"] = {}

    return JsonResponse(response_state)

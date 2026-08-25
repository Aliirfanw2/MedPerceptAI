"""
Presentation / demo tuning for MedPerceptAI.

Prioritizes smooth streaming, stable overlays, and fewer false-positive alerts.
All values overridable via environment variables.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(str(os.environ.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


def _float_env(*keys: str, default: float) -> float:
    """Read the first set env var from keys; fall back to default."""
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return default


# --- Detection confidence (first working pipeline defaults) ---
OBJECT_BASE_CONF = _float_env("OBJECT_BASE_CONF", "YOLO_CONFIDENCE", default=0.03)
OBJECT_CONF = _float_env("OBJECT_CONF", "YOLO_OBJECT_DISPLAY_CONF", default=0.25)
BED_CONF = _float_env("BED_CONF", default=0.05)
PERSON_CONF = _float_env("PERSON_CONF", default=0.30)
ROLE_CONF = _float_env("ROLE_CONF", "ROLE_CONFIDENCE_MIN", default=0.25)
PATIENT_CONF = _float_env("PATIENT_CONF", "ROLE_PATIENT_CONF", default=0.15)
DOCTOR_CONF = _float_env("DOCTOR_CONF", default=0.25)
NURSE_CONF = _float_env("NURSE_CONF", default=0.25)
POSE_CONF = _float_env("POSE_CONF", "POSE_MIN_SCORE", default=0.25)

# Legacy aliases (existing code paths)
YOLO_CONFIDENCE = OBJECT_BASE_CONF
YOLO_OBJECT_DISPLAY_CONF = OBJECT_CONF
ROLE_CONFIDENCE_MIN = ROLE_CONF
ROLE_PATIENT_CONF = PATIENT_CONF
POSE_MIN_SCORE = POSE_CONF
YOLO_IOU = _float("YOLO_IOU", 0.45)
YOLO_MAX_DETECTIONS = _int("YOLO_MAX_DETECTIONS", 3)
YOLO_MIN_BOX_AREA_PCT = _float("YOLO_MIN_BOX_AREA_PCT", 0.008)
YOLO_MAX_BOX_AREA_PCT = _float("YOLO_MAX_BOX_AREA_PCT", 0.72)
YOLO_MIN_ASPECT = _float("YOLO_MIN_ASPECT", 0.22)
YOLO_MAX_ASPECT = _float("YOLO_MAX_ASPECT", 4.5)
BBOX_SMOOTH_ALPHA = _float("BBOX_SMOOTH_ALPHA", 0.28)

ROLE_CONFIDENCE_SWITCH = _float_env("ROLE_CONFIDENCE_SWITCH", "ROLE_CONF", default=0.25)
ROLE_STABLE_TTL_FRAMES = _int("ROLE_STABLE_TTL_FRAMES", 60)
ROLE_MIN_DISPLAY_CONF = _float_env("ROLE_MIN_DISPLAY_CONF", default=0.45)

# --- Pose ---
POSE_MIN_VISIBILITY = _float("POSE_MIN_VISIBILITY", 0.30)
POSE_SMOOTH_ALPHA = _float("POSE_SMOOTH_ALPHA", 0.30)

# --- Confidence temporal smoothing ---
OBJECT_CONF_SMOOTH_ALPHA = _float("OBJECT_CONF_SMOOTH_ALPHA", 0.35)
ROLE_CONF_SMOOTH_ALPHA = _float("ROLE_CONF_SMOOTH_ALPHA", 0.30)
POSE_CONF_SMOOTH_ALPHA = _float("POSE_CONF_SMOOTH_ALPHA", 0.30)
OVERALL_CONF_SMOOTH_ALPHA = _float("OVERALL_CONF_SMOOTH_ALPHA", 0.25)

# --- Combined reasoning confidence (weighted 0–1) ---
WEIGHT_OBJECT = _float("CONF_WEIGHT_OBJECT", 0.38)
WEIGHT_ROLE = _float("CONF_WEIGHT_ROLE", 0.22)
WEIGHT_POSE = _float("CONF_WEIGHT_POSE", 0.40)

# Alert gates on overall confidence (standalone has no tier gate; these are looser defaults)
ALERT_CONFIDENCE_LOW = _float("ALERT_CONFIDENCE_LOW", 0.25)
ALERT_CONFIDENCE_MEDIUM = _float("ALERT_CONFIDENCE_MEDIUM", 0.35)
ALERT_CONFIDENCE_HIGH = _float("ALERT_CONFIDENCE_HIGH", 0.45)
REASONING_MIN_OBJECT_CONF = _float("REASONING_MIN_OBJECT_CONF", 0.15)

# --- Runtime performance (smooth demo) ---
INFERENCE_EVERY_N_FRAMES = max(1, _int("INFERENCE_EVERY_N_FRAMES", 6))
INFERENCE_EVERY_N_FRAMES_VIDEO = max(1, _int("INFERENCE_EVERY_N_FRAMES_VIDEO", 8))
RUNTIME_QUEUE_SIZE = max(1, _int("RUNTIME_QUEUE_SIZE", 1))
FUSION_TIMEOUT_MS = _int("FUSION_TIMEOUT_MS", 500)
REASONING_INTERVAL_SEC = _float_env("REASONING_INTERVAL_SEC", "REASON_INTERVAL", default=2.5)
HUMAN_READABLE_LOGS = os.environ.get("HUMAN_READABLE_LOGS", "0").strip().lower() in ("1", "true", "yes", "on")
FRAME_SLEEP_SECONDS = _float("FRAME_SLEEP_SECONDS", 0.033)
CAPTURE_MAX_FPS = _float("CAPTURE_MAX_FPS", 24.0)
ALERT_COOLDOWN_SECONDS = _float("ALERT_COOLDOWN_SECONDS", 1.0)
MONITOR_INTENT_COOLDOWN_SECONDS = _float("MONITOR_INTENT_COOLDOWN_SECONDS", 0.8)


def reasoning_enabled() -> bool:
    """LLM reasoning when ENABLE_REASONING=1; otherwise fallback mode."""
    return os.environ.get("ENABLE_REASONING", "0").strip().lower() in ("1", "true", "yes", "on")


def _is_bed_object_label(label: str) -> bool:
    text = (label or "").strip().lower().replace("_", " ")
    return "bed" in text or text in ("patient bed", "hospital bed")


def object_confidence_threshold(label: str) -> float:
    text = (label or "").strip().lower()
    if _is_bed_object_label(text):
        return BED_CONF
    if "person" in text or text in ("person", "patient", "human"):
        return PERSON_CONF
    return OBJECT_CONF


def role_confidence_threshold(role: str) -> float:
    text = (role or "").strip().lower()
    if text == "patient":
        return PATIENT_CONF
    if text == "doctor":
        return DOCTOR_CONF
    if text == "nurse":
        return NURSE_CONF
    return ROLE_CONF


DASHBOARD_POLL_MS = _int("DASHBOARD_POLL_MS", 800)
LLAMA_MAX_TOKENS = _int("LLAMA_MAX_TOKENS", 120)
MONITOR_FRAME_WIDTH = _int("MONITOR_FRAME_WIDTH", 640)
MONITOR_FRAME_HEIGHT = _int("MONITOR_FRAME_HEIGHT", 360)
YOLO_IMG_SIZE = _int("YOLO_IMG_SIZE", 640)
LLM_LAG_STALE_FRAMES = max(1, _int("LLM_LAG_STALE_FRAMES", 120))


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


USE_CUDA = _env_bool("USE_CUDA", True)
YOLO_DEVICE_ENV = os.environ.get("YOLO_DEVICE", "").strip().lower()
LLAMA_DEVICE_ENV = os.environ.get("LLAMA_DEVICE", "auto").strip().lower() or "auto"
LLAMA_N_GPU_LAYERS_ENV = os.environ.get("LLAMA_N_GPU_LAYERS", "auto").strip()
LLAMA_N_GPU_LAYERS_AUTO = max(0, _int("LLAMA_N_GPU_LAYERS_AUTO", 20))


def cuda_available() -> bool:
    if not USE_CUDA:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_yolo_device() -> str:
    """Resolve YOLO inference device from USE_CUDA / YOLO_DEVICE env."""
    explicit = YOLO_DEVICE_ENV
    if explicit == "cpu":
        return "cpu"
    if explicit == "cuda":
        if cuda_available():
            return "cuda"
        return "cpu"
    if cuda_available():
        return "cuda"
    return "cpu"


def scene_environment() -> str:
    """LLM scene context: home_demo or hospital_room."""
    explicit = os.environ.get("SCENE_ENVIRONMENT", "").strip().lower()
    if explicit in ("home_demo", "hospital_room"):
        return explicit
    if os.environ.get("MONITOR_INPUT_SOURCE", "").strip().lower() == "video":
        return "home_demo"
    return "hospital_room"


def expected_bed_present() -> bool:
    """Whether the monitored environment expects a bed (grounds missing-bed alerts)."""
    raw = os.environ.get("EXPECTED_BED_PRESENT", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return scene_environment() == "hospital_room"


def log_runtime_config(*, inference_stride: int = 1, video_mode: bool = False) -> None:
    """Print resolved runtime stride/timing config at worker startup."""
    stride = max(1, int(inference_stride))
    video_stride = INFERENCE_EVERY_N_FRAMES_VIDEO if video_mode else INFERENCE_EVERY_N_FRAMES
    llama_n_ctx = _int("LLAMA_N_CTX", 2048)
    logger.info(
        "[RUNTIME_CONFIG] capture_fps_target=%s inference_every_n_frames=%s "
        "object_every_n_frames=%s role_every_n_frames=%s pose_every_n_frames=%s "
        "reasoning_interval_sec=%s fusion_timeout_ms=%s dashboard_poll_ms=%s "
        "llama_n_ctx=%s llama_max_tokens=%s video_inference_stride=%s",
        CAPTURE_MAX_FPS,
        video_stride if video_mode else INFERENCE_EVERY_N_FRAMES,
        stride,
        stride,
        stride,
        REASONING_INTERVAL_SEC,
        FUSION_TIMEOUT_MS,
        DASHBOARD_POLL_MS,
        llama_n_ctx,
        LLAMA_MAX_TOKENS,
        INFERENCE_EVERY_N_FRAMES_VIDEO,
    )


def log_runtime_thresholds() -> None:
    reasoner = "llama" if reasoning_enabled() else "fallback"
    logger.info(
        "[runtime] conf: object_base=%.2f object=%.2f bed=%.2f person=%.2f "
        "role=%.2f patient=%.2f doctor=%.2f nurse=%.2f pose=%.2f "
        "reason_interval=%.1f reasoner=%s",
        OBJECT_BASE_CONF,
        OBJECT_CONF,
        BED_CONF,
        PERSON_CONF,
        ROLE_CONF,
        PATIENT_CONF,
        DOCTOR_CONF,
        NURSE_CONF,
        POSE_CONF,
        REASONING_INTERVAL_SEC,
        reasoner,
    )


class EmaSmoother:
    """Exponential moving average for stable dashboard confidence."""

    def __init__(self, alpha: float) -> None:
        self.alpha = max(0.05, min(1.0, float(alpha)))
        self._value: Optional[float] = None

    def update(self, value: float) -> float:
        v = max(0.0, min(1.0, float(value)))
        if self._value is None:
            self._value = v
        else:
            self._value = self.alpha * v + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


def reasoning_mode_label(
    run_llama_requested: bool,
    has_patient: bool,
    *,
    llama_available: bool = True,
    llama_used: bool = False,
    llama_backend: str = "",  # noqa: ARG001 — kept for call-site compat; hardware is auto-selected
    loading: bool = False,
) -> str:
    if not has_patient:
        return "idle"
    if loading:
        return "loading"
    if not run_llama_requested:
        return "heuristic"
    if llama_used:
        if llama_backend == "gpu":
            return "llama_gpu"
        return "llama_cpu"
    if not llama_available:
        return "llama_unavailable"
    return "heuristic_fallback"


REASONING_DISPLAY_LABELS = {
    "idle": "Awaiting patient",
    "loading": "Loading reasoning model…",
    "heuristic": "Fallback heuristic (Llama off)",
    "llama": "Llama active",
    "llama_cpu": "Llama active (CPU)",
    "llama_gpu": "Llama active (GPU)",
    "llama_unavailable": "Llama unavailable",
    "heuristic_fallback": "Fallback heuristic (Llama failed)",
}


def reasoning_display_text(mode: str) -> str:
    return REASONING_DISPLAY_LABELS.get(mode, mode.replace("_", " ").title())


def compute_confidence_scores(
    object_conf: float,
    role_conf: float,
    pose_score: float,
    pose_detected: bool,
) -> Dict[str, Any]:
    """Build per-stage and overall confidence (0–100%) for UI and alert gating."""
    obj = max(0.0, min(1.0, float(object_conf or 0.0)))
    role = max(0.0, min(1.0, float(role_conf or 0.0)))
    pose_raw = max(0.0, min(1.0, float(pose_score or 0.0)))
    if pose_detected and pose_raw < POSE_MIN_SCORE:
        pose_raw = max(pose_raw, POSE_MIN_SCORE)
    pose = pose_raw if pose_detected else 0.0

    weight_sum = WEIGHT_OBJECT + WEIGHT_ROLE + WEIGHT_POSE
    overall = (WEIGHT_OBJECT * obj + WEIGHT_ROLE * role + WEIGHT_POSE * pose) / max(weight_sum, 1e-6)

    if obj >= REASONING_MIN_OBJECT_CONF and pose_detected and overall < ALERT_CONFIDENCE_LOW:
        overall = max(overall, ALERT_CONFIDENCE_LOW)

    if overall >= ALERT_CONFIDENCE_HIGH:
        alert_tier = "high"
    elif overall >= ALERT_CONFIDENCE_MEDIUM:
        alert_tier = "medium"
    elif overall >= ALERT_CONFIDENCE_LOW:
        alert_tier = "low"
    else:
        alert_tier = "ignore"

    return {
        "object_pct": round(obj * 100, 1),
        "role_pct": round(role * 100, 1),
        "pose_pct": round(pose * 100, 1),
        "reasoning_pct": round(overall * 100, 1),
        "overall_pct": round(overall * 100, 1),
        "object": round(obj, 3),
        "role": round(role, 3),
        "pose": round(pose, 3),
        "overall": round(overall, 3),
        "alert_tier": alert_tier,
        "pose_detected": pose_detected,
    }


def apply_alert_gate(
    raw_alert: bool,
    intent: str,
    scores: Dict[str, Any],
    *,
    last_alert_ts: float,
    now: float,
) -> Tuple[bool, str, str]:
    """
    Map raw model alert + confidence to presentation-safe alert behavior.

    Returns: (alert_triggered, intent_display, monitor_status)
    """
    tier = scores.get("alert_tier", "ignore")
    overall = float(scores.get("overall") or 0.0)
    obj = float(scores.get("object") or 0.0)
    pose_ok = bool(scores.get("pose_detected"))
    cooldown_elapsed = (now - last_alert_ts) >= ALERT_COOLDOWN_SECONDS

    def _log_suppressed(reason: str) -> None:
        logger.info(
            "alert_gate: SUPPRESSED reason=%s raw_alert=%s tier=%s overall=%.3f "
            "obj=%.3f pose_ok=%s cooldown_elapsed=%s intent=%r",
            reason,
            raw_alert,
            tier,
            overall,
            obj,
            pose_ok,
            cooldown_elapsed,
            intent,
        )

    if tier == "ignore" and obj >= REASONING_MIN_OBJECT_CONF and pose_ok:
        _log_suppressed("tier_ignore_with_detection")
        return False, intent or "Patient under observation", "monitor"

    if tier == "ignore" or overall < ALERT_CONFIDENCE_LOW:
        if obj >= REASONING_MIN_OBJECT_CONF:
            _log_suppressed("tier_ignore_or_low_overall_has_object")
            return False, intent or "Patient under observation", "monitor"
        _log_suppressed("tier_ignore_or_low_overall_scanning")
        return False, "Scanning — awaiting patient", "idle"

    if tier == "low":
        _log_suppressed("tier_low")
        return False, intent or "Patient under observation", "monitor"

    # medium/high: emit when model raised raw_alert (closer to standalone safety_label=ALERT)
    if raw_alert and tier in ("medium", "high") and cooldown_elapsed:
        logger.info(
            "alert_gate: ALERT emitted tier=%s overall=%.3f intent=%r",
            tier,
            overall,
            intent,
        )
        return True, intent, "alert"

    if tier == "medium":
        if raw_alert:
            _log_suppressed("medium_raw_alert_cooldown" if not cooldown_elapsed else "medium_raw_alert_other")
        else:
            _log_suppressed("medium_no_raw_alert")
        return False, intent or "Monitoring — possible activity", "monitor"

    if raw_alert and not cooldown_elapsed:
        _log_suppressed("high_tier_cooldown")
        return False, intent or "Monitoring — alert cooling down", "monitor"

    if raw_alert:
        _log_suppressed("high_tier_other")
    else:
        _log_suppressed("high_tier_no_raw_alert")
    return False, intent or "Patient under observation", "stable"

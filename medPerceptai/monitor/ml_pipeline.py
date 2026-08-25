from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from monitor import llama_runtime, presentation_config as pc
from monitor.runtime.tensor_utils import coerce_landmarks
from monitor.presentation_config import compute_confidence_scores, EmaSmoother, reasoning_mode_label
from monitor.runtime.fusion import _is_bed_label, normalize_object_label

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_model_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


DEFAULT_OBJECT_MODEL_PATH = PROJECT_ROOT / "model_weights" / "obj.pt"
DEFAULT_ROLE_MODEL_PATH = PROJECT_ROOT / "model_weights" / "roles.pt"
DEFAULT_YOLO_POSE_MODEL_PATH = PROJECT_ROOT / "model_weights" / "yolov8n-pose.pt"
DEFAULT_REASONING_MODEL_PATH = PROJECT_ROOT / "model_weights" / "llama-3-8b-instruct.Q4_K_M.gguf"
DEFAULT_FALLBACK_VIDEO = PROJECT_ROOT / "media" / "test_video.mp4"

# Ultralytics YOLO pose uses COCO-17 keypoints.
POSE_CONNECTIONS = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
]


class PatientIntentPipeline:
    """YOLO object/role/pose + local reasoning pipeline for patient intent detection."""

    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(
        self,
        weights_path: Optional[str] = None,
        role_weights_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        reasoning_model_path: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
    ) -> None:
        self.device = pc.resolve_yolo_device()
        self._yolo_predict_device: Union[int, str] = 0 if self.device == "cuda" else "cpu"
        object_weights = weights_path or os.environ.get("YOLO_OBJECT_MODEL_PATH")
        role_weights = role_weights_path or os.environ.get("YOLO_ROLE_MODEL_PATH")
        self.weights_path = (
            _resolve_model_path(object_weights) if object_weights else DEFAULT_OBJECT_MODEL_PATH
        )
        self.role_weights_path = (
            _resolve_model_path(role_weights) if role_weights else DEFAULT_ROLE_MODEL_PATH
        )
        self.pose_model_path = _resolve_model_path(
            pose_model_path or os.environ.get("YOLO_POSE_MODEL_PATH", str(DEFAULT_YOLO_POSE_MODEL_PATH))
        )
        self.reasoning_model_path = _resolve_model_path(
            reasoning_model_path or os.environ.get("REASONING_MODEL_PATH", str(DEFAULT_REASONING_MODEL_PATH))
        )
        self.vlm_model_name = vlm_model_name or os.environ.get("VLM_MODEL_NAME")
        self.enable_vlm = os.environ.get("ENABLE_VLM", "0") == "1" or bool(self.vlm_model_name)
        self.enable_reasoning = pc.reasoning_enabled()
        self._last_reasoning_trace: Dict[str, Any] = {}
        self._last_llama_invoked = False
        self._last_llama_runtime_error: Optional[str] = None
        self._last_llama_latency_ms: Optional[int] = None
        self.min_confidence = pc.YOLO_CONFIDENCE
        self.iou_threshold = pc.YOLO_IOU
        self.max_detections = pc.YOLO_MAX_DETECTIONS

        self._stable_role = "unknown"
        self._stable_role_conf = 0.0
        self._smoothed_pose_score: Optional[float] = None
        self._smooth_box_state: Optional[Tuple[int, int, int, int]] = None
        self._obj_conf_smoother = EmaSmoother(pc.OBJECT_CONF_SMOOTH_ALPHA)
        self._role_conf_smoother = EmaSmoother(pc.ROLE_CONF_SMOOTH_ALPHA)
        self._pose_conf_smoother = EmaSmoother(pc.POSE_CONF_SMOOTH_ALPHA)
        self._overall_conf_smoother = EmaSmoother(pc.OVERALL_CONF_SMOOTH_ALPHA)
        self._last_llm_result: Any = None
        self._stable_role_frame_id = 0

        self.model_status: Dict[str, str] = {
            "object_detection": "pending",
            "role_classification": "pending",
            "pose_estimation": "pending",
            "intent_reasoning": "pending",
        }

        self.yolo_model: Optional[YOLO] = None
        self.role_model: Optional[YOLO] = None
        self.pose_model: Optional[YOLO] = None
        self.vlm_pipeline = None
        self.reasoning_pipeline = None
        self._reasoning_hw = llama_runtime.detect_hardware(probe_llama=False)
        self.reasoning_health: Dict[str, Any] = llama_runtime.build_health(
            model_path=self.reasoning_model_path,
            profile=self._reasoning_hw,
        )
        if not self._reasoning_hw.llama_cpp_installed:
            self.model_status["intent_reasoning"] = "llama_unavailable"
        self._reasoning_load_lock = threading.Lock()
        self._reasoning_load_started = False
        self._llama_gpu_disabled_session = False
        self._object_lock = threading.Lock()
        self._role_lock = threading.Lock()
        self._pose_lock = threading.Lock()
        self._reasoning_lock = threading.Lock()

        try:
            self.yolo_model = self._place_yolo_on_device(self._load_yolo_model(), "object")
            self.model_status["object_detection"] = "ready" if self.yolo_model is not None else "unavailable"
        except Exception as exc:
            logger.exception("Object detection model failed to initialize: %s", exc)
            self.model_status["object_detection"] = "error"

        try:
            self.role_model = self._place_yolo_on_device(self._load_role_model(), "role")
            self.model_status["role_classification"] = "ready" if self.role_model is not None else "unavailable"
        except Exception as exc:
            logger.exception("Role classification model failed to initialize: %s", exc)
            self.model_status["role_classification"] = "error"

        try:
            self.pose_model = self._place_yolo_on_device(self._load_pose_model(), "pose")
            self.model_status["pose_estimation"] = "ready" if self.pose_model is not None else "unavailable"
        except Exception as exc:
            logger.exception("Pose estimation model failed to initialize: %s", exc)
            self.pose_model = None
            self.model_status["pose_estimation"] = "error"

        if self.model_status.get("intent_reasoning") != "llama_unavailable":
            self.model_status["intent_reasoning"] = self.reasoning_health.get("display_mode", "pending")

        logger.info("Pipeline model status: %s", self.model_status)
        self._log_device_startup()
        pc.log_runtime_thresholds()
        gguf_path = llama_runtime.resolve_gguf_path(self.reasoning_model_path)
        logger.info(
            "[LLAMA] startup config ENABLE_REASONING=%s env=%s model_path=%s exists=%s "
            "resolved_gguf=%s llama_cpp_installed=%s (lazy load — not imported at startup)",
            self.enable_reasoning,
            os.environ.get("ENABLE_REASONING", "0"),
            self.reasoning_model_path,
            self.reasoning_model_path.exists(),
            gguf_path,
            self._reasoning_hw.llama_cpp_installed,
        )
        llama_runtime.log_startup_verification(self.reasoning_model_path, self.reasoning_health)
        self._log_llama_runtime_status()

    def _place_yolo_on_device(self, model: Optional[YOLO], label: str) -> Optional[YOLO]:
        if model is None:
            return None
        if self.device != "cuda":
            return model
        try:
            model.to("cuda")
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to("cuda")
        except Exception as exc:
            logger.error(
                "[YOLO_DEVICE] %s failed to move to CUDA: %s",
                label,
                exc,
                exc_info=True,
            )
        return model

    @staticmethod
    def _yolo_runtime_device(model: Optional[YOLO]) -> str:
        if model is None:
            return "unavailable"
        try:
            inner = getattr(model, "model", None)
            if inner is not None:
                return str(next(inner.parameters()).device)
        except StopIteration:
            pass
        except Exception:
            pass
        return "cpu"

    def _log_device_startup(self) -> None:
        obj_dev = self._yolo_runtime_device(self.yolo_model)
        role_dev = self._yolo_runtime_device(self.role_model)
        pose_dev = self._yolo_runtime_device(self.pose_model)
        llama_runtime.log_runtime_environment(
            yolo_object=obj_dev,
            yolo_role=role_dev,
            yolo_pose=pose_dev,
        )
        logger.info("[YOLO] predict_device=%s configured=%s", self._yolo_predict_device, self.device)
        if pc.USE_CUDA and not torch.cuda.is_available():
            logger.warning("[YOLO] USE_CUDA=1 but torch.cuda.is_available()=False — using CPU")
        elif self.device == "cuda":
            for name, dev in (("object", obj_dev), ("role", role_dev), ("pose", pose_dev)):
                if dev == "unavailable":
                    continue
                if "cuda" not in str(dev):
                    logger.warning(
                        "[YOLO_DEVICE] %s model is on %s but YOLO_DEVICE=cuda was requested",
                        name,
                        dev,
                    )

    def _load_yolo_model(self) -> Optional[YOLO]:
        if not self.weights_path.exists():
            logger.warning("Object detection model missing: %s", self.weights_path)
            return None
        try:
            return YOLO(str(self.weights_path))
        except Exception as exc:
            logger.error("Object detection model could not be loaded (%s): %s", self.weights_path, exc)
            return None

    def _load_role_model(self) -> Optional[YOLO]:
        if self.role_weights_path.exists():
            try:
                return YOLO(str(self.role_weights_path))
            except Exception as exc:
                logger.warning("Role classification model could not be loaded (%s): %s", self.role_weights_path, exc)
                return None

        logger.warning("Role classification model missing: %s", self.role_weights_path)
        return None

    def _load_pose_model(self) -> Optional[YOLO]:
        if not self.pose_model_path.exists():
            logger.warning("YOLO pose model missing: %s", self.pose_model_path)
            return None
        try:
            return YOLO(str(self.pose_model_path))
        except Exception as exc:
            logger.warning("YOLO pose model could not be loaded (%s): %s", self.pose_model_path, exc)
            return None

    def _load_vlm_pipeline(self):
        if not self.enable_vlm:
            return None

        if hf_pipeline is None:
            logger.warning("transformers is unavailable; VLM inference is disabled.")
            return None

        model_name = self.vlm_model_name or "Salesforce/blip-image-captioning-base"
        device = 0 if torch.cuda.is_available() else -1
        try:
            return hf_pipeline(
                task="image-to-text",
                model=model_name,
                framework="pt",
                device=device,
            )
        except Exception as exc:
            logger.warning("VLM pipeline could not be loaded (%s): %s", model_name, exc)
            return None

    def _reasoning_available(self) -> bool:
        return llama_runtime.resolve_gguf_path(self.reasoning_model_path) is not None

    def _log_llama_runtime_status(self) -> None:
        loaded = self.reasoning_pipeline is not None
        logger.info(
            "[LLAMA] ENABLE_REASONING=%s model_path=%s exists=%s loaded=%s",
            os.environ.get("ENABLE_REASONING", "0"),
            self.reasoning_model_path,
            self.reasoning_model_path.exists(),
            loaded,
        )

    def reset_capture_state(self) -> None:
        """Clear cached LLM result and YOLO role stability on new capture session."""
        self._stable_role = "unknown"
        self._stable_role_conf = 0.0
        self._stable_role_frame_id = 0
        self._last_llm_result = None

    def _should_run_llama(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """ENABLE_REASONING=1 enables Llama; session/config may opt out explicitly."""
        self.enable_reasoning = pc.reasoning_enabled()
        if not self.enable_reasoning:
            return False
        if not llama_runtime.is_llama_cpp_installed() or not self._reasoning_available():
            return False
        if config is not None and config.get("enable_ai_reasoning") is False:
            return False
        if config is not None and "enable_ai_reasoning" in config:
            return bool(config.get("enable_ai_reasoning"))
        return True

    def _update_reasoning_health(
        self,
        *,
        model_loaded: bool = False,
        load_error: Optional[str] = None,
        loading: bool = False,
    ) -> None:
        self.reasoning_health = llama_runtime.build_health(
            model_path=self.reasoning_model_path,
            profile=self._reasoning_hw,
            model_loaded=model_loaded,
            load_error=load_error,
            loading=loading,
        )
        self.model_status["intent_reasoning"] = self.reasoning_health.get("display_mode", "llama_unavailable")

    def _startup_load_reasoning(self) -> None:
        with self._reasoning_load_lock:
            if self._reasoning_load_started:
                return
            self._reasoning_load_started = True
        self._update_reasoning_health(loading=True)
        logger.info("Llama startup load: beginning automatic initialization")
        pipeline = self._get_reasoning_pipeline()
        if pipeline is not None:
            logger.info(
                "Llama startup load: SUCCESS backend=%s n_gpu_layers=%s",
                self._reasoning_hw.backend,
                self._reasoning_hw.n_gpu_layers,
            )
        else:
            logger.error(
                "Llama startup load: FAILED — %s",
                self.reasoning_health.get("load_error") or "unknown error",
            )
        self._log_llama_runtime_status()

    def probe_reasoning_load(self) -> bool:
        """Eager-load Llama for warmup (idempotent, single load)."""
        if not self._reasoning_available():
            return False
        return self._get_reasoning_pipeline() is not None

    def _disable_llama_gpu_for_session(self, *, error: str, frame_id: int) -> None:
        """Drop GPU Llama after a CUDA/ggml failure; subsequent loads use CPU only."""
        logger.error("[LLAMA_CUDA_ERROR] frame_id=%s error=%s", frame_id, error)
        self._llama_gpu_disabled_session = True
        with self._reasoning_load_lock:
            self.reasoning_pipeline = None
        self._reasoning_hw.n_gpu_layers = 0
        self._reasoning_hw.backend = "cpu"
        self._reasoning_hw.selection_reason = "Session GPU disabled after CUDA inference error"
        llama_runtime.log_llama_device(self._reasoning_hw)

    def _load_reasoning_pipeline(self):
        Llama = llama_runtime.get_llama_class()
        if Llama is None:
            err = "llama-cpp-python not installed"
            self._update_reasoning_health(load_error=err)
            self.model_status["intent_reasoning"] = "llama_unavailable"
            return None

        model_path = llama_runtime.resolve_gguf_path(self.reasoning_model_path)
        if model_path is None:
            err = f"GGUF not found: {self.reasoning_model_path}"
            logger.error("Llama load failed: %s", err)
            self._update_reasoning_health(load_error=err)
            self.model_status["intent_reasoning"] = "llama_unavailable"
            return None

        self._reasoning_hw = llama_runtime.detect_hardware(probe_llama=True)
        if self._llama_gpu_disabled_session:
            self._reasoning_hw.n_gpu_layers = 0
            self._reasoning_hw.backend = "cpu"
            self._reasoning_hw.selection_reason = "Session GPU disabled after CUDA inference error"
        n_ctx = int(os.environ.get("LLAMA_N_CTX", "2048"))
        n_batch = int(os.environ.get("LLAMA_N_BATCH", "256"))
        n_threads = int(os.environ.get("LLAMA_N_THREADS", "4"))
        load_attempts = llama_runtime.llama_load_attempts(
            profile=self._reasoning_hw,
            model_path=model_path,
        )
        if self._llama_gpu_disabled_session:
            load_attempts = [(layers, mmap) for layers, mmap in load_attempts if layers == 0]

        last_error: Optional[str] = None
        for n_gpu_layers, use_mmap in load_attempts:
            logger.info(
                "Loading Llama GGUF: path=%s n_gpu_layers=%s use_mmap=%s (%s)",
                model_path,
                n_gpu_layers,
                use_mmap,
                self._reasoning_hw.selection_reason,
            )
            try:
                pipeline = Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_batch=n_batch,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    chat_format="llama-3",
                    use_mmap=use_mmap,
                    use_mlock=False,
                    verbose=False,
                )
                self._reasoning_hw.n_gpu_layers = n_gpu_layers
                self._reasoning_hw.backend = "gpu" if n_gpu_layers != 0 else "cpu"
                logger.info(
                    "Llama model loaded: path=%s n_gpu_layers=%s use_mmap=%s backend=%s",
                    model_path,
                    n_gpu_layers,
                    use_mmap,
                    self._reasoning_hw.backend,
                )
                llama_runtime.log_llama_device(self._reasoning_hw)
                self._update_reasoning_health(model_loaded=True)
                return pipeline
            except OSError as exc:
                last_error = str(exc)
                win_mmap = "WinError 1" in last_error or "Incorrect function" in last_error
                if win_mmap and use_mmap:
                    logger.warning(
                        "Llama mmap load failed on Windows (%s) — retrying with use_mmap=False",
                        exc,
                    )
                elif n_gpu_layers != 0:
                    logger.warning(
                        "[LLaMA_DEVICE] GPU load failed (n_gpu_layers=%s use_mmap=%s) — "
                        "llama-cpp-python may lack CUDA support; retrying CPU. Error: %s",
                        n_gpu_layers,
                        use_mmap,
                        exc,
                    )
                else:
                    logger.warning(
                        "Llama load failed n_gpu_layers=%s use_mmap=%s (%s): %s",
                        n_gpu_layers,
                        use_mmap,
                        model_path,
                        exc,
                    )
            except Exception as exc:
                last_error = str(exc)
                if n_gpu_layers != 0:
                    logger.warning(
                        "[LLaMA_DEVICE] GPU load failed (n_gpu_layers=%s use_mmap=%s) — "
                        "llama-cpp-python may lack CUDA support; retrying CPU. Error: %s",
                        n_gpu_layers,
                        use_mmap,
                        exc,
                    )
                else:
                    logger.warning(
                        "Llama load failed n_gpu_layers=%s use_mmap=%s (%s): %s",
                        n_gpu_layers,
                        use_mmap,
                        model_path,
                        exc,
                    )

        logger.error("Llama load failed after all attempts (%s): %s", model_path, last_error)
        self._update_reasoning_health(load_error=last_error)
        return None

    def _get_vlm_pipeline(self):
        if self.vlm_pipeline is None:
            self.vlm_pipeline = self._load_vlm_pipeline()
        return self.vlm_pipeline

    def _get_reasoning_pipeline(self):
        if self.reasoning_pipeline is not None:
            return self.reasoning_pipeline
        with self._reasoning_load_lock:
            if self.reasoning_pipeline is not None:
                return self.reasoning_pipeline
            self._update_reasoning_health(loading=True)
            self.reasoning_pipeline = self._load_reasoning_pipeline()
        return self.reasoning_pipeline

    @staticmethod
    def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _safe_caption_text(caption_result: Any) -> str:
        if isinstance(caption_result, list) and caption_result:
            first = caption_result[0]
            if isinstance(first, dict):
                return str(first.get("generated_text") or first.get("caption") or "")
            return str(first)
        if isinstance(caption_result, dict):
            return str(caption_result.get("generated_text") or caption_result.get("caption") or "")
        return str(caption_result or "")

    @staticmethod
    def _safe_text_generation(output: Any) -> str:
        if isinstance(output, dict) and isinstance(output.get("choices"), list) and output["choices"]:
            first_choice = output["choices"][0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message") or {}
                if isinstance(message, dict):
                    content = message.get("content")
                    if content:
                        return str(content)
                text = first_choice.get("text")
                if text:
                    return str(text)
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                return str(first.get("generated_text") or first.get("text") or "")
            return str(first)
        if isinstance(output, dict):
            return str(output.get("generated_text") or output.get("text") or "")
        return str(output or "")

    @staticmethod
    def _landmark_xy(landmarks: Any, idx: int) -> Tuple[float, float]:
        try:
            pt = landmarks[idx]
            x = float(pt[0].item() if hasattr(pt[0], "item") else pt[0])
            y = float(pt[1].item() if hasattr(pt[1], "item") else pt[1])
            return x, y
        except Exception:
            return 0.0, 0.0

    def _classify_pose(
        self,
        landmarks: Any,
        pose_box: Tuple[int, int, int, int],
        *,
        frame_id: Optional[int] = None,
        confidence: float = 0.0,
        person_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        x1, y1, x2, y2 = pose_box
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        aspect_ratio = bbox_w / bbox_h

        valid = 0
        for idx in range(min(len(landmarks), 17)):
            lx, ly = self._landmark_xy(landmarks, idx)
            if lx > 1.0 and ly > 1.0:
                valid += 1
        pose_score = valid / 17.0

        shoulder_y = hip_y = knee_y = ankle_y = 0.0
        torso_angle = 0.0
        if len(landmarks) > self.RIGHT_ANKLE:
            lsx, lsy = self._landmark_xy(landmarks, self.LEFT_SHOULDER)
            rsx, rsy = self._landmark_xy(landmarks, self.RIGHT_SHOULDER)
            lhx, lhy = self._landmark_xy(landmarks, self.LEFT_HIP)
            rhx, rhy = self._landmark_xy(landmarks, self.RIGHT_HIP)
            lkx, lky = self._landmark_xy(landmarks, self.LEFT_KNEE)
            rkx, rky = self._landmark_xy(landmarks, self.RIGHT_KNEE)
            lax, lay = self._landmark_xy(landmarks, self.LEFT_ANKLE)
            rax, ray = self._landmark_xy(landmarks, self.RIGHT_ANKLE)
            shoulder_y = (lsy + rsy) / 2.0
            hip_y = (lhy + rhy) / 2.0
            knee_y = (lky + rky) / 2.0
            ankle_y = (lay + ray) / 2.0
            shoulder_x = (lsx + rsx) / 2.0
            hip_x = (lhx + rhx) / 2.0
            torso_angle = math.degrees(
                math.atan2(abs(hip_y - shoulder_y), max(abs(hip_x - shoulder_x), 1.0))
            )

        kp_inside_ratio = 0.0
        if person_box is not None:
            px1, py1, px2, py2 = person_box
            inside = 0
            visible = 0
            for idx in range(min(len(landmarks), 17)):
                lx, ly = self._landmark_xy(landmarks, idx)
                if lx <= 1.0 or ly <= 1.0:
                    continue
                visible += 1
                if px1 <= lx <= px2 and py1 <= ly <= py2:
                    inside += 1
            kp_inside_ratio = inside / visible if visible else 0.0

        features = {
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "aspect_ratio": round(aspect_ratio, 3),
            "shoulder_y": round(shoulder_y, 1),
            "hip_y": round(hip_y, 1),
            "knee_y": round(knee_y, 1),
            "ankle_y": round(ankle_y, 1),
            "torso_angle": round(torso_angle, 1),
            "keypoints_inside_ratio": round(kp_inside_ratio, 3),
            "pose_score": round(pose_score, 3),
        }
        logger.info(
            "[POSE][FEATURES] frame_id=%s bbox_w=%s bbox_h=%s aspect_ratio=%.3f shoulder_y=%.1f "
            "hip_y=%.1f knee_y=%.1f ankle_y=%.1f torso_angle=%.1f keypoints_inside_ratio=%.3f",
            frame_id if frame_id is not None else "?",
            bbox_w,
            bbox_h,
            aspect_ratio,
            shoulder_y,
            hip_y,
            knee_y,
            ankle_y,
            torso_angle,
            kp_inside_ratio,
        )

        weak = (
            pose_score < pc.POSE_CONF
            and confidence < pc.POSE_CONF
        ) or valid < 5
        if weak:
            reason = f"weak keypoints/conf (score={pose_score:.2f} conf={confidence:.2f} valid={valid})"
            logger.info("[POSE][CLASSIFY] frame_id=%s pose=unknown reason=%s", frame_id, reason)
            return "unknown", reason, features

        vertical_stack = (
            shoulder_y > 0
            and hip_y > shoulder_y + 8
            and knee_y > hip_y + 8
            and ankle_y > knee_y + 8
        )
        tall_bbox = bbox_h > bbox_w * 1.05
        wide_bbox = bbox_w > bbox_h * 1.1
        horizontal_torso = torso_angle < 35.0 and abs(hip_y - shoulder_y) < bbox_h * 0.2

        if tall_bbox and (vertical_stack or bbox_h > bbox_w * 1.25):
            reason = "vertical bbox with stacked shoulders→ankles"
            logger.info("[POSE][CLASSIFY] frame_id=%s pose=standing reason=%s", frame_id, reason)
            return "standing", reason, features
        if vertical_stack and bbox_h >= bbox_w:
            reason = "shoulders/hips/knees/ankles vertically stacked"
            logger.info("[POSE][CLASSIFY] frame_id=%s pose=standing reason=%s", frame_id, reason)
            return "standing", reason, features
        if wide_bbox or horizontal_torso:
            reason = "wide bbox or near-horizontal torso"
            logger.info("[POSE][CLASSIFY] frame_id=%s pose=lying reason=%s", frame_id, reason)
            return "lying", reason, features
        if aspect_ratio < 0.85 and pose_score >= pc.POSE_CONF:
            reason = "tall aspect with sufficient keypoints"
            logger.info("[POSE][CLASSIFY] frame_id=%s pose=standing reason=%s", frame_id, reason)
            return "standing", reason, features
        if 0.85 <= aspect_ratio <= 1.1:
            reason = "seated-like aspect ratio"
            logger.info("[POSE][CLASSIFY] frame_id=%s pose=sitting reason=%s", frame_id, reason)
            return "sitting", reason, features

        reason = "ambiguous geometry"
        logger.info("[POSE][CLASSIFY] frame_id=%s pose=unknown reason=%s", frame_id, reason)
        return "unknown", reason, features

    def _summarize_pose(
        self,
        pose_landmarks: Optional[Any],
        *,
        pose_box: Optional[Tuple[int, int, int, int]] = None,
        frame_id: Optional[int] = None,
        confidence: float = 0.0,
        person_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        if pose_landmarks is None:
            return {"available": False, "pose_detected": False, "pose_class": "unknown"}

        landmarks = pose_landmarks
        valid = 0
        for idx in range(min(len(landmarks), 17)):
            lx, ly = self._landmark_xy(landmarks, idx)
            if lx > 1.0 and ly > 1.0:
                valid += 1
        pose_score = max(0.0, min(1.0, valid / 17.0))
        pose_detected = pose_score > 0.05 and confidence >= pc.POSE_CONF * 0.5

        box = pose_box or (0, 0, 0, 0)
        if box == (0, 0, 0, 0):
            xs, ys = [], []
            for idx in range(min(len(landmarks), 17)):
                lx, ly = self._landmark_xy(landmarks, idx)
                if lx > 1.0 and ly > 1.0:
                    xs.append(lx)
                    ys.append(ly)
            if xs and ys:
                box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        pose_class, classify_reason, features = self._classify_pose(
            landmarks,
            box,
            frame_id=frame_id,
            confidence=confidence,
            person_box=person_box,
        )
        standing_hint = pose_class == "standing"
        fall_hint = pose_class == "fall_risk"

        return {
            "available": pose_detected and pose_class != "unknown",
            "pose_detected": pose_detected and pose_class != "unknown",
            "pose_score": round(pose_score, 3),
            "pose_class": pose_class,
            "classify_reason": classify_reason,
            "standing_hint": standing_hint,
            "fall_hint": fall_hint,
            "torso_length": features.get("torso_angle", 0.0),
            "shoulder_y": features.get("shoulder_y", 0.0),
            "hip_y": features.get("hip_y", 0.0),
            "knee_y": features.get("knee_y", 0.0),
            "ankle_y": features.get("ankle_y", 0.0),
            "aspect_ratio": features.get("aspect_ratio", 0.0),
            "keypoints_inside_ratio": features.get("keypoints_inside_ratio", 0.0),
        }

    def _is_valid_person_box(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
    ) -> bool:
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw < 8 or bh < 8:
            return False
        area_pct = (bw * bh) / max(width * height, 1)
        if area_pct < pc.YOLO_MIN_BOX_AREA_PCT or area_pct > pc.YOLO_MAX_BOX_AREA_PCT:
            return False
        aspect = bw / max(bh, 1)
        return pc.YOLO_MIN_ASPECT <= aspect <= pc.YOLO_MAX_ASPECT

    def _smooth_box(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if self._smooth_box_state is None:
            self._smooth_box_state = box
            return box
        alpha = pc.BBOX_SMOOTH_ALPHA
        smoothed = tuple(
            int(alpha * current + (1.0 - alpha) * previous)
            for current, previous in zip(box, self._smooth_box_state)
        )
        self._smooth_box_state = smoothed
        return smoothed

    @staticmethod
    def _sanitize_llama_narrative(
        reason: str,
        summary: str,
        scene: Dict[str, Any],
    ) -> Tuple[str, str]:
        staff_presence = str(scene.get("staff_presence") or "unknown")
        bed_relation = str(scene.get("bed_relation") or "unknown")
        missing = list(scene.get("missing_signals") or [])
        patient_status = str(scene.get("patient_status") or "unknown")
        expected_bed = pc.expected_bed_present()
        bed_unconfirmed = expected_bed and (bed_relation == "unknown" or "bed" in missing)
        bed_safe_text = (
            "Person is lying. Bed was not detected, so bed relation is unconfirmed."
            if patient_status == "lying"
            else "Bed was not detected, so bed relation is unconfirmed."
        )

        def clean(text: str) -> str:
            cleaned = (text or "").strip()
            if not cleaned:
                return cleaned
            lower = cleaned.lower()
            if staff_presence in ("no_staff", "staff_not_detected", "unknown_staff", "staff_detected_not_near"):
                staff_phrases = (
                    "staff is nearby",
                    "staff nearby",
                    "staff are nearby",
                    "medical staff is nearby",
                    "medical staff nearby",
                    "nurse is nearby",
                    "nurse nearby",
                    "caregiver is nearby",
                )
                if any(p in lower for p in staff_phrases):
                    cleaned = "Patient under observation; no staff detected in scene."
            if bed_unconfirmed:
                bed_phrases = (
                    "on bed",
                    "in bed",
                    "on the bed",
                    "in the bed",
                    "bed confirmed",
                    "bed sensor",
                    "resting in bed",
                    "lying in bed",
                    "lying on bed",
                    "on the floor",
                    "on floor",
                    "floor fall",
                    "away from bed",
                    "off the bed",
                )
                if any(p in lower for p in bed_phrases):
                    cleaned = bed_safe_text
            return cleaned.strip()

        return clean(reason), clean(summary)

    _LLM_JSON_SCHEMA_EXAMPLE = (
        '{"patient_status":"unknown","staff_presence":"unknown",'
        '"safety_label":"MONITOR","alert_type":"unknown","risk_level":"unknown",'
        '"risk_score":0,"reason":"short reason","summary":"short summary"}'
    )

    @staticmethod
    def _build_scene_reasoning_user_message(scene: Dict[str, Any], *, frame_id: Optional[int] = None) -> str:
        from monitor.runtime.display_fields import build_llm_scene_payload

        fid = int(frame_id if frame_id is not None else scene.get("frame_id") or 0)
        payload = build_llm_scene_payload(scene, frame_id=fid)
        return (
            "Patient safety monitor. Scene JSON:\n"
            f"{json.dumps(payload, default=str, separators=(',', ':'))}\n"
            "Reply with ONLY one valid compact JSON object. No markdown. No explanation. No text outside JSON.\n"
            "Use EXACTLY these keys and enum values:\n"
            "patient_status: patient_on_bed|patient_standing|patient_sitting|patient_lying_off_bed|not_detected|unknown\n"
            "staff_presence: staff_nearby|no_staff_nearby|unknown\n"
            "safety_label: SAFE|ALERT|MONITOR\n"
            "alert_type: no_alert|possible_fall_risk|fall_detected|unknown\n"
            "risk_level: no_risk|low|medium|high|unknown\n"
            "risk_score: integer 0-100\n"
            "reason: max 12 words\n"
            "summary: max 12 words\n"
            f"Example shape: {PatientIntentPipeline._LLM_JSON_SCHEMA_EXAMPLE}\n"
            "Definitions:\n"
            "- Doctor and Nurse are staff roles.\n"
            "- Patient is the monitored subject.\n"
            "- on_bed means the patient is located on a bed.\n"
            "- near means a person is close to another person or object.\n"
            "Decide all output fields from the scene JSON only. Start with { end with }."
        )

    @staticmethod
    def _build_fallback_structured(
        scene: Dict[str, Any],
        reason: str,
        *,
        no_person: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        from monitor import presentation_config as _pc

        scene_type = _pc.scene_environment()
        code = fallback_reason or reason
        if no_person:
            return {
                "scene_type": scene_type,
                "patient_status": "not_detected",
                "staff_presence": "unknown",
                "staff_activity": "not provided",
                "equipment_context": "not provided",
                "safety_label": "MONITOR",
                "alert_type": "unknown",
                "risk_level": "unknown",
                "risk_score": 0,
                "reason": f"Fallback: no person detected — {reason}",
                "summary": "Fallback: no person detected in scene",
                "decision_source": "fallback",
                "fallback_reason": code,
            }
        return {
            "scene_type": scene_type,
            "patient_status": "unknown",
            "staff_presence": "unknown",
            "staff_activity": "not provided",
            "equipment_context": "not provided",
            "safety_label": "MONITOR",
            "alert_type": "unknown",
            "risk_level": "unknown",
            "risk_score": 0,
            "reason": f"Fallback: {reason}",
            "summary": f"Fallback: {reason}",
            "decision_source": "fallback",
            "fallback_reason": code,
        }

    @staticmethod
    def _safety_label_is_alert(safety_label: Any) -> bool:
        return str(safety_label or "").strip().upper() == "ALERT"

    @staticmethod
    def _safety_label_is_warning(safety_label: Any, risk_level: Any) -> bool:
        label = str(safety_label or "").strip().upper()
        level = str(risk_level or "").strip().lower()
        return label in ("MONITOR", "WARNING") or level == "warning"

    def _normalize_llama_decision(self, parsed: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
        """LLM fields are authoritative — do not merge rule-engine defaults."""
        reason = str(parsed.get("reason") or "").strip()
        summary = str(parsed.get("summary") or reason or "").strip()
        if not reason and summary:
            reason = summary
        if not summary and reason:
            summary = reason
        reason, summary = self._sanitize_llama_narrative(reason, summary, scene)

        risk_score = parsed.get("risk_score")
        if risk_score is not None:
            try:
                risk_score = float(risk_score)
            except (TypeError, ValueError):
                risk_score = None

        from monitor import presentation_config as _pc

        def _short(text: str, limit: int = 120) -> str:
            words = str(text or "").split()
            if len(words) > 15:
                return " ".join(words[:15])
            return str(text or "").strip()[:limit]

        merged: Dict[str, Any] = {
            "scene_type": str(parsed.get("scene_type") or _pc.scene_environment()).strip(),
            "patient_status": str(parsed.get("patient_status") or "unknown").strip().lower(),
            "staff_presence": str(parsed.get("staff_presence") or "unknown").strip().lower(),
            "staff_activity": str(parsed.get("staff_activity") or "not provided").strip(),
            "equipment_context": str(parsed.get("equipment_context") or "not provided").strip(),
            "safety_label": str(parsed.get("safety_label") or "MONITOR").strip().upper(),
            "alert_type": str(parsed.get("alert_type") or "unknown").strip().lower(),
            "risk_level": str(parsed.get("risk_level") or "unknown").strip().lower(),
            "risk_score": risk_score if risk_score is not None else 0,
            "reason": _short(reason or "not provided"),
            "summary": _short(summary or reason or "not provided"),
            "decision_source": "llama",
        }
        return merged

    def _extract_structured_fields(self, text: str) -> Dict[str, Any]:
        """Best-effort field extraction when the model returns near-JSON text."""
        fields: Dict[str, Any] = {}
        string_keys = (
            "scene_type",
            "patient_status",
            "staff_presence",
            "staff_activity",
            "equipment_context",
            "safety_label",
            "alert_type",
            "risk_level",
            "reason",
            "summary",
        )
        for key in string_keys:
            pattern = rf'"{key}"\s*:\s*"([^"]+)"'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                fields[key] = match.group(1).strip()
        score_match = re.search(r'"risk_score"\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if score_match:
            fields["risk_score"] = score_match.group(1)
        return fields

    @staticmethod
    def _extract_json_blob(text: str) -> str:
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence:
            cleaned = fence.group(1).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    @staticmethod
    def _repair_truncated_json(blob: str) -> Optional[str]:
        """Best-effort repair for truncated Llama JSON (unterminated strings/braces)."""
        if not blob or "{" not in blob:
            return None
        trimmed = blob.rstrip().rstrip(",")
        candidates: List[str] = []
        if trimmed.count('"') % 2 == 1:
            candidates.append(trimmed + '"')
        candidates.append(trimmed)
        open_braces = max(0, trimmed.count("{") - trimmed.count("}"))
        open_brackets = max(0, trimmed.count("[") - trimmed.count("]"))
        candidates.append(trimmed + ("]" * open_brackets) + ("}" * open_braces))
        if trimmed.count('"') % 2 == 1:
            candidates.append(trimmed + '"' + ("}" * open_braces))
        for suffix in ('"}', '"}', '"}', '"}', '0}', '}'):
            candidates.append(trimmed + suffix)
        last_comma = trimmed.rfind('",')
        if last_comma > 0:
            candidates.append(trimmed[: last_comma + 1] + "}")
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
        return None

    def _parse_structured_reasoning(self, text: str, scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        blob = self._extract_json_blob(text)
        attempts = [blob]
        repaired = self._repair_truncated_json(blob)
        if repaired and repaired not in attempts:
            attempts.append(repaired)

        for candidate in attempts:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return self._normalize_llama_decision(parsed, scene)
            except json.JSONDecodeError as exc:
                logger.warning("[LLAMA] JSON decode failed: %s raw=%r", exc, candidate[:300])

        extracted = self._extract_structured_fields(blob)
        if extracted.get("safety_label") or extracted.get("patient_status") or extracted.get("summary"):
            logger.info("[LLAMA] recovered decision fields via regex: %s", list(extracted.keys()))
            return self._normalize_llama_decision(extracted, scene)
        return None

    def _invoke_llama_chat_completion(self, reasoning_pipeline: Any, user_message: str) -> str:
        generated = reasoning_pipeline.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a patient safety JSON API. Output ONLY valid compact JSON. "
                        "No markdown. No prose. No code fences."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            max_tokens=int(pc.LLAMA_MAX_TOKENS),
            temperature=0.0,
            top_p=0.9,
            stop=["</s>", "<|eot_id|>", "<|end_of_text|>", "\n\n"],
        )
        return self._safe_text_generation(generated).strip()

    @staticmethod
    def _normalize_fallback_reason(raw_reason: Optional[str]) -> str:
        reason = str(raw_reason or "structured_llama_failed")
        if reason == "llama_json_parse_failed" or "JSON parse failed" in reason:
            return "llama_json_parse_failed"
        if reason == "llama_runtime_error" or llama_runtime.is_llama_cuda_error_text(reason):
            return "llama_runtime_error"
        return reason

    def _run_structured_reasoning(self, scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._last_llama_invoked = False
        self._last_llama_runtime_error = None
        reasoning_pipeline = self._get_reasoning_pipeline()
        if reasoning_pipeline is None:
            self._last_llama_runtime_error = (
                self.reasoning_health.get("load_error") or "reasoning_pipeline is None"
            )
            logger.error("[LLAMA] inference skipped — model not loaded: %s", self._last_llama_runtime_error)
            return None
        frame_id = int(scene.get("frame_id") or 0)
        user_message = self._build_scene_reasoning_user_message(scene, frame_id=frame_id)
        import time as _time

        llama_started = _time.time()
        retry_cpu = False
        while True:
            try:
                text = self._invoke_llama_chat_completion(reasoning_pipeline, user_message)
                if not text:
                    self._last_llama_runtime_error = "empty model output"
                    logger.error("[LLAMA] inference returned empty output frame_id=%s", frame_id)
                    return None
                parsed = self._parse_structured_reasoning(text, scene)
                if parsed:
                    self._last_llama_invoked = True
                    self._last_llama_latency_ms = int((_time.time() - llama_started) * 1000)
                    logger.info(
                        "[LLAMA] inference OK frame_id=%s latency_ms=%s backend=%s summary=%r",
                        frame_id,
                        self._last_llama_latency_ms,
                        self._reasoning_hw.backend,
                        (str(parsed.get("summary") or ""))[:80],
                    )
                    return parsed
                self._last_llama_runtime_error = "llama_json_parse_failed"
                logger.error(
                    "[LLAMA] structured JSON parse failed frame_id=%s fallback=llama_json_parse_failed raw=%r",
                    frame_id,
                    text[:500],
                )
                return None
            except Exception as exc:
                err_text = f"{type(exc).__name__}: {exc}"
                if llama_runtime.is_llama_cuda_error(exc) and not retry_cpu:
                    self._disable_llama_gpu_for_session(error=err_text, frame_id=frame_id)
                    reasoning_pipeline = self._get_reasoning_pipeline()
                    if reasoning_pipeline is not None:
                        logger.warning(
                            "[LLAMA] retrying inference on CPU frame_id=%s n_gpu_layers=0 n_batch=%s",
                            frame_id,
                            os.environ.get("LLAMA_N_BATCH", "256"),
                        )
                        retry_cpu = True
                        continue
                    self._last_llama_runtime_error = "llama_runtime_error"
                    logger.error(
                        "[LLAMA] CPU reload failed after CUDA error frame_id=%s",
                        frame_id,
                    )
                    return None
                if llama_runtime.is_llama_cuda_error(exc):
                    self._last_llama_runtime_error = "llama_runtime_error"
                    logger.error(
                        "[LLAMA] CPU retry failed after CUDA error frame_id=%s error=%s",
                        frame_id,
                        err_text,
                    )
                    return None
                self._last_llama_runtime_error = err_text
                logger.exception(
                    "[LLAMA] inference exception frame_id=%s: %s",
                    frame_id,
                    exc,
                )
                return None

    def _infer_intent_from_scene(
        self,
        scene: Dict[str, Any],
        *,
        use_reasoning: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool, Dict[str, Any], Dict[str, Any]]:
        trace = {
            "requested": bool(use_reasoning),
            "llama_available": self._reasoning_available(),
            "llama_invoked": False,
            "fallback": False,
        }
        primary = scene.get("primary_person") or {}
        pose_summary = dict(primary.get("pose_summary") or {"available": False})
        has_patient = primary is not None and bool(primary.get("bbox"))

        if not use_reasoning:
            trace["fallback"] = True
            trace["fallback_reason"] = (
                "ENABLE_REASONING=0"
                if not pc.reasoning_enabled()
                else "enable_ai_reasoning=False in stream config"
            )
            self._last_reasoning_trace = trace
            structured = self._build_fallback_structured(
                scene, trace["fallback_reason"], no_person=not has_patient
            )
            intent = str(structured.get("summary") or structured.get("reason") or "not provided")
            return intent, False, pose_summary, structured

        if not trace["llama_available"] or self._get_reasoning_pipeline() is None:
            trace["fallback"] = True
            trace["fallback_reason"] = self.reasoning_health.get("load_error") or "llama_unavailable"
            self._last_reasoning_trace = trace
            structured = self._build_fallback_structured(
                scene, trace["fallback_reason"], no_person=not has_patient
            )
            intent = str(structured.get("summary") or structured.get("reason") or "not provided")
            return intent, False, pose_summary, structured

        llama_decision = self._run_structured_reasoning(scene)
        if llama_decision:
            trace["llama_invoked"] = True
            structured = llama_decision
        else:
            trace["fallback"] = True
            fb_reason = self._normalize_fallback_reason(self._last_llama_runtime_error)
            trace["fallback_reason"] = fb_reason
            logger.warning(
                "[LLAMA] decision fallback frame_id=%s reason=%s",
                scene.get("frame_id"),
                trace["fallback_reason"],
            )
            structured = self._build_fallback_structured(
                scene,
                trace["fallback_reason"],
                no_person=not has_patient,
                fallback_reason=trace["fallback_reason"],
            )
        self._last_reasoning_trace = trace

        intent = str(structured.get("summary") or structured.get("reason") or "not provided")
        return intent, trace.get("llama_invoked", False), pose_summary, structured

    @staticmethod
    def build_basic_reasoning_test_scene() -> Dict[str, Any]:
        """Fixed scene: patient on bed with nurse and doctor nearby."""
        patient = {
            "id": 0,
            "role": "patient",
            "role_for_llm": "Patient",
            "pose": "lying",
            "pose_conf": 0.92,
            "role_conf": 0.91,
            "relation_to_bed": "on_bed",
            "staff_near": True,
            "bbox": [120, 180, 420, 520],
        }
        nurse = {
            "id": 1,
            "role": "nurse",
            "role_for_llm": "Nurse",
            "pose": "standing",
            "pose_conf": 0.88,
            "role_conf": 0.9,
            "bbox": [480, 120, 620, 480],
        }
        doctor = {
            "id": 2,
            "role": "doctor",
            "role_for_llm": "Doctor",
            "pose": "standing",
            "pose_conf": 0.87,
            "role_conf": 0.89,
            "bbox": [640, 110, 780, 470],
        }
        return {
            "frame_id": 1,
            "expected_bed_present": True,
            "persons": [patient, nurse, doctor],
            "primary_person": dict(patient),
            "objects": [{"label": "bed", "confidence": 0.94}],
            "llm_relations": [
                {"subject": 0, "relation": "on_bed", "object": 0},
                {"subject": 1, "relation": "near", "object": 0},
                {"subject": 2, "relation": "near", "object": 0},
            ],
        }

    def run_basic_reasoning_test(self) -> Dict[str, Any]:
        """Manual LLM smoke test — patient on bed with staff nearby."""
        scene = self.build_basic_reasoning_test_scene()
        from monitor.runtime.display_fields import build_llm_scene_payload

        payload = build_llm_scene_payload(scene, frame_id=int(scene["frame_id"]))
        structured = self._run_structured_reasoning(scene)
        trace = dict(self._last_reasoning_trace or {})
        if structured is None:
            fb_reason = self._normalize_fallback_reason(self._last_llama_runtime_error)
            structured = self._build_fallback_structured(
                scene,
                fb_reason,
                fallback_reason=fb_reason,
            )
            trace = {
                "fallback": True,
                "fallback_reason": fb_reason,
                "llama_invoked": False,
                "llama_available": self._reasoning_available(),
            }
        return {
            "scene": scene,
            "scene_payload": payload,
            "structured": structured,
            "trace": trace,
            "llama_latency_ms": getattr(self, "_last_llama_latency_ms", None),
        }

    @staticmethod
    def _build_reasoning_prompt(detections: Dict[str, Any], pose_summary: Dict[str, Any], vlm_caption: str) -> str:
        bbox = detections.get("bbox") or {}
        intent_hint = detections.get("intent_hint", "unknown")
        role_hint = detections.get("role_hint", "unknown")
        parts = [
            "You are a patient-intent reasoning model in a hospital monitoring system.",
            f"Detection hint: {intent_hint}.",
            f"Role hint: {role_hint}.",
            f"Bounding box: {bbox}.",
            f"Pose summary: {pose_summary}.",
        ]
        if vlm_caption:
            parts.append(f"Visual caption: {vlm_caption}.")
        parts.append(
            "Return one short intent sentence only, such as: Patient attempting to stand, Patient resting, or Patient under observation."
        )
        return "\n".join(parts)

    def _run_reasoning_model(
        self,
        pose_landmarks: Optional[Any],
        crop: np.ndarray,
        detection_hint: str,
        role_hint: str,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        pose_summary: Dict[str, Any] = {"available": False}
        try:
            pose_summary = self._summarize_pose(pose_landmarks)
        except Exception as exc:
            logger.warning("Pose summary failed: %s", exc)

        caption_text = ""
        vlm_pipeline = self._get_vlm_pipeline()
        if vlm_pipeline is not None and crop.size > 0:
            try:
                caption = vlm_pipeline(crop)
                caption_text = self._safe_caption_text(caption).lower()
            except Exception as exc:
                logger.warning("VLM inference failed: %s", exc)

        reasoning_output = ""
        self._last_llama_invoked = False
        try:
            prompt = self._build_reasoning_prompt(
                {"bbox": None, "intent_hint": detection_hint, "role_hint": role_hint},
                pose_summary,
                caption_text,
            )
            reasoning_pipeline = self._get_reasoning_pipeline()
            if reasoning_pipeline is not None:
                logger.debug("Llama inference started (prompt_len=%d)", len(prompt))
                generated = reasoning_pipeline(
                    prompt,
                    max_tokens=int(pc.LLAMA_MAX_TOKENS),
                    temperature=0.1,
                    top_p=0.9,
                    stop=["</s>", "<|eot_id|>", "\n\n"],
                )
                reasoning_output = self._safe_text_generation(generated).strip()
                self._last_llama_invoked = True
                logger.info(
                    "Llama inference completed (output_len=%d preview=%r)",
                    len(reasoning_output),
                    (reasoning_output[:80] + "…") if len(reasoning_output) > 80 else reasoning_output,
                )
            else:
                logger.warning("Llama inference skipped: reasoning_pipeline is None")
        except Exception as exc:
            logger.warning("Llama intent reasoning failed: %s", exc)
            self.model_status["intent_reasoning"] = "error"

        standing_hint = bool(pose_summary.get("standing_hint"))
        intent_text = reasoning_output.lower()

        if intent_text:
            if any(keyword in intent_text for keyword in ("stand", "rising", "getting up", "leave bed")):
                return reasoning_output or "Patient attempting to stand", True, pose_summary
            if any(keyword in intent_text for keyword in ("rest", "lying", "sleep", "observe", "stable")):
                return reasoning_output or "Patient under observation", False, pose_summary

        if caption_text:
            if any(keyword in caption_text for keyword in ("standing", "stand", "getting up", "rise", "moving")):
                return "Patient attempting to stand", True, pose_summary
            if any(keyword in caption_text for keyword in ("lying", "sleep", "resting", "sitting")):
                return "Patient resting", False, pose_summary

        if standing_hint and float(pose_summary.get("pose_score") or 0) >= pc.POSE_MIN_SCORE:
            return "Patient attempting to stand", True, pose_summary

        if detection_hint:
            return detection_hint, False, pose_summary

        return "Patient under observation", False, pose_summary

    def _infer_intent_heuristic(
        self,
        pose_landmarks: Optional[Any],
        detection_hint: str,
        role_hint: str = "unknown",
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """Fast intent path: pose + detection only (no Llama / VLM)."""
        try:
            pose_summary = self._summarize_pose(pose_landmarks)
        except Exception as exc:
            logger.debug("Heuristic pose summary failed: %s", exc)
            pose_summary = {"available": False}

        standing_hint = bool(pose_summary.get("standing_hint"))
        pose_score = float(pose_summary.get("pose_score") or 0)
        if standing_hint and pose_score >= pc.POSE_MIN_SCORE:
            return "Patient attempting to stand", True, pose_summary

        hint_lower = (detection_hint or "").lower()
        if "detected" in hint_lower:
            if role_hint and role_hint not in {"unknown", ""}:
                return f"Patient under observation ({role_hint})", False, pose_summary
            return "Patient under observation", False, pose_summary

        if detection_hint:
            return detection_hint, False, pose_summary

        return "Patient under observation", False, pose_summary

    def _infer_intent(
        self,
        pose_landmarks: Optional[Any],
        crop: np.ndarray,
        detection_hint: str,
        role_hint: str = "unknown",
        *,
        use_reasoning: bool = True,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        trace = {
            "requested": bool(use_reasoning),
            "llama_available": self._reasoning_available(),
            "llama_invoked": False,
            "fallback": False,
        }
        if not use_reasoning:
            trace["fallback"] = True
            trace["fallback_reason"] = "llama_disabled_by_config"
            self._last_reasoning_trace = trace
            logger.info("reasoning: FALLBACK heuristic — Llama disabled by config")
            return self._infer_intent_heuristic(pose_landmarks, detection_hint, role_hint)

        if not trace["llama_available"]:
            trace["fallback"] = True
            trace["fallback_reason"] = self.reasoning_health.get("load_error") or "llama_unavailable"
            self._last_reasoning_trace = trace
            logger.warning(
                "reasoning: FALLBACK heuristic — Llama unavailable (%s)",
                trace["fallback_reason"],
            )
            return self._infer_intent_heuristic(pose_landmarks, detection_hint, role_hint)

        if self._get_reasoning_pipeline() is None:
            trace["fallback"] = True
            trace["fallback_reason"] = self.reasoning_health.get("load_error") or "load_failed"
            self._last_reasoning_trace = trace
            logger.warning(
                "reasoning: FALLBACK heuristic — load failed (%s)",
                trace["fallback_reason"],
            )
            return self._infer_intent_heuristic(pose_landmarks, detection_hint, role_hint)

        try:
            intent, alert, summary = self._run_reasoning_model(
                pose_landmarks, crop, detection_hint, role_hint
            )
            trace["llama_invoked"] = bool(self._last_llama_invoked)
            if not trace["llama_invoked"]:
                trace["fallback"] = True
                trace["fallback_reason"] = "empty_llama_output"
                logger.warning("reasoning: FALLBACK heuristic — Llama returned empty output")
            else:
                logger.info(
                    "reasoning: Llama inference OK backend=%s",
                    self._reasoning_hw.backend,
                )
            self._last_reasoning_trace = trace
            return intent, alert, summary
        except Exception as exc:
            trace["fallback"] = True
            self._last_reasoning_trace = trace
            logger.warning("Intent inference failed; using heuristic fallback: %s", exc)
            return self._infer_intent_heuristic(pose_landmarks, detection_hint, role_hint)

    def _infer_role_with_confidence(self, crop: np.ndarray) -> Tuple[str, float]:
        if self.role_model is None or crop.size == 0:
            return "unknown", 0.0

        try:
            predictions = self.role_model.predict(
                crop,
                verbose=False,
                conf=max(0.03, pc.OBJECT_BASE_CONF),
                device=self._yolo_predict_device,
            )
            if not predictions:
                return "unknown", 0.0
            boxes = predictions[0].boxes
            names = getattr(predictions[0], "names", {}) or {}
            if boxes is not None and len(boxes) and hasattr(boxes, "cls"):
                best_index = int(torch.argmax(boxes.conf).item()) if hasattr(boxes, "conf") and len(boxes.conf) else 0
                class_id = int(boxes.cls[best_index].item())
                conf = float(boxes.conf[best_index].item()) if hasattr(boxes, "conf") else 0.0
                role = str(names.get(class_id, f"class_{class_id}")).strip().lower()
                if conf < pc.role_confidence_threshold(role):
                    return "unknown", conf
                return role, conf
        except Exception as exc:
            logger.debug("Role model inference failed: %s", exc)
        return "unknown", 0.0

    def _infer_role_hint(self, crop: np.ndarray) -> str:
        role, _ = self._infer_role_with_confidence(crop)
        return role

    @staticmethod
    def _split_box_label(label: str) -> List[str]:
        text = str(label or "").strip()
        if not text:
            return []
        if " | " in text:
            head, tail = text.split(" | ", 1)
            return [head.strip()[:26], tail.strip()[:30]]
        return [text[:34]]

    @staticmethod
    def draw_detection_box(
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        label: str,
        alert_triggered: bool = False,
    ) -> None:
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]
        color = (0, 0, 255) if alert_triggered else (0, 200, 0)
        lines = PatientIntentPipeline._split_box_label(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        pad_x = 6
        pad_y = 5
        line_gap = 3
        box_gap = 4

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not lines:
            return

        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        block_w = max(size[0] for size in sizes) + pad_x * 2
        block_h = sum(size[1] for size in sizes) + pad_y * 2 + line_gap * max(0, len(lines) - 1)
        label_bottom = max(0, y1 - box_gap)
        label_top = max(0, label_bottom - block_h)
        label_left = max(0, min(x1, width - 1))
        label_right = min(width - 1, label_left + block_w)
        if label_right <= label_left:
            label_right = min(width - 1, label_left + 72)

        cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), color, -1)
        text_y = label_top + pad_y
        for idx, line in enumerate(lines):
            text_y += sizes[idx][1]
            cv2.putText(
                frame,
                line,
                (label_left + pad_x, text_y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            text_y += line_gap

    def _draw_yolo_box(self, frame: np.ndarray, box: Tuple[int, int, int, int], label: str, alert_triggered: bool) -> None:
        self.draw_detection_box(frame, box, label, alert_triggered)

    def _draw_skeleton(
        self,
        output: np.ndarray,
        pose_landmarks: Any,
        x1: int = 0,
        y1: int = 0,
        crop_w: int = 0,
        crop_h: int = 0,
        *,
        frame_space: bool = False,
    ) -> None:
        if pose_landmarks is None:
            return

        points: Dict[int, Tuple[int, int]] = {}
        try:
            for idx, pt in enumerate(pose_landmarks):
                raw_x = float(pt[0].item() if hasattr(pt[0], "item") else pt[0])
                raw_y = float(pt[1].item() if hasattr(pt[1], "item") else pt[1])
                if frame_space:
                    px = int(raw_x)
                    py = int(raw_y)
                else:
                    px = int(raw_x) + x1
                    py = int(raw_y) + y1
                if px > 0 and py > 0:
                    points[idx] = (px, py)

            for pt in points.values():
                cv2.circle(output, pt, 4, (0, 255, 255), -1)

            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx in points and end_idx in points:
                    cv2.line(output, points[start_idx], points[end_idx], (255, 255, 0), 2)
        except Exception as exc:
            logger.debug("Failed to draw skeleton: %s", exc)

    def _draw_pipeline_status(self, frame: np.ndarray, message: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        cv2.putText(
            frame,
            message,
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_model_status_bar(self, frame: np.ndarray) -> None:
        labels = [
            f"obj:{self.model_status.get('object_detection', '?')}",
            f"roles:{self.model_status.get('role_classification', '?')}",
            f"pose:{self.model_status.get('pose_estimation', '?')}",
            f"reason:{self.model_status.get('intent_reasoning', '?')}",
        ]
        text = " | ".join(labels)
        height = frame.shape[0]
        cv2.rectangle(frame, (8, height - 34), (min(frame.shape[1] - 8, 8 + len(text) * 9), height - 8), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (14, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

    def _extract_pose_landmarks(self, crop: np.ndarray, x1: int, y1: int) -> Optional[Any]:
        if crop.size == 0 or self.pose_model is None:
            return None

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_result = self.pose_model.predict(crop_rgb, verbose=False, device=self._yolo_predict_device)
            if pose_result:
                candidates = pose_result[0].keypoints
                if candidates is not None and len(candidates) and hasattr(candidates, "xy"):
                    return candidates.xy[0]
        except Exception as exc:
            logger.warning("Pose estimation inference failed: %s", exc)
            self.model_status["pose_estimation"] = "error"
        return None

    def detect_objects_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int, int, int]], Optional[np.ndarray], str, float]:
        """Object stage for parallel runtime (full-frame, per-model lock)."""
        with self._object_lock:
            detections = self._run_object_detection(frame)
        if not detections:
            return [], None, None, "no patient detected", 0.0
        person_dets = [
            d
            for d in detections
            if not _is_bed_label(str(d.get("label", "person")))
        ]
        primary = (person_dets or detections)[0]
        x1, y1, x2, y2 = self._smooth_box(primary["box"])
        confidence = float(self._obj_conf_smoother.update(float(primary["confidence"])))
        hint = (
            "patient detected"
            if confidence >= pc.PERSON_CONF
            else "patient under observation"
        )
        return detections, (x1, y1, x2, y2), None, hint, confidence

    @staticmethod
    def _landmarks_to_box(landmarks: Any, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        xs: List[float] = []
        ys: List[float] = []
        for pt in landmarks:
            try:
                x = float(pt[0].item() if hasattr(pt[0], "item") else pt[0])
                y = float(pt[1].item() if hasattr(pt[1], "item") else pt[1])
                if x > 1 and y > 1:
                    xs.append(x)
                    ys.append(y)
            except Exception:
                continue
        if not xs or not ys:
            return (0, 0, 0, 0)
        height = frame_shape[0]
        width = frame_shape[1]
        x1 = max(0, int(min(xs)))
        y1 = max(0, int(min(ys)))
        x2 = min(width - 1, int(max(xs)))
        y2 = min(height - 1, int(max(ys)))
        return (x1, y1, x2, y2)

    def _collect_role_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.role_model is None or frame.size == 0:
            return []
        detections: List[Dict[str, Any]] = []
        try:
            predictions = self.role_model.predict(
                frame,
                verbose=False,
                conf=max(0.03, pc.OBJECT_BASE_CONF),
                imgsz=pc.YOLO_IMG_SIZE,
                device=self._yolo_predict_device,
            )
            if not predictions:
                return detections
            boxes = predictions[0].boxes
            names = getattr(predictions[0], "names", {}) or {}
            if boxes is None or len(boxes) == 0:
                return detections
            height, width = frame.shape[:2]
            for index in range(len(boxes)):
                box_data = boxes.xyxy[index].tolist()
                confidence = float(boxes.conf[index].item()) if hasattr(boxes, "conf") else 0.0
                class_id = int(boxes.cls[index].item()) if hasattr(boxes, "cls") else 0
                role = str(names.get(class_id, f"class_{class_id}")).strip().lower()
                x1, y1, x2, y2 = self._clip_box(
                    int(box_data[0]),
                    int(box_data[1]),
                    int(box_data[2]),
                    int(box_data[3]),
                    width,
                    height,
                )
                threshold = pc.role_confidence_threshold(role)
                logger.info(
                    "[ROLE] raw detection class=%s conf=%.3f threshold=%.2f box=%s",
                    role,
                    confidence,
                    threshold,
                    (x1, y1, x2, y2),
                )
                if confidence < threshold:
                    logger.info(
                        "[ROLE] filtered class=%s conf=%.3f below threshold=%.2f",
                        role,
                        confidence,
                        threshold,
                    )
                    continue
                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "role": role,
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                )
            detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
        except Exception as exc:
            logger.debug("Full-frame role inference failed: %s", exc)
        return detections

    def detect_roles_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], str, float]:
        with self._role_lock:
            detections = self._collect_role_detections(frame)
        if not detections:
            logger.info("[ROLE] no roles.pt detections on full frame after class filtering")

        best_role = "unknown"
        best_conf = 0.0
        best_threshold = pc.ROLE_CONF
        for item in detections:
            role = str(item.get("role") or "unknown")
            conf = float(item.get("confidence") or 0.0)
            accept_conf = pc.role_confidence_threshold(role)
            if conf >= accept_conf and conf >= best_conf:
                best_role = role
                best_conf = conf
                best_threshold = accept_conf

        if best_role == "unknown":
            if detections:
                top = detections[0]
                logger.info(
                    "[ROLE] no role accepted — top=%s conf=%.3f below threshold=%.3f "
                    "(patient=%.3f doctor=%.3f nurse=%.3f role=%.3f)",
                    top.get("role"),
                    float(top.get("confidence") or 0.0),
                    best_threshold,
                    pc.PATIENT_CONF,
                    pc.DOCTOR_CONF,
                    pc.NURSE_CONF,
                    pc.ROLE_CONF,
                )
            if self._stable_role != "unknown":
                logger.info(
                    "[ROLE] keeping stable role=%s conf=%.3f",
                    self._stable_role,
                    self._stable_role_conf,
                )
                return detections, self._stable_role, self._stable_role_conf
        elif best_conf < pc.ROLE_CONFIDENCE_SWITCH and self._stable_role != "unknown":
            logger.info(
                "[ROLE] below switch threshold %.3f — keeping stable role=%s conf=%.3f",
                pc.ROLE_CONFIDENCE_SWITCH,
                self._stable_role,
                self._stable_role_conf,
            )
            return detections, self._stable_role, self._stable_role_conf
        elif (
            best_role != self._stable_role
            and best_role != "unknown"
            and self._stable_role != "unknown"
            and best_conf < pc.ROLE_CONFIDENCE_SWITCH + 0.12
        ):
            logger.info(
                "[ROLE] unstable switch blocked %s→%s conf=%.3f keeping stable=%s",
                self._stable_role,
                best_role,
                best_conf,
                self._stable_role,
            )
            return detections, self._stable_role, self._stable_role_conf

        if best_conf >= pc.ROLE_CONFIDENCE_SWITCH or self._stable_role == "unknown":
            self._stable_role = best_role
            self._stable_role_conf = best_conf
        return detections, self._stable_role, self._stable_role_conf

    def detect_poses_frame(
        self,
        frame: np.ndarray,
        *,
        frame_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Any], Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        primary_landmarks = None
        primary_summary: Dict[str, Any] = {"available": False}
        if self.pose_model is None or frame.size == 0:
            return detections, None, primary_summary

        with self._pose_lock:
            try:
                predictions = self.pose_model.predict(
                    frame,
                    verbose=False,
                    conf=max(0.03, pc.OBJECT_BASE_CONF),
                    imgsz=pc.YOLO_IMG_SIZE,
                    device=self._yolo_predict_device,
                )
            except Exception as exc:
                logger.warning("Full-frame pose inference failed: %s", exc)
                self.model_status["pose_estimation"] = "error"
                return detections, None, primary_summary

        if not predictions:
            return detections, None, primary_summary

        boxes = predictions[0].boxes
        keypoints = predictions[0].keypoints
        if keypoints is None or not hasattr(keypoints, "xy") or len(keypoints.xy) == 0:
            return detections, None, primary_summary

        height, width = frame.shape[:2]
        best_score = -1.0
        primary_landmarks = None
        for index in range(len(keypoints.xy)):
            landmarks = keypoints.xy[index]
            if boxes is not None and len(boxes) > index:
                box_data = boxes.xyxy[index].tolist()
                confidence = float(boxes.conf[index].item()) if hasattr(boxes, "conf") else 0.5
                x1, y1, x2, y2 = self._clip_box(
                    int(box_data[0]),
                    int(box_data[1]),
                    int(box_data[2]),
                    int(box_data[3]),
                    width,
                    height,
                )
            else:
                x1, y1, x2, y2 = self._landmarks_to_box(landmarks, frame.shape)
                confidence = 0.5
            pose_box = (x1, y1, x2, y2)
            summary = self._summarize_pose(
                landmarks,
                pose_box=pose_box,
                frame_id=frame_id,
                confidence=confidence,
            )
            raw_score = float(summary.get("pose_score") or 0.0)
            pose_class = str(summary.get("pose_class") or "unknown")
            landmark_list = coerce_landmarks(landmarks)
            kp_count = len(landmark_list or [])
            logger.info(
                "[POSE][RAW] frame_id=%s bbox=(%s,%s,%s,%s) keypoints_count=%s conf=%.3f "
                "pose_score=%.3f pose_class=%s",
                frame_id if frame_id is not None else "?",
                x1,
                y1,
                x2,
                y2,
                kp_count,
                confidence,
                raw_score,
                pose_class,
            )
            if pose_class == "unknown":
                continue
            if confidence < pc.POSE_CONF and raw_score < pc.POSE_CONF:
                continue
            detections.append(
                {
                    "box": pose_box,
                    "landmarks": coerce_landmarks(landmarks),
                    "pose_summary": summary,
                    "confidence": confidence,
                }
            )
            if raw_score > best_score:
                best_score = raw_score
                primary_landmarks = coerce_landmarks(landmarks)
                primary_summary = summary

        return detections, primary_landmarks, primary_summary

    def detect_role(self, crop: np.ndarray) -> Tuple[str, float]:
        with self._role_lock:
            role, conf = self._infer_role_with_confidence(crop)
            accept_conf = pc.role_confidence_threshold(role)
            if conf < accept_conf and self._stable_role != "unknown":
                return self._stable_role, self._stable_role_conf
            if conf >= pc.ROLE_CONFIDENCE_SWITCH or self._stable_role == "unknown":
                self._stable_role = role
                self._stable_role_conf = conf
            return self._stable_role, self._stable_role_conf

    def detect_pose(self, crop: np.ndarray, x1: int, y1: int) -> Tuple[Optional[Any], Dict[str, Any]]:
        with self._pose_lock:
            landmarks = self._extract_pose_landmarks(crop, x1, y1)
        try:
            summary = self._summarize_pose(landmarks)
        except Exception:
            summary = {"available": False, "pose_detected": False}

        raw_score = float(summary.get("pose_score") or 0.0)
        pose_detected = bool(summary.get("pose_detected")) or landmarks is not None
        if landmarks is not None and raw_score <= 0.0:
            raw_score = pc.POSE_MIN_SCORE

        alpha = pc.POSE_SMOOTH_ALPHA
        if self._smoothed_pose_score is None:
            self._smoothed_pose_score = raw_score
        else:
            self._smoothed_pose_score = alpha * raw_score + (1.0 - alpha) * self._smoothed_pose_score
        display_score = float(self._pose_conf_smoother.update(self._smoothed_pose_score))

        summary = dict(summary)
        summary["pose_score_raw"] = round(raw_score, 3)
        summary["pose_score"] = round(display_score, 3)
        summary["pose_detected"] = pose_detected
        summary["available"] = pose_detected
        return coerce_landmarks(landmarks), summary

    @staticmethod
    def _sync_pose_summary_from_scene(
        pose_summary: Dict[str, Any],
        scene: Dict[str, Any],
    ) -> Dict[str, Any]:
        pose_summary = dict(pose_summary or {})
        scene_status = str(scene.get("patient_status") or "").lower()
        if scene_status == "standing":
            pose_summary["standing_hint"] = True
            pose_summary["pose_class"] = "standing"
        elif scene_status in ("lying", "sitting", "fall_risk"):
            pose_summary["pose_class"] = scene_status
            pose_summary["standing_hint"] = False
        elif scene_status == "unknown":
            pose_summary["pose_class"] = "unknown"
            pose_summary["standing_hint"] = False
        return pose_summary

    @staticmethod
    def _attach_display_sections(
        scores: Dict[str, Any],
        structured: Dict[str, Any],
        scene: Dict[str, Any],
        intent: str,
        *,
        decision_frame_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        from monitor.runtime.display_fields import build_display_sections, evaluate_reasoning_consistency

        decision_fid = int(decision_frame_id or scores.get("frame_id") or 0)
        sections = build_display_sections(
            scene=scene,
            structured=structured,
            scores=scores,
            intent=intent,
            decision_frame_id=decision_fid,
            llm_scene_payload=scores.get("llm_scene_payload"),
        )
        scores = dict(scores)
        scores["display_sections"] = sections
        warn, warn_text = evaluate_reasoning_consistency(scene, structured, scores)
        scores["reasoning_consistency_warning"] = warn
        scores["warning_text"] = warn_text
        return scores

    def run_reasoning_from_context(self, context: Any) -> Any:
        """LLM/reasoner is the final decision maker for all visible dashboard fields."""
        from monitor.runtime.types import ReasoningContext, ReasoningResult

        if not isinstance(context, ReasoningContext):
            raise TypeError("context must be ReasoningContext")

        if self._last_llm_result and context.frame_id < self._last_llm_result.frame_id:
            return self._last_llm_result

        import copy

        scene = copy.deepcopy(context.scene or {})
        from monitor.runtime.display_fields import build_llm_scene_payload, evaluate_reasoning_consistency

        llm_scene_payload = build_llm_scene_payload(scene, frame_id=context.frame_id)
        has_patient = scene.get("primary_person") is not None or context.primary_box is not None
        pose_summary = self._sync_pose_summary_from_scene(
            dict((scene.get("primary_person") or {}).get("pose_summary") or context.pose_summary or {}),
            scene,
        )

        role_conf = float(
            self._role_conf_smoother.update(float(getattr(context.role_result, "role_confidence", 0.0) or 0.0))
        )
        object_conf = float(self._obj_conf_smoother.update(float(context.confidence or 0.0)))
        pose_score = float(pose_summary.get("pose_score") or 0.0)
        pose_detected = context.pose_landmarks is not None or bool(
            pose_summary.get("pose_detected") or context.pose_summary.get("pose_detected")
        )
        scores = compute_confidence_scores(object_conf, role_conf, pose_score, pose_detected)
        overall_smooth = self._overall_conf_smoother.update(float(scores["overall"]))
        scores["overall"] = round(overall_smooth, 3)
        scores["overall_pct"] = round(overall_smooth * 100, 1)
        scores["reasoning_pct"] = scores["overall_pct"]
        scores["scene"] = scene
        scores["frame_id"] = context.frame_id
        scores["llm_scene_payload"] = llm_scene_payload
        scores["llm_scene_frame_id"] = int(context.frame_id)
        scores["llm_output_frame_id"] = int(context.frame_id)
        scores["object_frame_id"] = int(context.object_result.frame_id)
        scores["role_frame_id"] = int(context.role_result.frame_id)
        scores["pose_frame_id"] = int(context.pose_result.frame_id)
        scores["pose_age_frames"] = max(0, int(context.frame_id) - int(context.pose_result.frame_id))
        scores["pose_updated_at"] = int(context.pose_result.frame_id)
        fps = max(1, int(os.environ.get("CAPTURE_MAX_FPS", "30") or 30))
        lag_frames = max(
            int(context.frame_id) - int(context.object_result.frame_id),
            int(context.frame_id) - int(context.role_result.frame_id),
            int(context.frame_id) - int(context.pose_result.frame_id),
        )
        scores["yolo_latency_ms"] = int((lag_frames / fps) * 1000)

        from monitor.runtime.pipeline_diagnostics import get_diagnostics

        diag = get_diagnostics()
        capture_fid = int(diag.snapshot().get("capture_frame_id") or context.frame_id)
        from monitor.runtime.display_fields import evaluate_llm_frame_sync, log_llm_sync

        reasoning_stale, lag_frames = evaluate_llm_frame_sync(
            scene_frame_id=int(context.frame_id),
            decision_frame_id=int(context.frame_id),
            capture_frame_id=capture_fid,
        )
        scores["reasoning_stale"] = reasoning_stale
        scores["llm_lag_frames"] = lag_frames
        log_llm_sync(
            scene_frame_id=int(context.frame_id),
            decision_frame_id=int(context.frame_id),
            capture_frame_id=capture_fid,
            stale=reasoning_stale,
        )
        scene_chars = len(json.dumps(llm_scene_payload, default=str, separators=(",", ":")))
        queued_ts = diag.get_context_queued_ts(int(context.frame_id))
        llm_queue_wait_ms = (
            int((time.time() - queued_ts) * 1000) if queued_ts is not None else None
        )
        scores["llm_queue_wait_ms"] = llm_queue_wait_ms
        diag.on_llm_input(
            int(context.frame_id),
            capture_frame_id=capture_fid,
            lag_frames=lag_frames,
            chars=scene_chars,
            llm_queue_wait_ms=llm_queue_wait_ms,
        )
        scores["capture_ts"] = diag.get_capture_ts(context.frame_id)
        run_llama = self._should_run_llama(context.config)
        llama_used = False
        trace: Dict[str, Any] = {}

        if not has_patient:
            structured = self._build_fallback_structured(scene, "no person detected", no_person=True)
            reasoning_mode = "fallback"
            llama_used = False
            trace = {"fallback": True, "fallback_reason": "no person detected", "llama_invoked": False}
        elif not run_llama:
            structured = self._build_fallback_structured(
                scene,
                "LLM disabled (ENABLE_REASONING=0 or enable_ai_reasoning=False)",
            )
            reasoning_mode = "fallback"
            llama_used = False
            trace = {"fallback": True, "fallback_reason": "llm_disabled", "llama_invoked": False}
        else:
            with self._reasoning_lock:
                intent_text, llama_used, _, structured = self._infer_intent_from_scene(
                    scene,
                    use_reasoning=True,
                    config=context.config,
                )
            trace = dict(self._last_reasoning_trace)
            loading = bool(self.reasoning_health.get("loading")) and not self.reasoning_health.get("model_loaded")
            reasoning_mode = reasoning_mode_label(
                True,
                True,
                llama_available=bool(trace.get("llama_available")),
                llama_used=bool(trace.get("llama_invoked")),
                llama_backend=self._reasoning_hw.backend if trace.get("llama_invoked") else "",
                loading=loading,
            )
            if trace.get("fallback"):
                reasoning_mode = "fallback"

        safety_label = structured.get("safety_label")
        risk_level = str(structured.get("risk_level") or "not provided")
        risk_score_raw = structured.get("risk_score")
        risk_score = float(risk_score_raw) if risk_score_raw is not None else None
        alert_triggered = self._safety_label_is_alert(safety_label) and not scores.get("reasoning_stale")

        intent_text = str(
            structured.get("summary")
            or structured.get("reason")
            or "not provided"
        )

        scores.update(
            {
                "structured_reasoning": structured,
                "reasoning_mode": reasoning_mode,
                "decision_source": structured.get("decision_source", "fallback"),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "safety_label": safety_label,
                "reasoning_health": dict(self.reasoning_health),
                "llama_runtime_error": trace.get("fallback_reason") if trace.get("fallback") else None,
                "fallback_reason": trace.get("fallback_reason") if trace.get("fallback") else None,
                "reasoning_trace_detail": trace,
            }
        )
        scores["llama_latency_ms"] = getattr(self, "_last_llama_latency_ms", None)
        scores["reasoning_latency_ms"] = scores.get("reasoning_latency_ms")
        scores = self._attach_display_sections(
            scores, structured, scene, intent_text, decision_frame_id=context.frame_id
        )

        bbox = None
        if context.primary_box is not None:
            x1, y1, x2, y2 = context.primary_box
            bbox = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": object_conf,
            }

        result = ReasoningResult(
            frame_id=context.frame_id,
            intent=intent_text,
            alert_triggered=alert_triggered,
            bbox=bbox,
            pose_summary=pose_summary,
            model_status={**dict(self.model_status), "intent_reasoning": reasoning_mode},
            use_reasoning=bool(llama_used),
            latency_ms=0,
            role_hint=context.role_hint,
            detection_list=list(context.object_result.detections or []),
            primary_box=context.primary_box,
            pose_landmarks=context.pose_landmarks,
            confidence_scores=scores,
            monitor_status="alert" if alert_triggered else "monitor",
            risk_score=risk_score,
            risk_level=risk_level,
            reasoning_trace=[],
        )
        self._last_llm_result = result
        return result

    def compose_overlay_frame(self, frame: np.ndarray, result: Any) -> np.ndarray:
        from monitor.runtime.types import ReasoningResult

        output = frame.copy()
        if not isinstance(result, ReasoningResult):
            return output

        scores = result.confidence_scores or {}
        structured = scores.get("structured_reasoning") or {}
        alert = self._safety_label_is_alert(structured.get("safety_label")) or bool(result.alert_triggered)
        bbox = result.bbox
        intent = str(result.intent or "Live stream active")
        role_hint = str(result.role_hint or "unknown").strip().lower()
        from monitor.runtime.context_builder import build_overlay_box_label

        capture_active = scores.get("capture_active", True)
        if result.pose_landmarks is not None and capture_active:
            self._draw_skeleton(output, result.pose_landmarks, frame_space=True)

        if bbox:
            box = (int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"]))
            conf = float(bbox.get("confidence") or 0.0)
            scene = scores.get("scene") if isinstance(scores.get("scene"), dict) else {}
            primary_label = build_overlay_box_label(role_hint, conf, intent, scene)
            self._draw_yolo_box(output, box, primary_label, alert)

        for rank, detection in enumerate(result.detection_list or []):
            if rank == 0:
                continue
            bx1, by1, bx2, by2 = detection["box"]
            det_conf = float(detection["confidence"])
            det_label = str(detection.get("label") or "track").strip().lower()
            self._draw_yolo_box(output, (bx1, by1, bx2, by2), f"{det_label} {det_conf:.2f}", False)

        return output

    def _run_object_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.yolo_model is None:
            return []

        height, width = frame.shape[:2]
        try:
            detections = self.yolo_model.predict(
                frame,
                verbose=False,
                conf=self.min_confidence,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                imgsz=pc.YOLO_IMG_SIZE,
                device=self._yolo_predict_device,
            )
        except Exception as exc:
            logger.warning("Object detection inference failed: %s", exc)
            self.model_status["object_detection"] = "error"
            return []

        if not detections:
            return []

        boxes = detections[0].boxes
        names = getattr(detections[0], "names", {}) or {}
        return self._collect_detection_boxes(boxes, width, height, names)

    def _collect_detection_boxes(
        self,
        boxes: Any,
        width: int,
        height: int,
        names: Optional[Dict[int, str]] = None,
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if boxes is None or len(boxes) == 0:
            return detections

        for index in range(len(boxes)):
            box_data = boxes.xyxy[index].tolist()
            confidence = float(boxes.conf[index].item()) if hasattr(boxes, "conf") else 0.0
            x1, y1, x2, y2 = self._clip_box(
                int(box_data[0]),
                int(box_data[1]),
                int(box_data[2]),
                int(box_data[3]),
                width,
                height,
            )
            class_id = int(boxes.cls[index].item()) if hasattr(boxes, "cls") else 0
            raw_label = str((names or {}).get(class_id, "person")).strip()
            label = normalize_object_label(raw_label)
            logger.info(
                "[OBJ] raw detection class=%s normalized=%s conf=%.3f box=(%s,%s,%s,%s)",
                raw_label,
                label,
                confidence,
                x1,
                y1,
                x2,
                y2,
            )
            if not self._is_valid_person_box(x1, y1, x2, y2, width, height) and not _is_bed_label(label):
                logger.info(
                    "[OBJ] skipped class=%s conf=%.3f — failed person box validation",
                    label,
                    confidence,
                )
                continue
            detections.append(
                {
                    "index": index,
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "label": label,
                    "class_id": class_id,
                }
            )
        detections.sort(key=lambda item: item["confidence"], reverse=True)

        beds: List[Dict[str, Any]] = []
        non_beds: List[Dict[str, Any]] = []
        for det in detections:
            label = str(det.get("label", ""))
            conf = float(det.get("confidence") or 0.0)
            threshold = pc.object_confidence_threshold(label)
            if _is_bed_label(label):
                if conf >= threshold:
                    logger.info(
                        "[OBJ] bed kept class=%s conf=%.3f threshold=%.2f box=%s",
                        label,
                        conf,
                        threshold,
                        det.get("box"),
                    )
                    beds.append(det)
                else:
                    logger.info(
                        "[OBJ] bed filtered class=%s conf=%.3f threshold=%.2f box=%s",
                        label,
                        conf,
                        threshold,
                        det.get("box"),
                    )
                continue
            if conf >= threshold:
                non_beds.append(det)
            else:
                logger.info(
                    "[OBJ] filtered class=%s conf=%.3f threshold=%.2f box=%s",
                    label,
                    conf,
                    threshold,
                    det.get("box"),
                )

        selected_non_beds = (non_beds or [])[: max(0, self.max_detections - len(beds))]
        return selected_non_beds + beds

    def process_frame(
        self,
        frame: np.ndarray,
        stream_config: Optional[Dict[str, Any]] = None,
        use_reasoning: bool = True,
    ) -> Dict[str, Any]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("process_frame expects a valid cv2 image array")

        run_llama = self._should_run_llama(stream_config)

        output = frame.copy()
        result: Dict[str, Any] = {
            "annotated_frame": output,
            "bbox": None,
            "intent": "no patient detected",
            "alert_triggered": False,
            "pose_summary": {"available": False},
            "model_status": dict(self.model_status),
        }

        detection_list: List[Dict[str, Any]] = []
        try:
            detection_list = self._run_object_detection(output)
        except Exception as exc:
            logger.exception("Object detection stage crashed: %s", exc)
            self._draw_pipeline_status(output, "Detection recovering", (0, 0, 255))
            return result

        if not detection_list:
            if self.yolo_model is None:
                self._draw_pipeline_status(output, "Object model unavailable", (0, 165, 255))
            else:
                self._draw_pipeline_status(output, "Scanning...")
            self._smooth_box_state = None
            return result

        primary = detection_list[0]
        x1, y1, x2, y2 = self._smooth_box(primary["box"])
        crop = output[y1:y2, x1:x2]
        confidence = float(self._obj_conf_smoother.update(float(primary["confidence"])))
        detection_hint = "patient detected" if confidence >= self.min_confidence else "patient under observation"

        pose_landmarks = None
        role_hint = "unknown"
        try:
            if crop.size > 0:
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="medpercept") as executor:
                    pose_future = executor.submit(self._extract_pose_landmarks, crop, x1, y1)
                    role_future = executor.submit(self.detect_role, crop)
                    pose_landmarks = pose_future.result()
                    role_hint, _role_conf = role_future.result()
            if crop.size > 0 and pose_landmarks is not None:
                crop_h, crop_w = crop.shape[:2]
                self._draw_skeleton(output, pose_landmarks, x1, y1, crop_w, crop_h)
        except Exception as exc:
            logger.warning("Parallel perception stage failed: %s", exc)

        intent_text = detection_hint
        alert_triggered = False
        pose_summary: Dict[str, Any] = {"available": False}
        try:
            intent_text, alert_triggered, pose_summary = self._infer_intent(
                pose_landmarks,
                crop,
                detection_hint,
                role_hint,
                use_reasoning=run_llama,
            )
        except Exception as exc:
            logger.warning("Intent reasoning stage failed: %s", exc)
            intent_text, alert_triggered, pose_summary = self._infer_intent_heuristic(
                pose_landmarks,
                detection_hint,
                role_hint,
            )

        try:
            for rank, detection in enumerate(detection_list):
                bx1, by1, bx2, by2 = detection["box"]
                det_conf = float(detection["confidence"])
                is_primary = rank == 0
                if is_primary:
                    label = f"patient {confidence:.2f} | {intent_text}"
                    self._draw_yolo_box(output, (x1, y1, x2, y2), label, alert_triggered)
                else:
                    label = f"track {det_conf:.2f}"
                    self._draw_yolo_box(output, (bx1, by1, bx2, by2), label, False)
        except Exception as exc:
            logger.warning("Annotation draw stage failed: %s", exc)

        trace = self._last_reasoning_trace
        self.model_status["intent_reasoning"] = reasoning_mode_label(
            run_llama,
            True,
            llama_available=bool(trace.get("llama_available")),
            llama_used=bool(trace.get("llama_invoked")),
        )
        status_snapshot = dict(self.model_status)
        try:
            self._draw_model_status_bar(output)
        except Exception as exc:
            logger.debug("Model status bar draw failed: %s", exc)

        result.update(
            {
                "annotated_frame": output,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                },
                "intent": intent_text,
                "alert_triggered": alert_triggered,
                "pose_summary": pose_summary,
                "model_status": status_snapshot,
                "use_reasoning": run_llama,
            }
        )
        return result

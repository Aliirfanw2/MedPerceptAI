from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp_python = None
    mp_vision = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OBJECT_MODEL_PATH = PROJECT_ROOT / "model_weights" / "obj.pt"
DEFAULT_ROLE_MODEL_PATH = PROJECT_ROOT / "model_weights" / "roles.pt"
DEFAULT_POSE_MODEL_PATH = PROJECT_ROOT / "model_weights" / "pose_landmarker_lite.task"
DEFAULT_YOLO_POSE_MODEL_PATH = PROJECT_ROOT / "model_weights" / "yolov8n-pose.pt"
DEFAULT_REASONING_MODEL_PATH = PROJECT_ROOT / "model_weights" / "llama-3-8b-instruct.Q4_K_M.gguf"
DEFAULT_FALLBACK_VIDEO = PROJECT_ROOT / "media" / "test_video.mp4"

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (27, 31),
    (28, 32),
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
]


class PatientIntentPipeline:
    """YOLO + MediaPipe + local reasoning pipeline for patient intent detection."""

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    def __init__(
        self,
        weights_path: Optional[str] = None,
        role_weights_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        yolo_pose_model_path: Optional[str] = None,
        reasoning_model_path: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        object_weights = weights_path or os.environ.get("YOLO_OBJECT_MODEL_PATH")
        role_weights = role_weights_path or os.environ.get("YOLO_ROLE_MODEL_PATH")
        self.weights_path = Path(object_weights) if object_weights else DEFAULT_OBJECT_MODEL_PATH
        self.role_weights_path = Path(role_weights) if role_weights else DEFAULT_ROLE_MODEL_PATH
        self.pose_model_path = Path(
            pose_model_path or os.environ.get("MEDIAPIPE_POSE_MODEL_PATH", str(DEFAULT_POSE_MODEL_PATH))
        )
        self.yolo_pose_model_path = Path(
            yolo_pose_model_path or os.environ.get("YOLO_POSE_MODEL_PATH", str(DEFAULT_YOLO_POSE_MODEL_PATH))
        )
        self.reasoning_model_path = Path(
            reasoning_model_path or os.environ.get("REASONING_MODEL_PATH", str(DEFAULT_REASONING_MODEL_PATH))
        )
        default_pose_backend = "yolo" if self.yolo_pose_model_path.exists() else "auto"
        self.pose_backend_mode = os.environ.get("POSE_BACKEND", default_pose_backend).strip().lower()
        self.vlm_model_name = vlm_model_name or os.environ.get("VLM_MODEL_NAME")
        self.enable_vlm = os.environ.get("ENABLE_VLM", "0") == "1" or bool(self.vlm_model_name)
        self.enable_reasoning = os.environ.get("ENABLE_REASONING", "1") != "0"
        self.min_confidence = float(os.environ.get("YOLO_CONFIDENCE", "0.25"))

        self.model_status: Dict[str, str] = {
            "object_detection": "pending",
            "role_classification": "pending",
            "pose_estimation": "pending",
            "intent_reasoning": "pending",
        }

        self.yolo_model: Optional[YOLO] = None
        self.role_model: Optional[YOLO] = None
        self.pose_backend: Dict[str, Any] = {"mode": "disabled", "instance": None}
        self.vlm_pipeline = None
        self.reasoning_pipeline = None

        try:
            self.yolo_model = self._load_yolo_model()
            self.model_status["object_detection"] = "ready" if self.yolo_model is not None else "unavailable"
        except Exception as exc:
            logger.exception("Object detection model failed to initialize: %s", exc)
            self.model_status["object_detection"] = "error"

        try:
            self.role_model = self._load_role_model()
            self.model_status["role_classification"] = "ready" if self.role_model is not None else "unavailable"
        except Exception as exc:
            logger.exception("Role classification model failed to initialize: %s", exc)
            self.model_status["role_classification"] = "error"

        try:
            self.pose_backend = self._load_pose_backend()
            pose_ready = self.pose_backend.get("mode") != "disabled" and self.pose_backend.get("instance") is not None
            self.model_status["pose_estimation"] = "ready" if pose_ready else "unavailable"
        except Exception as exc:
            logger.exception("Pose estimation backend failed to initialize: %s", exc)
            self.pose_backend = {"mode": "disabled", "instance": None}
            self.model_status["pose_estimation"] = "error"

        self.model_status["intent_reasoning"] = (
            "ready" if self.reasoning_model_path.exists() and Llama is not None else "unavailable"
        )

        logger.info("Pipeline model status: %s", self.model_status)

    def _load_yolo_model(self) -> Optional[YOLO]:
        candidate = self.weights_path
        if not candidate.exists():
            logger.warning("Object detection model missing: %s", candidate)
            fallback = PROJECT_ROOT / "yolov8n.pt"
            if fallback.exists():
                logger.warning("Falling back to %s", fallback)
                candidate = fallback
            else:
                logger.warning("No local YOLO fallback was found; ultralytics may attempt to resolve yolov8n.pt.")
                candidate = Path("yolov8n.pt")
        try:
            return YOLO(str(candidate))
        except Exception as exc:
            logger.warning("Falling back to default YOLO weights after load failure: %s", exc)
            try:
                return YOLO("yolov8n.pt")
            except Exception as fallback_exc:
                logger.error("Default YOLO weights could not be loaded: %s", fallback_exc)
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

    def _ensure_pose_model_file(self) -> Optional[Path]:
        if self.pose_model_path.exists():
            return self.pose_model_path

        logger.warning("MediaPipe pose model missing: %s", self.pose_model_path)
        return None

    def _load_pose_backend(self) -> Dict[str, Any]:
        if self.pose_backend_mode in {"mediapipe", "auto"} and mp_python is not None and mp_vision is not None:
            model_path = self._ensure_pose_model_file()
            if model_path is not None:
                try:
                    options = mp_vision.PoseLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    return {"mode": "tasks", "instance": mp_vision.PoseLandmarker.create_from_options(options)}
                except Exception as exc:
                    logger.warning("MediaPipe tasks pose backend could not be loaded: %s", exc)

        if self.pose_backend_mode in {"yolo", "auto"}:
            if self.yolo_pose_model_path.exists():
                try:
                    return {"mode": "yolo", "instance": YOLO(str(self.yolo_pose_model_path))}
                except Exception as exc:
                    logger.warning("YOLO pose backend could not be loaded (%s): %s", self.yolo_pose_model_path, exc)
            else:
                logger.warning("YOLO pose model missing: %s", self.yolo_pose_model_path)

        logger.warning("Pose backend unavailable; pipeline will fall back to heuristic intent scoring only.")
        return {"mode": "disabled", "instance": None}

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

    def _load_reasoning_pipeline(self):
        if not self.enable_reasoning:
            return None

        if Llama is None:
            logger.warning("llama-cpp-python is unavailable; reasoning inference is disabled.")
            return None

        if not self.reasoning_model_path.exists():
            logger.warning(
                "Reasoning model path does not exist: %s. Set REASONING_MODEL_PATH to your local .gguf file or folder.",
                self.reasoning_model_path,
            )
            return None

        model_path = self.reasoning_model_path
        if model_path.is_dir():
            gguf_files = sorted(model_path.glob("*.gguf"))
            if not gguf_files:
                logger.warning("No .gguf files found in reasoning model directory: %s", model_path)
                return None
            model_path = gguf_files[0]

        requested_gpu_layers = int(os.environ.get("LLAMA_N_GPU_LAYERS", "0"))
        if requested_gpu_layers == -1:
            if torch.cuda.is_available():
                n_gpu_layers = -1
            else:
                logger.warning("LLAMA_N_GPU_LAYERS=-1 requested, but CUDA is unavailable; falling back to CPU mode.")
                n_gpu_layers = 0
        else:
            n_gpu_layers = max(requested_gpu_layers, 0)

        n_ctx = int(os.environ.get("LLAMA_N_CTX", "2048"))
        n_batch = int(os.environ.get("LLAMA_N_BATCH", "512"))
        n_threads = int(os.environ.get("LLAMA_N_THREADS", "4"))

        try:
            return Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("Reasoning pipeline could not be loaded (%s): %s", model_path, exc)
            return None

    def _get_vlm_pipeline(self):
        if self.vlm_pipeline is None:
            self.vlm_pipeline = self._load_vlm_pipeline()
        return self.vlm_pipeline

    def _get_reasoning_pipeline(self):
        if self.reasoning_pipeline is None:
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

    def _summarize_pose(self, pose_landmarks: Optional[Any]) -> Dict[str, Any]:
        if not pose_landmarks:
            return {"available": False}

        landmarks = pose_landmarks
        pose_score = sum(getattr(point, "visibility", 0.0) for point in landmarks) / max(len(landmarks), 1)

        if len(landmarks) <= self.RIGHT_KNEE:
            return {"available": True, "pose_score": round(float(pose_score), 3)}

        nose = landmarks[self.NOSE]
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        left_knee = landmarks[self.LEFT_KNEE]
        right_knee = landmarks[self.RIGHT_KNEE]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        knee_y = (left_knee.y + right_knee.y) / 2.0
        torso_length = abs(hip_y - shoulder_y)
        leg_extension = abs(knee_y - hip_y)
        standing_hint = torso_length < 0.28 and nose.y < hip_y and leg_extension > 0.08 and pose_score > 0.25

        return {
            "available": True,
            "pose_score": round(float(pose_score), 3),
            "standing_hint": bool(standing_hint),
            "shoulder_y": round(float(shoulder_y), 3),
            "hip_y": round(float(hip_y), 3),
            "knee_y": round(float(knee_y), 3),
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
        try:
            prompt = self._build_reasoning_prompt(
                {"bbox": None, "intent_hint": detection_hint, "role_hint": role_hint},
                pose_summary,
                caption_text,
            )
            reasoning_pipeline = self._get_reasoning_pipeline()
            if reasoning_pipeline is not None:
                generated = reasoning_pipeline(
                    prompt,
                    max_tokens=64,
                    temperature=0.1,
                    top_p=0.9,
                    stop=["</s>", "<|eot_id|>", "\n\n"],
                )
                reasoning_output = self._safe_text_generation(generated).strip()
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

        if standing_hint:
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
        if standing_hint:
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
        if not use_reasoning:
            return self._infer_intent_heuristic(pose_landmarks, detection_hint, role_hint)

        try:
            return self._run_reasoning_model(pose_landmarks, crop, detection_hint, role_hint)
        except Exception as exc:
            logger.warning("Intent inference failed; using heuristic fallback: %s", exc)
            return self._infer_intent_heuristic(pose_landmarks, detection_hint, role_hint)

    def _infer_role_hint(self, crop: np.ndarray) -> str:
        if self.role_model is None or crop.size == 0:
            return "unknown"

        try:
            predictions = self.role_model.predict(crop, verbose=False)
            if not predictions:
                return "unknown"
            boxes = predictions[0].boxes
            names = getattr(predictions[0], "names", {}) or {}
            if boxes is not None and len(boxes) and hasattr(boxes, "cls"):
                best_index = int(torch.argmax(boxes.conf).item()) if hasattr(boxes, "conf") and len(boxes.conf) else 0
                class_id = int(boxes.cls[best_index].item())
                return str(names.get(class_id, f"class_{class_id}"))
        except Exception as exc:
            logger.debug("Role model inference failed: %s", exc)
        return "unknown"

    def _draw_yolo_box(self, frame: np.ndarray, box: Tuple[int, int, int, int], label: str, alert_triggered: bool) -> None:
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if alert_triggered else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_bg_y1 = max(0, y1 - 28)
        cv2.rectangle(frame, (x1, label_bg_y1), (min(frame.shape[1] - 1, x1 + 420), y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 8, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_skeleton(self, output: np.ndarray, pose_landmarks: Any, x1: int, y1: int, crop_w: int, crop_h: int) -> None:
        if pose_landmarks is None:
            return

        points: Dict[int, Tuple[int, int]] = {}
        try:
            if self.pose_backend["mode"] == "tasks":
                for idx, lm in enumerate(pose_landmarks):
                    visibility = getattr(lm, "visibility", 1.0)
                    if visibility < 0.3:
                        continue
                    px = int(lm.x * crop_w) + x1
                    py = int(lm.y * crop_h) + y1
                    points[idx] = (px, py)

            elif self.pose_backend["mode"] == "yolo":
                for idx, pt in enumerate(pose_landmarks):
                    px = int(pt[0].item()) + x1
                    py = int(pt[1].item()) + y1
                    if px > x1 and py > y1:
                        points[idx] = (px, py)

            for pt in points.values():
                cv2.circle(output, pt, 4, (0, 255, 255), -1)

            connections = POSE_CONNECTIONS
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                connections = mp.solutions.pose.POSE_CONNECTIONS

            for start_idx, end_idx in connections:
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
        if crop.size == 0:
            return None

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if self.pose_backend["mode"] == "tasks" and self.pose_backend.get("instance") is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                pose_result = self.pose_backend["instance"].detect(mp_image)
                if pose_result.pose_landmarks:
                    return pose_result.pose_landmarks[0]
                return None

            if self.pose_backend["mode"] == "yolo" and self.pose_backend.get("instance") is not None:
                pose_result = self.pose_backend["instance"].predict(crop_rgb, verbose=False)
                if pose_result:
                    candidates = pose_result[0].keypoints
                    if candidates is not None and len(candidates) and hasattr(candidates, "xy"):
                        return candidates.xy[0]
        except Exception as exc:
            logger.warning("Pose estimation inference failed: %s", exc)
            self.model_status["pose_estimation"] = "error"
        return None

    def _run_object_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.yolo_model is None:
            return []

        height, width = frame.shape[:2]
        try:
            detections = self.yolo_model.predict(frame, verbose=False, conf=self.min_confidence)
        except Exception as exc:
            logger.warning("Object detection inference failed: %s", exc)
            self.model_status["object_detection"] = "error"
            return []

        if not detections:
            return []

        boxes = detections[0].boxes
        return self._collect_detection_boxes(boxes, width, height)

    def _collect_detection_boxes(self, boxes: Any, width: int, height: int) -> List[Dict[str, Any]]:
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
            detections.append(
                {
                    "index": index,
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                }
            )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def process_frame(
        self,
        frame: np.ndarray,
        stream_config: Optional[Dict[str, Any]] = None,
        use_reasoning: bool = True,
    ) -> Dict[str, Any]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("process_frame expects a valid cv2 image array")

        if stream_config is not None and "enable_ai_reasoning" in stream_config:
            use_reasoning = bool(stream_config.get("enable_ai_reasoning"))
        elif stream_config is not None and "use_reasoning" in stream_config:
            use_reasoning = bool(stream_config.get("use_reasoning"))

        run_llama = bool(use_reasoning and self.enable_reasoning)
        reasoning_status = "ready" if run_llama else "skipped"
        if run_llama and self.reasoning_model_path.exists() and Llama is None:
            reasoning_status = "unavailable"

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
            return result

        primary = detection_list[0]
        x1, y1, x2, y2 = primary["box"]
        crop = output[y1:y2, x1:x2]
        confidence = float(primary["confidence"])
        detection_hint = "patient detected" if confidence >= self.min_confidence else "patient under observation"

        pose_landmarks = None
        role_hint = "unknown"
        try:
            if crop.size > 0:
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="medpercept") as executor:
                    pose_future = executor.submit(self._extract_pose_landmarks, crop, x1, y1)
                    role_future = executor.submit(self._infer_role_hint, crop)
                    pose_landmarks = pose_future.result()
                    role_hint = role_future.result()
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
                    label = f"patient {det_conf:.2f} | {intent_text}"
                    self._draw_yolo_box(output, (bx1, by1, bx2, by2), label, alert_triggered)
                else:
                    label = f"track {det_conf:.2f}"
                    self._draw_yolo_box(output, (bx1, by1, bx2, by2), label, False)
        except Exception as exc:
            logger.warning("Annotation draw stage failed: %s", exc)

        self.model_status["intent_reasoning"] = reasoning_status
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

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from monitor.runtime.types import FrameJob, ReasoningResult

OVERLAY_MAX_FRAME_LAG = 45
VIDEO_OVERLAY_MAX_FRAME_LAG = 180


class StateManager:
    """Thread-safe shared buffers for frames, overlays, and API-facing inference state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._frame_id = 0
        self._latest_job: Optional[FrameJob] = None
        self._latest_display_chunk: Optional[bytes] = None
        self._latest_reasoning: Optional[ReasoningResult] = None
        self._latest_perception: Optional[ReasoningResult] = None
        self._stream_fps = 0
        self._frame_size: Tuple[int, int] = (0, 0)
        self._capture_active = False
        self._capture_frame_id = 0
        self._capture_exit_reason = ""

    def next_frame_id(self) -> int:
        with self._lock:
            self._frame_id += 1
            return self._frame_id

    def note_capture_frame(self, frame_id: int) -> None:
        with self._lock:
            self._capture_active = True
            self._capture_frame_id = frame_id

    def mark_capture_ended(self, reason: str) -> None:
        with self._lock:
            self._capture_active = False
            self._capture_exit_reason = reason or "unknown"

    def is_capture_active(self) -> bool:
        with self._lock:
            return self._capture_active

    def get_capture_frame_id(self) -> int:
        with self._lock:
            return self._capture_frame_id

    def get_capture_exit_reason(self) -> str:
        with self._lock:
            return self._capture_exit_reason

    def publish_capture_job(self, job: FrameJob) -> None:
        with self._lock:
            self._latest_job = job

    def get_latest_job(self) -> Optional[FrameJob]:
        with self._lock:
            return self._latest_job

    def set_display_chunk(self, chunk: Optional[bytes]) -> None:
        if not chunk:
            return
        with self._lock:
            self._latest_display_chunk = chunk

    def get_display_chunk(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_display_chunk

    @staticmethod
    def _has_llm_decision(result: Optional[ReasoningResult]) -> bool:
        if result is None:
            return False
        scores = result.confidence_scores or {}
        structured = scores.get("structured_reasoning") or {}
        if str(structured.get("decision_source") or "") in ("llama", "fallback"):
            return True
        mode = str(scores.get("reasoning_mode") or "")
        return bool(result.use_reasoning) or mode.startswith("llama") or mode == "fallback"

    def publish_reasoning_result(self, result: ReasoningResult) -> None:
        """Store overlay state; LLM decisions must not be displaced by perception previews."""
        with self._lock:
            current = self._latest_reasoning
            incoming_llm = self._has_llm_decision(result)
            current_llm = self._has_llm_decision(current)

            if current is None:
                self._latest_reasoning = result
                return
            if incoming_llm and not current_llm:
                self._latest_reasoning = result
                return
            if not incoming_llm and current_llm:
                return
            if result.frame_id >= int(current.frame_id or 0):
                self._latest_reasoning = result

    def publish_perception_preview(self, result: ReasoningResult) -> None:
        """Fresh YOLO/fusion overlay for the MJPEG stream (independent of slow LLM decisions)."""
        with self._lock:
            current = self._latest_perception
            if current is None or int(result.frame_id or 0) >= int(current.frame_id or 0):
                self._latest_perception = result

    def get_video_overlay_snapshot(self, display_frame_id: int) -> Optional[ReasoningResult]:
        """Overlay for live video — uses fast perception buffer, not throttled LLM results."""
        with self._lock:
            result = self._latest_perception or self._latest_reasoning
            if result is None:
                return None
            lag = display_frame_id - int(result.frame_id or 0)
            exit_reason = self._capture_exit_reason or None
            stale = (
                (not self._capture_active)
                or bool(exit_reason)
                or lag > VIDEO_OVERLAY_MAX_FRAME_LAG
            )
            scores = dict(result.confidence_scores or {})
            scores["capture_frame_id"] = display_frame_id
            scores["overlay_frame_id"] = int(result.frame_id or 0)
            scores["overlay_stale"] = stale
            scores["capture_active"] = self._capture_active
            scores["capture_exit_reason"] = exit_reason
            scores["video_overlay"] = True
            return replace(
                result,
                alert_triggered=False if stale else result.alert_triggered,
                confidence_scores=scores,
            )

    def get_perception_scene(self, capture_frame_id: int = 0) -> Dict[str, Any]:
        """Latest fused scene from overlay buffer (may be ahead of LLM decision frame)."""
        overlay = self.get_overlay_snapshot(int(capture_frame_id or 0))
        if overlay is None:
            return {}
        scores = dict(overlay.confidence_scores or {})
        scene = scores.get("scene")
        return dict(scene) if isinstance(scene, dict) else {}

    def get_overlay_snapshot(self, display_frame_id: int) -> Optional[ReasoningResult]:
        with self._lock:
            result = self._latest_reasoning
            if result is None:
                return None
            lag = display_frame_id - int(result.frame_id or 0)
            exit_reason = self._capture_exit_reason or None
            stale = (
                (not self._capture_active)
                or bool(exit_reason)
                or lag > OVERLAY_MAX_FRAME_LAG
            )
            scores = dict(result.confidence_scores or {})
            scores["capture_frame_id"] = display_frame_id
            scores["overlay_frame_id"] = int(result.frame_id or 0)
            scores["overlay_stale"] = stale
            scores["capture_active"] = self._capture_active
            scores["capture_exit_reason"] = exit_reason
            if stale:
                scores["safety_label"] = "STALE"
            intent = result.intent
            if stale and not self._capture_active:
                intent = f"Capture stopped — stale frame ({exit_reason or 'inactive'})"
            elif stale:
                intent = "Awaiting fresh perception…"
            return replace(
                result,
                intent=intent,
                alert_triggered=False if stale else result.alert_triggered,
                confidence_scores=scores,
            )

    def update_stream_metrics(self, stream_fps: int, width: int, height: int) -> None:
        with self._lock:
            self._stream_fps = stream_fps
            self._frame_size = (width, height)

    def get_stream_metrics(self) -> Tuple[int, Tuple[int, int]]:
        with self._lock:
            return self._stream_fps, self._frame_size

    def begin_capture_session(self) -> None:
        with self._lock:
            self._capture_active = True
            self._capture_exit_reason = ""
            self._capture_frame_id = 0

    def live_snapshot(self, overlay_frame_id: int = 0) -> Dict[str, Any]:
        """Authoritative capture fields for API/logs. Stale = capture stopped only (not frame lag)."""
        with self._lock:
            capture_active = self._capture_active
            capture_fid = self._capture_frame_id
            exit_reason = (self._capture_exit_reason or "").strip() or None
            if capture_active:
                exit_reason = None
            lag = max(0, int(capture_fid or 0) - int(overlay_frame_id or 0))
            # LLM runs on throttled fused frames; large frame lag is normal while capture is live.
            overlay_stale = not capture_active
            return {
                "capture_active": capture_active,
                "capture_frame_id": capture_fid,
                "capture_exit_reason": exit_reason,
                "overlay_stale": overlay_stale,
                "overlay_frame_id": int(overlay_frame_id or 0),
                "overlay_frame_lag": lag,
            }

    def reset(self) -> None:
        with self._lock:
            self._frame_id = 0
            self._latest_job = None
            self._latest_display_chunk = None
            self._latest_reasoning = None
            self._latest_perception = None
            self._stream_fps = 0
            self._frame_size = (0, 0)
            self._capture_active = False
            self._capture_frame_id = 0
            self._capture_exit_reason = ""

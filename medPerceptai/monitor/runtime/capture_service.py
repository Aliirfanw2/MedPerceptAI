from __future__ import annotations

import logging
import threading
import time
from queue import Full, Queue
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from monitor import presentation_config as pc
from monitor.presentation_config import FRAME_SLEEP_SECONDS
from monitor.runtime.state_manager import StateManager
from monitor.runtime.types import FrameJob, ReasoningResult

logger = logging.getLogger(__name__)


class CaptureService:
    """Reads OpenCV frames at target FPS and publishes FrameJob packets without blocking on AI."""

    def __init__(
        self,
        state: StateManager,
        ingress_queues: list[Queue],
        stop_event: threading.Event,
        read_frame_fn: Callable,
        video_interval_fn: Callable,
        encode_display_fn: Callable[[np.ndarray, Dict[str, Any], ReasoningResult | None], Optional[bytes]],
        on_stream_metrics: Callable[[int, int, int], None],
        on_offline: Callable[[], None],
        inference_stride: int,
        on_capture_ended: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._state = state
        self._ingress_queues = ingress_queues
        self._stop = stop_event
        self._read_frame = read_frame_fn
        self._video_interval = video_interval_fn
        self._encode_display = encode_display_fn
        self._on_stream_metrics = on_stream_metrics
        self._on_offline = on_offline
        self._on_capture_ended = on_capture_ended
        self._inference_stride = max(1, inference_stride)
        self._thread: Optional[threading.Thread] = None

    def start(self, capture: cv2.VideoCapture, source_name: str, config: Dict[str, Any]) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop,
            args=(capture, source_name, config),
            name="medpercept-capture-service",
            daemon=True,
        )
        logger.info("[runtime] CaptureService thread starting for %s", source_name)
        self._thread.start()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    @staticmethod
    def _resize_frame(frame: np.ndarray) -> np.ndarray:
        target_w = int(pc.MONITOR_FRAME_WIDTH or 0)
        target_h = int(pc.MONITOR_FRAME_HEIGHT or 0)
        if target_w <= 0 or target_h <= 0:
            return frame
        height, width = frame.shape[:2]
        if width == target_w and height == target_h:
            return frame
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _loop(self, capture: cv2.VideoCapture, source_name: str, config: Dict[str, Any]) -> None:
        video_mode = source_name.startswith("video:")
        frame_interval = self._video_interval(capture) if video_mode else FRAME_SLEEP_SECONDS
        consecutive_failures = 0
        fps_window_start = time.time()
        fps_frame_count = 0
        stream_fps = 0
        exit_reason = "unknown"

        logger.info("[runtime] CaptureService loop entered (video_mode=%s)", video_mode)
        try:
            while not self._stop.is_set():
                if not capture.isOpened():
                    exit_reason = "source closed"
                    logger.warning("[runtime] CaptureService exit reason=%s", exit_reason)
                    break

                success, frame = self._read_frame(capture, source_name)
                if not success or frame is None:
                    consecutive_failures += 1
                    if video_mode:
                        if consecutive_failures >= 3:
                            logger.info(
                                "[runtime] CaptureService video ended — rewinding source (failures=%s)",
                                consecutive_failures,
                            )
                        try:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        except cv2.error as exc:
                            exit_reason = "video ended"
                            logger.warning(
                                "[runtime] CaptureService exit reason=%s rewind failed: %s",
                                exit_reason,
                                exc,
                            )
                            break
                        time.sleep(frame_interval)
                        continue
                    self._on_offline()
                    if consecutive_failures >= 30:
                        exit_reason = "frame read failed"
                        logger.warning(
                            "[runtime] CaptureService exit reason=%s (failures=%s)",
                            exit_reason,
                            consecutive_failures,
                        )
                        break
                    time.sleep(0.2)
                    continue

                consecutive_failures = 0
                frame = self._resize_frame(frame)
                frame_id = self._state.next_frame_id()
                self._state.note_capture_frame(frame_id)
                frame_ts = time.time()
                from monitor.runtime.pipeline_diagnostics import get_diagnostics

                get_diagnostics().note_capture_frame(frame_id, ts=frame_ts)
                fps_frame_count += 1
                elapsed = time.time() - fps_window_start
                if elapsed >= 1.0:
                    stream_fps = int(round(fps_frame_count / elapsed))
                    fps_frame_count = 0
                    fps_window_start = time.time()

                job = FrameJob(
                    frame_id=frame_id,
                    frame=frame,
                    config=config,
                    timestamp=frame_ts,
                )
                self._state.publish_capture_job(job)

                if frame_id % self._inference_stride == 0:
                    self._enqueue_job(job)

                overlay = self._state.get_video_overlay_snapshot(frame_id)
                chunk = self._encode_display(frame, config, overlay)
                if chunk:
                    self._state.set_display_chunk(chunk)
                else:
                    logger.warning("[runtime] JPEG encode returned empty frame_id=%s", frame_id)

                height, width = frame.shape[:2]
                self._state.update_stream_metrics(stream_fps, width, height)
                self._on_stream_metrics(stream_fps, width, height)

                time.sleep(frame_interval)
            else:
                if self._stop.is_set():
                    exit_reason = "stop requested (browser disconnected or orchestrator stop)"
                    logger.info("[runtime] CaptureService exit reason=%s", exit_reason)
        except cv2.error as exc:
            exit_reason = "opencv error"
            logger.exception("[runtime] CaptureService exit reason=%s error=%s", exit_reason, exc)
        except Exception as exc:
            exit_reason = "capture loop crashed"
            logger.exception("[runtime] CaptureService exit reason=%s error=%s", exit_reason, exc)
        finally:
            if exit_reason == "unknown":
                exit_reason = "capture loop exited"
            self._state.mark_capture_ended(exit_reason)
            if self._on_capture_ended:
                try:
                    self._on_capture_ended(exit_reason)
                except Exception as exc:
                    logger.warning("[runtime] on_capture_ended callback failed: %s", exc)
            logger.warning("[runtime] CaptureService loop exited for %s reason=%s", source_name, exit_reason)
            try:
                if capture.isOpened():
                    capture.release()
            except Exception:
                pass

    def _enqueue_job(self, job: FrameJob) -> None:
        for queue in self._ingress_queues:
            try:
                queue.put_nowait(job)
            except Full:
                try:
                    queue.get_nowait()
                except Exception:
                    pass
                try:
                    queue.put_nowait(job)
                except Exception:
                    pass

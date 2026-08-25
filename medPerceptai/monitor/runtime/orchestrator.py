from __future__ import annotations

import logging
import threading
from queue import Queue
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from monitor import presentation_config as pc
from monitor.presentation_config import RUNTIME_QUEUE_SIZE
from monitor.runtime.aggregator import PerceptionAggregator
from monitor.runtime.context_builder import context_to_overlay_result
from monitor.runtime.capture_service import CaptureService
from monitor.runtime.reasoning_worker import ReasoningWorker
from monitor.runtime.state_manager import StateManager
from monitor.runtime.types import ReasoningResult
from monitor.runtime.workers.object_worker import ObjectDetectionWorker
from monitor.runtime.workers.pose_worker import PoseWorker
from monitor.runtime.workers.role_worker import RoleWorker

logger = logging.getLogger(__name__)


class StreamOrchestrator:
    """Starts capture + parallel full-frame perception workers + async reasoning."""

    def __init__(
        self,
        *,
        state: StateManager,
        pipeline_getter: Callable,
        read_frame_fn: Callable,
        video_interval_fn: Callable,
        build_overlay_frame_fn: Callable[[np.ndarray, Dict[str, Any], ReasoningResult | None], np.ndarray],
        encode_chunk_fn: Callable[[np.ndarray, Dict[str, Any], bool, str], Optional[bytes]],
        on_reasoning_result: Callable[[ReasoningResult, Dict[str, Any], str], None],
        on_stream_metrics: Callable[[int, int, int], None],
        on_offline: Callable[[], None],
        on_capture_ended: Optional[Callable[[str], None]] = None,
        inference_stride: int = 2,
        queue_size: int = RUNTIME_QUEUE_SIZE,
    ) -> None:
        self.state = state
        self._pipeline_getter = pipeline_getter
        self._read_frame = read_frame_fn
        self._video_interval = video_interval_fn
        self._build_overlay = build_overlay_frame_fn
        self._encode_chunk = encode_chunk_fn
        self._on_reasoning_result = on_reasoning_result
        self._on_stream_metrics = on_stream_metrics
        self._on_offline = on_offline
        self._on_capture_ended = on_capture_ended
        self._inference_stride = inference_stride
        self._stop = threading.Event()
        self._config: Dict[str, Any] = {}
        self._source_name = ""

        self._object_queue: Queue = Queue(maxsize=queue_size)
        self._role_queue: Queue = Queue(maxsize=queue_size)
        self._pose_queue: Queue = Queue(maxsize=queue_size)
        self._reasoning: Queue = Queue(maxsize=queue_size)

        self._aggregator = PerceptionAggregator(
            self._reasoning,
            on_context_ready=self._publish_perception_preview,
        )
        self._capture: Optional[CaptureService] = None
        self._object_worker: Optional[ObjectDetectionWorker] = None
        self._role_worker: Optional[RoleWorker] = None
        self._pose_worker: Optional[PoseWorker] = None
        self._reasoning_worker: Optional[ReasoningWorker] = None

    def start(self, capture: cv2.VideoCapture, source_name: str, config: Dict[str, Any]) -> None:
        self._stop.clear()
        self._config = config
        self._source_name = source_name
        self.state.begin_capture_session()
        from monitor.runtime.pipeline_diagnostics import get_diagnostics

        get_diagnostics().begin_capture_session()
        self._object_worker = ObjectDetectionWorker(
            self._object_queue,
            self._aggregator,
            self._pipeline_getter,
            self._stop,
        )
        self._role_worker = RoleWorker(self._role_queue, self._aggregator, self._pipeline_getter, self._stop)
        self._pose_worker = PoseWorker(self._pose_queue, self._aggregator, self._pipeline_getter, self._stop)
        self._reasoning_worker = ReasoningWorker(
            self._reasoning,
            self._pipeline_getter,
            self._handle_reasoning_result,
            self._stop,
        )

        logger.info(
            "[runtime] Starting workers: object, role, pose, reasoning "
            "(full-frame parallel, fusion_timeout_ms=%s)",
            pc.FUSION_TIMEOUT_MS,
        )
        self._object_worker.start()
        self._role_worker.start()
        self._pose_worker.start()
        self._reasoning_worker.start()

        def encode_display(frame: np.ndarray, cfg: Dict[str, Any], overlay: Optional[ReasoningResult]) -> Optional[bytes]:
            display = self._build_overlay(frame, cfg, overlay)
            return self._encode_chunk(display, cfg, False, "live")

        self._capture = CaptureService(
            state=self.state,
            ingress_queues=[self._object_queue, self._role_queue, self._pose_queue],
            stop_event=self._stop,
            read_frame_fn=self._read_frame,
            video_interval_fn=self._video_interval,
            encode_display_fn=encode_display,
            on_stream_metrics=self._on_stream_metrics,
            on_offline=self._on_offline,
            inference_stride=self._inference_stride,
            on_capture_ended=self._on_capture_ended,
        )
        logger.info("[runtime] Starting CaptureService for %s", source_name)
        self._capture.start(capture, source_name, config)
        logger.info("[runtime] StreamOrchestrator started for %s", source_name)

    def stop(self) -> None:
        self._stop.set()
        exit_reason = "stop requested (browser disconnected or orchestrator stop)"
        self.state.mark_capture_ended(exit_reason)
        if self._on_capture_ended:
            try:
                self._on_capture_ended(exit_reason)
            except Exception as exc:
                logger.warning("[runtime] on_capture_ended callback failed: %s", exc)
        for worker in (self._capture, self._object_worker, self._role_worker, self._pose_worker, self._reasoning_worker):
            if worker:
                worker.join(timeout=3.0)
        self._aggregator.clear()
        self._drain_queues()
        self.state.reset()
        logger.info("StreamOrchestrator stopped")

    def _publish_perception_preview(self, context) -> None:
        """Overlay-only preview — dashboard decisions come from ReasoningWorker (LLM)."""
        try:
            preview = context_to_overlay_result(context)
            self.state.publish_perception_preview(preview)
        except Exception as exc:
            logger.warning(
                "Perception preview failed frame_id=%s: %s",
                context.frame_id,
                exc,
            )

    def _handle_reasoning_result(self, result: ReasoningResult) -> None:
        logger.info(
            "[runtime] Reasoning completed frame_id=%s intent=%s alert=%s latency_ms=%s",
            result.frame_id,
            (result.intent or "")[:48],
            result.alert_triggered,
            result.latency_ms,
        )
        self.state.publish_reasoning_result(result)
        self._on_reasoning_result(result, self._config, self._source_name)

    def _drain_queues(self) -> None:
        for queue in (self._object_queue, self._role_queue, self._pose_queue, self._reasoning):
            while True:
                try:
                    queue.get_nowait()
                except Exception:
                    break

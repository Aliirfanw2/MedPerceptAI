from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

from monitor import presentation_config as pc
from monitor.runtime.aggregator import PerceptionAggregator
from monitor.runtime.types import FrameJob, PoseResult

logger = logging.getLogger(__name__)


class PoseWorker:
    def __init__(
        self,
        ingress_queue: Queue,
        aggregator: PerceptionAggregator,
        pipeline_getter: Callable,
        stop_event: threading.Event,
    ) -> None:
        self._queue = ingress_queue
        self._aggregator = aggregator
        self._get_pipeline = pipeline_getter
        self._stop = stop_event
        self._thread: Optional[threading.Thread] = None
        logger.info("[PIPELINE][POSE] thresholds pose_conf=%.2f object_base=%.2f", pc.POSE_CONF, pc.OBJECT_BASE_CONF)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="medpercept-pose-worker", daemon=True)
        self._thread.start()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job: FrameJob = self._queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                pipeline = self._get_pipeline()
                pose_dets, landmarks, summary = pipeline.detect_poses_frame(
                    job.frame,
                    frame_id=job.frame_id,
                )
                self._aggregator.record_pose(
                    PoseResult(
                        frame_id=job.frame_id,
                        pose_landmarks=landmarks,
                        pose_summary=summary,
                        pose_detections=pose_dets,
                    )
                )
            except Exception as exc:
                logger.exception("Pose worker failed frame_id=%s: %s", job.frame_id, exc)
                self._aggregator.record_pose(PoseResult(frame_id=job.frame_id))

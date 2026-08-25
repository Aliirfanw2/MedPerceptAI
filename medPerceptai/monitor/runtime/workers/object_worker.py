from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

from monitor import presentation_config as pc
from monitor.runtime.aggregator import PerceptionAggregator
from monitor.runtime.types import FrameJob, ObjectResult

logger = logging.getLogger(__name__)


class ObjectDetectionWorker:
    def __init__(
        self,
        ingress_queue: Queue,
        aggregator: PerceptionAggregator,
        pipeline_getter: Callable,
        stop_event: threading.Event,
    ) -> None:
        self._ingress = ingress_queue
        self._aggregator = aggregator
        self._get_pipeline = pipeline_getter
        self._stop = stop_event
        self._thread: Optional[threading.Thread] = None
        logger.info(
            "[PIPELINE][OBJECT] thresholds person_conf=%.2f bed_conf=%.2f object_base=%.2f",
            pc.PERSON_CONF,
            pc.BED_CONF,
            pc.OBJECT_BASE_CONF,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="medpercept-object-worker", daemon=True)
        self._thread.start()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job: FrameJob = self._ingress.get(timeout=0.25)
            except Empty:
                continue

            try:
                pipeline = self._get_pipeline()
                detections, primary, _crop, hint, confidence = pipeline.detect_objects_frame(job.frame)
                result = ObjectResult(
                    frame_id=job.frame_id,
                    detections=detections,
                    primary_box=primary,
                    detection_hint=hint,
                    confidence=confidence,
                )
                self._aggregator.record_object(result, frame=job.frame, config=job.config)
            except Exception as exc:
                logger.exception("Object worker failed frame_id=%s: %s", job.frame_id, exc)
                empty = ObjectResult(frame_id=job.frame_id, detection_hint="no patient detected")
                self._aggregator.record_object(empty, frame=job.frame, config=job.config)

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

from monitor import presentation_config as pc
from monitor.runtime.aggregator import PerceptionAggregator
from monitor.runtime.types import FrameJob, RoleResult

logger = logging.getLogger(__name__)


class RoleWorker:
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
        logger.info(
            "[PIPELINE][ROLE] thresholds role=%.2f patient=%.2f doctor=%.2f nurse=%.2f",
            pc.ROLE_CONF,
            pc.PATIENT_CONF,
            pc.DOCTOR_CONF,
            pc.NURSE_CONF,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="medpercept-role-worker", daemon=True)
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
                role_dets, role_hint, role_conf = pipeline.detect_roles_frame(job.frame)
                self._aggregator.record_role(
                    RoleResult(
                        frame_id=job.frame_id,
                        role_hint=role_hint,
                        role_confidence=role_conf,
                        role_detections=role_dets,
                    )
                )
            except Exception as exc:
                logger.exception("Role worker failed frame_id=%s: %s", job.frame_id, exc)
                self._aggregator.record_role(
                    RoleResult(frame_id=job.frame_id, role_hint="unknown", role_confidence=0.0)
                )

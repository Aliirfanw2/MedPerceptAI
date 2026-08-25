from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

from monitor import presentation_config as pc
from monitor.runtime.human_log import emit_human_reasoning_log
from monitor.runtime.types import ReasoningContext, ReasoningResult

logger = logging.getLogger(__name__)


class ReasoningWorker:
    """Runs Llama/heuristic reasoning on a throttled latest-scene snapshot."""

    def __init__(
        self,
        reasoning_queue: Queue,
        pipeline_getter: Callable,
        on_result: Callable[[ReasoningResult], None],
        stop_event: threading.Event,
    ) -> None:
        self._queue = reasoning_queue
        self._get_pipeline = pipeline_getter
        self._on_result = on_result
        self._stop = stop_event
        self._thread: Optional[threading.Thread] = None
        self._interval = max(0.5, float(pc.REASONING_INTERVAL_SEC))
        logger.info(
            "[PIPELINE][REASONING] interval=%.1fs reasoner=%s",
            self._interval,
            "llama" if pc.reasoning_enabled() else "fallback",
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="medpercept-reasoning-worker", daemon=True)
        self._thread.start()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _drain_latest(self, seed: Optional[ReasoningContext]) -> Optional[ReasoningContext]:
        latest = seed
        while True:
            try:
                candidate: ReasoningContext = self._queue.get_nowait()
                if latest is None or candidate.frame_id >= latest.frame_id:
                    latest = candidate
            except Empty:
                break
        return latest

    def _loop(self) -> None:
        latest_context: Optional[ReasoningContext] = None
        last_run = 0.0

        while not self._stop.is_set():
            try:
                latest_context = self._queue.get(timeout=0.25)
                latest_context = self._drain_latest(latest_context)
            except Empty:
                pass

            if latest_context is None:
                continue

            now = time.time()
            if now - last_run < self._interval:
                continue

            context = latest_context
            latest_context = None
            last_run = now
            started = time.time()
            from monitor.runtime.pipeline_diagnostics import get_diagnostics

            queued_ts = get_diagnostics().get_context_queued_ts(context.frame_id)
            llm_queue_wait_ms = (
                int((started - queued_ts) * 1000) if queued_ts is not None else None
            )

            try:
                pipeline = self._get_pipeline()
                result = pipeline.run_reasoning_from_context(context)
                result.latency_ms = int((time.time() - started) * 1000)
                scores = dict(result.confidence_scores or {})
                scores["reasoning_latency_ms"] = result.latency_ms
                from monitor.runtime.display_fields import enrich_scores_with_capture
                from monitor.views import _RUNTIME_STATE

                snap = _RUNTIME_STATE.live_snapshot(int(result.frame_id or 0))
                structured = dict(scores.get("structured_reasoning") or {})
                import copy

                scene = copy.deepcopy(scores.get("scene") or context.scene or {})
                if int(scene.get("frame_id") or 0) != int(context.frame_id):
                    scene = copy.deepcopy(context.scene or {})
                live_scene = _RUNTIME_STATE.get_perception_scene(int(result.frame_id or context.frame_id))
                scores = enrich_scores_with_capture(
                    scores,
                    structured,
                    scene,
                    snap,
                    intent=str(result.intent or ""),
                    decision_frame_id=int(result.frame_id or context.frame_id),
                    live_scene=live_scene if int(live_scene.get("frame_id") or 0) <= int(result.frame_id or 0) else scene,
                )
                from monitor.runtime.types import ReasoningResult
                from dataclasses import replace

                scores["llm_queue_wait_ms"] = llm_queue_wait_ms
                result = replace(result, confidence_scores=scores)
                structured = scores.get("structured_reasoning") or {}
                capture_fid = int(snap.get("capture_frame_id") or result.frame_id)
                lag = max(0, capture_fid - int(result.frame_id or 0))
                get_diagnostics().on_llm_final(
                    int(result.frame_id),
                    capture_frame_id=capture_fid,
                    lag_frames=lag,
                    llama_latency_ms=scores.get("llama_latency_ms"),
                    safety_label=structured.get("safety_label"),
                )
                risk_result = {
                    "intent": result.intent,
                    "alert_triggered": result.alert_triggered,
                    "risk_score": scores.get("risk_score", result.risk_score),
                    "risk_level": scores.get("risk_level", result.risk_level),
                    "confidence_scores": scores,
                    "monitor_status": result.monitor_status,
                }
                emit_human_reasoning_log(context, result, risk_result)
                self._on_result(result)
            except Exception as exc:
                logger.exception("Reasoning worker failed frame_id=%s: %s", context.frame_id, exc)

"""Minimal pipeline logging — [LLM_INPUT], [LLM_FINAL], [API_STATE] only."""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PipelineDiagnostics:
    """Lightweight capture/LLM timing for the three approved log lines."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session_id = 0
        self._capture_ts: dict[int, float] = {}
        self._context_queued_ts: dict[int, float] = {}
        self._capture_frame_id: Optional[int] = None

    def begin_capture_session(self) -> int:
        with self._lock:
            self._session_id += 1
            self._capture_ts.clear()
            self._context_queued_ts.clear()
            self._capture_frame_id = None
            return self._session_id

    def note_capture_frame(self, frame_id: int, *, ts: float) -> None:
        with self._lock:
            self._capture_ts[int(frame_id)] = float(ts)
            self._capture_frame_id = int(frame_id)

    def note_context_queued(self, frame_id: int, *, ts: float) -> None:
        with self._lock:
            self._context_queued_ts[int(frame_id)] = float(ts)

    def get_context_queued_ts(self, frame_id: int) -> Optional[float]:
        with self._lock:
            return self._context_queued_ts.get(int(frame_id))

    def get_capture_ts(self, frame_id: int) -> Optional[float]:
        with self._lock:
            return self._capture_ts.get(int(frame_id))

    def on_llm_input(
        self,
        frame_id: int,
        *,
        capture_frame_id: int,
        lag_frames: int,
        chars: int,
        llm_queue_wait_ms: Optional[int] = None,
    ) -> None:
        with self._lock:
            logger.info(
                "[LLM_INPUT] session=%s frame=%s capture_frame=%s lag_frames=%s chars=%s queue_wait_ms=%s",
                self._session_id,
                frame_id,
                capture_frame_id,
                lag_frames,
                chars,
                llm_queue_wait_ms if llm_queue_wait_ms is not None else "—",
            )

    def on_llm_final(
        self,
        frame_id: int,
        *,
        capture_frame_id: int,
        lag_frames: int,
        llama_latency_ms: Optional[int],
        safety_label: Any,
    ) -> None:
        with self._lock:
            logger.info(
                "[LLM_FINAL] session=%s frame=%s capture_frame=%s lag_frames=%s llama_latency_ms=%s safety=%s",
                self._session_id,
                frame_id,
                capture_frame_id,
                lag_frames,
                llama_latency_ms if llama_latency_ms is not None else "—",
                safety_label or "not provided",
            )

    def on_api_publish(
        self,
        *,
        llm_frame_id: int,
        capture_frame_id: int,
        lag_frames: int,
        capture_active: bool,
        reasoning_stale: bool,
        api_publish_ms: int,
    ) -> None:
        with self._lock:
            logger.info(
                "[API_STATE] session=%s llm_frame=%s capture_frame=%s lag_frames=%s "
                "capture_active=%s reasoning_stale=%s api_publish_ms=%s",
                self._session_id,
                llm_frame_id,
                capture_frame_id,
                lag_frames,
                capture_active,
                reasoning_stale,
                api_publish_ms,
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self._capture_frame_id is None:
                return {}
            return {"capture_frame_id": self._capture_frame_id}


_DIAG = PipelineDiagnostics()


def get_diagnostics() -> PipelineDiagnostics:
    return _DIAG

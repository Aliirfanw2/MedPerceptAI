from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, Optional

from monitor import presentation_config as pc
from monitor.runtime.context_builder import build_empty_context, build_reasoning_context
from monitor.runtime.fusion import fuse_scene
from monitor.runtime.types import ObjectResult, PoseResult, ReasoningContext, RoleResult

logger = logging.getLogger(__name__)


class PerceptionAggregator:
    """Late-fusion aggregator: merges full-frame object/role/pose by frame_id with timeout."""

    def __init__(
        self,
        reasoning_queue: Queue,
        on_context_ready: Optional[Callable[[ReasoningContext], None]] = None,
        *,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self._reasoning_queue = reasoning_queue
        self._on_context_ready = on_context_ready
        self._timeout_sec = max(0.05, (timeout_ms or pc.FUSION_TIMEOUT_MS) / 1000.0)
        self._lock = threading.Lock()
        self._pending: Dict[int, Dict[str, object]] = {}
        self._meta: Dict[int, Dict[str, Any]] = {}
        logger.info(
            "[PIPELINE][AGGREGATOR] fusion_timeout_ms=%s reason_interval=%.1fs",
            self._timeout_sec * 1000,
            pc.REASONING_INTERVAL_SEC,
        )

    def record_object(
        self,
        result: ObjectResult,
        *,
        frame: Any,
        config: Dict[str, Any],
    ) -> None:
        with self._lock:
            slot = self._pending.setdefault(result.frame_id, {})
            slot["object"] = result
            slot["frame"] = frame
            slot["config"] = config
            self._meta.setdefault(result.frame_id, {"first_seen": time.time()})
            if result.primary_box is None:
                self._try_emit_locked(result.frame_id, force=True)
            else:
                self._try_emit_locked(result.frame_id)

    def record_role(self, result: RoleResult) -> None:
        with self._lock:
            slot = self._pending.setdefault(result.frame_id, {})
            slot["role"] = result
            self._meta.setdefault(result.frame_id, {"first_seen": time.time()})
            self._try_emit_locked(result.frame_id)

    def record_pose(self, result: PoseResult) -> None:
        with self._lock:
            slot = self._pending.setdefault(result.frame_id, {})
            slot["pose"] = result
            self._meta.setdefault(result.frame_id, {"first_seen": time.time()})
            self._try_emit_locked(result.frame_id)

    def _try_emit_locked(self, frame_id: int, force: bool = False) -> None:
        slot = self._pending.get(frame_id)
        if not slot:
            return

        meta = self._meta.get(frame_id, {})
        elapsed = time.time() - float(meta.get("first_seen", time.time()))
        has_object = "object" in slot
        has_role = "role" in slot
        has_pose = "pose" in slot
        timed_out = elapsed >= self._timeout_sec
        all_ready = has_object and has_role and has_pose

        if not force and not all_ready and not timed_out:
            return

        frame = slot.get("frame")
        config = slot.get("config") or {}
        if frame is None:
            return

        missing: list[str] = []
        if not has_object:
            missing.append("object")
        if not has_role:
            missing.append("role")
        if not has_pose:
            missing.append("pose")

        object_result = slot.get("object")
        if not isinstance(object_result, ObjectResult):
            object_result = ObjectResult(frame_id=frame_id, detection_hint="no patient detected")

        role_result = slot.get("role")
        if not isinstance(role_result, RoleResult):
            role_result = RoleResult(frame_id=frame_id)

        pose_result = slot.get("pose")
        if not isinstance(pose_result, PoseResult):
            pose_result = PoseResult(frame_id=frame_id)

        merge_reason = "forced_no_patient" if force and object_result.primary_box is None else (
            "all_ready" if all_ready else "timeout"
        )

        if force and object_result.primary_box is None:
            context = build_empty_context(frame_id, frame, config, object_result)
        else:
            scene = fuse_scene(
                frame_id=frame_id,
                object_result=object_result,
                role_result=role_result,
                pose_result=pose_result,
                missing_signals=missing,
            )
            context = build_reasoning_context(
                frame_id=frame_id,
                frame=frame,
                config=config,
                scene=scene,
                object_result=object_result,
                role_result=role_result,
                pose_result=pose_result,
            )
        self._emit(context, merge_reason=merge_reason, missing=missing)
        self._pending.pop(frame_id, None)
        self._meta.pop(frame_id, None)

    def _emit(
        self,
        context: ReasoningContext,
        *,
        merge_reason: str = "unknown",
        missing: Optional[list[str]] = None,
    ) -> None:
        if self._on_context_ready is not None:
            try:
                self._on_context_ready(context)
            except Exception as exc:
                logger.warning("Perception preview publish failed frame_id=%s: %s", context.frame_id, exc)
        from monitor.runtime.pipeline_diagnostics import get_diagnostics

        get_diagnostics().note_context_queued(int(context.frame_id), ts=time.time())
        try:
            self._reasoning_queue.put_nowait(context)
        except Full:
            try:
                self._reasoning_queue.get_nowait()
            except Empty:
                pass
            try:
                self._reasoning_queue.put_nowait(context)
            except Full:
                logger.debug("Reasoning queue saturated; dropped context frame_id=%s", context.frame_id)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._meta.clear()

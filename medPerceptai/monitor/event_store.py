from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from django.contrib.auth.models import User
from django.db import close_old_connections
from django.utils import timezone

from accounts.permissions import filter_events_for_user, user_is_system_admin
from monitor.models import MonitoringEvent

logger = logging.getLogger(__name__)


def format_relative_time(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "—"
    delta = max(0, int(time.time() - float(timestamp)))
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60} min ago"
    return f"{delta // 3600}h ago"


def _classify_severity(intent: str, alert: bool, *, safety_label: Optional[str] = None) -> str:
    label = str(safety_label or "").strip().upper()
    if alert and label == "ALERT":
        return MonitoringEvent.Severity.CRITICAL
    if label == "MONITOR":
        return MonitoringEvent.Severity.WARNING
    return MonitoringEvent.Severity.INFO


def _is_llm_alert_event(event: Dict[str, Any]) -> bool:
    if not event.get("alert"):
        return False
    bbox = event.get("bbox") if isinstance(event.get("bbox"), dict) else {}
    label = str(bbox.get("safety_label") or "").strip().upper()
    return label == "ALERT"


def event_to_dict(event: MonitoringEvent) -> Dict[str, Any]:
    ts = event.created_at.timestamp() if event.created_at else time.time()
    bbox = event.bbox if isinstance(event.bbox, dict) else {}
    return {
        "id": event.pk,
        "ts": ts,
        "intent": event.intent,
        "alert": event.alert,
        "camera_id": event.camera_id,
        "source": event.source,
        "building": event.building,
        "floor": event.floor,
        "floor_number": event.floor,
        "room_number": event.room_number,
        "bbox": event.bbox,
        "latency_ms": event.latency_ms,
        "frame_id": event.frame_id,
        "role_hint": event.role_hint,
        "severity": event.severity,
        "event": event.event_type,
        "risk_score": bbox.get("risk_score"),
        "risk_level": bbox.get("risk_level"),
    }


def _events_queryset_for_user(user: Optional[User]):
    qs = MonitoringEvent.objects.all().order_by("-created_at")
    if user is None or not user.is_authenticated:
        return qs.none()
    if user_is_system_admin(user):
        return qs
    try:
        profile = user.staff_profile
    except Exception:
        return qs.none()
    return qs.filter(
        building=profile.assigned_building,
        floor=profile.assigned_floor,
    )


def persist_monitoring_event(
    *,
    intent: str,
    alert: bool,
    camera_id: str,
    source: str,
    building: str,
    floor: str,
    room_number: str,
    bbox: Optional[Dict[str, Any]],
    latency_ms: Optional[int],
    frame_id: Optional[int] = None,
    role_hint: str = "",
    safety_label: Optional[str] = None,
) -> Optional[MonitoringEvent]:
    """Write pipeline output to the database (safe from background worker threads)."""
    close_old_connections()
    try:
        label = str(safety_label or "").strip().upper()
        if bbox is not None and safety_label:
            bbox = dict(bbox)
            bbox["safety_label"] = safety_label
        severity = _classify_severity(intent, alert, safety_label=safety_label)
        event_type = (
            MonitoringEvent.EventType.ALERT if alert and label == "ALERT" else MonitoringEvent.EventType.INFERENCE
        )

        if alert and label == "ALERT":
            recent_alert = (
                MonitoringEvent.objects.filter(camera_id=camera_id, alert=True, building=building, floor=floor)
                .order_by("-created_at")
                .first()
            )
            if (
                recent_alert
                and recent_alert.intent == intent
                and recent_alert.created_at
                and (timezone.now() - recent_alert.created_at).total_seconds() < 30
            ):
                recent_alert.intent = intent
                recent_alert.severity = severity
                recent_alert.bbox = bbox
                recent_alert.latency_ms = latency_ms
                recent_alert.frame_id = frame_id
                recent_alert.role_hint = role_hint or recent_alert.role_hint
                recent_alert.save(
                    update_fields=["intent", "severity", "bbox", "latency_ms", "frame_id", "role_hint"]
                )
                return recent_alert

        if not alert:
            recent = (
                MonitoringEvent.objects.filter(
                    camera_id=camera_id,
                    building=building,
                    floor=floor,
                    alert=False,
                )
                .order_by("-created_at")
                .first()
            )
            # Only merge updates for the same inference frame (keeps log stream growing per frame).
            if (
                recent
                and frame_id is not None
                and recent.frame_id == frame_id
                and recent.created_at
                and (timezone.now() - recent.created_at).total_seconds() < 2.0
            ):
                recent.intent = intent
                recent.severity = severity
                recent.event_type = event_type
                recent.source = source
                recent.room_number = room_number
                recent.bbox = bbox
                recent.latency_ms = latency_ms
                recent.frame_id = frame_id
                recent.role_hint = role_hint or recent.role_hint
                recent.save(
                    update_fields=[
                        "intent",
                        "severity",
                        "event_type",
                        "source",
                        "room_number",
                        "bbox",
                        "latency_ms",
                        "frame_id",
                        "role_hint",
                    ]
                )
                return recent

        return MonitoringEvent.objects.create(
            intent=intent,
            alert=alert,
            severity=severity,
            event_type=event_type,
            camera_id=camera_id,
            building=building,
            floor=floor,
            room_number=room_number,
            source=source,
            bbox=bbox,
            latency_ms=latency_ms,
            frame_id=frame_id,
            role_hint=role_hint or "",
        )
    except Exception as exc:
        logger.exception("Failed to persist monitoring event: %s", exc)
        return None
    finally:
        close_old_connections()


def fetch_recent_events(limit: int = 40, user: Optional[User] = None) -> List[Dict[str, Any]]:
    close_old_connections()
    try:
        qs = _events_queryset_for_user(user)[: max(limit, 1)]
        events = [event_to_dict(row) for row in qs]
        if user is not None:
            events = filter_events_for_user(user, events)
        return events[:limit]
    finally:
        close_old_connections()


def count_active_alerts(user: Optional[User] = None, window_seconds: int = 3600) -> int:
    close_old_connections()
    try:
        since = timezone.now() - timezone.timedelta(seconds=window_seconds)
        qs = _events_queryset_for_user(user).filter(created_at__gte=since, alert=True)
        count = 0
        for event in qs[:200]:
            row = event_to_dict(event)
            if _is_llm_alert_event(row):
                count += 1
        return count
    finally:
        close_old_connections()


def fetch_recent_alerts(
    limit: int = 12,
    user: Optional[User] = None,
    *,
    current_safety_label: Optional[str] = None,
    current_capture_active: bool = True,
    current_reasoning_stale: bool = False,
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    seen_summaries: Dict[str, float] = {}
    current_label = str(current_safety_label or "").strip().upper()

    for event in fetch_recent_events(120, user=user):
        intent_text = str(event.get("intent") or "Alert detected")
        is_critical = _is_llm_alert_event(event)
        if not is_critical:
            continue

        dedupe_key = intent_text.strip().lower()[:120]
        ts = float(event.get("ts") or 0.0)
        last_ts = seen_summaries.get(dedupe_key)
        if last_ts is not None and abs(ts - last_ts) < 30:
            continue
        seen_summaries[dedupe_key] = ts

        age_sec = max(0, int(time.time() - ts)) if ts else 0
        is_live = (
            current_capture_active
            and not current_reasoning_stale
            and current_label == "ALERT"
            and age_sec < 30
        )
        alerts.append(
            {
                "intent": intent_text,
                "camera_id": event.get("camera_id", ""),
                "room_number": event.get("room_number", ""),
                "floor": event.get("floor", ""),
                "alert": is_critical,
                "updated_label": format_relative_time(event.get("ts")),
                "severity": "Critical" if is_critical else "Warning",
                "severity_class": "badge-critical" if is_critical else "badge-warning",
                "historical": not is_live,
                "live": is_live,
            }
        )
        if len(alerts) >= limit:
            break
    return alerts


def fetch_patient_history_rows(limit: int = 25, user: Optional[User] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_cameras: set[str] = set()
    for event in fetch_recent_events(200, user=user):
        camera_id = str(event.get("camera_id") or "")
        if camera_id in seen_cameras:
            continue
        seen_cameras.add(camera_id)
        intent_text = str(event.get("intent") or "Under observation")
        rows.append(
            {
                "patient_label": f"Patient @ {camera_id or 'Cam'}",
                "unit": f"Floor {event.get('floor', '')} • Room {event.get('room_number', '')}",
                "camera_id": camera_id,
                "last_event": intent_text,
                "risk": "Critical" if event.get("alert") else ("Warning" if "stand" in intent_text.lower() else "OK"),
                "risk_class": "crit" if event.get("alert") else ("warn" if "stand" in intent_text.lower() else "ok"),
                "updated_label": format_relative_time(event.get("ts")),
            }
        )
        if len(rows) >= limit:
            break
    return rows

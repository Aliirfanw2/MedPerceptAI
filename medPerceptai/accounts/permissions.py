from __future__ import annotations

from typing import Any, Dict, List, Optional

from django.contrib.auth.models import User

from accounts.models import StaffProfile


def get_staff_profile(user: User) -> Optional[StaffProfile]:
    if not user or not user.is_authenticated:
        return None
    try:
        return user.staff_profile
    except StaffProfile.DoesNotExist:
        return None


def user_is_system_admin(user: User) -> bool:
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    profile = get_staff_profile(user)
    return bool(profile and profile.is_system_admin)


def user_can_manage_system_settings(user: User) -> bool:
    return user_is_system_admin(user)


def user_can_view_location(user: User, building: str, floor: str) -> bool:
    if not user or not user.is_authenticated:
        return False
    if user_is_system_admin(user):
        return True
    profile = get_staff_profile(user)
    if not profile:
        return False
    return profile.matches_location(building, floor)


def user_can_view_event(user: User, event: Dict[str, Any]) -> bool:
    if not user or not user.is_authenticated:
        return False
    if user_is_system_admin(user):
        return True
    profile = get_staff_profile(user)
    if not profile:
        return False
    return profile.matches_event(event)


def filter_events_for_user(user: User, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [event for event in events if user_can_view_event(user, event)]


def apply_location_filter_to_inference_state(user: User, state: Dict[str, Any]) -> Dict[str, Any]:
    """Hide alert/intent data when the active stream is outside the staff member's unit."""
    filtered = dict(state)
    building = str(filtered.get("building") or "").strip()
    floor = str(filtered.get("floor") or filtered.get("floor_number") or "").strip()

    if user_is_system_admin(user):
        filtered["routing_status"] = "delivered" if filtered.get("alert") else "idle"
        filtered["visible_to_request_user"] = True
        filtered["location_access"] = "all_units"
        return filtered

    profile = get_staff_profile(user)
    matches = bool(profile and profile.matches_location(building, floor))

    filtered["visible_to_request_user"] = matches or not filtered.get("alert")
    filtered["location_access"] = "assigned_unit" if profile else "none"

    if not matches:
        filtered["alert"] = False
        filtered["bbox"] = None
        filtered["intent"] = "No live activity for your assigned unit"
        filtered["routing_status"] = "suppressed_wrong_location"
        filtered["active_alerts"] = 0

    if matches:
        filtered["routing_status"] = "delivered" if filtered.get("alert") else "idle"
    else:
        filtered["routing_status"] = "idle"

    filtered["target_building"] = building or None
    filtered["target_floor"] = floor or None
    return filtered

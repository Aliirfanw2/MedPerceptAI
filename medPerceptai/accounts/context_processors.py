from __future__ import annotations


def layout_context(request):
    if not getattr(request, "user", None) or not request.user.is_authenticated:
        return {}

    from accounts.permissions import get_staff_profile, user_is_system_admin

    profile = get_staff_profile(request.user)
    if profile:
        staff_role = profile.get_role_display()
        shift_label = f"{staff_role} • {profile.assigned_building} • Floor {profile.assigned_floor}"
    elif user_is_system_admin(request.user):
        staff_role = "Admin"
        shift_label = "Admin • All units"
    else:
        staff_role = "Staff"
        shift_label = "Staff • Contact admin for assignment"

    registered_cameras = 0
    try:
        from monitor.views import build_monitor_camera_context

        _, _, registered_cameras = build_monitor_camera_context(request.session)
    except Exception:
        pass

    return {
        "layout_staff_role": staff_role,
        "layout_shift_label": shift_label,
        "layout_registered_cameras": registered_cameras,
    }

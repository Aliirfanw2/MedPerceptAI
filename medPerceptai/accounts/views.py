import os

import time

from pathlib import Path



from django.contrib import messages

from django.conf import settings

from django.contrib.auth.decorators import login_required

from django.http import HttpResponseForbidden

from django.shortcuts import redirect

from django.shortcuts import render



from accounts.permissions import get_staff_profile, user_can_manage_system_settings, user_is_system_admin





DEFAULT_MONITOR_SOURCE = os.environ.get("MONITOR_INPUT_SOURCE", "camera").strip().lower()

if DEFAULT_MONITOR_SOURCE not in {"camera", "video"}:

    DEFAULT_MONITOR_SOURCE = "camera"

DEFAULT_FLOOR_NUMBER = "3"

DEFAULT_ROOM_NUMBER = "302"

DEFAULT_CAMERA_ID = "Cam-1"





def _safe_int(value: object, fallback: int = 0) -> int:

    try:

        return int(str(value).strip())

    except Exception:

        return fallback





def _session_or_env(session, session_key, env_key, default):

    if session_key in session and session.get(session_key) not in (None, ""):

        return session.get(session_key)

    env_value = os.environ.get(env_key)

    if env_value is not None and str(env_value).strip() != "":

        return env_value

    return default





def _resolve_video_path(session):

    candidates = [

        session.get("monitor_video_path") if session else None,

        os.environ.get("MONITOR_VIDEO_PATH"),

        str(Path(settings.BASE_DIR) / "media" / "demo.mp4"),

        str(Path(settings.BASE_DIR) / "media" / "test_video.mp4"),

    ]

    for candidate in candidates:

        if not candidate:

            continue

        path = Path(str(candidate).strip())

        if not path.is_absolute():

            path = Path(settings.BASE_DIR) / path

        if path.exists():

            return str(path)

    fallback = Path(str(candidates[1] or candidates[2]).strip())

    if not fallback.is_absolute():

        fallback = Path(settings.BASE_DIR) / fallback

    return str(fallback)





DEFAULT_BUILDING = "Main Building"


def _monitoring_preferences(session):
    source = str(_session_or_env(session, "monitor_source", "MONITOR_INPUT_SOURCE", DEFAULT_MONITOR_SOURCE)).strip().lower()
    if source not in {"camera", "video"}:
        source = DEFAULT_MONITOR_SOURCE

    building = str(_session_or_env(session, "monitor_building", "MONITOR_BUILDING", DEFAULT_BUILDING)).strip() or DEFAULT_BUILDING
    floor_number = str(_session_or_env(session, "monitor_floor_number", "MONITOR_FLOOR_NUMBER", DEFAULT_FLOOR_NUMBER)).strip() or DEFAULT_FLOOR_NUMBER
    room_number = str(_session_or_env(session, "monitor_room_number", "MONITOR_ROOM_NUMBER", DEFAULT_ROOM_NUMBER)).strip() or DEFAULT_ROOM_NUMBER
    camera_id = str(_session_or_env(session, "monitor_camera_id", "MONITOR_CAMERA_ID", DEFAULT_CAMERA_ID)).strip() or DEFAULT_CAMERA_ID
    camera_index = _safe_int(_session_or_env(session, "monitor_camera_index", "MONITOR_CAMERA_INDEX", "0"))
    video_path = _resolve_video_path(session)

    source_label = "Live Camera Feed" if source == "camera" else "Pre-recorded Video"
    monitoring_display = f"{building} • Floor {floor_number} • Room {room_number} • {camera_id}"

    return {
        "monitor_source": source,
        "monitor_source_label": source_label,
        "monitor_building": building,
        "monitor_floor_number": floor_number,
        "monitor_room_number": room_number,
        "monitor_camera_id": camera_id,
        "monitor_camera_index": camera_index,
        "monitor_video_path": video_path,
        "monitoring_display": monitoring_display,
    }





def _staff_settings_context(user):
    profile = get_staff_profile(user)
    if profile:
        return {
            "staff_profile": profile,
            "staff_role": profile.get_role_display(),
            "staff_role_code": profile.role,
            "assigned_building": profile.assigned_building,
            "assigned_floor": profile.assigned_floor,
            "has_staff_profile": True,
        }
    if user_is_system_admin(user):
        return {
            "staff_profile": None,
            "staff_role": "Admin",
            "staff_role_code": "admin",
            "assigned_building": "All units",
            "assigned_floor": "—",
            "has_staff_profile": False,
        }
    return {
        "staff_profile": None,
        "staff_role": "Not assigned",
        "staff_role_code": "",
        "assigned_building": "—",
        "assigned_floor": "—",
        "has_staff_profile": False,
    }





def home(request):

    return render(request, "home.html")





@login_required

def dashboard(request):

    from monitor.views import build_monitor_camera_context, _snapshot_latest_inference_state



    monitoring = _monitoring_preferences(request.session)

    selected_camera = request.GET.get("camera") or request.session.get("selected_camera") or monitoring["monitor_camera_id"]

    stream_state = _snapshot_latest_inference_state()

    camera_feeds, connected_cameras, registered_cameras = build_monitor_camera_context(request.session)

    context = {

        "selected_camera": selected_camera,

        "camera_feeds": camera_feeds,

        "connected_cameras": connected_cameras,

        "registered_cameras": registered_cameras,

        "stream_status": stream_state.get("status") or "idle",

        "stream_cache_key": request.session.get("stream_cache_key", int(time.time())),

        **monitoring,

        **_staff_settings_context(request.user),

    }

    return render(request, "dashboard.html", context)





@login_required

def live_monitor(request):

    from monitor.views import build_monitor_camera_context, _snapshot_latest_inference_state



    monitoring = _monitoring_preferences(request.session)

    stream_state = _snapshot_latest_inference_state()

    camera_feeds, connected_cameras, registered_cameras = build_monitor_camera_context(request.session)

    context = {

        **monitoring,

        "camera_feeds": camera_feeds,

        "connected_cameras": connected_cameras,

        "registered_cameras": registered_cameras,

        "stream_status": stream_state.get("status") or "idle",

        **_staff_settings_context(request.user),

    }

    return render(request, "live_monitor.html", context)





@login_required
def settings_page(request):
    from django.contrib.auth.models import User

    from accounts.forms import StaffAccountForm
    from monitor.views import _ai_reasoning_enabled, _stop_capture_worker

    is_admin = user_can_manage_system_settings(request.user)
    staff_context = _staff_settings_context(request.user)
    monitoring_prefs = _monitoring_preferences(request.session) if is_admin else {}
    staff_form = StaffAccountForm(
        initial={
            "assigned_building": monitoring_prefs.get("monitor_building", DEFAULT_BUILDING),
            "assigned_floor": monitoring_prefs.get("monitor_floor_number", DEFAULT_FLOOR_NUMBER),
        }
    )

    if request.method == "POST":
        action = request.POST.get("form_action", "save_monitoring")

        if action == "create_staff":
            if not is_admin:
                return HttpResponseForbidden("Only administrators can create staff accounts.")
            staff_form = StaffAccountForm(request.POST)
            if staff_form.is_valid():
                user = staff_form.save()
                messages.success(
                    request,
                    f"Staff account created. Username: {user.username} — share the password you set so they can login at /login/",
                )
                return redirect("settings")
            messages.error(request, "Could not create staff account. Fix the errors below.")

        elif action == "save_monitoring":
            if not is_admin:
                return HttpResponseForbidden("Only administrators may change system settings.")

            monitor_source = str(request.POST.get("monitor_source", DEFAULT_MONITOR_SOURCE)).strip().lower()
            if monitor_source not in {"camera", "video"}:
                monitor_source = DEFAULT_MONITOR_SOURCE

            request.session["monitor_source"] = monitor_source
            request.session["monitor_building"] = str(request.POST.get("building_name", DEFAULT_BUILDING)).strip() or DEFAULT_BUILDING
            request.session["monitor_floor_number"] = str(request.POST.get("floor_number", DEFAULT_FLOOR_NUMBER)).strip() or DEFAULT_FLOOR_NUMBER
            request.session["monitor_room_number"] = str(request.POST.get("room_number", DEFAULT_ROOM_NUMBER)).strip() or DEFAULT_ROOM_NUMBER
            request.session["monitor_camera_id"] = str(request.POST.get("camera_id", DEFAULT_CAMERA_ID)).strip() or DEFAULT_CAMERA_ID
            request.session["monitor_camera_index"] = _safe_int(
                request.POST.get("camera_index"),
                _safe_int(request.session.get("monitor_camera_index", 0)),
            )
            posted_video = str(request.POST.get("video_path", "")).strip()
            if monitor_source == "video":
                request.session["monitor_video_path"] = (
                    posted_video
                    or request.session.get("monitor_video_path")
                    or os.environ.get("MONITOR_VIDEO_PATH", str(Path(settings.BASE_DIR) / "media" / "demo.mp4"))
                )
            request.session["enable_ai_reasoning"] = request.POST.get("enable_ai_reasoning") == "on"
            request.session["stream_cache_key"] = int(time.time())
            request.session.modified = True
            _stop_capture_worker()
            messages.success(request, "Monitoring settings saved. Open Dashboard to see the live stream.")
            return redirect("settings")

    monitoring = _monitoring_preferences(request.session) if is_admin else {}
    enable_ai_reasoning = _ai_reasoning_enabled(request.session)

    staff_accounts = []
    if is_admin:
        for profile in User.objects.filter(staff_profile__isnull=False).select_related("staff_profile").order_by("username")[:20]:
            staff_accounts.append(
                {
                    "username": profile.username,
                    "name": profile.get_full_name() or profile.username,
                    "role": profile.staff_profile.get_role_display(),
                    "building": profile.staff_profile.assigned_building,
                    "floor": profile.staff_profile.assigned_floor,
                }
            )

    context = {
        "is_system_admin": is_admin,
        "can_edit_system_settings": is_admin,
        "enable_ai_reasoning": enable_ai_reasoning,
        "staff_form": staff_form,
        "staff_accounts": staff_accounts,
        **staff_context,
        **monitoring,
    }
    return render(request, "settings.html", context)





@login_required

def alerts_page(request):

    from monitor.views import get_recent_alerts



    alert_events = get_recent_alerts(12, user=request.user)

    return render(

        request,

        "alerts.html",

        {

            "alert_events": alert_events,

            **_staff_settings_context(request.user),

        },

    )





@login_required

def patient_history_page(request):

    from monitor.views import get_patient_history_rows



    history_rows = get_patient_history_rows(25, user=request.user)

    return render(

        request,

        "patient_history.html",

        {

            "history_rows": history_rows,

            **_staff_settings_context(request.user),

        },

    )



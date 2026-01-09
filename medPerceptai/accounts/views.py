from django.shortcuts import render
from django.contrib.auth.decorators import login_required

def home(request):
    return render(request, 'home.html')

@login_required
def dashboard(request):
    selected_camera = request.GET.get("camera") or "ICU-CAM-01"
    return render(request, "dashboard.html", {"selected_camera": selected_camera})


@login_required
def live_monitor(request):
    return render(request, 'live_monitor.html')


@login_required
def settings_page(request):
    return render(request, 'settings.html')


@login_required
def alerts_page(request):
    return render(request, 'alerts.html')


@login_required
def patient_history_page(request):
    return render(request, 'patient_history.html')

"""
URL configuration for medPerceptai project.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include
from django.urls import path
from django.urls import reverse_lazy
from django.contrib.auth import views as auth_views
from accounts import views

urlpatterns = [
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("live-monitor/", views.live_monitor, name="live_monitor"),
    path("alerts/", views.alerts_page, name="alerts"),
    path("patient-history/", views.patient_history_page, name="patient_history"),
    path("settings/", views.settings_page, name="settings"),
    path("profile/", views.profile_page, name="profile"),
    path("", include("monitor.urls")),
    path("login/", auth_views.LoginView.as_view(template_name="login.html", redirect_authenticated_user=True), name="login"),
    path("logout/", auth_views.LogoutView.as_view(next_page=reverse_lazy("home")), name="logout"),
    path("admin/", admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
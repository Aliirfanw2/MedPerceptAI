"""
URL configuration for medPerceptai project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
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
    path("", include("monitor.urls")),
    path("login/", auth_views.LoginView.as_view(template_name="login.html", redirect_authenticated_user=True), name="login"),
    path("logout/", auth_views.LogoutView.as_view(next_page=reverse_lazy("home")), name="logout"),
    path('admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

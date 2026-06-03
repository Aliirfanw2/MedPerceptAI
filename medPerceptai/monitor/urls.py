from django.urls import path

from monitor import views

urlpatterns = [
    path("stream/", views.live_stream_feed, name="live_stream_feed"),
    path("api/latest-alert/", views.get_latest_alert, name="get_latest_alert"),
]

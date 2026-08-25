import os

from django.apps import AppConfig


class MonitorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "monitor"
    verbose_name = "Monitor"

    def ready(self) -> None:
        # StatReloader spawns a parent watcher process; only warm up the real server.
        if os.environ.get("RUN_MAIN") != "true":
            return
        if not getattr(self, "_warmup_started", False):
            MonitorConfig._warmup_started = True
            try:
                from monitor.views import _warm_pipeline_async

                _warm_pipeline_async()
            except Exception:
                pass

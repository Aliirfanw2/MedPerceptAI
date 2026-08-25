from django.contrib import admin

from monitor.models import Camera
from monitor.models import MonitoringEvent
from monitor.models import Room


@admin.register(Room)
class RoomAdmin(admin.ModelAdmin):
    list_display = ("building", "floor", "room_number", "display_name", "is_active")
    list_filter = ("building", "floor", "is_active")
    search_fields = ("building", "floor", "room_number", "display_name")


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("camera_identifier", "building", "floor", "room_number", "display_name", "is_active")
    list_filter = ("building", "floor", "is_active")
    search_fields = ("camera_identifier", "building", "floor", "room_number", "display_name")


@admin.register(MonitoringEvent)
class MonitoringEventAdmin(admin.ModelAdmin):
    list_display = ("created_at", "severity", "camera_id", "building", "floor", "intent_short", "alert")
    list_filter = ("severity", "alert", "building", "floor", "event_type")
    search_fields = ("intent", "camera_id", "building", "floor", "room_number")
    readonly_fields = ("created_at",)

    @admin.display(description="Intent")
    def intent_short(self, obj: MonitoringEvent) -> str:
        return (obj.intent or "")[:80]

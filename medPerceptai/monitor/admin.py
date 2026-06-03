from django.contrib import admin

from monitor.models import Camera
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

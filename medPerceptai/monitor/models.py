from django.db import models


class Room(models.Model):
    building = models.CharField(max_length=120)
    floor = models.CharField(max_length=30)
    room_number = models.CharField(max_length=30)
    display_name = models.CharField(max_length=120, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["building", "floor", "room_number"]
        constraints = [
            models.UniqueConstraint(fields=["building", "floor", "room_number"], name="unique_room_location"),
        ]

    def __str__(self) -> str:
        label = self.display_name.strip() if self.display_name else self.room_number
        return f"{self.building} - Floor {self.floor} - Room {label}"


class Camera(models.Model):
    camera_identifier = models.CharField(max_length=80, unique=True)
    building = models.CharField(max_length=120)
    floor = models.CharField(max_length=30)
    room_number = models.CharField(max_length=30, blank=True)
    display_name = models.CharField(max_length=120, blank=True)
    is_active = models.BooleanField(default=True)
    room = models.ForeignKey(Room, on_delete=models.SET_NULL, null=True, blank=True, related_name="cameras")

    class Meta:
        ordering = ["building", "floor", "camera_identifier"]
        indexes = [
            models.Index(fields=["building", "floor"]),
            models.Index(fields=["camera_identifier"]),
        ]

    def save(self, *args, **kwargs):
        if self.room is not None:
            self.building = self.room.building
            self.floor = self.room.floor
            self.room_number = self.room.room_number
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        label = self.display_name.strip() if self.display_name else self.camera_identifier
        return f"{label} ({self.building}, Floor {self.floor})"


class MonitoringEvent(models.Model):
    """Persisted inference and alert events from the live pipeline."""

    class EventType(models.TextChoices):
        INFERENCE = "inference_update", "Inference update"
        ALERT = "alert_emitted", "Alert emitted"

    class Severity(models.TextChoices):
        INFO = "info", "Info"
        WARNING = "warning", "Warning"
        CRITICAL = "critical", "Critical"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    intent = models.TextField()
    alert = models.BooleanField(default=False, db_index=True)
    severity = models.CharField(
        max_length=16,
        choices=Severity.choices,
        default=Severity.INFO,
        db_index=True,
    )
    event_type = models.CharField(max_length=32, choices=EventType.choices)
    camera_id = models.CharField(max_length=80, db_index=True)
    building = models.CharField(max_length=120, db_index=True)
    floor = models.CharField(max_length=30, db_index=True)
    room_number = models.CharField(max_length=30, blank=True)
    source = models.CharField(max_length=255, blank=True)
    bbox = models.JSONField(null=True, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    frame_id = models.PositiveIntegerField(null=True, blank=True)
    role_hint = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at", "building", "floor"]),
            models.Index(fields=["alert", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.get_severity_display()} @ {self.camera_id}: {self.intent[:60]}"

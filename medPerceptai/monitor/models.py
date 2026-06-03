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

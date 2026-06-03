from django.conf import settings
from django.db import models


class StaffProfile(models.Model):
    class Role(models.TextChoices):
        NURSE = "nurse", "Nurse"
        DOCTOR = "doctor", "Doctor"
        ADMIN = "admin", "Admin"

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="staff_profile",
    )
    role = models.CharField(
        max_length=32,
        choices=Role.choices,
        default=Role.NURSE,
    )
    assigned_building = models.CharField(max_length=120)
    assigned_floor = models.CharField(max_length=30)

    class Meta:
        ordering = ["assigned_building", "assigned_floor", "user__username"]
        indexes = [
            models.Index(fields=["assigned_building", "assigned_floor"]),
            models.Index(fields=["role"]),
        ]

    def __str__(self) -> str:
        return (
            f"{self.user.get_username()} ({self.get_role_display()}) "
            f"-> {self.assigned_building} / Floor {self.assigned_floor}"
        )

    @property
    def is_system_admin(self) -> bool:
        return self.role == self.Role.ADMIN

    def matches_location(self, building: str, floor: str) -> bool:
        if not building or not floor:
            return False
        return (
            self.assigned_building.strip().lower() == str(building).strip().lower()
            and self.assigned_floor.strip().lower() == str(floor).strip().lower()
        )

    def matches_event(self, event: dict) -> bool:
        return self.matches_location(
            str(event.get("building") or ""),
            str(event.get("floor") or event.get("floor_number") or ""),
        )

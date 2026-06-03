from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from django.contrib.auth.models import User

from accounts.forms import StaffCreationForm
from accounts.models import StaffProfile


class StaffProfileInline(admin.StackedInline):
    model = StaffProfile
    can_delete = False
    extra = 0
    fields = ("role", "assigned_building", "assigned_floor")
    verbose_name = "Staff profile"
    verbose_name_plural = "Staff profile (role + location for alerts)"


@admin.register(StaffProfile)
class StaffProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "assigned_building", "assigned_floor")
    list_filter = ("role", "assigned_building", "assigned_floor")
    search_fields = (
        "user__username",
        "user__first_name",
        "user__last_name",
        "assigned_building",
        "assigned_floor",
    )


class CustomUserAdmin(DjangoUserAdmin):
    add_form = StaffCreationForm
    inlines = (StaffProfileInline,)

    list_display = DjangoUserAdmin.list_display + ("staff_role", "staff_building", "staff_floor")
    list_select_related = ("staff_profile",)

    add_fieldsets = (
        (
            "Login credentials",
            {
                "classes": ("wide",),
                "description": "Create username and password — staff uses these at the login page.",
                "fields": ("username", "password1", "password2"),
            },
        ),
        (
            "Personal info",
            {"fields": ("first_name", "last_name", "email")},
        ),
        (
            "Role & location",
            {
                "description": "Building and floor must match the live stream settings for alerts.",
                "fields": ("role", "assigned_building", "assigned_floor"),
            },
        ),
    )

    @admin.display(description="Role", ordering="staff_profile__role")
    def staff_role(self, obj):
        profile = getattr(obj, "staff_profile", None)
        return profile.get_role_display() if profile else "—"

    @admin.display(description="Building", ordering="staff_profile__assigned_building")
    def staff_building(self, obj):
        profile = getattr(obj, "staff_profile", None)
        return profile.assigned_building if profile else "—"

    @admin.display(description="Floor", ordering="staff_profile__assigned_floor")
    def staff_floor(self, obj):
        profile = getattr(obj, "staff_profile", None)
        return profile.assigned_floor if profile else "—"

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        if change:
            return

        assigned_building = form.cleaned_data.get("assigned_building")
        assigned_floor = form.cleaned_data.get("assigned_floor")
        role = form.cleaned_data.get("role", StaffProfile.Role.NURSE)
        if assigned_building and assigned_floor:
            StaffProfile.objects.update_or_create(
                user=obj,
                defaults={
                    "role": role,
                    "assigned_building": assigned_building.strip(),
                    "assigned_floor": assigned_floor.strip(),
                },
            )
            obj.is_staff = role == StaffProfile.Role.ADMIN
            obj.save(update_fields=["is_staff"])


admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

admin.site.site_header = "MedPerceptAI Admin"
admin.site.site_title = "MedPerceptAI"
admin.site.index_title = "Staff accounts & system data"

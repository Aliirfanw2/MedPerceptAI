from django.db import migrations, models


def map_legacy_roles(apps, schema_editor):
    StaffProfile = apps.get_model("accounts", "StaffProfile")
    StaffProfile.objects.filter(role__in=["technician", "supervisor"]).update(role="nurse")


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0003_staffprofile_role"),
    ]

    operations = [
        migrations.RunPython(map_legacy_roles, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="staffprofile",
            name="role",
            field=models.CharField(
                choices=[
                    ("nurse", "Nurse"),
                    ("doctor", "Doctor"),
                    ("admin", "Admin"),
                ],
                default="nurse",
                max_length=32,
            ),
        ),
    ]

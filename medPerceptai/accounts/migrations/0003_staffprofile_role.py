from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_alter_staffprofile_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="staffprofile",
            name="role",
            field=models.CharField(
                choices=[
                    ("nurse", "Nurse"),
                    ("doctor", "Doctor"),
                    ("admin", "Administrator"),
                    ("technician", "Technician"),
                    ("supervisor", "Supervisor"),
                ],
                default="nurse",
                max_length=32,
            ),
        ),
        migrations.AddIndex(
            model_name="staffprofile",
            index=models.Index(fields=["role"], name="accounts_st_role_4b8e62_idx"),
        ),
    ]

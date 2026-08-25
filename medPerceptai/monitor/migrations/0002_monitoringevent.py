from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("monitor", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="MonitoringEvent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                ("intent", models.TextField()),
                ("alert", models.BooleanField(db_index=True, default=False)),
                (
                    "severity",
                    models.CharField(
                        choices=[
                            ("info", "Info"),
                            ("warning", "Warning"),
                            ("critical", "Critical"),
                        ],
                        db_index=True,
                        default="info",
                        max_length=16,
                    ),
                ),
                (
                    "event_type",
                    models.CharField(
                        choices=[
                            ("inference_update", "Inference update"),
                            ("alert_emitted", "Alert emitted"),
                        ],
                        max_length=32,
                    ),
                ),
                ("camera_id", models.CharField(db_index=True, max_length=80)),
                ("building", models.CharField(db_index=True, max_length=120)),
                ("floor", models.CharField(db_index=True, max_length=30)),
                ("room_number", models.CharField(blank=True, max_length=30)),
                ("source", models.CharField(blank=True, max_length=255)),
                ("bbox", models.JSONField(blank=True, null=True)),
                ("latency_ms", models.PositiveIntegerField(blank=True, null=True)),
                ("frame_id", models.PositiveIntegerField(blank=True, null=True)),
                ("role_hint", models.CharField(blank=True, max_length=64)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="monitoringevent",
            index=models.Index(fields=["-created_at", "building", "floor"], name="monitor_mon_created_6e2f01_idx"),
        ),
        migrations.AddIndex(
            model_name="monitoringevent",
            index=models.Index(fields=["alert", "-created_at"], name="monitor_mon_alert_2c8b44_idx"),
        ),
    ]

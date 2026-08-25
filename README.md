# MedPerceptAI

MedPerceptAI is a real-time patient monitoring system built with Django, YOLO perception models, and optional Llama-based reasoning.

## What This Project Does

- Streams live camera or video input in the dashboard.
- Detects patient/staff and posture context using YOLO models.
- Produces monitor/alert decisions from runtime reasoning output.
- Plays alarm audio only when final alert state is active.

## Repository Status

- Frontend monitoring pages and alert flow are integrated.
- Backend runtime pipeline, event store, and dashboard APIs are integrated.
- Alarm logic is finalized to trigger only on alert.

## Important Reasoning Model Note

- The Llama reasoning model file is not included in this repository.
- Reasoning-based final decisions require adding the external GGUF model at runtime.
- Team reference source is Hugging Face user/account: waqas69.

If the reasoning model is not added, the system can still run in fallback mode, but full reasoning behavior will be limited.

## Default Model Paths

Expected model files in medPerceptai/model_weights:

- obj.pt
- roles.pt
- yolov8n-pose.pt
- llama-3-8b-instruct.Q4_K_M.gguf (reasoning model, external)

You can override paths with environment variables:

- YOLO_OBJECT_MODEL_PATH
- YOLO_ROLE_MODEL_PATH
- YOLO_POSE_MODEL_PATH
- REASONING_MODEL_PATH

## Key Runtime Environment Variables

- ENABLE_REASONING=1 enables Llama reasoning; 0 keeps fallback mode.
- MONITOR_INPUT_SOURCE=camera or video.
- MONITOR_CAMERA_INDEX=0 for default camera.
- MONITOR_VIDEO_PATH=<path> for prerecorded video mode.
- CAPTURE_MAX_FPS and INFERENCE_EVERY_N_FRAMES tune performance.

## Quick Start (Windows)

1. Open project root:

```powershell
cd MedPerceptAI
```

2. Activate virtual environment:

```powershell
.\env\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run migrations:

```powershell
cd medPerceptai
python manage.py migrate
```

5. Start server:

```powershell
python manage.py runserver
```

6. Open in browser:

- Dashboard: http://127.0.0.1:8000/
- Live monitor page: http://127.0.0.1:8000/live-monitor/

## Alert Audio Behavior

- Alarm file path in templates: /media/alarm.mp3
- Alarm plays only when final alert state is true.
- Monitor/warning states do not play alarm audio.

## Troubleshooting

- If stream is offline, check camera index or MONITOR_VIDEO_PATH.
- If reasoning stays fallback, verify ENABLE_REASONING and REASONING_MODEL_PATH.
- If audio does not start, click once on page to unlock browser autoplay policy.
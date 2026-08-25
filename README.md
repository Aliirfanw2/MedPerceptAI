# MedPerceptAI

MedPerceptAI is a real-time patient monitoring system built with Django, YOLO perception models, and a Llama-based final reasoning model.

## What This Project Does

- Streams live camera or video input in the dashboard.
- Detects patient/staff and posture context using YOLO models.
- Produces monitor/alert decisions from runtime reasoning output.
- Plays alarm audio only when final alert state is active.

## Repository Status

- Frontend monitoring pages and alert flow are integrated.
- Backend runtime pipeline, event store, and dashboard APIs are integrated.
- Alarm logic is finalized to trigger only on alert.

## Main Reasoning Model Note

- The Llama reasoning model is the main final decision model in this project.
- It generates NLP-style final output such as patient posture/status, risk level, alert type, and summary.
- This GGUF model file is not included in the repository because of large model size.
- Team reference source is Hugging Face user/account: waqas69.

If the reasoning model is not added, the system runs in fallback mode only and full reasoning behavior is not available.

## Model Inventory and Integration Flow

This project uses 4 model components in total.

1. Object detection model: obj.pt
- Purpose: Detects scene objects and person bounding boxes.
- Runtime role: Provides primary detections for downstream context.

2. Role classification model: roles.pt
- Purpose: Classifies detected person role, such as patient, nurse, or doctor.
- Runtime role: Adds identity/role context to each person track.

3. Pose estimation model: yolov8n-pose.pt
- Purpose: Extracts pose landmarks and posture cues.
- Runtime role: Contributes posture safety signals used in decisions.

4. Reasoning model (main final decision model): llama-3-8b-instruct.Q4_K_M.gguf
- Purpose: Converts structured scene context into final reasoning output.
- Runtime role: Produces final reasoning fields such as safety_label, alert_type, risk_level, reason, and summary.
- Source: Hugging Face account waqas69 (kept outside repo due to model size).

How integration works in runtime:

1. Input source starts from camera or prerecorded video.
2. Per-frame inference runs on object, role, and pose models.
3. Outputs are fused into a single structured scene context.
4. If ENABLE_REASONING=1 and reasoning model is available, Llama generates final decision fields.
5. If reasoning model is unavailable, system runs fallback reasoning logic.
6. Dashboard API publishes final state and UI renders alert/monitor cards.
7. Alarm audio is triggered only when final alert state is true.

Minimum requirement for full reasoning mode:

- Keep all 3 YOLO models available in model_weights.
- Add the GGUF reasoning model and set REASONING_MODEL_PATH if custom path is used.
- Enable ENABLE_REASONING=1.

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
# MedPerceptAI

Real-time patient monitoring dashboard with camera/video ingestion, YOLO-based perception, and reasoning-assisted alerting.

## Current Status

- Live monitoring UI and alert audio flow are configured.
- Alarm audio now plays only when final alert state is active.

## Important Note About Reasoning Model

- The Llama reasoning model is not pushed in this repository.
- Reasoning-based decisions depend on adding the external model artifacts at runtime.
- Source reference: Hugging Face account/name shared by team as `waqas69`.

If the Llama reasoning model is correctly added and configured, the full reasoning pipeline and alert decisions will work as expected.
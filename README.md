# Engine YOLO

An inference engine for YOLO models.

## Configuration

- `MODEL_PATH`: path to a YOLO `.pt` file.
  OR `MODEL_DIR`: path to a directory containing a YOLO `model.pt` file.
- `DEFAULT_SCORE_THRESHOLD` (default: `0.5`): the default confidence score threshold.
- `UVICORN_HOST` (default: `0.0.0.0`)
- `UVICORN_PORT` (default: `8080`)

Task type is inferred from the loaded model.

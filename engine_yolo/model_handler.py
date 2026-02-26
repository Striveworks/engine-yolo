import logging
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from threading import Lock

import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo import YOLO

from engine_yolo.protocol import DecodedInput, SupportedTaskType, get_score_threshold
from engine_yolo.result_mapper import map_yolo_results

logger = logging.getLogger(__name__)


class ModelHandler:
    model: YOLO

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise ValueError(f"model_path {str(model_path)!r} does not exist")
        if model_path.suffix != ".pt":
            raise ValueError(f"expected model_path to point to a .pt file, got {str(model_path)!r}")

        logger.info(f"Loading model from {str(model_path)!r}")
        self.model = YOLO(model_path)

        model_task = getattr(self.model, "task", None)
        task_type_by_model_task = {
            "detect": SupportedTaskType.OBJECT_DETECTION,
            "obb": SupportedTaskType.ORIENTED_OBJECT_DETECTION,
        }
        if model_task not in task_type_by_model_task:
            raise RuntimeError(
                f"Unsupported YOLO model task {model_task!r}. Supported model tasks are "
                "'detect' and 'obb'."
            )
        self.task_type = task_type_by_model_task[model_task]

        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using device {torch.cuda.current_device()}.")
        else:
            logger.info("CUDA is not available. Using CPU.")
        self.device = torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self._predict_lock = Lock()

    def inference(self, data: list[tuple]) -> list[Results]:
        # Group consecutive images that share a threshold to avoid duplicate model calls.
        with self._predict_lock:
            return [
                output
                for threshold, image_threshold_pairs in groupby(data, key=itemgetter(1))
                for output in self.model.predict(
                    source=[image for image, _ in image_threshold_pairs],
                    conf=threshold,
                    save_conf=True,
                    device=self.device,
                )
            ]

    def handle(self, data: list[DecodedInput]) -> list[list[dict]]:
        model_inputs = [
            (decoded.image, get_score_threshold(decoded.parameters)) for decoded in data
        ]
        model_output = self.inference(model_inputs)
        return map_yolo_results(self.task_type, self.device, model_output)

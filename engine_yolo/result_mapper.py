import logging
from typing import Any

import torch
from ultralytics.engine.results import Results

from engine_yolo.protocol import SupportedTaskType

logger = logging.getLogger(__name__)


def map_yolo_results(
    task_type: SupportedTaskType,
    device: torch.device,
    results: list[Results],
) -> list[list[dict[str, Any]]]:
    if task_type == SupportedTaskType.OBJECT_DETECTION:
        return [_map_object_detection_result(device, result) for result in results]

    if task_type == SupportedTaskType.ORIENTED_OBJECT_DETECTION:
        return [_map_oriented_object_detection_result(device, result) for result in results]

    raise ValueError(f"Unsupported task type {task_type!r}")


def _map_object_detection_result(device: torch.device, result: Results) -> list[dict[str, Any]]:
    if not result.boxes:
        return []

    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        xyxy = box.xyxy[0].to(device)
        detections.append(
            {
                "label": result.names[int(box.cls.item())],
                "score": float(box.conf.item()),
                "xmin": float(xyxy[0].item()),
                "ymin": float(xyxy[1].item()),
                "xmax": float(xyxy[2].item()),
                "ymax": float(xyxy[3].item()),
            }
        )
    return detections


def _map_oriented_object_detection_result(
    device: torch.device, result: Results
) -> list[dict[str, Any]]:
    if not result.obb:
        return []

    detections: list[dict[str, Any]] = []
    for box in result.obb:
        try:
            score = float(box.conf.item())
        except AttributeError:
            logger.warning("Missing box.conf in OBB output, using score=0.0")
            score = 0.0

        xywhr = box.xywhr[0].to(device)
        orig_height = box.orig_shape[0]
        orig_width = box.orig_shape[1]
        xywhr_scaled = xywhr / torch.tensor(
            [orig_width, orig_height, orig_width, orig_height, 1],
            device=device,
        )
        detections.append(
            {
                "label": result.names[int(box.cls.item())],
                "score": score,
                "cx": float(xywhr_scaled[0].item()),
                "cy": float(xywhr_scaled[1].item()),
                "w": float(xywhr_scaled[2].item()),
                "h": float(xywhr_scaled[3].item()),
                "r": float(xywhr_scaled[4].item()),
            }
        )
    return detections

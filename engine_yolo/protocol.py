import base64
import io
import os
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from PIL import Image

DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", 0.5))


class SupportedTaskType(StrEnum):
    OBJECT_DETECTION = "Object Detection"
    ORIENTED_OBJECT_DETECTION = "Oriented Object Detection"


@dataclass(frozen=True)
class DecodedInput:
    name: str
    image: Image.Image
    parameters: dict[str, Any]


def decode_kserve_inputs(body: Mapping[str, Any]) -> list[DecodedInput]:
    raw_inputs = body.get("inputs")
    if not isinstance(raw_inputs, list):
        raise ValueError("expected request body field 'inputs' to be a list")

    return [_decode_kserve_input(input_) for input_ in raw_inputs]


def _decode_kserve_input(input_: Any) -> DecodedInput:
    if not isinstance(input_, dict):
        raise TypeError("expected each input item to be a JSON object")

    raw_data = input_.get("data")
    encoded_image = _get_first_data_item(raw_data)
    if not isinstance(encoded_image, str):
        raise TypeError("expected input.data[0] to be a base64 string")

    image_bytes = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    parameters = input_.get("parameters", {})
    if not isinstance(parameters, dict):
        raise TypeError("expected input.parameters to be a JSON object")

    return DecodedInput(name=str(input_.get("name", "")), image=image, parameters=parameters)


def _get_first_data_item(raw_data: Any) -> Any:
    if isinstance(raw_data, list):
        if not raw_data:
            raise ValueError("expected input.data to include at least one element")
        return raw_data[0]
    if raw_data is None:
        raise ValueError("expected input.data field")
    return raw_data


def get_score_threshold(parameters: Mapping[str, Any]) -> float:
    raw_threshold = parameters.get("score_threshold", DEFAULT_SCORE_THRESHOLD)
    if not isinstance(raw_threshold, (float, int)):
        raise TypeError("expected parameters.score_threshold to be a number")
    return float(raw_threshold)


def get_request_id(body: Mapping[str, Any], request_id_header: str | None) -> str:
    raw_request_id = body.get("id")
    if isinstance(raw_request_id, str) and raw_request_id:
        return raw_request_id
    if request_id_header:
        return request_id_header
    return str(uuid.uuid4())


def encode_kserve_output(name: str, detections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [1],
        "datatype": "BYTES",
        "data": [detections],
    }


def encode_kserve_response(
    model_name: str,
    request_id: str,
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_version": "1.0.0",
        "id": request_id,
        "parameters": None,
        "outputs": outputs,
    }

import base64
import io
import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fastapi.responses import JSONResponse, Response
from PIL import Image

DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", 0.5))
INFERENCE_PARAMETERS_HEADER = "chariot-inference-parameters"


class SupportedTaskType(StrEnum):
    OBJECT_DETECTION = "Object Detection"
    ORIENTED_OBJECT_DETECTION = "Oriented Object Detection"


@dataclass(frozen=True)
class DecodedInput:
    name: str
    image: Image.Image
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ParsedRequest:
    inputs: list[DecodedInput]
    request_id: str


RequestParser = Callable[[bytes, str | None, Mapping[str, str] | None], ParsedRequest]
ResponseRenderer = Callable[[str, str, list[DecodedInput], list[list[dict[str, Any]]]], Response]


@dataclass(frozen=True)
class ProtocolHandlers:
    parse_request: RequestParser
    render_response: ResponseRenderer


class InferenceProtocol(StrEnum):
    CHARIOT_V2 = "chariot-v2"
    CHARIOT_V2_KSERVE = "chariot-v2-kserve"


def get_inference_protocol(raw_protocol: str | None = None) -> InferenceProtocol:
    return InferenceProtocol(
        raw_protocol or os.getenv("CHARIOT_INFERENCE_PROTOCOL", "chariot-v2-kserve")
    )


def get_protocol_handlers(protocol: InferenceProtocol) -> ProtocolHandlers:
    match protocol:
        case InferenceProtocol.CHARIOT_V2_KSERVE:
            return ProtocolHandlers(
                parse_request=parse_kserve_request,
                render_response=render_kserve_response,
            )
        case InferenceProtocol.CHARIOT_V2:
            return ProtocolHandlers(
                parse_request=parse_chariot_v2_request,
                render_response=render_chariot_v2_response,
            )
        case _:
            raise ValueError(f"unsupported inference protocol: {protocol}")


def get_score_threshold(parameters: Mapping[str, Any]) -> float:
    raw_threshold = parameters.get("score_threshold", DEFAULT_SCORE_THRESHOLD)
    if not isinstance(raw_threshold, (float, int)):
        raise TypeError("expected parameters.score_threshold to be a number")
    return float(raw_threshold)


def parse_kserve_request(
    body: bytes,
    content_type: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> ParsedRequest:
    del content_type
    request_id_header = headers.get("X-Request-ID") if headers is not None else None
    raw_body = _decode_json_object(body)
    raw_inputs = raw_body.get("inputs")
    if not isinstance(raw_inputs, list):
        raise ValueError("expected request body field 'inputs' to be a list")
    if not raw_inputs:
        raise ValueError("expected request body field 'inputs' to be a non-empty list")

    raw_request_id = raw_body.get("id")
    request_id = request_id_header or str(uuid.uuid4())
    if isinstance(raw_request_id, str) and raw_request_id:
        request_id = raw_request_id

    return ParsedRequest(
        inputs=[_decode_kserve_input(input_) for input_ in raw_inputs],
        request_id=request_id,
    )


def parse_chariot_v2_request(
    body: bytes,
    content_type: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> ParsedRequest:
    request_id_header = headers.get("X-Request-ID") if headers is not None else None
    normalized_content_type = _normalize_content_type(content_type)
    if normalized_content_type == "application/jsonl":
        inputs = [
            _decode_chariot_v2_json_input(_decode_json_object(line.encode("utf-8")))
            for line in body.decode("utf-8").splitlines()
        ]
    elif normalized_content_type in {"", "application/json"}:
        inputs = [_decode_chariot_v2_json_input(_decode_json_object(body))]
    else:
        inputs = [_decode_raw_input(body, headers or {})]
    if not inputs:
        raise ValueError("expected request body to contain at least one input")

    return ParsedRequest(
        inputs=inputs,
        request_id=request_id_header or str(uuid.uuid4()),
    )


def render_kserve_response(
    model_name: str,
    request_id: str,
    inputs: list[DecodedInput],
    mapped_outputs: list[list[dict[str, Any]]],
) -> Response:
    outputs = [
        {
            "name": decoded_input.name,
            "shape": [1],
            "datatype": "BYTES",
            "data": [output],
        }
        for decoded_input, output in zip(inputs, mapped_outputs)
    ]
    if not outputs:
        raise ValueError("expected outputs to contain at least one item")

    return JSONResponse(
        content={
            "model_name": model_name,
            "model_version": "1.0.0",
            "id": request_id,
            "parameters": None,
            "outputs": outputs,
        }
    )


def render_chariot_v2_response(
    model_name: str,
    request_id: str,
    inputs: list[DecodedInput],
    mapped_outputs: list[list[dict[str, Any]]],
) -> Response:
    del model_name, request_id, inputs
    outputs = [{"output": output} for output in mapped_outputs]
    if not outputs:
        raise ValueError("expected outputs to contain at least one item")

    if len(outputs) == 1:
        return JSONResponse(content=outputs[0])
    return Response(
        content="\n".join(json.dumps(output) for output in outputs) + "\n",
        media_type="application/jsonl",
    )


def _decode_kserve_input(input_: Any) -> DecodedInput:
    if not isinstance(input_, dict):
        raise TypeError("expected each input item to be a JSON object")

    raw_data = input_.get("data")
    if isinstance(raw_data, list):
        if not raw_data:
            raise ValueError("expected input.data to include at least one element")
        raw_data = raw_data[0]
    elif raw_data is None:
        raise ValueError("expected input.data field")

    if not isinstance(raw_data, str):
        raise TypeError("expected input.data[0] to be a base64 string")

    parameters = input_.get("parameters", {})
    if not isinstance(parameters, dict):
        raise TypeError("expected input.parameters to be a JSON object")

    return DecodedInput(
        name=str(input_.get("name", "")),
        image=_decode_base64_image(raw_data),
        parameters=parameters,
    )


def _decode_chariot_v2_json_input(input_: Mapping[str, Any]) -> DecodedInput:
    encoded_image = input_.get("input")
    if not isinstance(encoded_image, str):
        raise TypeError("expected request body field 'input' to be a base64 string")

    parameters = input_.get("parameters", {})
    if not isinstance(parameters, dict):
        raise TypeError("expected request body field 'parameters' to be a JSON object")

    return DecodedInput(
        name="",
        image=_decode_base64_image(encoded_image),
        parameters=parameters,
    )


def _decode_raw_input(body: bytes, headers: Mapping[str, str]) -> DecodedInput:
    raw_parameters = next(
        (
            value
            for key, value in headers.items()
            if key.lower() == INFERENCE_PARAMETERS_HEADER.lower()
        ),
        "",
    )
    parameters: dict[str, Any] = {}
    if raw_parameters:
        try:
            parsed_parameters = json.loads(raw_parameters)
        except json.JSONDecodeError as exc:
            raise TypeError(
                f"expected {INFERENCE_PARAMETERS_HEADER} header to contain a JSON object"
            ) from exc
        if not isinstance(parsed_parameters, dict):
            raise TypeError(
                f"expected {INFERENCE_PARAMETERS_HEADER} header to contain a JSON object"
            )
        parameters = parsed_parameters

    return DecodedInput(
        name="",
        image=Image.open(io.BytesIO(body)).convert("RGB"),
        parameters=parameters,
    )


def _decode_base64_image(encoded_image: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded_image))).convert("RGB")


def _decode_json_object(body: bytes) -> Mapping[str, Any]:
    parsed_body = json.loads(body)
    if not isinstance(parsed_body, dict):
        raise TypeError("expected request body to be a JSON object")
    return parsed_body


def _normalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()

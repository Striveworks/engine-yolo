import logging
import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from engine_yolo.model_handler import ModelHandler
from engine_yolo.protocol import (
    InferenceProtocol,
    get_inference_protocol,
    get_protocol_handlers,
)

logger = logging.getLogger(__name__)


class Handler:
    def __init__(
        self,
        model_path: Path,
        protocol: InferenceProtocol,
        model_name: str = "model",
        model_handler: ModelHandler | None = None,
    ):
        self.model_name = model_name
        protocol_handlers = get_protocol_handlers(protocol)
        self.parse_request = protocol_handlers.parse_request
        self.render_response = protocol_handlers.render_response
        self.model_handler = model_handler or ModelHandler(model_path)

    async def infer(self, request: Request) -> Response:
        request_body = await request.body()
        try:
            parsed_request = self.parse_request(
                request_body,
                request.headers.get("Content-Type"),
                request.headers,
            )
            mapped_outputs = await run_in_threadpool(
                self.model_handler.handle, parsed_request.inputs
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        num_inputs = len(parsed_request.inputs)
        num_outputs = len(mapped_outputs)
        if num_inputs != num_outputs:
            raise HTTPException(
                status_code=500,
                detail=f"input/output mismatch: {num_inputs} != {num_outputs}",
            )

        response = self.render_response(
            self.model_name,
            parsed_request.request_id,
            parsed_request.inputs,
            mapped_outputs,
        )
        return response

    async def ready(self) -> JSONResponse:
        return JSONResponse(content={"status": "ready"})


def build_app(handler: Handler) -> FastAPI:
    router = APIRouter()
    router.add_api_route("/ready", handler.ready, methods=["GET"])
    router.add_api_route("/infer", handler.infer, methods=["POST"])
    app = FastAPI()
    app.include_router(router)
    return app


def get_model_path() -> Path:
    if "MODEL_PATH" in os.environ:
        model_path = Path(os.environ["MODEL_PATH"])
        if not model_path.exists():
            raise ValueError(f"MODEL_PATH {model_path!r} does not exist")
        if not model_path.is_file():
            raise ValueError(f"MODEL_PATH {model_path!r} is not a file")
    elif "MODEL_DIR" in os.environ:
        model_dir = Path(os.environ["MODEL_DIR"])
        if not model_dir.exists():
            raise ValueError(f"MODEL_DIR {model_dir!r} does not exist")
        if not model_dir.is_dir():
            raise ValueError(f"MODEL_DIR {model_dir!r} is not a directory")
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            raise ValueError(f"MODEL_DIR {model_dir!r} does not contain a model.pt file")
        if not model_path.is_file():
            raise ValueError(f"MODEL_DIR {model_dir!r} contains a non-file {model_path!r}")
    else:
        raise ValueError("MODEL_PATH or MODEL_DIR is not set")
    return model_path


def main() -> None:
    import uvicorn

    model_path = get_model_path()
    protocol: InferenceProtocol = get_inference_protocol()
    handler = Handler(model_path, protocol)
    app = build_app(handler)

    uvicorn.run(
        app=app,
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", 8080)),
        workers=int(os.getenv("WEB_CONCURRENCY", 1)),
        log_config=None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

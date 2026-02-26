import logging
import os
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from engine_yolo.model_handler import ModelHandler
from engine_yolo.protocol import (
    decode_kserve_inputs,
    encode_kserve_output,
    encode_kserve_response,
    get_request_id,
)

logger = logging.getLogger(__name__)


class Handler:
    def __init__(
        self,
        model_path: Path,
        model_name: str = "model",
        model_handler: ModelHandler | None = None,
    ):
        self.model_name = model_name
        self.model_handler = model_handler or ModelHandler(model_path)

    async def infer(self, request: Request) -> JSONResponse:
        request_body = await request.json()
        try:
            decoded_inputs = decode_kserve_inputs(request_body)
            mapped_outputs = await run_in_threadpool(self.model_handler.handle, decoded_inputs)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if len(decoded_inputs) != len(mapped_outputs):
            raise HTTPException(
                status_code=500,
                detail=f"input/output mismatch: {len(decoded_inputs)} != {len(mapped_outputs)}",
            )

        outputs = [
            encode_kserve_output(decoded_input.name, output)
            for decoded_input, output in zip(decoded_inputs, mapped_outputs)
        ]
        response_body = encode_kserve_response(
            model_name=self.model_name,
            request_id=get_request_id(request_body, request.headers.get("X-Request-ID")),
            outputs=outputs,
        )
        return JSONResponse(content=response_body)

    async def ready(self) -> JSONResponse:
        return JSONResponse(content={"status": "ready"})


def build_app(handler: Handler) -> FastAPI:
    router = APIRouter()
    router.add_api_route("/ready", handler.ready, methods=["GET"])
    router.add_api_route("/infer", handler.infer, methods=["POST"])
    app = FastAPI()
    app.include_router(router)
    return app


def main() -> None:
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
    handler = Handler(model_path, model_name="model")
    app = build_app(handler)

    uvicorn.run(
        app,
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", 8080)),
        workers=int(os.getenv("WEB_CONCURRENCY", 1)),
        log_config=None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

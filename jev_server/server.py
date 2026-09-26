import argparse
import hmac
import json
import os
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from jev_server.api.generated import ModelMetadataList
from jev_server.api.types import SystemOneRequest, SystemOneResponse
from jev_server.runtime import Runtime


def create_app(runtime, api_key=None):
    app = FastAPI(title="Jev-style models OpenVINO NPU server")
    lock = threading.Lock()

    @app.middleware("http")
    async def boundary(request, call_next):
        request_id = uuid.uuid4().hex
        if (
            api_key
            and request.url.path.startswith("/v1")
            and not hmac.compare_digest(
                request.headers.get("authorization", ""), f"Bearer {api_key}"
            )
        ):
            response = JSONResponse({"detail": "Invalid bearer token"}, status_code=401)
        elif len(await request.body()) > 65536:
            response = JSONResponse(
                {"detail": "Request body exceeds 64 KiB"}, status_code=413
            )
        else:
            response = await call_next(request)
        response.headers["x-typesafe-request-id"] = request_id
        return response

    @app.get("/healthz")
    def health():
        return {
            "ready": True,
            **runtime.evidence,
            "limits": {
                "tokens_per_sequence": runtime.config["length"],
                "options": runtime.config["max_options"],
                "questions": 8,
            },
        }

    @app.get("/v1/models", response_model=ModelMetadataList)
    def models():
        return {
            "models": [
                {
                    "name": name,
                    "description": runtime.config["model_id"],
                    "release_date": "2026-09-26",
                }
                for name in runtime.models
            ]
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(req: SystemOneRequest):
        if req.model not in runtime.models:
            raise HTTPException(404, "Unknown model; see /v1/models")
        if len(req.questions) > 8:
            raise HTTPException(422, "Maximum 8 questions per request")
        try:
            prepared = runtime.prepare(req)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if not lock.acquire(blocking=False):
            raise HTTPException(
                429, "Model busy; retry later", headers={"Retry-After": "1"}
            )
        try:
            started = time.perf_counter()
            answers = runtime.answer(prepared)
            elapsed = (time.perf_counter() - started) * 1000
        finally:
            lock.release()
        output_tokens = len(
            runtime.tokenizer.encode(json.dumps(answers), add_special_tokens=False).ids
        )
        return {
            "model": req.model,
            "answers": answers,
            "usage": {
                "input_tokens": prepared.input_tokens,
                "output_tokens": output_tokens,
            },
            "latency_ms": round(elapsed, 3),
        }

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/models")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--backend", choices=["openvino"], default="openvino")
    ap.add_argument("--device", choices=["NPU", "CPU"], default="NPU")
    args = ap.parse_args()
    runtime = Runtime(args.model_dir, args.backend, args.device)
    uvicorn.run(
        create_app(runtime, os.getenv("JEV_API_KEY")), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()

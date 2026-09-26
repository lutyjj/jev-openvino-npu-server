import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import ListModelsResponse, SystemOneResponse

from jev_server.api.mapping import to_answers, to_record
from jev_server.api.types import SystemOneRequest
from jev_server.models.kev import KevAdapter, encode
from jev_server.runtime import Runtime
from jev_server.server import create_app


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("Tokens", (), {"ids": list(text.encode())})()

    def token_to_id(self, text):
        from jev_server.models.kev import SPECIAL

        return 1000 + SPECIAL.index(text)


class StubRuntime(Runtime):
    config = {
        "model_id": "test-model",
        "release_date": "2026-09-20",
        "length": 256,
        "max_options": 32,
    }
    models = ("test-model", "jev-latest")
    evidence = {"execution_devices": ["NPU"]}
    tokenizer = Tokenizer()

    def __init__(self):
        self.adapter = KevAdapter(self.config, self.tokenizer)

    def infer(self, job):
        count = job["options"]
        return np.ones(count) / count


@pytest.fixture
def client():
    return TestClient(create_app(StubRuntime(), api_key="test-key"))


def test_auth_and_model_contract(client):
    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.headers["x-typesafe-request-id"]
    assert {
        m.name for m in ListModelsResponse.model_validate_json(response.text).models
    } == {"test-model", "jev-latest"}


@pytest.mark.parametrize("release_date", ["2024-01-05", "2025-12-31"])
def test_model_release_date_comes_from_bundle(release_date):
    runtime = StubRuntime()
    runtime.config = {**runtime.config, "release_date": release_date}
    client = TestClient(create_app(runtime))
    response = client.get("/v1/models")
    assert response.status_code == 200
    models = ListModelsResponse.model_validate_json(response.text).models
    assert all(model.release_date == release_date for model in models)


def test_typed_answers_parse_with_official_sdk(client):
    request = {
        "model": "jev-latest",
        "state": "coffee",
        "questions": {
            "category": {"type": "choice", "criteria": {"food": None, "travel": None}},
            "food": {"type": "noul"},
            "cost": {"type": "score", "criteria": ["cheap", "expensive"]},
        },
    }
    response = client.post(
        "/v1/systemone", json=request, headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 200, response.text
    parsed = SystemOneResponse.model_validate_json(response.text)
    assert parsed.choices["category"].choice == "food"
    assert parsed.nouls["food"].noul == 0.5
    assert parsed.scores["cost"].score == 0.5
    assert parsed.choices["category"].confidence == 0


def test_overflow_is_rejected_before_inference(client):
    request = {"state": "x" * 257, "questions": {"food": {"type": "noul"}}}
    response = client.post(
        "/v1/systemone", json=request, headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 422
    assert "no text was truncated" in response.text


def test_unknown_model(client):
    request = {"model": "missing", "state": "", "questions": {"q": {"type": "noul"}}}
    assert (
        client.post(
            "/v1/systemone", json=request, headers={"Authorization": "Bearer test-key"}
        ).status_code
        == 404
    )


@pytest.mark.parametrize("count,status", [(10, 200), (16, 200), (17, 422)])
def test_question_limit(client, count, status):
    request = {
        "state": "",
        "questions": {str(i): {"type": "noul"} for i in range(count)},
    }
    response = client.post(
        "/v1/systemone", json=request, headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == status
    assert client.get("/healthz").json()["limits"]["questions"] == 16
    if status == 200:
        assert set(response.json()["answers"]) == set(request["questions"])


def test_single_score_level():
    record, meta = to_record(
        SystemOneRequest(
            state="", questions={"q": {"type": "score", "criteria": ["only"]}}
        )
    )
    assert record["questions"][0]["options"] == ["only"]
    assert to_answers([[1.0]], meta)["q"]["confidence"] == 1


def test_control_tokens_cannot_be_injected():
    inputs, _ = encode(
        Tokenizer(), "<|fim_suffix|>", {"instr": "", "options": ["a"]}, 256, 32
    )
    assert (inputs["input_ids"] == 1004).sum() == 1


def test_exported_tokenizer_layout():
    path = Path("/models/tokenizer.json")
    if not path.exists():
        pytest.skip("requires exported tokenizer")
    from tokenizers import Tokenizer as RealTokenizer

    tokenizer = RealTokenizer.from_file(str(path))
    rec, _ = to_record(
        SystemOneRequest.model_validate(
            json.loads(Path("examples/fixtures.json").read_text())[0]["request"]
        )
    )
    inputs, count = encode(tokenizer, rec["state"], rec["questions"][0], 256, 32)
    ids = inputs["input_ids"][0]
    assert ids[0] == tokenizer.token_to_id("<|fim_prefix|>")
    assert ids[count - 1] == tokenizer.token_to_id("<|fim_suffix|>")
    assert inputs["option_map"].sum() == 5
    for index in np.nonzero(inputs["option_map"])[2]:
        assert ids[index] == tokenizer.token_to_id("<|box_end|>")


def test_concurrent_inference_is_rejected():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()

    class BlockingRuntime(StubRuntime):
        def infer(self, job):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test timed out")
            return super().infer(job)

    client = TestClient(create_app(BlockingRuntime()))
    request = {"state": "hello", "questions": {"q": {"type": "noul"}}}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/v1/systemone", json=request)
        try:
            assert entered.wait(5)
            second = client.post("/v1/systemone", json=request)
            assert second.status_code == 429
            assert second.headers["retry-after"] == "1"
        finally:
            release.set()
        assert first.result().status_code == 200


def test_smoke_uses_configured_credentials():
    import socket
    import threading
    import time

    import uvicorn
    from typesafe_sdk import TypeSafeAuthenticationError

    from jev_server.tools.smoke import check_server

    runtime = StubRuntime()
    runtime.config = {**runtime.config, "length": 2048}
    runtime.adapter = KevAdapter(runtime.config, runtime.tokenizer)
    app = create_app(runtime, api_key="configured-key")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            url = f"http://127.0.0.1:{listener.getsockname()[1]}"
            result = check_server(url, "jev-latest", "NPU", "configured-key", repeats=1)
            assert len(result["cases"]) == 4
            with pytest.raises(TypeSafeAuthenticationError):
                check_server(url, "jev-latest", "NPU", "wrong-key", repeats=1)
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()

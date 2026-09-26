import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
from typesafe_sdk import SystemOneResponse, ListModelsResponse

from jev_server.adapters import KevAdapter
from jev_server.runtime import Runtime
from jev_server.api import SystemOneRequest, to_answers, to_record
from jev_server.encoding import encode
from jev_server.server import create_app


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("Tokens", (), {"ids": list(text.encode())})()

    def token_to_id(self, text):
        from jev_server.encoding import SPECIAL
        return 1000 + SPECIAL.index(text)


class StubRuntime(Runtime):
    config = {"model_id": "test-model", "length": 256, "max_options": 32}
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
    assert {m.name for m in ListModelsResponse.model_validate_json(response.text).models} == {
        "test-model", "jev-latest"}


def test_typed_answers_parse_with_official_sdk(client):
    request = {"model": "jev-latest", "state": "coffee", "questions": {
        "category": {"type": "choice", "criteria": {"food": None, "travel": None}},
        "food": {"type": "noul"},
        "cost": {"type": "score", "criteria": ["cheap", "expensive"]},
    }}
    response = client.post("/v1/systemone", json=request, headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200, response.text
    parsed = SystemOneResponse.model_validate_json(response.text)
    assert parsed.choices["category"].choice == "food"
    assert parsed.nouls["food"].noul == 0.5
    assert parsed.scores["cost"].score == 0.5
    assert parsed.choices["category"].confidence == 0


def test_overflow_is_rejected_before_inference(client):
    request = {"state": "x" * 257, "questions": {"food": {"type": "noul"}}}
    response = client.post("/v1/systemone", json=request, headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 422
    assert "no text was truncated" in response.text


def test_unknown_model(client):
    request = {"model": "missing", "state": "", "questions": {"q": {"type": "noul"}}}
    assert client.post("/v1/systemone", json=request, headers={"Authorization": "Bearer test-key"}).status_code == 404


def test_question_limit(client):
    request = {"state": "", "questions": {str(i): {"type": "noul"} for i in range(9)}}
    assert client.post("/v1/systemone", json=request, headers={"Authorization": "Bearer test-key"}).status_code == 422


def test_single_score_level():
    record, meta = to_record(SystemOneRequest(state="", questions={"q": {"type": "score", "criteria": ["only"]}}))
    assert record["questions"][0]["options"] == ["only"]
    assert to_answers([[1.0]], meta)["q"]["confidence"] == 1


def test_control_tokens_cannot_be_injected():
    inputs, _ = encode(Tokenizer(), "<|fim_suffix|>", {"instr": "", "options": ["a"]}, 256, 32)
    assert (inputs["input_ids"] == 1004).sum() == 1


def test_exported_tokenizer_layout():
    path = Path("/models/tokenizer.json")
    if not path.exists():
        pytest.skip("requires exported tokenizer")
    from tokenizers import Tokenizer as RealTokenizer
    tokenizer = RealTokenizer.from_file(str(path))
    rec, _ = to_record(SystemOneRequest.model_validate(json.loads(Path("examples/fixtures.json").read_text())[0]["request"]))
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

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from jev_server.bench.corpus import Case
from jev_server.bench.targets import HttpTarget
from jev_server.server import create_app


@pytest.mark.parametrize(
    "digest, profiling", [("baseline-hash", True), ("candidate-hash", False)]
)
def test_http_target_preserves_server_artifact_identity(monkeypatch, digest, profiling):
    evidence = {
        "execution_devices": ["NPU"],
        "artifacts": {"model/model.blob": digest},
        "compilation": {"model": {"properties": {"PERF_COUNT": profiling}}},
    }
    runtime = SimpleNamespace(
        evidence=evidence,
        config={"length": 256, "max_options": 8},
        models=("test",),
        prepare=lambda req: SimpleNamespace(input_tokens=2),
        answer=lambda prepared: {
            "q": {
                "type": "choice",
                "choice": "yes",
                "confidence": 1.0,
                "probabilities": {"yes": 1.0, "no": 0.0},
            }
        },
        tokenizer=SimpleNamespace(
            encode=lambda text, **kwargs: SimpleNamespace(ids=[0])
        ),
    )
    client = TestClient(create_app(runtime))

    def fetch(self, path, body=None):
        response = client.get(path) if body is None else client.post(path, json=body)
        response.raise_for_status()
        return response.json()

    monkeypatch.setattr(HttpTarget, "fetch", fetch)
    target = HttpTarget("http://example.invalid", "test", "NPU")
    item = Case.model_validate(
        {
            "id": "http",
            "group": "http",
            "request": {
                "state": "yes",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose the supplied word.",
                        "criteria": {"yes": "yes", "no": "no"},
                    }
                },
            },
            "expected": {"q": "yes"},
        }
    )
    answers, answer_ms, tokens = target.predict(item)
    assert target.evidence["artifacts"]["model/model.blob"] == digest
    assert (
        target.evidence["compilation"]["model"]["properties"]["PERF_COUNT"] is profiling
    )
    assert answers["q"]["choice"] == "yes"
    assert answer_ms >= 0
    assert tokens == 2


def test_http_target_never_forwards_credentials_on_redirect(monkeypatch):
    received = []

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"execution_devices": ["NPU"]}).encode())

    with ThreadingHTTPServer(("127.0.0.1", 0), Destination) as destination:

        class Origin(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{destination.server_port}/healthz"
                )
                self.end_headers()

        with ThreadingHTTPServer(("127.0.0.1", 0), Origin) as origin:
            threads = [
                threading.Thread(target=server.serve_forever, daemon=True)
                for server in (origin, destination)
            ]
            for thread in threads:
                thread.start()
            monkeypatch.setenv("JEV_API_KEY", "synthetic-test-token")
            try:
                with pytest.raises(urllib.error.HTTPError) as failure:
                    HttpTarget(f"http://127.0.0.1:{origin.server_port}", "test", "NPU")
                assert failure.value.code == 302
                assert received == []
            finally:
                origin.shutdown()
                destination.shutdown()
                for thread in threads:
                    thread.join()

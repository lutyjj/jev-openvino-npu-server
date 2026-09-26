import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from jev_server.adapters import NanoJevAdapter, load_adapter
from jev_server.api import SystemOneRequest
from jev_server.backends import load_backend
from jev_server.runtime import Runtime


def test_unknown_adapter_and_backend_rejected(tmp_path):
    with pytest.raises(ValueError, match="adapter"):
        load_adapter({"adapter": "missing"}, None)
    with pytest.raises(ValueError, match="backend"):
        load_backend("automatic", tmp_path, ("model",), "NPU")


@pytest.fixture
def nano():
    path = Path("/nano")
    if not (path / "config.json").exists():
        pytest.skip("requires NanoJev bundle")
    return NanoJevAdapter(json.loads((path / "config.json").read_text()), Tokenizer.from_file(str(path / "tokenizer.json")))


def test_nano_boolean_is_one_path(nano):
    request = SystemOneRequest(state="A complete outage.", questions={"urgent": {
        "type": "noul", "instructions": "Is this urgent?", "criteria": {"true": "Needs immediate attention"}}})
    prepared = nano.prepare(request)
    job = prepared.jobs[0]
    assert job["options"] == 2
    assert len(job["paths"]) == 1
    text = nano.tokenizer.decode(job["paths"][0], skip_special_tokens=False)
    assert "Question type: boolean\n" in text
    assert "True criterion: Needs immediate attention\n" in text
    assert "Candidate:\nThe proposition is true.\nDecision:" in text
    assert prepared.input_tokens == len(job["paths"][0])


def test_nano_candidates_are_isolated(nano):
    request = SystemOneRequest(state="hello", questions={"q": {"type": "choice", "instructions": "Choose.",
                                "criteria": {"alpha": "first", "beta": "second"}}})
    paths = nano.prepare(request).jobs[0]["paths"]
    assert "beta" not in nano.tokenizer.decode(paths[0])
    assert "alpha" not in nano.tokenizer.decode(paths[1])


@pytest.mark.parametrize("question", [
    {"type": "choice", "instructions": "Choose.", "criteria": {"a": None, "b": "text"}},
    {"type": "score", "instructions": "Choose.", "criteria": ["only"]},
    {"type": "noul"},
])
def test_nano_rejects_unsupported_semantics(nano, question):
    with pytest.raises(ValueError):
        nano.prepare(SystemOneRequest(state="hello", questions={"q": question}))


def test_runtime_rejects_invalid_output():
    runtime = object.__new__(Runtime)
    runtime.backend = None
    class Adapter:
        def infer(self, job, backend):
            return np.array([np.nan, 1])
    runtime.adapter = Adapter()
    with pytest.raises(RuntimeError, match="invalid probabilities"):
        runtime.infer({})


def test_no_npu_fallback(tmp_path):
    import openvino as ov
    if "NPU" in ov.Core().available_devices:
        pytest.skip("requires a machine without NPU")
    with pytest.raises(RuntimeError, match="NPU unavailable"):
        load_backend("openvino", tmp_path, ("model",), "NPU")


def test_nano_export_matches_upstream_reference(nano):
    golden = json.loads(Path("tests/fixtures/nanojev-reference.json").read_text())
    assert hashlib.sha256(Path("examples/fixtures.json").read_bytes()).hexdigest() == golden["fixtures_sha256"]
    references = json.loads(Path("/nano/reference.json").read_text())
    assert nano.config["pins"]["revision"] == golden["checkpoint_revision"]
    assert len(references) == len(golden["probabilities"])
    for reference, expected in zip(references, golden["probabilities"]):
        np.testing.assert_allclose(reference["probabilities"], expected, atol=1e-4, rtol=0)


def test_nano_rotary_configuration_is_preserved(tmp_path):
    from jev_server.export_nanojev import load_backbone_config
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "rope_parameters": {"rope_type": "default", "rope_theta": 1000000}}))
    config = load_backbone_config(tmp_path)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    rotary = Qwen3RotaryEmbedding(config)
    expected = 1000000 ** (-np.arange(0, config.head_dim, 2) / config.head_dim)
    np.testing.assert_allclose(rotary.inv_freq.numpy(), expected, rtol=1e-6)

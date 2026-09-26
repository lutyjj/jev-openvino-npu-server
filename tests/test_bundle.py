import json

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from jev_server.bundle import save_metadata
from jev_server.models.kev import SPECIAL
from jev_server.runtime import Runtime


class Backend:
    evidence = {"execution_devices": ["CPU"]}

    def __init__(self):
        self.calls = []

    def run(self, graph, inputs):
        self.calls.append((graph, inputs))
        if graph == "backbone":
            return np.zeros((1, 1, 4), dtype=np.float32)
        return np.array([[0.25, 0.75]], dtype=np.float32)


@pytest.fixture(params=["kev-qwen3", "nanojev-qwen3"])
def bundle(request, tmp_path, monkeypatch):
    words = ["[UNK]", "marker", "choose", "alpha", "beta", *SPECIAL]
    tokenizer = Tokenizer(
        WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    config = {
        "adapter": request.param,
        "model_id": "test",
        "length": 64,
        "max_options": 2,
        "hidden_size": 4,
        "eos_token_id": 0,
    }
    reference = {
        "request": {
            "state": "marker",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "choose",
                    "criteria": {"a": "alpha", "b": "beta"},
                }
            },
        },
        "probabilities": [0.25, 0.75],
    }
    save_metadata(tmp_path, config, tokenizer, [reference])
    backend = Backend()
    monkeypatch.setattr("jev_server.runtime.load_backend", lambda *args: backend)
    return tmp_path, backend


def test_startup_prepares_reference_requests(bundle):
    directory, backend = bundle
    runtime = Runtime(directory, device="CPU")
    assert runtime.evidence["reference_argmax_matches"] == 1
    token_inputs = [
        inputs["input_ids"] for _, inputs in backend.calls if "input_ids" in inputs
    ]
    assert token_inputs
    assert all(1 in tokens for tokens in token_inputs)


def test_replaced_tokenizer_fails_before_inference(bundle):
    directory, backend = bundle
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.save(str(directory / "tokenizer.json"))
    with pytest.raises(ValueError, match="tokenizer digest mismatch"):
        Runtime(directory, device="CPU")
    assert backend.calls == []


def test_invalid_preparation_fails_before_inference(bundle):
    directory, backend = bundle
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config["length"] = 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="no text was truncated"):
        Runtime(directory, device="CPU")
    assert backend.calls == []


def test_old_bundle_requires_reexport(bundle):
    directory, backend = bundle
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config["format_version"] = 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="re-export"):
        Runtime(directory, device="CPU")
    assert backend.calls == []

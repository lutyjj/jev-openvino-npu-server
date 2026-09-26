import copy
import json
from pathlib import Path

import pytest

from jev_server.bench.corpus import (
    Case,
    import_kev,
    load_corpus,
    synthetic,
    write_corpus,
)
from jev_server.bench.metrics import compare, distributions
from jev_server.bench.runner import run
from jev_server.bundle import Bundle
from jev_server.models import load_adapter


def case():
    return Case.model_validate(
        {
            "id": "one",
            "group": "example",
            "request": {
                "state": "red",
                "questions": {
                    "color": {
                        "type": "choice",
                        "instructions": "Which color?",
                        "criteria": {"r": "red", "b": "blue"},
                    }
                },
            },
            "expected": {"color": "r"},
        }
    )


class Target:
    def __init__(self, p=0.8):
        self.evidence = {
            "mode": "runtime",
            "execution_devices": ["NPU"],
            "artifacts": {"model/model.blob": "test-digest"},
        }
        self.p = p
        self.calls = 0

    def predict(self, item):
        self.calls += 1
        return (
            {
                "color": {
                    "type": "choice",
                    "probabilities": {"r": self.p, "b": 1 - self.p},
                }
            },
            1.0,
            10,
        )

    def memory(self):
        return {}


def test_corpus_is_reproducible_valid_and_has_all_primitives(tmp_path):
    a, b = synthetic(), synthetic()
    assert a == b
    assert a != synthetic(seed=18)
    assert len(a) == 256
    assert {q.type for c in a for q in c.request.questions.values()} == {
        "choice",
        "noul",
        "score",
    }
    path = tmp_path / "corpus.jsonl"
    write_corpus(path, a)
    loaded, digest = load_corpus(path)
    assert loaded == a
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        write_corpus(path, b)


@pytest.mark.parametrize("directory", ["/models", "/nano"])
def test_default_corpus_fits_supported_default_bundles(directory):
    bundle = Bundle.load(Path(directory))
    adapter = load_adapter(bundle.config, bundle.tokenizer)
    for item in synthetic():
        prepared = adapter.prepare(item.request)
        assert len(prepared.jobs) == len(item.expected)


def test_large_corpus_fits_2048_export_shape():
    bundle = Bundle.load(Path("/models"))
    config = {**bundle.config, "length": 2048, "max_options": 32}
    adapter = load_adapter(config, bundle.tokenizer)
    for item in synthetic(profile="context2048"):
        adapter.prepare(item.request)


def test_duplicate_cases_and_wrong_labels_fail(tmp_path):
    path = tmp_path / "bad.jsonl"
    write_corpus(path, [case(), case()])
    with pytest.raises(ValueError, match="unique"):
        load_corpus(path)
    raw = case().model_dump()
    raw["expected"] = {"color": "missing"}
    with pytest.raises(ValueError, match="Invalid label"):
        Case.model_validate(raw)


def test_warmup_excluded_and_quality_has_independent_labels():
    target = Target()
    report = run(target, [case()], "digest", repeats=2, warmup=3)
    assert target.calls == 5
    assert report["summary"]["measured_requests"] == 2
    assert report["summary"]["quality"]["questions"] == 1
    assert report["summary"]["quality"]["accuracy"] == 1
    assert report["summary"]["quality"]["brier"] == pytest.approx(0.08)


def test_comparison_detects_decision_and_probability_changes():
    a = run(Target(0.8), [case()], "digest", repeats=1, warmup=0)
    b = run(Target(0.3), [case()], "digest", repeats=1, warmup=0)
    a["cases"][0]["samples"][0]["elapsed_ms"] = 20
    b["cases"][0]["samples"][0]["elapsed_ms"] = 10
    result = compare(a, b)
    assert result["paired_case_speedup"]["p50"] == 2
    assert result["decision_agreement"] == 0
    assert result["max_probability_delta"] == pytest.approx(0.5)
    assert result["accuracy_delta"] == -1


@pytest.mark.parametrize("mutation", ["corpus", "settings", "failure", "mode"])
def test_comparison_rejects_incomparable_runs(mutation):
    a = run(Target(), [case()], "digest", repeats=1, warmup=0)
    b = copy.deepcopy(a)
    if mutation == "corpus":
        b["corpus_sha256"] = "different"
    elif mutation == "settings":
        b["settings"]["repeats"] = 2
    elif mutation == "failure":
        b["cases"][0]["error"] = {"type": "ValueError"}
    else:
        b["target"]["mode"] = "http"
    with pytest.raises(ValueError):
        compare(a, b)


def test_invalid_predictions_are_reported_as_failures():
    report = run(Target(float("nan")), [case()], "digest", repeats=1, warmup=0)
    assert report["summary"]["failed_cases"] == 1
    assert report["summary"]["request_ms"] is None
    assert report["summary"]["quality"]["accuracy"] is None


def test_noul_and_score_distributions():
    item = Case.model_validate(
        {
            "id": "types",
            "group": "types",
            "request": {
                "state": "test",
                "questions": {
                    "b": {"type": "noul", "instructions": "True?"},
                    "s": {
                        "type": "score",
                        "instructions": "Level?",
                        "criteria": ["low", "high"],
                    },
                },
            },
            "expected": {"b": "true", "s": "1"},
        }
    )
    result = distributions(
        item,
        {
            "b": {"type": "noul", "noul": 0.75},
            "s": {"type": "score", "probabilities": {"0": 0.2, "1": 0.8}},
        },
    )
    assert result["b"] == {"false": 0.25, "true": 0.75}
    assert result["s"]["1"] == 0.8


def test_import_requires_checksum_and_removes_labels_from_request(tmp_path):
    import hashlib

    raw = {
        "state": "hello",
        "questions": {
            "q": {
                "type": "noul",
                "instructions": "Greeting?",
                "label": 1,
                "src": "example",
            }
        },
        "_meta": {"source": "example"},
    }
    path = tmp_path / "external.jsonl"
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        import_kev(path, "wrong")
    imported = import_kev(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert imported[0].expected == {"q": "true"}
    assert "label" not in imported[0].request.model_dump()["questions"]["q"]

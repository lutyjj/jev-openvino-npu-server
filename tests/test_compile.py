import json
from types import SimpleNamespace

import openvino as ov
import pytest

from jev_server.backends.openvino import OpenVINOBackend, load_npu
from jev_server.tools.compile import compile_graph


def test_compile_records_applied_settings(monkeypatch, tmp_path):
    observed = {}

    class Compiled:
        def export_model(self, output):
            output.write(b"compiled")

        def get_property(self, name):
            return "PLUGIN"

    class Core:
        def compile_model(self, model, device, properties):
            observed.update(properties)
            assert device == "NPU"
            return Compiled()

    monkeypatch.setattr(ov, "Core", Core)
    compile_graph(tmp_path, "3720", 2, False)
    metadata = json.loads((tmp_path / "compile.json").read_text())
    assert metadata["properties"] == observed
    assert observed["NPU_COMPILATION_MODE_PARAMS"] == "optimization-level=2"
    assert observed["PERF_COUNT"] is False
    assert (tmp_path / "model.blob").read_bytes() == b"compiled"


@pytest.mark.parametrize(
    "properties, expected", [({}, True), ({"PERF_COUNT": False}, False)]
)
def test_blob_import_preserves_profiling_setting(tmp_path, properties, expected):
    (tmp_path / "model.blob").write_bytes(b"compiled")
    (tmp_path / "compile.json").write_text(
        json.dumps(
            {"openvino": ov.__version__, "platform": "3720", "properties": properties}
        )
    )
    observed = {}

    def import_model(tensor, device, settings):
        observed.update(settings)
        return "model"

    core = SimpleNamespace(
        get_property=lambda device, name: "3720", import_model=import_model
    )
    compiled, storage = load_npu(core, tmp_path)
    assert compiled == "model"
    assert observed["PERF_COUNT"] is expected
    assert storage


def test_backend_owns_artifact_and_compiler_evidence(monkeypatch, tmp_path):
    (tmp_path / "model.blob").write_bytes(b"abc")
    metadata = {
        "openvino": ov.__version__,
        "platform": "3720",
        "properties": {"PERF_COUNT": False},
    }
    (tmp_path / "compile.json").write_text(json.dumps(metadata))
    compiled = SimpleNamespace(
        get_property=lambda name: ["NPU"], create_infer_request=lambda: object()
    )
    core = SimpleNamespace(
        available_devices=["NPU"],
        get_property=lambda device, name: (
            "3720" if name == "DEVICE_ARCHITECTURE" else "test-device"
        ),
        import_model=lambda *args: compiled,
    )
    monkeypatch.setattr(ov, "Core", lambda: core)
    backend = OpenVINOBackend(tmp_path, ["model"], "NPU")
    assert backend.evidence["artifacts"] == {
        "model/model.blob": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    }
    assert backend.evidence["compilation"] == {"model": metadata}
    assert backend.profile() == {"model": []}

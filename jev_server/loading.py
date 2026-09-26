import json
from pathlib import Path

import numpy as np
import openvino as ov


def execution_devices(compiled):
    devices = compiled.get_property("EXECUTION_DEVICES")
    return [devices] if isinstance(devices, str) else list(devices)


def load_npu(core, directory):
    directory = Path(directory)
    metadata = json.loads((directory / "compile.json").read_text())
    if metadata["openvino"] != ov.__version__:
        raise RuntimeError(f"Blob requires OpenVINO {metadata['openvino']}; running {ov.__version__}")
    architecture = core.get_property("NPU", "DEVICE_ARCHITECTURE")
    if metadata["platform"] != architecture:
        raise RuntimeError(f"Blob targets {metadata['platform']}; device is {architecture}")
    mapped = np.memmap(directory / "model.blob", dtype=np.uint8, mode="c")
    tensor = ov.Tensor(mapped, shared_memory=True)
    compiled = core.import_model(tensor, "NPU", {"PERF_COUNT": True})
    return compiled, (mapped, tensor)

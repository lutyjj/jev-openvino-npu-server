import json
import time
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
        raise RuntimeError(
            f"Blob requires OpenVINO {metadata['openvino']}; running {ov.__version__}"
        )
    architecture = core.get_property("NPU", "DEVICE_ARCHITECTURE")
    if metadata["platform"] != architecture:
        raise RuntimeError(
            f"Blob targets {metadata['platform']}; device is {architecture}"
        )
    mapped = np.memmap(directory / "model.blob", dtype=np.uint8, mode="c")
    tensor = ov.Tensor(mapped, shared_memory=True)
    compiled = core.import_model(tensor, "NPU", {"PERF_COUNT": True})
    return compiled, (mapped, tensor)


class OpenVINOBackend:
    def __init__(self, directory, graphs, device):
        if device not in {"NPU", "CPU"}:
            raise ValueError(
                "Select NPU or CPU explicitly; automatic fallback is forbidden"
            )
        self.core = ov.Core()
        if device not in self.core.available_devices:
            raise RuntimeError(f"{device} unavailable; automatic fallback is forbidden")
        started = time.perf_counter()
        self.models = {}
        self.storage = []
        self.requests = {}
        for name in graphs:
            if not name.isidentifier():
                raise ValueError("Invalid graph name")
            path = directory if name == "model" else directory / name
            if device == "NPU":
                compiled, storage = load_npu(self.core, path)
                self.storage.append(storage)
            else:
                compiled = self.core.compile_model(
                    str(path / "model.xml"),
                    "CPU",
                    {
                        "PERFORMANCE_HINT": "LATENCY",
                        "PERF_COUNT": True,
                        "INFERENCE_NUM_THREADS": 2,
                        "INFERENCE_PRECISION_HINT": "f32",
                    },
                )
            devices = execution_devices(compiled)
            if not devices or any(d.split(".")[0] != device for d in devices):
                raise RuntimeError(f"Unexpected execution devices: {devices}")
            self.models[name] = compiled
            self.requests[name] = compiled.create_infer_request()
        self.evidence = {
            "backend": "openvino",
            "execution_devices": [device],
            "graphs": list(graphs),
            "device_name": self.core.get_property(device, "FULL_DEVICE_NAME"),
            "openvino": ov.__version__,
            "load_seconds": round(time.perf_counter() - started, 3),
        }
        if device == "NPU":
            self.evidence["architecture"] = self.core.get_property(
                device, "DEVICE_ARCHITECTURE"
            )

    def run(self, graph, inputs):
        request = self.requests[graph]
        request.infer(inputs)
        return request.get_output_tensor(0).data.copy()

    def profile(self):
        return {
            name: [
                {
                    "node": p.node_name,
                    "execution_type": p.exec_type,
                    "status": str(p.status),
                    "real_us": p.real_time.total_seconds() * 1e6,
                }
                for p in request.profiling_info
            ]
            for name, request in self.requests.items()
        }

import argparse
import io
import json
import time
from pathlib import Path

import openvino as ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("/models"))
    ap.add_argument("--platform", default="3720")
    ap.add_argument("--optimization-level", type=int, choices=[0, 1, 2])
    ap.add_argument("--no-profiling", action="store_true")
    args = ap.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    from jev_server.models import load_adapter

    adapter = load_adapter(config, None)
    for name in adapter.graphs:
        directory = args.model_dir if name == "model" else args.model_dir / name
        compile_graph(
            directory,
            args.platform,
            args.optimization_level,
            not args.no_profiling,
        )


def compile_graph(directory, platform, optimization_level=None, profiling=True):
    core = ov.Core()
    print(f"Offline NPU compilation with OpenVINO {ov.__version__}", flush=True)
    start = time.perf_counter()
    properties = {
        "NPU_PLATFORM": platform,
        "NPU_COMPILER_TYPE": "PLUGIN",
        "PERFORMANCE_HINT": "LATENCY",
        "PERF_COUNT": profiling,
        "COMPILATION_NUM_THREADS": 2,
    }
    if optimization_level is not None:
        properties["NPU_COMPILATION_MODE_PARAMS"] = (
            f"optimization-level={optimization_level}"
        )
    compiled = core.compile_model(str(directory / "model.xml"), "NPU", properties)
    output = io.BytesIO()
    compiled.export_model(output)
    temporary = directory / "model.blob.tmp"
    temporary.write_bytes(output.getbuffer())
    temporary.replace(directory / "model.blob")
    metadata = {
        "platform": platform,
        "openvino": ov.__version__,
        "compiler": str(compiled.get_property("NPU_COMPILER_TYPE")),
        "properties": properties,
        "compile_seconds": time.perf_counter() - start,
    }
    (directory / "compile.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()

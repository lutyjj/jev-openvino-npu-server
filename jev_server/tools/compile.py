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
    args = ap.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    from jev_server.models import load_adapter

    adapter = load_adapter(config, None)
    for name in adapter.graphs:
        directory = args.model_dir if name == "model" else args.model_dir / name
        compile_graph(directory, args.platform)


def compile_graph(directory, platform):
    core = ov.Core()
    print(f"Offline NPU compilation with OpenVINO {ov.__version__}", flush=True)
    start = time.perf_counter()
    compiled = core.compile_model(
        str(directory / "model.xml"),
        "NPU",
        {
            "NPU_PLATFORM": platform,
            "NPU_COMPILER_TYPE": "PLUGIN",
            "PERFORMANCE_HINT": "LATENCY",
            "PERF_COUNT": True,
            "COMPILATION_NUM_THREADS": 2,
        },
    )
    output = io.BytesIO()
    compiled.export_model(output)
    temporary = directory / "model.blob.tmp"
    temporary.write_bytes(output.getbuffer())
    temporary.replace(directory / "model.blob")
    metadata = {
        "platform": platform,
        "openvino": ov.__version__,
        "compiler": str(compiled.get_property("NPU_COMPILER_TYPE")),
        "compile_seconds": time.perf_counter() - start,
    }
    (directory / "compile.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()

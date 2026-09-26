import argparse
import json
import time
from pathlib import Path

import numpy as np

from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/models"))
    parser.add_argument("--backend", default="openvino", choices=["openvino"])
    parser.add_argument("--device", choices=["CPU", "NPU"], default="NPU")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    runtime = Runtime(args.model_dir, args.backend, args.device)
    refs = json.loads((args.model_dir / "reference.json").read_text())
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        runtime.infer(refs[0]["job"])
        timings.append((time.perf_counter() - start) * 1000)
    print(json.dumps({"passed": True, **runtime.evidence, "repeats": args.repeats,
                      "latency_ms": {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95))},
                      "profile": runtime.profile()}), flush=True)


if __name__ == "__main__":
    main()

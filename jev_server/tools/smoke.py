import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

from typesafe_sdk import TypeSafeClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18009")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--device", choices=["CPU", "NPU"], default="NPU")
    args = ap.parse_args()
    result = check_server(
        args.url,
        args.model,
        args.device,
        os.getenv("JEV_API_KEY") or "local-experiment",
    )
    print(json.dumps(result, indent=2))


def check_server(url, model, device, api_key, repeats=20):
    with urllib.request.urlopen(url + "/healthz") as response:
        health = json.load(response)
    if health["execution_devices"] != [device]:
        raise RuntimeError(f"Unexpected execution devices: {health}")
    cases = []
    fixtures = json.loads(Path("examples/fixtures.json").read_text())
    with TypeSafeClient(base_url=url, api_key=api_key, timeout=30) as client:
        models = client.models.list()
        for fixture in fixtures:
            request = fixture["request"]
            start = time.perf_counter()
            response = client.system_one(
                state=request["state"], questions=request["questions"], model=model
            )
            cases.append(
                {
                    "response": response.model_dump(mode="json"),
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                }
            )
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            request = fixtures[0]["request"]
            client.system_one(
                state=request["state"], questions=request["questions"], model=model
            )
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "health": health,
        "models": models.model_dump(mode="json"),
        "cases": cases,
        "http_median_ms": statistics.median(timings),
        "repeats": len(timings),
    }


if __name__ == "__main__":
    main()

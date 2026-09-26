import argparse
import json
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
    with urllib.request.urlopen(args.url + "/healthz") as response:
        health = json.load(response)
    assert health["execution_devices"] == [args.device], health
    cases = []
    fixtures = json.loads(Path("examples/fixtures.json").read_text())
    with TypeSafeClient(base_url=args.url, api_key="local-experiment", timeout=30) as client:
        models = client.models.list()
        for fixture in fixtures:
            request = fixture["request"]
            start = time.perf_counter()
            response = client.system_one(state=request["state"], questions=request["questions"], model=args.model)
            cases.append({"response": response.model_dump(mode="json"),
                          "elapsed_ms": (time.perf_counter() - start) * 1000})
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            request = fixtures[0]["request"]
            client.system_one(state=request["state"], questions=request["questions"], model=args.model)
            timings.append((time.perf_counter() - start) * 1000)
    print(json.dumps({"health": health, "models": models.model_dump(mode="json"), "cases": cases,
                      "http_median_ms": statistics.median(timings), "repeats": len(timings)}, indent=2))


if __name__ == "__main__":
    main()

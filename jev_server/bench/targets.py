import json
import os
import resource
import time
import urllib.request

from jev_server.runtime import Runtime


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RuntimeTarget:
    def __init__(self, directory, device):
        self.runtime = Runtime(directory, device=device)
        self.evidence = {
            "mode": "runtime",
            **self.runtime.evidence,
            "config": self.runtime.config,
        }

    def predict(self, case):
        prepared = self.runtime.prepare(case.request)
        start = time.perf_counter()
        answers = self.runtime.answer(prepared)
        answer_ms = (time.perf_counter() - start) * 1000
        return answers, answer_ms, prepared.input_tokens

    def memory(self):
        return {
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "scope": "Linux benchmark process including model loading; excludes device allocations",
        }


class HttpTarget:
    def __init__(self, url, model, device, timeout=120):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.opener = urllib.request.build_opener(NoRedirects())
        self.evidence = {"mode": "http", **self.fetch("/healthz")}
        if self.evidence.get("execution_devices") != [device]:
            raise ValueError("Server execution device differs from requested device")

    def fetch(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if key := os.getenv("JEV_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def predict(self, case):
        body = case.request.model_dump(mode="json", exclude_none=True)
        body["model"] = self.model
        response = self.fetch("/v1/systemone", body)
        return (
            response["answers"],
            response.get("latency_ms"),
            response["usage"]["input_tokens"],
        )

    def memory(self):
        return {
            "process_peak_rss_bytes": None,
            "scope": "Remote server memory is not observable through this API",
        }

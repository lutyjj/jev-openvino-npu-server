import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .adapters import load_adapter
from .api import to_answers
from .backends import load_backend


class Runtime:
    def __init__(self, directory, backend="openvino", device="NPU"):
        directory = Path(directory)
        self.config = json.loads((directory / "config.json").read_text())
        if self.config.get("format_version") != 1:
            raise ValueError("Unsupported bundle format; re-export the model")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.adapter = load_adapter(self.config, self.tokenizer)
        self.models = (self.config["model_id"], "jev-latest")
        self.backend = load_backend(backend, directory, self.adapter.graphs, device)
        self.evidence = {**self.backend.evidence, "model_id": self.config["model_id"], "adapter": self.config["adapter"]}
        refs = json.loads((directory / "reference.json").read_text())
        if not refs:
            raise ValueError("Bundle requires reference cases")
        errors = []
        same = []
        for ref in refs:
            actual = self.infer(ref["job"])
            expected = np.asarray(ref["probabilities"])
            if actual.shape != expected.shape or not np.isfinite(expected).all():
                raise ValueError("Invalid reference probabilities")
            errors.append(float(np.max(np.abs(actual - expected))))
            same.append(int(np.argmax(actual)) == int(np.argmax(expected)))
        if not all(same) or max(errors) > 0.05:
            raise RuntimeError(f"Reference parity failed: errors={errors}, agreement={same}")
        self.evidence.update({"reference_max_probability_error": max(errors), "reference_argmax_matches": sum(same),
                              "reference_cases": len(refs)})
        print(json.dumps(self.evidence), flush=True)

    def prepare(self, request):
        return self.adapter.prepare(request)

    def infer(self, job):
        probabilities = self.adapter.infer(job, self.backend)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise RuntimeError("Backend returned invalid probabilities")
        if abs(float(probabilities.sum()) - 1) > 0.02:
            raise RuntimeError("Backend returned unnormalized probabilities")
        return probabilities

    def answer(self, prepared):
        return to_answers([self.infer(job).tolist() for job in prepared.jobs], prepared.meta)

    def profile(self):
        return self.backend.profile()

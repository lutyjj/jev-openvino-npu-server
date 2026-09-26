import numpy as np

from jev_server.api.mapping import to_record
from jev_server.models.base import Prepared


class NanoJevAdapter:
    graphs = ("backbone", "head")

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def prepare(self, request):
        _, meta = to_record(request)
        jobs = []
        count = 0
        if not isinstance(request.state, str) or not request.state.strip():
            raise ValueError("NanoJev requires a nonempty text state")
        for question in request.questions.values():
            q = question.model_dump(exclude_none=True)
            instructions = q.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("NanoJev requires nonempty text instructions")
            typ = "boolean" if q["type"] == "noul" else q["type"]
            criteria = q.get("criteria", {})
            if typ == "boolean":
                if set(criteria) - {"false", "true"}:
                    raise ValueError("Boolean criteria must use false or true keys")
                texts = ["The proposition is true."]
                options = 2
            elif typ == "choice":
                texts = [f"{key}: {value}" for key, value in criteria.items()]
                options = len(texts)
            else:
                texts = criteria
                options = len(texts)
            values = criteria.values() if isinstance(criteria, dict) else criteria
            if any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError("NanoJev requires nonempty text criteria")
            if options < 2 or options > min(
                self.config["max_options"], 10 if typ == "score" else 255
            ):
                raise ValueError("NanoJev option count exceeds the model limits")
            segments = [
                f"State:\n{request.state}\n",
                f"Question type: {typ}\nQuestion:\n{instructions}\n",
            ]
            if typ == "boolean":
                for key, label in (("false", "False"), ("true", "True")):
                    if key in criteria:
                        segments[1] += f"{label} criterion: {criteria[key]}\n"
            prefix = sum(
                [
                    self.tokenizer.encode(t, add_special_tokens=False).ids
                    for t in segments
                ],
                [],
            )
            paths = []
            for text in texts:
                ids = (
                    prefix
                    + self.tokenizer.encode(
                        f"Candidate:\n{text}\nDecision:", add_special_tokens=False
                    ).ids
                )
                ids += [self.config["eos_token_id"]]
                if len(ids) > self.config["length"]:
                    raise ValueError("Candidate exceeds context; no text was truncated")
                count += len(ids)
                paths.append(ids)
            jobs.append({"paths": paths, "options": options, "type": typ})
        return Prepared(jobs, meta, count)

    def infer(self, job, backend):
        length = self.config["length"]
        width = self.config["max_options"]
        leaves = np.zeros((1, width, self.config["hidden_size"]), dtype=np.float32)
        for index, ids in enumerate(job["paths"]):
            tokens = np.zeros((1, length), dtype=np.int64)
            tokens[0, : len(ids)] = ids
            selector = np.zeros((1, 1, length), dtype=np.float32)
            selector[0, 0, len(ids) - 1] = 1
            leaves[0, index] = backend.run(
                "backbone", {"input_ids": tokens, "leaf_map": selector}
            )[0, 0]
        valid = np.zeros((1, width), dtype=np.float32)
        valid[0, : job["options"]] = 1
        return backend.run(
            "head",
            {
                "leaves": leaves,
                "valid": valid,
                "is_choice": np.array([[job["type"] == "choice"]], dtype=np.float32),
                "is_boolean": np.array([[job["type"] == "boolean"]], dtype=np.float32),
            },
        )[0, : job["options"]]

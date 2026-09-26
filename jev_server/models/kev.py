import re

import numpy as np

from jev_server.api.mapping import to_record
from jev_server.models.base import Prepared

SPECIAL = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
]


def encode(tokenizer, state, question, length, max_options):

    def tokens(text):
        text = re.sub("<\\|([A-Za-z0-9_]+)\\|>", "<¦\\1¦>", text)
        return tokenizer.encode(text, add_special_tokens=False).ids

    s, q, o, end, decide = [tokenizer.token_to_id(t) for t in SPECIAL]
    if None in (s, q, o, end, decide):
        raise ValueError("tokenizer lacks Kev control tokens")
    options = question["options"]
    if not 1 <= len(options) <= max_options:
        raise ValueError(f"question requires 1..{max_options} options")
    ids = [s] + tokens(state) + [q] + tokens(question["instr"])
    ends = []
    for option in options:
        ids += [o] + tokens(option) + [end]
        ends.append(len(ids) - 1)
    ids += [decide]
    if len(ids) > length:
        raise ValueError(
            f"state and question need {len(ids)} tokens; maximum is {length}; no text was truncated"
        )
    inputs = {
        "input_ids": np.zeros((1, length), dtype=np.int64),
        "decide_map": np.zeros((1, 1, length), dtype=np.float32),
        "option_map": np.zeros((1, max_options, length), dtype=np.float32),
    }
    inputs["input_ids"][0, : len(ids)] = ids
    inputs["decide_map"][0, 0, len(ids) - 1] = 1
    inputs["option_map"][0, range(len(ends)), ends] = 1
    return (inputs, len(ids))


class KevAdapter:
    graphs = ("model",)

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def prepare(self, request):
        record, meta = to_record(request)
        encoded = [
            encode(
                self.tokenizer,
                record["state"],
                q,
                self.config["length"],
                self.config["max_options"],
            )
            for q in record["questions"]
        ]
        return Prepared(
            [
                {"inputs": inputs, "options": len(q["options"])}
                for (inputs, _), q in zip(encoded, record["questions"])
            ],
            meta,
            sum(count for _, count in encoded),
        )

    def infer(self, job, backend):
        inputs = {
            k: np.asarray(v, dtype=np.int64 if k == "input_ids" else np.float32)
            for k, v in job["inputs"].items()
        }
        return backend.run("model", inputs)[0, : job["options"]]

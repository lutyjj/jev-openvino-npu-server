import hashlib
import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jev_server.api.mapping import question_keys
from jev_server.api.types import SystemOneRequest


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    group: str = Field(min_length=1)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    request: SystemOneRequest
    expected: dict[str, str]

    @model_validator(mode="after")
    def validate_labels(self):
        if set(self.expected) != set(self.request.questions):
            raise ValueError("Every question requires a label")
        for name, question in self.request.questions.items():
            if self.expected[name] not in question_keys(
                question.type, question.criteria
            ):
                raise ValueError(f"Invalid label for {name}")
        return self


def load_corpus(path):
    raw = Path(path).read_bytes()
    cases = [
        Case.model_validate_json(line) for line in raw.splitlines() if line.strip()
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Corpus must be nonempty with unique IDs")
    return cases, hashlib.sha256(raw).hexdigest()


def write_corpus(path, cases):
    with Path(path).open("x") as output:
        output.writelines(
            case.model_dump_json(exclude_none=True) + "\n" for case in cases
        )


def import_kev(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Source checksum mismatch")
    cases = []
    for index, line in enumerate(raw.splitlines()):
        record = json.loads(line)
        questions, expected = {}, {}
        for name, question in record["questions"].items():
            questions[name] = {
                k: v
                for k, v in question.items()
                if k in {"type", "instructions", "criteria"}
            }
            label = question["label"]
            expected[name] = (
                ["false", "true"][int(label)]
                if question["type"] == "noul"
                else str(label)
            )
        cases.append(
            Case(
                id=f"{sha256[:12]}:{index}",
                group=record["_meta"]["source"],
                source_sha256=sha256,
                request=SystemOneRequest(state=record["state"], questions=questions),
                expected=expected,
            )
        )
    return cases


def synthetic(count=256, seed=17, profile="small"):
    if profile not in {"small", "context2048"}:
        raise ValueError("Unknown synthetic profile")
    rng = random.Random(seed)
    cases = []
    topics = {
        "billing": "A payment was taken twice. Refund the duplicate charge.",
        "delivery": "The parcel has not arrived. Where is it?",
        "access": "I forgot my password and cannot sign in.",
        "damage": "The package arrived but the item inside is broken.",
    }
    for index in range(count):
        family = index % 4
        noise = rng.choice([0, 2, 4, 8] if profile == "small" else [0, 16, 64, 160])
        distractors = "\n".join(
            f"Archive {n}: routine inspection completed." for n in range(noise)
        )
        questions, expected = {}, {}
        if family == 0:
            size = rng.choice([2, 4, 8] if profile == "small" else [2, 8, 16, 32])
            names = [f"item_{i}" for i in range(size)]
            target = rng.choice(names)
            rng.shuffle(names)
            state = f"The selected item is {target}."
            questions["item"] = {
                "type": "choice",
                "instructions": "Which item is selected?",
                "criteria": {name: name for name in names},
            }
            expected["item"] = target
            group = f"lookup/{size}-options"
        elif family == 1:
            active, approved = rng.choice([True, False]), rng.choice([True, False])
            state = f"Active: {str(active).lower()}. Approved: {str(approved).lower()}."
            for name, instruction, label in [
                ("both", "Are active and approved both true?", active and approved),
                (
                    "either",
                    "Is at least one of active and approved true?",
                    active or approved,
                ),
                ("inactive", "Is active false?", not active),
            ]:
                questions[name] = {"type": "noul", "instructions": instruction}
                expected[name] = str(label).lower()
            group = "boolean/multiple-questions"
        elif family == 2:
            label = rng.choice(list(topics))
            keys = list(topics)
            rng.shuffle(keys)
            state = topics[label]
            questions["team"] = {
                "type": "choice",
                "instructions": "Route the request to the relevant team.",
                "criteria": {key: key for key in keys},
            }
            expected["team"] = label
            group = "routing"
        else:
            threshold = rng.randrange(1, 99)
            value = threshold + rng.choice([-1, 0, 1])
            state = f"Measured value: {value}. Threshold: {threshold}."
            label = 0 if value < threshold else 1 if value == threshold else 2
            questions["relation"] = {
                "type": "score",
                "instructions": "Compare the measured value with the threshold.",
                "criteria": ["Below", "Equal", "Above"],
            }
            expected["relation"] = str(label)
            group = "numeric/score"
        state = f"{distractors}\n{state}" if index % 2 else f"{state}\n{distractors}"
        cases.append(
            Case(
                id=f"synthetic-{profile}-{seed}-{index}",
                group=f"{group}/noise-{noise}",
                request=SystemOneRequest(state=state, questions=questions),
                expected=expected,
            )
        )
    return cases

from dataclasses import dataclass


@dataclass
class Prepared:
    jobs: list
    meta: list
    input_tokens: int

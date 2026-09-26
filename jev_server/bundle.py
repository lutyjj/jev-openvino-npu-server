import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, TypeAdapter
from tokenizers import Tokenizer

from jev_server.api.types import SystemOneRequest

FORMAT_VERSION = 2


class BundleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    format_version: Literal[2]
    model_id: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    length: PositiveInt
    max_options: PositiveInt
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Reference(BaseModel):
    request: SystemOneRequest
    probabilities: list[float] = Field(min_length=1)
    torch_error: float | None = None


REFERENCES = TypeAdapter(list[Reference])


@dataclass
class Bundle:
    config: dict
    tokenizer: Tokenizer
    references: list[Reference]

    @classmethod
    def load(cls, directory: Path):
        config = json.loads((directory / "config.json").read_text())
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported bundle format; re-export the model")
        config = BundleConfig.model_validate(config).model_dump()
        tokenizer_path = directory / "tokenizer.json"
        digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        if digest != config.get("tokenizer_sha256"):
            raise ValueError("Bundle tokenizer digest mismatch; re-export the model")
        references = REFERENCES.validate_json(
            (directory / "reference.json").read_text()
        )
        if not references:
            raise ValueError("Bundle requires reference requests")
        return cls(config, Tokenizer.from_file(str(tokenizer_path)), references)


def save_metadata(
    directory: Path, config: dict, tokenizer: Tokenizer, references: list[dict]
):
    tokenizer_path = directory / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = {
        **config,
        "format_version": FORMAT_VERSION,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }
    references = REFERENCES.validate_python(references)
    (directory / "reference.json").write_bytes(
        REFERENCES.dump_json(references, indent=2) + b"\n"
    )
    (directory / "config.json").write_text(
        BundleConfig.model_validate(config).model_dump_json(indent=2) + "\n"
    )

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: dict[str, str | dict[str, Any] | list[Any] | None]


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class NoulCriteria(BaseModel):
    true: str | dict[str, Any] | list[Any] | None = None
    false: str | dict[str, Any] | list[Any] | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: NoulCriteria | None = None


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, str | dict[str, Any] | list[Any]]
    probabilities: dict[str, float]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: list[str | dict[str, Any] | list[Any]] = Field(..., min_length=1)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ValidationError(BaseModel):
    loc: list[str | int]
    msg: str
    type: str
    input: Any | None = None
    ctx: dict[str, Any] | None = None


class HTTPValidationError(BaseModel):
    detail: list[ValidationError] | None = None


class SystemOneRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    model: str
    questions: dict[str, NoulQuestion | ChoiceQuestion | ScoreQuestion] = Field(
        ..., min_length=1
    )


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, NoulAnswer | ScoreAnswer | ChoiceAnswer] = Field(
        ..., min_length=1
    )
    usage: Usage

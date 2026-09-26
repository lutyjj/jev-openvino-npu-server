from pydantic import model_validator

from jev_server.api.generated import SystemOneRequest as WireRequest
from jev_server.api.generated import SystemOneResponse as WireResponse


class SystemOneRequest(WireRequest):
    model: str = "jev-latest"

    @model_validator(mode="after")
    def validate_option_count(self):
        for question in self.questions.values():
            if question.type != "noul" and not 1 <= len(question.criteria) <= 255:
                raise ValueError("criteria must have 1..255 options")
        return self


class SystemOneResponse(WireResponse):
    latency_ms: float

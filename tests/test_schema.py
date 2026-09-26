from pathlib import Path

import pytest
from pydantic import ValidationError

from jev_server.api.types import SystemOneRequest
from jev_server.tools.generate_api import generate


def test_generated_models_match_pinned_specification(tmp_path):
    output = tmp_path / "generated.py"
    generate(output)
    assert output.read_text() == Path("jev_server/api/generated.py").read_text()


@pytest.mark.parametrize("state", [None, True, 42])
def test_request_rejects_states_outside_wire_contract(state):
    with pytest.raises(ValidationError):
        SystemOneRequest(state=state, questions={"q": {"type": "noul"}})


def test_noul_criteria_preserve_wire_keys():
    request = SystemOneRequest(
        state="hello",
        questions={"q": {"type": "noul", "criteria": {"true": "yes", "false": "no"}}},
    )
    assert request.model_dump()["questions"]["q"]["criteria"] == {
        "true": "yes",
        "false": "no",
    }

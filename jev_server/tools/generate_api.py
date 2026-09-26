import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def compact_schema(value):
    if isinstance(value, list):
        return [compact_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: {name: compact_schema(schema) for name, schema in item.items()}
        if key in {"properties", "schemas"}
        else compact_schema(item)
        for key, item in value.items()
        if key not in {"description", "examples", "title"}
    }


def generate(output: Path):
    schema = compact_schema(json.loads(Path("schemas/systemone.json").read_text()))
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "systemone.json"
        source.write_text(json.dumps(schema))
        subprocess.run(
            [
                "datamodel-codegen",
                "--input",
                str(source),
                "--input-file-type",
                "openapi",
                "--output",
                str(output.resolve()),
                "--output-model-type",
                "pydantic_v2.BaseModel",
                "--target-python-version",
                "3.12",
                "--collapse-root-models",
                "--use-union-operator",
                "--use-standard-collections",
                "--field-constraints",
                "--custom-file-header",
                "\n",
                "--formatters",
                "ruff-format",
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("jev_server/api/generated.py")
    )
    generate(parser.parse_args().output)

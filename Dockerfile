FROM python:3.13-slim AS dependencies
RUN pip install --no-cache-dir uv==0.12.19
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-header --no-annotate -o /tmp/runtime.txt

FROM dependencies AS export
RUN uv sync --frozen --extra export --extra test --extra codegen
COPY jev_server ./jev_server
COPY examples ./examples
COPY assets.lock.json ./
COPY models ./models
COPY schemas ./schemas
COPY LICENSE NOTICE.md ./
COPY licenses ./licenses
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/cache/huggingface
ENTRYPOINT ["python", "-m", "jev_server.exporters.kev"]

FROM openvino/ubuntu24_dev@sha256:91a9411e9c8aecdbb8a35426c9d1f1938c5680eca977da9e7b9cf519e5e0f21a AS runtime
USER root
COPY --from=dependencies /tmp/runtime.txt /tmp/runtime.txt
RUN python3 -c "from importlib.metadata import version; from pathlib import Path; assert any(line.split()[0] == 'openvino==' + version('openvino') for line in Path('/tmp/runtime.txt').read_text().splitlines() if line.strip())"
RUN pip install --no-cache-dir --require-hashes -r /tmp/runtime.txt
WORKDIR /app
COPY jev_server ./jev_server
COPY LICENSE NOTICE.md ./
COPY licenses ./licenses
USER openvino
ENTRYPOINT ["python3", "-m", "jev_server.server"]

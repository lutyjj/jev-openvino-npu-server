FROM python:3.13-slim AS export
RUN pip install --no-cache-dir uv==0.12.19
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra export --extra test
COPY jev_server ./jev_server
COPY examples ./examples
COPY assets.lock.json ./
COPY models ./models
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/cache/huggingface
ENTRYPOINT ["python", "-m", "jev_server.export"]

FROM openvino/ubuntu24_dev@sha256:91a9411e9c8aecdbb8a35426c9d1f1938c5680eca977da9e7b9cf519e5e0f21a AS runtime
USER root
RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.54.0 tokenizers==0.23.2 numpy==2.5.3
WORKDIR /app
COPY jev_server ./jev_server
USER openvino
ENTRYPOINT ["python3", "-m", "jev_server.server"]

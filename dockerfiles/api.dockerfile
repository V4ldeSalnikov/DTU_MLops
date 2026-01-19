FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY models models/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

# Expose Gradio default port
EXPOSE 7860

# Run Gradio app
ENTRYPOINT ["uv", "run", "python", "src/dtu_mlops/api.py"]

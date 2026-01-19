FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

# Set working directory for mounted volumes
WORKDIR /inference

# Default entrypoint - can be overridden
ENTRYPOINT ["uv", "run", "python", "-m", "dtu_mlops.predict"]

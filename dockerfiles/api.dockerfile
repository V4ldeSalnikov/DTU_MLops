FROM ghcr.io/astral-sh/uv:python3.12-bookworm

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

# Set environment variable for Gradio server (to see it locally)
ENV GRADIO_SERVER_NAME=0.0.0.0

# Run Gradio app
ENTRYPOINT ["uv", "run", "python", "-u", "src/dtu_mlops/api.py"]

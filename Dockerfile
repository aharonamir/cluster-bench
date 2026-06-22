# ClusterBench — single-box server image (FR-28).
#
# Ships the mock path by default (no GPU/Docker/downloads, FR-29). The real
# path (mini-swe-agent + SWE-bench Verified scoring) is opt-in at build time
# via INSTALL_EXTRA=real and needs the host Docker socket mounted at run time,
# because mini-swe-agent spawns one container per task:
#
#   docker run --rm -p 8000:8000 \
#       -v /var/run/docker.sock:/var/run/docker.sock \
#       -v "$PWD/results:/app/results" \
#       clusterbench --real --base-url http://litellm:4000/v1
#
# Mock path (CI / demo):
#
#   docker build -t clusterbench .
#   docker run --rm -p 8000:8000 clusterbench
#
# Build the real path:
#
#   docker build --build-arg INSTALL_EXTRA=real -t clusterbench:real .

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# uv knobs: copy (not symlink) into the venv so the layer is self-contained,
# and compile bytecode at build time for faster cold starts.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Dependency layer (cached unless lockfile or pyproject changes) ---------
# Copy only the manifest + lock first so the (slow) dependency install is
# cached across source edits.
COPY pyproject.toml uv.lock ./

# INSTALL_EXTRA=real adds mini-swe-agent + swebench; default is the mock path.
ARG INSTALL_EXTRA=""
# --no-install-project: install deps only here; the project itself is added in
# the next layer so source edits don't bust the dependency cache. --no-dev:
# the runtime image doesn't need pytest/playwright.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$INSTALL_EXTRA" ]; then \
        uv sync --locked --no-dev --no-install-project --extra "$INSTALL_EXTRA"; \
    else \
        uv sync --locked --no-dev --no-install-project; \
    fi

# --- Application layer ------------------------------------------------------
COPY clusterbench ./clusterbench
COPY mock_litellm.py mock_minisweagent.py run_server.py ./

# Install the project itself into the venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$INSTALL_EXTRA" ]; then \
        uv sync --locked --no-dev --extra "$INSTALL_EXTRA"; \
    else \
        uv sync --locked --no-dev; \
    fi

# Persisted RunReports land here; mount a volume to keep them across runs.
RUN mkdir -p /app/results
VOLUME ["/app/results"]

# Put the venv on PATH so the entrypoint doesn't need `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Bind 0.0.0.0 so the server is reachable from outside the container. The
# default flags point at a LiteLLM/mock on the host; override at run time.
ENTRYPOINT ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]

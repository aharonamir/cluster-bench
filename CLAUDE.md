# ClusterBench

## Spec Kit
Full spec lives in `.spec/`. Read order:
1. `.spec/README.md` — architecture + locked decisions
2. `.spec/spec.md` — what to build (FRs + ACs)
3. `.spec/plan.md` — how (architecture, modules, data shapes)
4. `.spec/tasks.md` — phased work with verification gates

## Working convention
- One phase at a time. Run the gate. Do not advance on a red gate.
- Write tests with the code.
- Isolate all LiteLLM series parsing in `clusterbench/metrics/litellm.py`
- Isolate all mini-swe-agent CLI calls in `clusterbench/miniswerunner.py`
- Pin both versions in `pyproject.toml` (uv-managed project).

## Toolchain — uv
This is a uv project. `pyproject.toml` is the source of truth; `uv.lock` is
committed.
- Install / refresh deps: `uv sync` (mock path) or `uv sync --extra real`
  (adds `mini-swe-agent` + `swebench` for real runs).
- Run anything in the venv: `uv run pytest`, `uv run python ...`,
  `uv run uvicorn mock_litellm:app`.
- Add a dep: `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (test only).
- Adding to the `real` extra: `uv add --optional real <pkg>`.

### TLS quirk on this host
uv does not pick up the host's corporate CA via `SSL_CERT_FILE` (the env points
at `toga-ai-ca.pem` only, which lacks the public CA chain PyPI needs). Until
the env is fixed globally, prefix uv commands that hit PyPI with:
```
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv sync
```
(`--native-tls` alone is not enough — the existing env var shadows it.)

## Current status
Phases 0–4 done. Phase 5 (dashboard) next.


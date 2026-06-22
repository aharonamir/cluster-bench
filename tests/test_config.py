"""Tests for clusterbench.config.ServerConfig + run_server config resolution."""
import argparse

import pytest

from clusterbench.config import ServerConfig


def _write(tmp_path, text: str):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_defaults_are_mock_oriented():
    cfg = ServerConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.base_url == "http://localhost:4000/v1"
    assert cfg.metrics_url == "http://localhost:4000/metrics"
    assert cfg.api_key == "sk-mock"
    assert cfg.real is False
    assert cfg.model == "gpt-4o-mini"


def test_load_from_yaml(tmp_path):
    p = _write(
        tmp_path,
        """
host: 0.0.0.0
port: 9001
base_url: http://litellm:4000/v1
api_key: sk-prod-123
metrics_url: http://litellm:4000/metrics
real: true
model: qwen2.5-coder
""",
    )
    cfg = ServerConfig.load(p)
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9001
    assert cfg.base_url == "http://litellm:4000/v1"
    assert cfg.api_key == "sk-prod-123"
    assert cfg.real is True
    assert cfg.model == "qwen2.5-coder"
    # Unspecified keys keep their defaults.
    assert cfg.streaming is True
    assert cfg.scrape_interval_s == 1.0


def test_empty_yaml_yields_defaults(tmp_path):
    p = _write(tmp_path, "")
    cfg = ServerConfig.load(p)
    assert cfg == ServerConfig()


def test_unknown_key_is_rejected(tmp_path):
    p = _write(tmp_path, "base_ur: http://typo\n")  # typo'd key
    with pytest.raises(ValueError, match="unknown config key"):
        ServerConfig.load(p)


def test_bad_log_level_is_rejected(tmp_path):
    p = _write(tmp_path, "log_level: verbose\n")
    with pytest.raises(ValueError, match="log_level"):
        ServerConfig.load(p)


def test_non_mapping_yaml_is_rejected(tmp_path):
    p = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        ServerConfig.load(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ServerConfig.load(tmp_path / "nope.yaml")


def test_merge_overrides_cli_wins():
    cfg = ServerConfig(base_url="http://from-file:4000/v1", port=8000)
    merged = cfg.merge_overrides({"port": 9999, "base_url": None})
    # Explicit override wins; None is ignored (keeps file value).
    assert merged.port == 9999
    assert merged.base_url == "http://from-file:4000/v1"


def test_merge_overrides_rejects_unknown_key():
    cfg = ServerConfig()
    with pytest.raises(ValueError, match="unknown override key"):
        cfg.merge_overrides({"bogus": 1})


def test_resolve_config_layers_file_then_cli(tmp_path):
    """run_server.resolve_config: file provides base, CLI flags override."""
    import run_server

    p = _write(
        tmp_path,
        "base_url: http://file:4000/v1\nport: 8000\nmodel: file-model\n",
    )
    args = argparse.Namespace(
        config=str(p),
        host=None,
        port=9090,  # CLI override
        log_level=None,
        results_dir=None,
        runner_root=None,
        base_url=None,  # not overridden -> keep file value
        metrics_url=None,
        api_key=None,
        model="cli-model",  # CLI override
        real=None,
        step_limit=None,
    )
    cfg = run_server.resolve_config(args)
    assert cfg.base_url == "http://file:4000/v1"  # from file
    assert cfg.port == 9090  # from CLI
    assert cfg.model == "cli-model"  # CLI wins over file


def test_resolve_config_no_file_uses_defaults_plus_cli():
    import run_server

    args = argparse.Namespace(
        config=None,
        host=None,
        port=None,
        log_level=None,
        results_dir=None,
        runner_root=None,
        base_url="http://cli:4000/v1",
        metrics_url=None,
        api_key=None,
        model=None,
        real=True,
        step_limit=None,
    )
    cfg = run_server.resolve_config(args)
    assert cfg.base_url == "http://cli:4000/v1"
    assert cfg.real is True
    assert cfg.port == 8000  # default


# ---------------------------------------------------------------------------
# Server run-defaults: model set once at startup, inherited by POST bodies.
# ---------------------------------------------------------------------------


def test_build_run_config_inherits_server_default_model():
    """A POST body that omits `model` inherits the server's configured default."""
    from clusterbench.web.server import StartRunBody, _build_run_config

    body = StartRunBody(mode="sweep", levels=[1, 2])  # no model/streaming set
    cfg = _build_run_config(
        "rid", body, {"model": "qwen2.5-coder", "streaming": False}
    )
    assert cfg.miniswe.model == "qwen2.5-coder"
    assert cfg.miniswe.streaming is False


def test_build_run_config_body_overrides_server_default():
    """An explicit `model` in the body wins over the server default."""
    from clusterbench.web.server import StartRunBody, _build_run_config

    body = StartRunBody(mode="sweep", levels=[1], model="explicit-model")
    cfg = _build_run_config("rid", body, {"model": "server-default"})
    assert cfg.miniswe.model == "explicit-model"


def test_build_run_config_falls_back_to_hardcoded_without_defaults():
    """No body value and no server default → the built-in default."""
    from clusterbench.web.server import StartRunBody, _build_run_config

    body = StartRunBody(mode="sweep", levels=[1])
    cfg = _build_run_config("rid", body, None)
    assert cfg.miniswe.model == "gpt-4o-mini"
    assert cfg.miniswe.streaming is True
    assert cfg.scrape_interval_s == 1.0

import os

import pytest

from mcp_server_qdrant.main import DEPRECATED_ENV_VARS, apply_deprecated_env_vars


@pytest.fixture(autouse=True)
def clean_env():
    # apply_deprecated_env_vars() writes to os.environ directly, so restore the
    # whole environment afterwards instead of relying on monkeypatch.
    saved = os.environ.copy()
    for old_name, new_name in DEPRECATED_ENV_VARS.items():
        os.environ.pop(old_name, None)
        os.environ.pop(new_name, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_old_name_is_copied_to_new_name(monkeypatch):
    monkeypatch.setenv("FASTMCP_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("FASTMCP_SERVER_PORT", "1234")

    with pytest.warns(FutureWarning) as record:
        apply_deprecated_env_vars()

    messages = [str(w.message) for w in record]
    assert any("FASTMCP_SERVER_HOST is deprecated" in m for m in messages)
    assert any("FASTMCP_SERVER_PORT is deprecated" in m for m in messages)
    assert os.environ["FASTMCP_HOST"] == "0.0.0.0"
    assert os.environ["FASTMCP_PORT"] == "1234"


def test_new_name_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("FASTMCP_SERVER_PORT", "1111")
    monkeypatch.setenv("FASTMCP_PORT", "2222")

    with pytest.warns(FutureWarning, match="FASTMCP_SERVER_PORT is deprecated"):
        apply_deprecated_env_vars()

    assert os.environ["FASTMCP_PORT"] == "2222"


def test_nothing_happens_without_old_names(monkeypatch, recwarn):
    monkeypatch.setenv("FASTMCP_PORT", "2222")

    apply_deprecated_env_vars()

    assert os.environ["FASTMCP_PORT"] == "2222"
    assert "FASTMCP_HOST" not in os.environ
    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]

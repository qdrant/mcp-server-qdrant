"""End-to-end MCP protocol smoke test.

Spawns the installed ``mcp-server-qdrant`` binary in stdio mode (the default
transport) and asserts that:

* the JSON-RPC ``initialize`` handshake completes and reports a server name,
  protocol version, and ``tools`` capability; and
* the documented tools (``qdrant-find``, ``qdrant-store``) are advertised in
  ``tools/list`` with valid object input schemas.

This catches regressions that the existing unit tests miss — broken entry
scripts, FastMCP API drift, and ``mcp.run()`` wiring issues — without booting
a real Qdrant instance: the test uses an in-memory store via
``QDRANT_LOCAL_PATH``.

Skipped automatically if ``pytest-mcp-plugin`` is not installed, so this file
is safe to land before adding the dev dependency.
"""

from __future__ import annotations

import shutil

import pytest

mcp_test = pytest.importorskip(
    "mcp_test",
    reason="install pytest-mcp-plugin (`uv add --dev pytest-mcp-plugin`) to run this test",
)


pytestmark = pytest.mark.skipif(
    shutil.which("mcp-server-qdrant") is None,
    reason="mcp-server-qdrant entry point not on PATH; run `uv sync` first",
)


@pytest.fixture
def server_env(tmp_path, monkeypatch):
    """Minimal env to run the server fully in-process, no network."""
    monkeypatch.setenv("COLLECTION_NAME", "smoke-test")
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)


def test_initialize_handshake_succeeds(server_env):
    """Server completes the MCP initialize handshake over stdio."""
    with mcp_test.MCPTestClient(
        ["mcp-server-qdrant", "--transport", "stdio"],
        startup_timeout=30.0,
    ) as client:
        assert client.server_info.get("name"), "serverInfo.name must be set"
        assert client.server_version, "server must advertise a protocolVersion"
        assert "tools" in client.server_capabilities, (
            "server must advertise the 'tools' capability"
        )


def test_documented_tools_are_listed(server_env):
    """The tools advertised in README are reachable via tools/list."""
    with mcp_test.MCPTestClient(
        ["mcp-server-qdrant", "--transport", "stdio"],
        startup_timeout=30.0,
    ) as client:
        tools = client.list_tools()

        assert tools.find("qdrant-find") is not None, (
            "qdrant-find tool must be advertised (documented in README)"
        )
        assert tools.find("qdrant-store") is not None, (
            "qdrant-store tool must be advertised (documented in README)"
        )

        for tool in tools:
            assert tool.input_schema, f"{tool.name} missing inputSchema"
            assert tool.input_schema.get("type") == "object", (
                f"{tool.name} inputSchema.type must be 'object'"
            )

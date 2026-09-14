"""tools/list caches generated tool schemas instead of rebuilding every call.

Generating a tool's JSON schema means reflecting over its type hints; doing that
for every tool on every tools/list request is wasted work when the signatures
never change. These tests pin the memoisation behaviour.
"""

import pathlib
import sys
from typing import Annotated

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _mcp_spec_support import McpServer  # noqa: E402


def _server_with_spy():
    srv = McpServer("cache-test")

    @srv.tool
    def alpha(x: Annotated[int, "x value"]) -> int:
        """Alpha."""
        return x

    @srv.tool
    def beta(s: Annotated[str, "s value"]) -> str:
        """Beta."""
        return s

    calls = {"n": 0}
    original = srv._generate_tool_schema

    def spy(name, func):
        calls["n"] += 1
        return original(name, func)

    srv._generate_tool_schema = spy  # type: ignore[method-assign]
    return srv, calls


def test_schema_generated_once_per_tool_across_calls():
    srv, calls = _server_with_spy()
    first = srv._mcp_tools_list()
    second = srv._mcp_tools_list()
    # Two tools: generated twice total, not once per tool per call (would be 4).
    assert calls["n"] == 2
    assert first == second
    assert sorted(t["name"] for t in first["tools"]) == ["alpha", "beta"]


def test_cache_reuses_the_same_schema_objects():
    srv, _ = _server_with_spy()
    a = srv._mcp_tools_list()["tools"]
    b = srv._mcp_tools_list()["tools"]
    assert all(x is y for x, y in zip(a, b))


def test_new_tool_is_picked_up_after_warming_cache():
    srv, calls = _server_with_spy()
    srv._mcp_tools_list()  # warm (n == 2)

    @srv.tool
    def gamma(y: Annotated[int, "y value"]) -> int:
        """Gamma."""
        return y

    result = srv._mcp_tools_list()
    assert sorted(t["name"] for t in result["tools"]) == ["alpha", "beta", "gamma"]
    assert calls["n"] == 3  # only gamma was generated on the third pass


def test_reregistered_function_refreshes_its_schema():
    srv, calls = _server_with_spy()
    srv._mcp_tools_list()  # warm (n == 2)

    def alpha(x: Annotated[str, "now a string"]) -> str:
        """Alpha version two."""
        return x

    srv.tools.methods["alpha"] = alpha  # same name, different function object
    result = srv._mcp_tools_list()
    assert calls["n"] == 3  # alpha regenerated (identity changed), beta cached
    alpha_schema = next(t for t in result["tools"] if t["name"] == "alpha")
    assert "version two" in alpha_schema["description"]

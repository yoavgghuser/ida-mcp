import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from _mcp_spec_support import load_ida_rpc_module


@pytest.fixture
def rpc(monkeypatch):
    module = load_ida_rpc_module()
    monkeypatch.setattr(module, "_output_cache", {})
    monkeypatch.setattr(module, "_output_text_cache", {})
    monkeypatch.setattr(module, "OUTPUT_CACHE_MAX_SIZE", 100)
    return module


def test_chunks_reconstruct_large_unicode_output_without_retruncation(rpc):
    expected = {"code": '\n"\\\u2603' * 20000, "items": list(range(100))}
    rpc._cache_output("large", expected)
    offset = 0
    chunks = []
    while offset is not None:
        page = rpc.output_read("large", offset)
        assert len(json.dumps(page)) < rpc.OUTPUT_LIMIT_MAX_CHARS
        chunks.append(page["text"])
        offset = page["next_offset"]
    assert json.loads("".join(chunks)) == expected
    assert rpc.output_read("large", 10**9)["text"] == ""


def test_eviction_removes_both_cache_entries(rpc):
    rpc.OUTPUT_CACHE_MAX_SIZE = 2
    for i in range(3):
        rpc._cache_output(str(i), {"value": i})
    assert rpc.get_cached_output("0") is None
    with pytest.raises(rpc.McpToolError, match="expired"):
        rpc.output_read("0")
    assert set(rpc._output_cache) == set(rpc._output_text_cache) == {"1", "2"}


def test_concurrent_writes_keep_cache_bounded_and_consistent(rpc):
    rpc.OUTPUT_CACHE_MAX_SIZE = 4
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: rpc._cache_output(str(i), {"i": i}), range(100)))
    assert len(rpc._output_cache) == 4
    assert set(rpc._output_cache) == set(rpc._output_text_cache)
    for key, value in rpc._output_cache.items():
        assert json.loads(rpc.output_read(key)["text"]) == value


@pytest.mark.parametrize("kwargs", [{"offset": -1}, {"count": 0}, {"count": 4001}])
def test_invalid_page(rpc, kwargs):
    with pytest.raises(rpc.McpToolError):
        rpc.output_read("absent", **kwargs)


def test_registered_tool_works_through_real_truncation_middleware(rpc):
    rpc._cache_output("example", {"hello": "world"})
    response = rpc.MCP_SERVER.registry.methods["tools/call"](
        "output_read", {"output_id": "example"}
    )
    assert not response.get("isError")
    assert json.loads(response["structuredContent"]["text"]) == {"hello": "world"}

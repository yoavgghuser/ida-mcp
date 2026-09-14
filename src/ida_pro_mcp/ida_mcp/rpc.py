import json
import os
import threading
from typing import Annotated, Any, Optional, TypedDict
from .zeromcp import (
    McpRpcRegistry,
    McpServer,
    McpToolError,
    McpHttpRequestHandler,
    get_current_request_external_base_url,
)

MCP_UNSAFE: set[str] = set()
MCP_EXTENSIONS: dict[str, set[str]] = {}  # group -> set of function names
MCP_SERVER = McpServer("ida-pro-mcp", extensions=MCP_EXTENSIONS)

# ============================================================================
# Output Size Limiting
# ============================================================================

OUTPUT_LIMIT_MAX_CHARS = 50000
OUTPUT_CACHE_MAX_SIZE = 100
_output_cache: dict[str, Any] = {}
_output_text_cache: dict[str, str] = {}
_output_cache_lock = threading.Lock()
_download_base_url: str = os.environ.get("IDA_MCP_URL", "http://127.0.0.1:13337")


def set_download_base_url(url: str) -> None:
    global _download_base_url
    _download_base_url = url.rstrip("/")


def get_download_base_url() -> str:
    return get_current_request_external_base_url() or _download_base_url


def get_current_transport_session_id() -> str | None:
    return MCP_SERVER.get_current_transport_session_id()


def _generate_output_id() -> str:
    import uuid

    return str(uuid.uuid4())


OUTPUT_LIMIT_PREVIEW_ITEMS = 10
OUTPUT_LIMIT_PREVIEW_STR_LEN = 1000


def _truncate_value(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return value

    if isinstance(value, str) and len(value) > OUTPUT_LIMIT_PREVIEW_STR_LEN:
        return value[:OUTPUT_LIMIT_PREVIEW_STR_LEN] + f"... [{len(value)} chars total]"

    if isinstance(value, list):
        # IMPORTANT: Do not inject sentinel objects like {"_truncated": "..."} into lists.
        # Many tool schemas constrain list item shapes (additionalProperties: false),
        # so sentinels can break structured output validation. Truncation is reported
        # via _meta.ida_mcp and the download_hint content.
        return [
            _truncate_value(item, depth + 1)
            for item in value[:OUTPUT_LIMIT_PREVIEW_ITEMS]
        ]

    if isinstance(value, dict):
        return {k: _truncate_value(v, depth + 1) for k, v in value.items()}

    return value


def _build_download_meta(output_id: str, total_chars: int) -> dict:
    download_url = f"{get_download_base_url()}/output/{output_id}.json"
    return {
        "output_truncated": True,
        "total_chars": total_chars,
        "output_id": output_id,
        "download_url": download_url,
        "download_hint": (
            f'Output truncated. Read chunks with output_read(output_id="{output_id}") '
            f"and follow next_offset, or run: curl -o .ida-mcp/{output_id}.json {download_url}"
        ),
    }


def get_cached_output(output_id: str) -> Optional[Any]:
    with _output_cache_lock:
        return _output_cache.get(output_id)


def _cache_output(output_id: str, data: Any, serialized: str | None = None) -> None:
    text = serialized if serialized is not None else json.dumps(data)
    with _output_cache_lock:
        if output_id not in _output_cache and len(_output_cache) >= OUTPUT_CACHE_MAX_SIZE:
            oldest_key = next(iter(_output_cache))
            del _output_cache[oldest_key]
            _output_text_cache.pop(oldest_key, None)
        _output_cache[output_id] = data
        _output_text_cache[output_id] = text


def _install_tools_call_patch() -> None:
    original = MCP_SERVER.registry.methods["tools/call"]

    def patched(
        name: str, arguments: Optional[dict] = None, _meta: Optional[dict] = None
    ) -> dict:
        response = original(name, arguments, _meta)

        if response.get("isError"):
            return response

        structured = response.get("structuredContent")
        if structured is None:
            return response

        serialized = json.dumps(structured)
        if len(serialized) <= OUTPUT_LIMIT_MAX_CHARS:
            return response

        output_id = _generate_output_id()
        _cache_output(output_id, structured, serialized)

        preview = _truncate_value(structured)
        download_meta = _build_download_meta(output_id, len(serialized))

        content = [{
            "type": "text",
            "text": json.dumps(preview, separators=(",", ":")),
        }, {
            "type": "text",
            "text": download_meta["download_hint"],
        }]

        return {
            "structuredContent": preview,
            "content": content,
            "isError": False,
            "_meta": {"ida_mcp": download_meta},
        }

    MCP_SERVER.registry.methods["tools/call"] = patched


# Install the output limiting patch
_install_tools_call_patch()


# ============================================================================
# Decorators
# ============================================================================


def tool(func):
    return MCP_SERVER.tool(func)


class OutputReadResult(TypedDict):
    output_id: str
    text: str
    offset: int
    next_offset: int | None
    total_chars: int


@tool
def output_read(
    output_id: Annotated[str, "output_id from a truncated tool response's _meta.ida_mcp"],
    offset: Annotated[int, "Character offset; resume with next_offset"] = 0,
    count: Annotated[int, "JSON characters to return (1-4000)"] = 4000,
) -> OutputReadResult:
    """Read cached full output in chunks without rerunning IDA or downloading a URL.

    Concatenate text chunks, then parse the combined JSON. Offsets are Unicode
    character positions, not byte positions. Cache entries expire on eviction
    or server restart; use the same IDA instance that produced the output.
    """
    if offset < 0:
        raise McpToolError("offset must be non-negative")
    if not 1 <= count <= 4000:
        raise McpToolError("count must be between 1 and 4000")
    with _output_cache_lock:
        serialized = _output_text_cache.get(output_id)
    if serialized is None:
        raise McpToolError("Output not found or expired; rerun the original tool")
    end = min(offset + count, len(serialized))
    return {
        "output_id": output_id,
        "text": serialized[offset:end],
        "offset": offset,
        "next_offset": end if end < len(serialized) else None,
        "total_chars": len(serialized),
    }


def resource(uri):
    return MCP_SERVER.resource(uri)


def unsafe(func):
    MCP_UNSAFE.add(func.__name__)
    return func


def ext(group: str):
    """Mark a tool as belonging to an extension group.

    Tools in extension groups are hidden by default. Enable via ?ext=group query param.
    Example: @ext("dbg") marks debugger tools that require ?ext=dbg to be visible.
    """

    def decorator(func):
        if group not in MCP_EXTENSIONS:
            MCP_EXTENSIONS[group] = set()
        MCP_EXTENSIONS[group].add(func.__name__)
        return func

    return decorator


__all__ = [
    "McpRpcRegistry",
    "McpServer",
    "McpToolError",
    "McpHttpRequestHandler",
    "MCP_SERVER",
    "MCP_UNSAFE",
    "MCP_EXTENSIONS",
    "tool",
    "unsafe",
    "ext",
    "resource",
    "get_cached_output",
    "set_download_base_url",
    "get_download_base_url",
    "get_current_transport_session_id",
]

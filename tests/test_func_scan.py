"""Exercise the real scan body with an indexed fake IDB; no IDA license needed."""

import ast
import bisect
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any
import time

import pytest


@pytest.fixture
def scan():
    path = Path(__file__).resolve().parents[1] / "src/ida_pro_mcp/ida_mcp/api_core.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "func_scan")
    function.decorator_list = []
    addresses = range(16, 1600016, 16)  # 100,000 functions
    calls = []

    def get_func(addr):
        if addr in addresses:
            return SimpleNamespace(start_ea=addr, end_ea=addr + 8)
        return None

    def get_next(addr):
        calls.append(addr)
        index = bisect.bisect_right(addresses, addr)
        return get_func(addresses[index]) if index < len(addresses) else None

    env = dict(
        Annotated=Annotated, Any=Any, IDAError=ValueError,
        parse_address=lambda value: int(value, 0),
        get_tool_deadline=lambda: None, time=time,
        idaapi=SimpleNamespace(BADADDR=2**64 - 1),
        ida_funcs=SimpleNamespace(get_func=get_func, get_next_func=get_next,
                                  get_func_name=lambda addr: f"fn_{addr}"),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), env)
    return env["func_scan"], calls, env


def test_pages_resume_without_duplicates_or_prefix_rescan(scan):
    run, calls, _ = scan
    first = run(count=2)
    calls.clear()
    second = run(start=first["next_addr"], count=2)
    assert [r["addr"] for r in first["data"] + second["data"]] == ["0x10", "0x20", "0x30", "0x40"]
    assert len(calls) == 2
    assert second["stop_reason"] == "count"


def test_sparse_filter_returns_resumable_empty_page(scan):
    run, calls, _ = scan
    page = run(name_contains="missing", scan_limit=3)
    assert page == dict(data=[], next_addr="0x40", scanned=3, stop_reason="scan_limit")
    assert len(calls) == 4


def test_last_page_and_end(scan):
    run, _, _ = scan
    last = run(start=hex(1600000))
    assert len(last["data"]) == 1
    assert last["next_addr"] is None
    assert run(start=hex(1600001))["data"] == []


def test_literal_case_insensitive_filter_and_size(scan):
    run, _, _ = scan
    assert run(name_contains="FN_16", count=1)["data"][0]["addr"] == "0x10"
    assert run(min_size=9, scan_limit=2)["data"] == []
    assert run(name_contains=".*", scan_limit=2)["data"] == []


def test_deadline_preserves_first_unprocessed_address(scan):
    run, _, env = scan
    env["get_tool_deadline"] = lambda: time.monotonic() - 1
    assert run() == dict(data=[], next_addr="0x10", scanned=0, stop_reason="deadline")


@pytest.mark.parametrize("kwargs", [dict(count=0), dict(count=1001), dict(scan_limit=0),
                                    dict(scan_limit=100001), dict(min_size=-1),
                                    dict(start="-1"), dict(start=hex(2**64 - 1))])
def test_invalid_arguments(scan, kwargs):
    with pytest.raises(ValueError):
        scan[0](**kwargs)

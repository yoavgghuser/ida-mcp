"""Run production function bodies with fake decompiler dependencies."""
import ast
from pathlib import Path
from typing import Annotated
import time

import pytest


class CancelledError(Exception):
    pass


class IDASyncError(Exception):
    pass


@pytest.fixture
def api():
    path = Path(__file__).resolve().parents[1] / "src/ida_pro_mcp/ida_mcp/api_analysis.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {"_decompile_one", "decompile", "decompile_batch"}]
    for node in nodes:
        node.decorator_list = []
    calls = []

    def decompiler(addr, include_addresses):
        calls.append((addr, include_addresses))
        if addr == 2:
            return None, "Unavailable decompiler"
        return f"code {addr}", None

    env = dict(Annotated=Annotated, DecompileResult=dict, DecompileBatchResult=dict,
               IDAError=ValueError, IDASyncError=IDASyncError, CancelledError=CancelledError,
               time=time, get_tool_deadline=lambda: None,
               parse_address=lambda addr: int(addr, 0), decompile_function_safe=decompiler)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env, calls


def test_batch_resume_errors_and_duplicate_targets(api):
    env, calls = api
    run = env["decompile_batch"]
    targets = ["1", "bad", "2", "1"]
    first = run(targets, count=2)
    second = run(targets, offset=first["next_offset"], count=2)
    assert first["data"][0]["code"] == "code 1"
    assert first["data"][1]["error"]
    assert second["data"][0]["error"] == "Unavailable decompiler"
    assert second["data"][1]["code"] == "code 1"
    assert second["next_offset"] is None
    assert calls == [(1, False), (2, False), (1, False)]


def test_deadline_returns_completed_work_and_resume_index(api):
    env, _ = api
    env["get_tool_deadline"] = lambda: 100
    ticks = iter([0, 100])
    env["time"] = type("Clock", (), {"monotonic": staticmethod(lambda: next(ticks))})
    page = env["decompile_batch"](["1", "3"])
    assert len(page["data"]) == 1
    assert page["next_offset"] == 1
    assert page["stop_reason"] == "deadline"


def test_empty_and_end_page(api):
    run = api[0]["decompile_batch"]
    assert run([]) == dict(data=[], next_offset=None, total=0, stop_reason="complete")
    assert run(["1"], offset=1)["data"] == []


@pytest.mark.parametrize("error", [CancelledError, IDASyncError])
def test_control_exceptions_are_not_swallowed(api, error):
    env, _ = api
    def fail(*args, **kwargs):
        raise error("stopped")
    env["decompile_function_safe"] = fail
    with pytest.raises(error):
        env["decompile_batch"](["1", "3"])


@pytest.mark.parametrize("kwargs", [dict(offset=-1), dict(offset=2), dict(count=0), dict(count=21)])
def test_invalid_page(api, kwargs):
    with pytest.raises(ValueError):
        api[0]["decompile_batch"](["1"], **kwargs)

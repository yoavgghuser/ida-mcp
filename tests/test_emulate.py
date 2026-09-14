"""Exercise the real emulation core with synthetic shellcode; no IDA needed.

`_emulate_core` takes the Unicorn module and a plain [(addr, bytes)] segment
list, so we load just that function out of api_emulate.py (dropping the relative
imports and the @tool wrapper that need IDA) and run real x86-64 code through it.
"""

import ast
from pathlib import Path

import pytest

unicorn = pytest.importorskip("unicorn")

BASE = 0x400000


def _load_core():
    path = Path(__file__).resolve().parents[1] / "src/ida_pro_mcp/ida_mcp/api_emulate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kept = []
    for node in tree.body:
        # Drop relative imports (.rpc/.sync/.utils) that pull in IDA.
        if isinstance(node, ast.ImportFrom) and node.level:
            continue
        # Drop the @tool wrapper; we test the pure core.
        if isinstance(node, ast.FunctionDef) and node.name == "emulate":
            continue
        kept.append(node)
    module = ast.Module(body=kept, type_ignores=[])
    namespace: dict = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_emulate_core"]


core = _load_core()


def test_add_and_return():
    # mov rax, rdi ; add rax, rsi ; ret
    code = bytes.fromhex("4889f8" "4801f0" "c3")
    r = core(unicorn, [(BASE, code)], start=BASE, regs={"rdi": 2, "rsi": 3})
    assert r["stop_reason"] == "function_returned"
    assert int(r["regs"]["rax"], 16) == 5
    assert r["instructions"] == 3
    assert r["calls"] == []


def test_external_call_is_skipped():
    # call rel32 -> 0x0 (unmapped) ; ret
    # rel32 = 0 - (BASE+5) = 0xffbffffb (LE: fb ff bf ff)
    code = bytes.fromhex("e8fbffbfff" "c3")
    r = core(unicorn, [(BASE, code)], start=BASE, call_return=0x1234)
    assert r["stop_reason"] == "function_returned"
    assert r["calls"] == ["0x0"]
    assert int(r["regs"]["rax"], 16) == 0x1234


def test_store_through_pointer_and_dump():
    # mov [rdi], rsi ; ret   (rdi points into an unmapped page -> auto-mapped)
    code = bytes.fromhex("488937" "c3")
    r = core(
        unicorn,
        [(BASE, code)],
        start=BASE,
        regs={"rdi": 0x150000, "rsi": 0xDEADBEEF},
        reads=[{"addr": "@rdi", "size": 8}],
    )
    assert r["stop_reason"] == "function_returned"
    assert "0x150000" in r["auto_mapped"]
    dump = r["reads"][0]
    assert dump["addr"] == "0x150000"
    raw = bytes(int(b, 16) for b in dump["hex"].split())
    assert int.from_bytes(raw, "little") == 0xDEADBEEF


def test_write_param_sets_memory():
    # mov al, [rdi] ; ret   -- read back a byte we staged via `write`
    code = bytes.fromhex("8a07" "c3")
    r = core(
        unicorn,
        [(BASE, code)],
        start=BASE,
        regs={"rdi": 0x160000},
        writes=[{"addr": "0x160000", "data": "7f"}],
    )
    assert r["stop_reason"] == "function_returned"
    assert int(r["regs"]["rax"], 16) & 0xFF == 0x7F


def test_max_instructions_cap():
    # jmp $  (infinite loop)
    code = bytes.fromhex("ebfe")
    r = core(unicorn, [(BASE, code)], start=BASE, max_instructions=50)
    assert r["stop_reason"] == "max_instructions"
    assert r["instructions"] >= 50


def test_end_address_stops_run():
    # nop ; nop ; nop ; ret  -- stop at the 3rd nop, before ret
    code = bytes.fromhex("90" "90" "90" "c3")
    r = core(unicorn, [(BASE, code)], start=BASE, end_ea=BASE + 2)
    assert r["stop_reason"] == "end_reached"
    assert int(r["regs"]["rip"], 16) == BASE + 2
    assert r["instructions"] == 2

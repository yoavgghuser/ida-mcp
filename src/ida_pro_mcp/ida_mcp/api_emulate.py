"""CPU emulation (x86-64) for IDA Pro MCP.

Runs a range of code from the current IDB inside the Unicorn engine without
launching the target process. Every IDB segment is mapped at its real address
with its real bytes, so intra-binary calls, jump tables and data references
resolve naturally. Two design choices make it work on "anything":

- Calls into unmapped memory (unresolved imports, indirect thunks) are trapped,
  logged, and returned from with a configurable RAX -- so a self-contained
  routine that happens to call a helper API keeps going instead of crashing.
- Stray reads/writes to unmapped pages are satisfied by lazily mapping zero
  pages on demand, so uninitialised scratch buffers don't abort the run.

This lets an agent *execute* the hard-to-read routines -- string/config
decryptors, hashing stubs, checksum/opaque-predicate blocks -- and read the
result straight out of registers or memory, instead of reasoning about carry-bit
arithmetic in the decompiler by hand.

The heavy lifting lives in `_emulate_core`, which takes the Unicorn module and a
plain list of (address, bytes) segments so it can be unit-tested with synthetic
shellcode and no IDA present. The `emulate` tool is a thin wrapper that gathers
segments from the live IDB and calls it.
"""

from typing import Annotated, Any, NotRequired, TypedDict

from .rpc import tool
from .sync import idasync
from .utils import parse_address, read_bytes_bss_safe

_PAGE = 0x1000
_MASK64 = (1 << 64) - 1
_CANON_MAX = 0x0000_8000_0000_0000  # first non-canonical address on x86-64


class MemDump(TypedDict):
    addr: str
    size: int
    hex: str
    ascii: str
    error: NotRequired[str]


class EmulateResult(TypedDict):
    start: str
    stop_reason: str
    instructions: int
    regs: dict[str, str]
    calls: list[str]
    reads: list[MemDump]
    auto_mapped: list[str]
    trace: NotRequired[list[str]]
    error: NotRequired[str]


# Registers reported back (and settable via `regs`).
_GP = [
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
]
_OUT_REGS = _GP + ["rip", "eflags"]


def _emulate_core(
    uc_mod: Any,
    segments: list[tuple[int, bytes]],
    *,
    start: int,
    end_ea: int | None = None,
    regs: dict[str, Any] | None = None,
    writes: list[dict[str, Any]] | None = None,
    reads: list[dict[str, Any]] | None = None,
    stack_base: int = 0,
    stack_size: int = 0x100000,
    max_instructions: int = 200000,
    timeout_ms: int = 5000,
    call_return: int = 0,
    trace_limit: int = 0,
    auto_map: bool = True,
) -> EmulateResult:
    """Emulate x86-64 code. Pure w.r.t. IDA: `segments` is [(addr, bytes)]."""
    import time

    Uc = uc_mod.Uc
    UcError = uc_mod.UcError
    xc = uc_mod.x86_const

    def _reg(name: str) -> int:
        return getattr(xc, "UC_X86_REG_" + name.upper())

    REG_RSP = _reg("rsp")
    REG_RIP = _reg("rip")
    REG_RAX = _reg("rax")
    REG_EFLAGS = _reg("eflags")

    def _align_down(a: int) -> int:
        return a & ~(_PAGE - 1)

    def _align_up(a: int) -> int:
        return (a + _PAGE - 1) & ~(_PAGE - 1)

    def _to_int(v: Any) -> int:
        if isinstance(v, int):
            return v
        return int(str(v).strip(), 0)

    def _parse_hex(data: Any) -> bytes:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, list):
            return bytes(int(x) & 0xFF for x in data)
        text = str(data).strip().replace("0x", "").replace(",", " ")
        text = "".join(text.split())
        if len(text) % 2:
            raise ValueError(f"hex string has odd length: {data!r}")
        return bytes.fromhex(text)

    uc = Uc(uc_mod.UC_ARCH_X86, uc_mod.UC_MODE_64)

    mapped: list[tuple[int, int]] = []

    def _is_mapped(page: int) -> bool:
        return any(s <= page < e for s, e in mapped)

    def _map_range(addr: int, size: int) -> None:
        """Map every page in [addr, addr+size) that isn't mapped yet."""
        p = _align_down(addr)
        end = _align_up(addr + size)
        while p < end:
            if not _is_mapped(p):
                run_end = p
                while run_end < end and not _is_mapped(run_end):
                    run_end += _PAGE
                uc.mem_map(p, run_end - p)
                mapped.append((p, run_end))
                p = run_end
            else:
                p += _PAGE

    def _find_free(size: int, hint: int) -> int:
        cand = _align_up(hint) or _PAGE
        size = _align_up(size)
        while cand + size < _CANON_MAX:
            clash = next(((s, e) for s, e in mapped if s < cand + size and cand < e), None)
            if clash is None:
                return cand
            cand = _align_up(clash[1])
        raise RuntimeError("no free address space for stack/sentinel")

    # 1. Map all program segments with their real bytes.
    for seg_addr, data in segments:
        if not data:
            continue
        _map_range(seg_addr, len(data))
        uc.mem_write(seg_addr, data)

    # 2. Stack + a sentinel return address that cleanly ends the run.
    if stack_base:
        _map_range(stack_base, stack_size)
    else:
        stack_base = _find_free(stack_size, 0x7000_0000)
        _map_range(stack_base, stack_size)
    sentinel = _find_free(_PAGE, 0x5555_0000)  # intentionally left unmapped

    rsp = (stack_base + stack_size - _PAGE) & ~0xF
    rsp -= 8  # push sentinel as the return address
    uc.mem_write(rsp, sentinel.to_bytes(8, "little"))
    uc.reg_write(REG_RSP, rsp)
    uc.reg_write(REG_EFLAGS, 0x202)

    # 3. Caller-supplied register + memory state.
    for name, value in (regs or {}).items():
        try:
            uc.reg_write(_reg(name), _to_int(value) & _MASK64)
        except AttributeError:
            raise ValueError(f"unknown register: {name!r}")
    for item in writes or []:
        addr = _to_int(item["addr"])
        blob = _parse_hex(item.get("data", ""))
        if blob:
            _map_range(addr, len(blob))
            uc.mem_write(addr, blob)

    state: dict[str, Any] = {
        "count": 0,
        "reason": None,
        "calls": [],
        "trace": [],
        "auto_mapped": [],
        "error": None,
    }
    auto_map_budget = 4096

    def _hook_code(uc_, address, size, _user):
        state["count"] += 1
        if trace_limit and len(state["trace"]) < trace_limit:
            state["trace"].append(address)
        if max_instructions and state["count"] > max_instructions:
            state["reason"] = "max_instructions"
            uc_.emu_stop()

    def _hook_fetch(uc_, _access, address, _size, _value, _user):
        nonlocal auto_map_budget
        # Landing on the sentinel means the top-level routine returned.
        if address == sentinel:
            state["reason"] = "function_returned"
            uc_.emu_stop()
            return False
        # Otherwise: a call/jump into unmapped memory (unresolved import/thunk).
        # Map a page of `ret` (0xC3) at the target so the CPU returns to the
        # caller naturally -- redirecting RIP from a fetch-unmapped hook is not
        # honoured by Unicorn, but mapping the faulting page (return True) is.
        if auto_map_budget <= 0:
            return False
        base = _align_down(address)
        try:
            _map_range(base, _PAGE)
            uc_.mem_write(base, b"\xc3" * _PAGE)
        except UcError:
            state["reason"] = "bad_call_target"
            uc_.emu_stop()
            return False
        auto_map_budget -= 1
        state["calls"].append(address)
        uc_.reg_write(REG_RAX, call_return & _MASK64)
        return True

    def _hook_mem(uc_, _access, address, size, _value, _user):
        nonlocal auto_map_budget
        if not auto_map or auto_map_budget <= 0:
            return False
        base = _align_down(address)
        span = _align_up(address + max(size, 1)) - base
        try:
            _map_range(base, span)
            state["auto_mapped"].append(hex(base))
            auto_map_budget -= span // _PAGE
        except UcError:
            return False
        return True

    uc.hook_add(uc_mod.UC_HOOK_CODE, _hook_code)
    uc.hook_add(uc_mod.UC_HOOK_MEM_FETCH_UNMAPPED, _hook_fetch)
    uc.hook_add(
        uc_mod.UC_HOOK_MEM_READ_UNMAPPED | uc_mod.UC_HOOK_MEM_WRITE_UNMAPPED,
        _hook_mem,
    )

    until = end_ea if end_ea is not None else sentinel
    started = time.monotonic()
    try:
        uc.emu_start(
            start,
            until,
            timeout=timeout_ms * 1000,
            count=(max_instructions + 16) if max_instructions else 0,
        )
    except UcError as exc:
        if state["reason"] is None:
            state["reason"] = "exception"
            state["error"] = f"{exc} @ {hex(uc.reg_read(REG_RIP))}"
    elapsed_ms = (time.monotonic() - started) * 1000

    pc = uc.reg_read(REG_RIP)
    if state["reason"] is None:
        if end_ea is not None and pc == end_ea:
            state["reason"] = "end_reached"
        elif pc == sentinel:
            state["reason"] = "function_returned"
        elif max_instructions and state["count"] >= max_instructions:
            state["reason"] = "max_instructions"
        elif elapsed_ms >= timeout_ms:
            state["reason"] = "timeout"
        else:
            state["reason"] = "stopped"

    final = {name: uc.reg_read(_reg(name)) for name in _OUT_REGS}
    out_regs = {name: hex(final[name]) for name in _OUT_REGS}

    # 4. Post-run memory dumps.
    dumps: list[MemDump] = []
    for item in reads or []:
        raw_addr = item.get("addr", "")
        size = int(item.get("size", 0) or 0)
        entry: MemDump = {"addr": str(raw_addr), "size": size, "hex": "", "ascii": ""}
        try:
            if isinstance(raw_addr, str) and raw_addr.startswith("@"):
                ea = final[raw_addr[1:].lower()]
            else:
                ea = _to_int(raw_addr)
            data = bytes(uc.mem_read(ea, size)) if size else b""
            entry["addr"] = hex(ea)
            entry["hex"] = " ".join(f"{b:02x}" for b in data)
            entry["ascii"] = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
        except Exception as exc:  # noqa: BLE001 - report per-item, keep going
            entry["error"] = str(exc)
        dumps.append(entry)

    result: EmulateResult = {
        "start": hex(start),
        "stop_reason": state["reason"],
        "instructions": state["count"],
        "regs": out_regs,
        "calls": [hex(c) for c in state["calls"]],
        "reads": dumps,
        "auto_mapped": state["auto_mapped"],
    }
    if trace_limit:
        result["trace"] = [hex(a) for a in state["trace"]]
    if state["error"]:
        result["error"] = state["error"]
    return result


@tool
@idasync
def emulate(
    start: Annotated[str, "Address or name to start execution at"],
    end: Annotated[
        str,
        "Address/name to stop at. Omit to run until the starting routine returns "
        "(or a limit is hit).",
    ] = "",
    regs: Annotated[
        dict | None,
        "Initial register values, e.g. {'rdi':'0x140001000','rsi':16}",
    ] = None,
    write: Annotated[
        list | None,
        "Memory to set before running: [{'addr':'0x..','data':'aa bb cc'}]. "
        "data is a hex string (spaces/0x optional) or a byte list.",
    ] = None,
    read: Annotated[
        list | None,
        "Memory to dump after running: [{'addr':'0x..'|name|'@rax','size':N}]. "
        "Prefix an address with '@' to use a final register value as the pointer.",
    ] = None,
    stack_size: Annotated[int, "Emulated stack size in bytes (default 0x100000)"] = 0x100000,
    max_instructions: Annotated[int, "Instruction cap (default 200000; 0=unlimited)"] = 200000,
    timeout_ms: Annotated[int, "Wall-clock timeout in ms (default 5000)"] = 5000,
    call_return: Annotated[
        int, "RAX value substituted when an unresolved external call is skipped (default 0)"
    ] = 0,
    trace_limit: Annotated[
        int, "Record up to N executed instruction addresses (default 0 = none)"
    ] = 0,
) -> EmulateResult:
    """Emulate x86-64 code from the IDB with Unicorn.

    Maps every segment at its real address, so intra-binary calls and data refs
    just work; unresolved external calls are skipped (logged in `calls`) and
    stray memory accesses lazily map zero pages, so self-contained routines
    (decryptors, hashers, checksums) run to completion. Read the answer out of
    the returned registers or `read` memory dumps.
    """
    try:
        import unicorn
    except ImportError:
        import sys

        return {  # type: ignore[return-value]
            "start": str(start),
            "stop_reason": "error",
            "instructions": 0,
            "regs": {},
            "calls": [],
            "reads": [],
            "auto_mapped": [],
            "error": (
                "The 'unicorn' package is not installed in IDA's Python. Install it with: "
                f"\"{sys.executable}\" -m pip install unicorn"
            ),
        }

    import idautils
    import idc

    try:
        import ida_ida

        is_64 = ida_ida.inf_is_64bit()
    except Exception:
        try:
            is_64 = idc.__EA64__  # type: ignore[attr-defined]
        except Exception:
            is_64 = True
    if not is_64:
        return {  # type: ignore[return-value]
            "start": str(start),
            "stop_reason": "error",
            "instructions": 0,
            "regs": {},
            "calls": [],
            "reads": [],
            "auto_mapped": [],
            "error": "emulate currently supports 64-bit (x86-64) databases only.",
        }

    segments: list[tuple[int, bytes]] = []
    for seg_ea in idautils.Segments():
        seg_end = idc.get_segm_end(seg_ea)
        size = seg_end - seg_ea
        if size <= 0:
            continue
        segments.append((seg_ea, read_bytes_bss_safe(seg_ea, size)))

    start_ea = parse_address(start)
    end_ea = parse_address(end) if end else None

    return _emulate_core(
        unicorn,
        segments,
        start=start_ea,
        end_ea=end_ea,
        regs=regs,
        writes=write,
        reads=read,
        stack_size=stack_size,
        max_instructions=max_instructions,
        timeout_ms=timeout_ms,
        call_return=call_return,
        trace_limit=trace_limit,
    )

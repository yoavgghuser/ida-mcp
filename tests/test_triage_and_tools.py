"""Unit tests for the pure cores of list_strings, find_crypto,
detect_capabilities and export_patched_binary -- extracted from their modules
so they run without IDA present.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "ida_pro_mcp" / "ida_mcp"

_SAFE_IMPORT_ROOTS = {"idaapi", "idc", "idautils"}


def _load_names(rel_path: str, names: set[str]) -> dict:
    """Exec only the named top-level defs/consts of a module, dropping IDA imports."""
    path = SRC / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(
                not (a.name.split(".")[0] in _SAFE_IMPORT_ROOTS or a.name.startswith("ida_"))
                for a in node.names
            ):
                kept.append(node)
            continue
        if isinstance(node, ast.ImportFrom):
            if not node.level:  # keep stdlib (typing, re, ...), drop relative
                kept.append(node)
            continue
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                kept.append(node)
            continue
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in names:
                kept.append(node)
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            kept.append(node)
    ns: dict = {}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), "exec"), ns)
    return ns


# --------------------------------------------------------------------------- #
# list_strings: _filter_strings
# --------------------------------------------------------------------------- #

_filter_strings = _load_names("api_core.py", {"_filter_strings"})["_filter_strings"]


def _mk(text, seg=".rdata"):
    return {"addr": 0x1000 + hash(text) % 0x1000, "text": text, "seg": seg}


def test_filter_min_length_and_regex():
    items = [_mk("hi"), _mk("password"), _mk("GetProcAddress"), _mk("aaa")]
    r = _filter_strings(items, pattern="pass|Proc", min_length=4)
    texts = {i["text"] for i in r["items"]}
    assert texts == {"password", "GetProcAddress"}
    assert r["total"] == 2


def test_filter_segment_and_pagination():
    items = [_mk(f"str{i}", seg=".rdata") for i in range(5)] + [_mk("other", seg=".data")]
    r = _filter_strings(items, segment=".rdata", offset=0, count=2)
    assert r["total"] == 5
    assert len(r["items"]) == 2
    assert r["next_offset"] == 2
    last = _filter_strings(items, segment=".rdata", offset=4, count=2)
    assert last["next_offset"] is None


# --------------------------------------------------------------------------- #
# find_crypto: _scan_signatures + CRYPTO_SIGNATURES
# --------------------------------------------------------------------------- #

_triage = _load_names(
    "api_triage.py",
    {"_scan_signatures", "CRYPTO_SIGNATURES", "_MAX_HITS_PER_SIG",
     "_evaluate_capabilities", "CAPABILITY_RULES"},
)
_scan_signatures = _triage["_scan_signatures"]
CRYPTO_SIGNATURES = _triage["CRYPTO_SIGNATURES"]
_evaluate_capabilities = _triage["_evaluate_capabilities"]
CAPABILITY_RULES = _triage["CAPABILITY_RULES"]


def test_scan_finds_aes_and_base64():
    aes = next(p for n, _, p in CRYPTO_SIGNATURES if n == "AES S-box")
    b64 = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    blob = b"\x00" * 16 + aes + b"\x11" * 8 + b64
    hits = _scan_signatures([(0x140000000, blob)], CRYPTO_SIGNATURES)
    names = {h["name"]: h["addr"] for h in hits}
    assert names["AES S-box"] == 0x140000000 + 16
    assert names["base64 alphabet"] == 0x140000000 + 16 + len(aes) + 8


def test_scan_respects_max_hits():
    sig = [("x", "", b"AB")]
    data = b"AB" * 100
    hits = _scan_signatures([(0, data)], sig, max_hits_per_sig=3)
    assert len(hits) == 3


# --------------------------------------------------------------------------- #
# detect_capabilities: _evaluate_capabilities + CAPABILITY_RULES
# --------------------------------------------------------------------------- #

def test_capabilities_process_injection():
    caps = _evaluate_capabilities(
        {"VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread", "GetProcAddress", "LoadLibraryA"},
        [],
        CAPABILITY_RULES,
    )
    names = {c["capability"] for c in caps}
    assert "Process injection" in names
    assert "Dynamic API resolution" in names
    inj = next(c for c in caps if c["capability"] == "Process injection")
    assert inj["mitre"] == "T1055"
    assert "VirtualAllocEx" in inj["evidence"]


def test_capabilities_string_only_rule_fires():
    caps = _evaluate_capabilities(
        set(), ["HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run"], CAPABILITY_RULES
    )
    assert "Registry Run-key persistence" in {c["capability"] for c in caps}


def test_capabilities_empty_binary_matches_nothing():
    caps = _evaluate_capabilities(set(), [], CAPABILITY_RULES)
    assert caps == []


# --------------------------------------------------------------------------- #
# export_patched_binary: _apply_patches_to_buffer
# --------------------------------------------------------------------------- #

_apply = _load_names("api_modify.py", {"_apply_patches_to_buffer"})["_apply_patches_to_buffer"]


def test_apply_patches_writes_in_range_only():
    buf = bytearray(b"\x00" * 8)
    applied = _apply(buf, [(0, 0xAA), (3, 0x1BB), (100, 0xCC), (-1, 0xDD)])
    assert applied == 2
    assert buf[0] == 0xAA
    assert buf[3] == 0xBB  # 0x1BB masked to a byte
    assert buf[7] == 0x00

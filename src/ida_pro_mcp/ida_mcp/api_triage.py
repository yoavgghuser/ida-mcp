"""Triage helpers: crypto-constant detection and capability fingerprinting.

Two tools that turn raw survey output into a verdict an agent can reason about:

- ``find_crypto`` scans mapped bytes for well-known cryptographic constants
  (AES S-boxes, SHA/MD5 tables and IVs, base64 alphabets, CRC32 table, ChaCha
  sigma) so you can locate crypto without recognising the maths by hand.
- ``detect_capabilities`` fingerprints behaviour from imported APIs and strings
  against a small capa-style ruleset, labelling each hit with a MITRE ATT&CK
  technique and the evidence that fired it.

The matching cores (``_scan_signatures`` / ``_evaluate_capabilities``) are pure
functions over plain data, so they unit-test without IDA present.
"""

from typing import Annotated, Any, NotRequired, TypedDict

from .rpc import tool
from .sync import idasync
from .utils import read_bytes_bss_safe


# ---------------------------------------------------------------------------
# Crypto constant signatures: (name, note, bytes-to-find)
# ---------------------------------------------------------------------------

CRYPTO_SIGNATURES: list[tuple[str, str, bytes]] = [
    ("AES S-box", "forward substitution box",
     bytes.fromhex("637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0")),
    ("AES inverse S-box", "inverse substitution box",
     bytes.fromhex("52096ad53036a538bf40a39e81f3d7fb7ce339829b2fff87348e4344c4dee9cb")),
    ("SHA-256 constants (K)", "round constants",
     bytes.fromhex("982f8a4291443771cffbc0b5a5dbb5e95bc25639f111f159a4823f92d55e1cab")),
    ("SHA-256 init (H0)", "initial hash values",
     bytes.fromhex("67e6096a85ae67bb72f36e3c3af54fa57f520e518c68059babd9831f19cde05b")),
    ("SHA-512 constants (K)", "first round constant",
     bytes.fromhex("22ae28d7982f8a42")),
    ("SHA-1 init", "initial hash values (incl. C3D2E1F0)",
     bytes.fromhex("0123456789abcdeffedcba9876543210f0e1d2c3")),
    ("MD5 T-table", "sine-derived constants",
     bytes.fromhex("78a46ad756b7c7e8db702024eecebdc1")),
    ("CRC32 table", "IEEE 802.3 polynomial table",
     bytes.fromhex("00000000" "96300777" "2c610eee" "ba510999")),
    ("base64 alphabet", "standard RFC 4648 alphabet",
     b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"),
    ("base64 url-safe alphabet", "url-safe RFC 4648 alphabet",
     b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"),
    ("ChaCha/Salsa20 sigma", "\"expand 32-byte k\" constant",
     b"expand 32-byte k"),
]

_MAX_HITS_PER_SIG = 8


# ---------------------------------------------------------------------------
# Capability rules: imports/strings -> behaviour + MITRE technique
# ---------------------------------------------------------------------------

CAPABILITY_RULES: list[dict[str, Any]] = [
    {"name": "Process injection", "mitre": "T1055",
     "all_apis": ["VirtualAllocEx", "WriteProcessMemory"],
     "any_apis": ["CreateRemoteThread", "NtCreateThreadEx", "QueueUserAPC", "RtlCreateUserThread"]},
    {"name": "Process hollowing", "mitre": "T1055.012",
     "all_apis": ["NtUnmapViewOfSection", "WriteProcessMemory", "SetThreadContext"]},
    {"name": "Dynamic API resolution", "mitre": "T1027",
     "all_apis": ["LoadLibraryA", "GetProcAddress"]},
    {"name": "Registry Run-key persistence", "mitre": "T1547.001",
     "any_apis": ["RegSetValueExA", "RegSetValueExW", "RegCreateKeyExA", "RegCreateKeyExW"],
     "any_strings": ["\\CurrentVersion\\Run", "Software\\Microsoft\\Windows\\CurrentVersion\\Run"]},
    {"name": "Service creation", "mitre": "T1543.003",
     "any_apis": ["CreateServiceA", "CreateServiceW", "OpenSCManagerA", "OpenSCManagerW"]},
    {"name": "Scheduled task", "mitre": "T1053.005",
     "any_strings": ["schtasks", "\\Microsoft\\Windows\\TaskCache"]},
    {"name": "Anti-debugging", "mitre": "T1622",
     "any_apis": ["IsDebuggerPresent", "CheckRemoteDebuggerPresent", "NtQueryInformationProcess",
                  "OutputDebugStringA"]},
    {"name": "Keylogging", "mitre": "T1056.001",
     "any_apis": ["SetWindowsHookExA", "SetWindowsHookExW", "GetAsyncKeyState", "GetKeyState"]},
    {"name": "Screen capture", "mitre": "T1113",
     "any_apis": ["BitBlt", "CreateCompatibleBitmap", "GetDC", "GetDIBits"]},
    {"name": "Clipboard access", "mitre": "T1115",
     "any_apis": ["OpenClipboard", "GetClipboardData"]},
    {"name": "Network communication", "mitre": "T1071",
     "any_apis": ["InternetOpenA", "InternetOpenUrlA", "HttpSendRequestA", "WinHttpOpen",
                  "WSAStartup", "socket", "connect", "send", "recv"]},
    {"name": "DNS resolution", "mitre": "T1071.004",
     "any_apis": ["DnsQuery_A", "getaddrinfo", "gethostbyname"]},
    {"name": "Cryptography (Win API)", "mitre": "T1027",
     "any_apis": ["CryptEncrypt", "CryptDecrypt", "CryptAcquireContextA", "CryptGenKey",
                  "BCryptEncrypt", "BCryptDecrypt", "NCryptCreatePersistedKey"]},
    {"name": "Token/privilege manipulation", "mitre": "T1134",
     "any_apis": ["AdjustTokenPrivileges", "OpenProcessToken", "LookupPrivilegeValueA"]},
    {"name": "Process discovery", "mitre": "T1057",
     "any_apis": ["CreateToolhelp32Snapshot", "Process32First", "Process32Next", "EnumProcesses"]},
    {"name": "File/directory discovery", "mitre": "T1083",
     "any_apis": ["FindFirstFileA", "FindFirstFileW", "FindNextFileA", "FindNextFileW"]},
    {"name": "Command execution", "mitre": "T1059",
     "any_apis": ["ShellExecuteA", "ShellExecuteW", "WinExec", "CreateProcessA", "CreateProcessW", "system"],
     "any_strings": ["cmd.exe", "cmd /c", "powershell"]},
    {"name": "Ransomware indicators", "mitre": "T1486",
     "any_strings": ["your files have been encrypted", "bitcoin", ".onion", "readme.txt", "decrypt"]},
    {"name": "Screenshot/desktop", "mitre": "T1113",
     "any_apis": ["GetDesktopWindow", "PrintWindow"]},
]


def _scan_signatures(
    segments: list[tuple[int, bytes]],
    signatures: list[tuple[str, str, bytes]],
    max_hits_per_sig: int = _MAX_HITS_PER_SIG,
) -> list[dict]:
    """Find each signature's bytes in the segments. Pure over (addr, bytes)."""
    hits = []
    for name, note, pat in signatures:
        if not pat:
            continue
        found = 0
        for base, data in segments:
            idx = data.find(pat)
            while idx != -1 and found < max_hits_per_sig:
                hits.append({"name": name, "note": note, "addr": base + idx, "size": len(pat)})
                found += 1
                idx = data.find(pat, idx + 1)
            if found >= max_hits_per_sig:
                break
    return hits


def _evaluate_capabilities(
    imports: set[str],
    strings: list[str],
    rules: list[dict],
) -> list[dict]:
    """Match imports/strings against capability rules. Pure over plain data."""
    imps = {i.lower() for i in imports}
    stext = "\n".join(strings).lower()
    out = []
    for rule in rules:
        evidence: list[str] = []
        ok = True
        for api in rule.get("all_apis", []):
            if api.lower() in imps:
                evidence.append(api)
            else:
                ok = False
                break
        if not ok:
            continue
        any_apis = rule.get("any_apis", [])
        if any_apis:
            matched = [a for a in any_apis if a.lower() in imps]
            if not matched and not evidence:
                # rule keyed on any_apis but none present
                if not rule.get("any_strings"):
                    continue
            evidence.extend(matched)
        any_strings = rule.get("any_strings", [])
        if any_strings:
            matched_s = [s for s in any_strings if s.lower() in stext]
            evidence.extend(f'"{s}"' for s in matched_s)
        # A rule fires only if it produced some evidence.
        if evidence:
            out.append({"capability": rule["name"], "mitre": rule.get("mitre"), "evidence": evidence})
    return out


class CryptoHit(TypedDict):
    name: str
    note: str
    addr: str
    segment: str
    size: int


@tool
@idasync
def find_crypto() -> dict:
    """Scan the database for well-known cryptographic constants.

    Detects AES S-boxes, SHA-256/512 and MD5/SHA-1 tables and IVs, the CRC32
    table, base64 alphabets and the ChaCha/Salsa sigma constant. Returns each
    hit with its address and segment.
    """
    import idautils
    import idc
    import ida_segment

    segments = []
    for seg_ea in idautils.Segments():
        end = idc.get_segm_end(seg_ea)
        if end > seg_ea:
            segments.append((seg_ea, read_bytes_bss_safe(seg_ea, end - seg_ea)))

    hits: list[CryptoHit] = []
    for hit in _scan_signatures(segments, CRYPTO_SIGNATURES):
        seg = ida_segment.getseg(hit["addr"])
        hits.append(
            {
                "name": hit["name"],
                "note": hit["note"],
                "addr": hex(hit["addr"]),
                "segment": ida_segment.get_segm_name(seg) if seg else "",
                "size": hit["size"],
            }
        )
    return {"count": len(hits), "hits": hits}


@tool
@idasync
def detect_capabilities() -> dict:
    """Fingerprint behaviour from imports and strings (capa-style + MITRE).

    Evaluates a curated ruleset over the binary's imported APIs and strings and
    returns matched capabilities, each with a MITRE ATT&CK technique id and the
    imports/strings that triggered it. Use it as a triage layer on top of
    survey_binary.
    """
    import ida_nalt

    imports: set[str] = set()
    for i in range(ida_nalt.get_import_module_qty()):
        def _cb(ea, name, ordinal, acc=imports):
            if name:
                acc.add(name)
            return True

        ida_nalt.enum_import_names(i, _cb)

    try:
        from .api_core import _get_strings_cache

        strings = [text for _ea, text in _get_strings_cache()]
    except Exception:  # noqa: BLE001
        strings = []

    caps = _evaluate_capabilities(imports, strings, CAPABILITY_RULES)
    return {"count": len(caps), "capabilities": caps, "imports_seen": len(imports)}

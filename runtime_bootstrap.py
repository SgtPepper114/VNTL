"""Load a compatible Windows C++ runtime before Qt alters DLL search paths.

PyQt6's wheel may ship an older msvcp140.dll than ONNX Runtime requires. Loading
the trusted system runtime first prevents a native access violation. Frozen
builds carry the same checked build-host runtime in their application directory.
No system file or global DLL search policy is modified.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_runtime_handle = None


def system_msvc_path() -> Path | None:
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    kernel32.GetSystemDirectoryW.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if not length or length >= len(buffer):
        return None
    path = Path(buffer.value) / "msvcp140.dll"
    return path if path.is_file() else None


def preload_windows_runtime() -> bool:
    global _runtime_handle
    if sys.platform != "win32" or _runtime_handle is not None:
        return True
    bundled = Path(sys._MEIPASS) / "msvcp140.dll" if getattr(sys, "frozen", False) else None
    path = bundled if bundled is not None and bundled.is_file() else system_msvc_path()
    if path is None:
        return False
    # Absolute path and constrained dependency lookup: application DLL directory
    # (for frozen redistributables) plus Windows System32, never the current dir.
    _runtime_handle = ctypes.WinDLL(str(path), winmode=0x00000100 | 0x00000800)
    return True

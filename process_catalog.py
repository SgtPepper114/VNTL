"""Read-only process/window inventory for the game process picker.

An application here means a process with a visible, titled, non-tool desktop
window. Minimized windows still qualify; this is not a full Task Manager API.
No process is opened or injected into while building this inventory.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass
from functools import lru_cache
import logging
import sys

from hooker import list_processes

logger = logging.getLogger(__name__)

_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x00000080
_DWMWA_CLOAKED = 14
_ENUM_CALLBACK = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(wt.BOOL, wt.HWND, wt.LPARAM)


@dataclass(frozen=True)
class ProcessEntry:
    pid: int
    name: str
    window_titles: tuple[str, ...] = ()

    @property
    def is_application(self) -> bool:
        return bool(self.window_titles)


class _WindowAPI:
    def __init__(self):
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.enum_windows = self.user32.EnumWindows
        self.enum_windows.argtypes = [_ENUM_CALLBACK, wt.LPARAM]
        self.enum_windows.restype = wt.BOOL
        self.is_visible = self.user32.IsWindowVisible
        self.is_visible.argtypes = [wt.HWND]
        self.is_visible.restype = wt.BOOL
        self.title_length = self.user32.GetWindowTextLengthW
        self.title_length.argtypes = [wt.HWND]
        self.title_length.restype = ctypes.c_int
        self.get_title = self.user32.GetWindowTextW
        self.get_title.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        self.get_title.restype = ctypes.c_int
        self.get_pid = self.user32.GetWindowThreadProcessId
        self.get_pid.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        self.get_pid.restype = wt.DWORD
        # GetWindowLongPtrW is exported only by 64-bit user32.
        self.get_style = getattr(self.user32, "GetWindowLongPtrW", None)
        if self.get_style is None:
            self.get_style = self.user32.GetWindowLongW
        self.get_style.argtypes = [wt.HWND, ctypes.c_int]
        self.get_style.restype = ctypes.c_ssize_t
        self.get_cloaked = None
        try:
            self.dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
            self.get_cloaked = self.dwmapi.DwmGetWindowAttribute
            self.get_cloaked.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]
            self.get_cloaked.restype = ctypes.c_long  # HRESULT
        except (AttributeError, OSError):
            # Older Windows or unavailable DWM: other window checks still work.
            pass


@lru_cache(maxsize=1)
def _window_api():
    return _WindowAPI()


def _window_titles_by_pid() -> dict[int, tuple[str, ...]]:
    if sys.platform != "win32":
        return {}
    api = _window_api()
    titles: dict[int, set[str]] = {}

    def collect(hwnd, _lparam):
        # Windows can close between any two calls. Skip that window and keep
        # enumerating; Python exceptions must never escape a ctypes callback.
        try:
            if not api.is_visible(hwnd) or api.get_style(hwnd, _GWL_EXSTYLE) & _WS_EX_TOOLWINDOW:
                return True
            if api.get_cloaked is not None:
                cloaked = wt.DWORD(0)
                hr = api.get_cloaked(hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
                if hr == 0 and cloaked.value:
                    return True
            length = api.title_length(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(min(length + 1, 32768))
            if api.get_title(hwnd, buffer, len(buffer)) <= 0:
                return True
            title = buffer.value.strip()
            if not title:
                return True
            pid = wt.DWORD(0)
            if api.get_pid(hwnd, ctypes.byref(pid)) and pid.value:
                titles.setdefault(pid.value, set()).add(title)
        except Exception:
            # A disappearing/inaccessible window must not remove its process.
            pass
        return True

    callback = _ENUM_CALLBACK(collect)
    if not api.enum_windows(callback, 0):
        return {}
    return {pid: tuple(sorted(names, key=lambda value: (value.casefold(), value)))
            for pid, names in titles.items()}


def list_process_entries() -> list[ProcessEntry]:
    """Return every process, applications first, without reading process memory.

    Window enumeration failures affect grouping only: the original process
    snapshot remains available as background entries. Multiple windows for a
    PID collapse into one entry with distinct, deterministically sorted titles.
    """
    processes = list_processes()
    try:
        titles = _window_titles_by_pid()
    except Exception as exc:
        logger.debug("Window inventory unavailable (%s); keeping process list", type(exc).__name__)
        titles = {}
    entries = [ProcessEntry(pid, name, titles.get(pid, ())) for pid, name in processes]
    return sorted(entries, key=lambda entry: (not entry.is_application, entry.name.casefold(), entry.name, entry.pid))

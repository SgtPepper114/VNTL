"""Windows global shortcuts without installing a keyboard hook."""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys

from PyQt6.QtCore import QAbstractNativeEventFilter, QTimer


class GlobalHotkeys(QAbstractNativeEventFilter):
    def __init__(self, app, translate, toggle_visible):
        super().__init__()
        self._app = app
        self._callbacks = {}
        self.failed = []
        if sys.platform != "win32":
            self.failed = ["Ctrl+Alt+T", "Ctrl+Alt+H"]
            return
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.RegisterHotKey.argtypes = [wt.HWND, ctypes.c_int, wt.UINT, wt.UINT]
        self._user32.RegisterHotKey.restype = wt.BOOL
        self._user32.UnregisterHotKey.argtypes = [wt.HWND, ctypes.c_int]
        self._user32.UnregisterHotKey.restype = wt.BOOL
        for identifier, key, label, callback in [
            (0x5654, ord("T"), "Ctrl+Alt+T", translate),
            (0x5655, ord("H"), "Ctrl+Alt+H", toggle_visible),
        ]:
            # MOD_CONTROL | MOD_ALT | MOD_NOREPEAT
            if self._user32.RegisterHotKey(None, identifier, 0x4003, key):
                self._callbacks[identifier] = callback
            else:
                self.failed.append(label)
        app.installNativeEventFilter(self)
        app.aboutToQuit.connect(self.close)

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            msg = wt.MSG.from_address(int(message))
            if msg.message == 0x0312 and msg.wParam in self._callbacks:
                QTimer.singleShot(0, self._callbacks[msg.wParam])
                return True, 0
        return False, 0

    def close(self):
        for identifier in self._callbacks:
            self._user32.UnregisterHotKey(None, identifier)
        self._callbacks.clear()
        self._app.removeNativeEventFilter(self)

"""Independent Windows backdrop with a capture-compatible color-key foreground.

Both windows use redirected backing stores rather than UpdateLayeredWindow.
Only the backdrop has a global alpha; foreground glyphs and controls stay at 1.0.
"""
from __future__ import annotations

import ctypes
import logging
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QApplication, QWidget

logger = logging.getLogger(__name__)
# A near-black matte avoids bright color-key fringes around antialiased glyphs.
FOREGROUND_KEY = (1, 2, 3)


def _native_api():
    if sys.platform != "win32" or QApplication.platformName() != "windows":
        return None
    import ctypes.wintypes as wt
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.GetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int]
    api.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    api.SetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_ssize_t]
    api.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    api.SetLayeredWindowAttributes.argtypes = [wt.HWND, wt.DWORD, wt.BYTE, wt.DWORD]
    api.SetLayeredWindowAttributes.restype = wt.BOOL
    api.SetWindowPos.argtypes = [wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, wt.UINT]
    api.SetWindowPos.restype = wt.BOOL
    return api


class BackgroundLayer(QWidget):
    """A nonactivating backdrop kept directly below its UI."""

    def __init__(self, foreground):
        super().__init__()
        self._foreground = foreground
        self._api = _native_api()
        self._color = QColor(20, 20, 25)
        self._key_ready = False
        self._key_attempted = False
        self._warned = False
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        foreground.destroyed.connect(self.deleteLater)

    def enable_foreground(self):
        if self._key_attempted:
            return self._key_ready
        self._key_attempted = True
        if self._api is None:  # Offscreen unit tests use ordinary Qt painting.
            self._key_ready = True
            return True
        hwnd = int(self._foreground.winId())
        style = self._api.GetWindowLongPtrW(hwnd, -20)
        self._api.SetWindowLongPtrW(hwnd, -20, style | 0x00080000)  # WS_EX_LAYERED
        r, g, b = FOREGROUND_KEY
        self._key_ready = bool(self._api.SetLayeredWindowAttributes(
            hwnd, r | (g << 8) | (b << 16), 255, 0x1))  # LWA_COLORKEY
        if not self._key_ready and not self._warned:
            error = ctypes.get_last_error()
            # A newly layered window stays invisible until its attributes are
            # initialized. Restore the previous opaque style on failure.
            self._api.SetWindowLongPtrW(hwnd, -20, style)
            logger.warning("Background transparency unavailable: Windows error %s", error)
            self._warned = True
        return self._key_ready

    def set_color(self, rgba):
        self._color = QColor(*rgba[:3])
        self.setWindowOpacity(max(0, min(255, rgba[3])) / 255)
        self.update()

    def sync(self):
        owner = self._foreground
        self.setGeometry(owner.geometry())
        if not owner.isVisible() or owner.isMinimized() or not self.enable_foreground():
            self.hide()
            return
        if not self.isVisible():
            self.show()
        if self._api is not None:
            # Keep both topmost windows together without stealing game focus.
            self._api.SetWindowPos(int(self.winId()), int(owner.winId()), 0, 0, 0, 0,
                                   0x1 | 0x2 | 0x10 | 0x200)  # NOSIZE|NOMOVE|NOACTIVATE|NOOWNERZORDER

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._color)
        painter.end()

    # A visible backdrop still behaves like the original draggable surface.
    # At zero alpha the always-visible foreground control bar is the drag area.
    def mousePressEvent(self, event):
        self._foreground.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._foreground.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._foreground.mouseReleaseEvent(event)

"""Read-only Switch 2 Pro USB back-button input; never injects game input.

The tested 057e:2069 common 0x05 report has GR/GL in byte 8, bits 0/1.
Reference: libsdl-org/SDL, src/joystick/hidapi/SDL_hidapi_switch2.c,
HandleSwitchProState. Other report formats and Bluetooth are not assumed.
"""
from __future__ import annotations

import logging
import threading
import time

import hid
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger("vntl.controller")
VENDOR_ID = 0x057E
PRODUCT_ID = 0x2069


def rear_buttons(report) -> tuple[bool, bool] | None:
    if len(report) < 64 or report[0] != 0x05:
        return None
    return bool(report[8] & 1), bool(report[8] & 2)


class PressEdge:
    """Require a release before the first press, with a short bounce guard."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._previous = None
        self._last_press = float("-inf")
        self._released_since = None

    def update(self, down: bool, now: float) -> bool:
        previous, self._previous = self._previous, down
        if not down:
            if previous is not False:
                self._released_since = now
            return False
        if (previous is False and self._released_since is not None
                and now - self._released_since >= 0.03
                and now - self._last_press >= 0.35):
            self._last_press = now
            return True
        return False


class Switch2Reader:
    """Pollable read-only backend, separate from the UI and translation API."""
    def __init__(self, backend=hid):
        self._backend = backend
        self._device = None
        self._edge = PressEdge()
        self._selection = None
        self._next_connect = 0.0
        self._last_report = 0.0
        self._streaming = False
        self.status = "正在查找 Switch 2 Pro 手柄…"

    def close(self):
        device, self._device = self._device, None
        if device is not None:
            try:
                device.close()
            except (OSError, ValueError):
                pass
        self._edge.reset()
        self._streaming = False

    def poll(self, enabled: bool, button: str, now: float) -> bool:
        selection = (enabled, button)
        if selection != self._selection:
            self._selection = selection
            self.close()
            self._next_connect = 0.0
        if not enabled:
            self.status = "手柄翻译已关闭"
            return False
        if self._device is None:
            if now < self._next_connect:
                return False
            self._next_connect = now + 2.0
            try:
                devices = [d for d in self._backend.enumerate(VENDOR_ID, PRODUCT_ID)
                           if d.get("usage_page") == 1 and d.get("usage") == 5
                           and d.get("interface_number") == 0]
                if not devices:
                    self.status = "未找到 Switch 2 Pro，请检查 USB 连接"
                    return False
                # Multiple identical pads must not silently select someone else's.
                if len(devices) != 1:
                    self.status = "检测到多个 Switch 2 Pro，请只连接要使用的手柄"
                    return False
                device = self._backend.device()
                self._device = device
                device.open_path(devices[0]["path"])
                device.set_nonblocking(True)
                self._last_report = now
                self.status = "手柄已连接，正在等待按键数据…"
            except (OSError, ValueError):
                self.close()
                self.status = "无法读取手柄，将自动重试"
                return False
        pressed = False
        try:
            # A bounded drain prevents the input stream from monopolizing a poll.
            for _ in range(64):
                report = self._device.read(512)
                if not report:
                    break
                state = rear_buttons(report)
                if state is None:
                    continue
                self._last_report = now
                if not self._streaming:
                    self._edge.reset()
                self._streaming = True
                self.status = f"已连接 · {'右背键 GR' if button == 'GR' else '左背键 GL'} 翻译"
                if self._edge.update(state[0 if button == "GR" else 1], now):
                    pressed = True
        except (OSError, ValueError):
            self.close()
            self._next_connect = now + 2.0
            self.status = "手柄已断开，等待重新连接"
            return False
        if now - self._last_report > 3.0:
            self._streaming = False
            self._edge.reset()
            self.status = "未收到按键数据，请确认手柄已能在模拟器中使用"
            # Reopen a stale handle so reconnects don't need an app restart.
            self.close()
            self._next_connect = now + 2.0
        return pressed


class ControllerService(QObject):
    pressed = pyqtSignal()
    _status_changed = pyqtSignal(str)

    def __init__(self, app, translate, *, enabled=True, button="GR"):
        super().__init__(app)
        self.status = "正在查找 Switch 2 Pro 手柄…"
        self._settings_lock = threading.Lock()
        self._settings = (bool(enabled), button if button in ("GR", "GL") else "GR")
        self._stop = threading.Event()
        self._status_changed.connect(self._set_status)
        self.pressed.connect(translate)
        self._thread = threading.Thread(target=self._run, name="vntl-controller", daemon=True)
        self._thread.start()
        app.aboutToQuit.connect(self.close)

    def configure(self, enabled: bool, button: str):
        with self._settings_lock:
            self._settings = (bool(enabled), button if button in ("GR", "GL") else "GR")

    def _set_status(self, status):
        self.status = status
        logger.info("Controller: %s", status)

    def _run(self):
        reader = Switch2Reader()
        last_status = None
        try:
            while not self._stop.is_set():
                with self._settings_lock:
                    enabled, button = self._settings
                if reader.poll(enabled, button, time.monotonic()):
                    self.pressed.emit()
                if reader.status != last_status:
                    last_status = reader.status
                    self._status_changed.emit(last_status)
                self._stop.wait(0.008 if enabled else 0.2)
        except Exception:
            logger.exception("Controller listener stopped")
            self._status_changed.emit("手柄监听已停止，请重新启动 VNTL")
        finally:
            reader.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.5)

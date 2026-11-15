"""Background controller input via SDL, with a Switch 2 USB rear-button adapter."""
from __future__ import annotations

import logging
import hashlib
import copy
import threading
import time

import hid
from PyQt6.QtCore import QObject, pyqtSignal
from controller_binding import binding_label, legacy_binding, normalize_binding
from controller_input import BindingLearner, BindingTrigger, PressEdge

logger = logging.getLogger("vntl.controller")
VENDOR_ID = 0x057E
PRODUCT_ID = 0x2069


def rear_buttons(report) -> tuple[bool, bool] | None:
    if len(report) < 64 or report[0] != 0x05:
        return None
    return bool(report[8] & 1), bool(report[8] & 2)


def _switch2_devices(backend=hid):
    return [d for d in backend.enumerate(VENDOR_ID, PRODUCT_ID)
            if d.get("usage_page") == 1 and d.get("usage") == 5
            and d.get("interface_number") == 0]


def _switch2_id(device):
    path = device["path"]
    if isinstance(path, str):
        path = path.encode("utf-8")
    return "switch2:" + hashlib.sha256(path).hexdigest()


def switch2_devices(backend=hid):
    return [{"id": _switch2_id(d), "name": "Switch 2 Pro USB · 原始背键", "backend": "switch2"}
            for d in _switch2_devices(backend)]


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
        self.state = None
        self.states = []
        self._connection = 0
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
        self.state = None
        self.states = []

    def poll(self, enabled: bool, button: str, now: float, device_id="switch2:auto") -> bool:
        self.states = []
        selection = (enabled, button, device_id)
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
                devices = _switch2_devices(self._backend)
                if device_id != "switch2:auto":
                    devices = [d for d in devices if _switch2_id(d) == device_id]
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
                self._connection += 1
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
                self.state = {"buttons": state, "hats": (), "axes": (), "connection": self._connection}
                self.states.append(self.state)
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
    binding_learned = pyqtSignal(object)
    _triggered = pyqtSignal(int)
    _binding_found = pyqtSignal(int, object)
    _learning_expired = pyqtSignal(int)
    _status_changed = pyqtSignal(str)

    def __init__(self, app, translate, *, enabled=True, button="GR", binding=None):
        super().__init__(app)
        self.status = "正在查找手柄…"
        self.learning_message = ""
        self._settings_lock = threading.Lock()
        self._settings = (bool(enabled), normalize_binding(binding) if binding is not None else legacy_binding(button))
        self._version = 0
        self._devices = []
        self._learning_request = None
        self._stop = threading.Event()
        self._status_changed.connect(self._set_status)
        self._triggered.connect(self._deliver_press)
        self._binding_found.connect(self._deliver_binding)
        self._learning_expired.connect(self._expire_learning)
        self.pressed.connect(translate)
        self._thread = threading.Thread(target=self._run, name="vntl-controller", daemon=True)
        self._thread.start()
        app.aboutToQuit.connect(self.close)

    def configure(self, enabled: bool, button="GR", *, binding=None):
        with self._settings_lock:
            self._settings = (bool(enabled), normalize_binding(binding) if binding is not None else legacy_binding(button))
            self._learning_request = None
            self._version += 1

    def devices(self):
        with self._settings_lock:
            return copy.deepcopy(self._devices)

    @property
    def learning(self):
        with self._settings_lock:
            return self._learning_request is not None

    def start_learning(self, device_id: str, include_axes=False):
        with self._settings_lock:
            self._learning_request = (device_id, bool(include_axes))
            self._version += 1
        self.learning_message = "请先松开所有按键，让摇杆和扳机静止。"

    def cancel_learning(self):
        with self._settings_lock:
            self._learning_request = None
            self._version += 1

    def _deliver_press(self, version):
        with self._settings_lock:
            allowed = (not self._stop.is_set() and version == self._version
                       and self._settings[0] and self._learning_request is None)
        if allowed:
            self.pressed.emit()

    def _deliver_binding(self, version, binding):
        with self._settings_lock:
            if self._stop.is_set() or version != self._version or self._learning_request is None:
                return
            self._learning_request = None
            self._version += 1
        self.learning_message = "录入完成，点击保存后生效。"
        self.binding_learned.emit(binding)

    def _expire_learning(self, version):
        with self._settings_lock:
            if version != self._version or self._learning_request is None:
                return
            self._learning_request = None
            self._version += 1
        self.learning_message = "录入已超时，请点击录入按键重试。"

    def _set_status(self, status):
        self.status = status
        logger.info("Controller: %s", status)

    def _run(self):
        from controller_sdl import SDLReader
        reader, sdl = Switch2Reader(), SDLReader()
        trigger = BindingTrigger()
        last_status = None
        version_seen = None
        learner = None
        learning_finished = False
        next_scan = 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                with self._settings_lock:
                    enabled, binding = self._settings
                    version, learning = self._version, self._learning_request
                if now >= next_scan:
                    next_scan = now + 1.0
                    devices = sdl.list_devices()
                    try:
                        devices += switch2_devices()
                    except (OSError, ValueError):
                        pass
                    with self._settings_lock:
                        self._devices = devices
                if version != version_seen:
                    version_seen = version
                    trigger.reset()
                    learner = None
                    learning_finished = False
                    reader.close()
                    sdl.release()
                device_id = learning[0] if learning else binding.get("device", "")
                active = bool(learning or (enabled and binding))
                state = None
                states = []
                if active and device_id.startswith("switch2:"):
                    reader.poll(True, "GR", now, device_id)
                    state, status = reader.state, reader.status
                    states = reader.states
                elif active and device_id.startswith("sdl:"):
                    reader.close()
                    state, status = sdl.read(device_id), sdl.status
                else:
                    reader.close()
                    status = "手柄翻译已关闭" if not enabled else "请在设置 → 手柄中录入翻译按键"
                if learning:
                    if learner is None:
                        device = next((d for d in self.devices() if d["id"] == device_id), None)
                        device = device or {"id": device_id, "backend": device_id.split(":")[0],
                                            "name": binding.get("name", "手柄")}
                        learner = BindingLearner(device, learning[1], now)
                    if not learning_finished:
                        learned = None
                        for sample in states or [state]:
                            learned = learner.update(sample, now)
                            if learned or learner.expired:
                                break
                        self.learning_message = learner.message
                        if learned:
                            learning_finished = True
                            self._binding_found.emit(version, learned)
                        elif learner.expired:
                            learning_finished = True
                            self._learning_expired.emit(version)
                elif active:
                    triggered = False
                    for sample in states or [state]:
                        triggered = trigger.update(binding, sample, now) or triggered
                    if triggered:
                        self._triggered.emit(version)
                    if state is not None:
                        status = "已连接 · " + binding_label(binding)
                if status != last_status:
                    last_status = status
                    self._status_changed.emit(last_status)
                self._stop.wait(0.008 if active else 0.1)
        except Exception:
            logger.exception("Controller listener stopped")
            self._status_changed.emit("手柄监听已停止，请重新启动 VNTL")
        finally:
            reader.close()
            sdl.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.5)

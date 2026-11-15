"""Windowless SDL joystick polling for configurable controller inputs.

Construct, poll, enumerate, and close a reader on one worker thread. SDL's
joystick subsystem updates devices directly; the Qt application owns the UI.
"""
from __future__ import annotations

import hashlib
import importlib
import logging
import time

logger = logging.getLogger("vntl.controller.sdl")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip()
    return str(value or "").strip()


class SDLReader:
    """A read-only joystick adapter; only the selected device is opened.

    ``id`` combines the hardware GUID and SDL device path when available.
    Paths are hashed before being returned or persisted. An absent path uses
    the GUID alone, with duplicate identities treated as ambiguous. SDL's
    per-connection instance ID is returned separately as ``connection``.
    """

    def __init__(self, backend=None, *, clock=time.monotonic):
        self.status = "正在查找通用手柄…"
        self._api = None
        self._clock = clock
        self._initialized = False
        self._closed = False
        self._devices = {}
        self._next_scan = 0.0
        self._handle = None
        self._selection = None
        try:
            self._api = backend if backend is not None else importlib.import_module("sdl2")
            api = self._api
            api.SDL_SetHint(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", b"1")
            if api.SDL_InitSubSystem(api.SDL_INIT_JOYSTICK) < 0:
                raise RuntimeError("SDL joystick initialization failed")
            self._initialized = True
            api.SDL_JoystickEventState(api.SDL_IGNORE)
        except Exception as exc:
            logger.warning("SDL joystick initialization failed: %s", type(exc).__name__)
            if self._initialized:
                self._quit_subsystem()
            self._initialized = False
            self.status = "通用手柄组件加载失败，请检查 SDL 依赖或更新程序"

    @staticmethod
    def _identity(guid, path) -> str:
        identity = "sdl:" + bytes(guid.data).hex()
        if path:
            raw = path if isinstance(path, bytes) else str(path).encode("utf-8")
            identity += ":" + hashlib.sha256(raw).hexdigest()[:24]
        return identity

    def _scan(self):
        api = self._api
        devices = {}
        for index in range(api.SDL_NumJoysticks()):
            instance = api.SDL_JoystickGetDeviceInstanceID(index)
            if instance < 0:  # A device may disappear during enumeration.
                continue
            identity = self._identity(api.SDL_JoystickGetDeviceGUID(index),
                                      api.SDL_JoystickPathForIndex(index))
            devices.setdefault(identity, []).append({
                "index": index, "instance": instance,
                "name": _text(api.SDL_JoystickNameForIndex(index)) or "通用手柄",
            })
        self._devices = devices
        self._next_scan = self._clock() + 0.5

    def list_devices(self) -> list[dict]:
        if not self._initialized or self._closed:
            return []
        try:
            self._api.SDL_JoystickUpdate()
            self._scan()
            result = []
            for identity, matches in self._devices.items():
                name = matches[0]["name"]
                if len(matches) > 1:
                    name += f"（{len(matches)} 台同标识设备）"
                result.append({"id": identity, "name": name, "backend": "sdl",
                               "ambiguous": len(matches) > 1})
            return result
        except Exception as exc:
            logger.warning("SDL joystick enumeration failed: %s", type(exc).__name__)
            self._devices = {}
            self.status = "手柄列表读取失败，正在等待重试"
            return []

    def _close_handle(self):
        handle, self._handle = self._handle, None
        self._selection = None
        if handle is not None:
            try:
                self._api.SDL_JoystickClose(handle)
            except Exception as exc:
                logger.warning("SDL joystick close failed: %s", type(exc).__name__)

    def _quit_subsystem(self):
        try:
            self._api.SDL_QuitSubSystem(self._api.SDL_INIT_JOYSTICK)
        except Exception as exc:
            logger.warning("SDL joystick shutdown failed: %s", type(exc).__name__)

    @staticmethod
    def _count(value):
        if not 0 <= value <= 512:
            raise ValueError("Invalid SDL control count")
        return value

    def read(self, device_id: str) -> dict | None:
        if not self._initialized or self._closed:
            return None
        api = self._api
        try:
            api.SDL_JoystickUpdate()
            if self._clock() >= self._next_scan:
                self._scan()
            if self._handle is not None and not api.SDL_JoystickGetAttached(self._handle):
                self._close_handle()
                self._scan()
            matches = self._devices.get(device_id, [])
            if len(matches) != 1:
                self._close_handle()
                self.status = ("检测到同标识手柄，请暂时保留要使用的一只" if matches else
                               "正在等待所选手柄连接…")
                return None
            selected = matches[0]
            selection = (device_id, selected["instance"])
            if selection != self._selection:
                self._close_handle()
                # Recheck the index before opening: hotplug can reorder it.
                if api.SDL_JoystickGetDeviceInstanceID(selected["index"]) != selected["instance"]:
                    self._next_scan = 0.0
                    self.status = "手柄连接正在更新…"
                    return None
                handle = api.SDL_JoystickOpen(selected["index"])
                if not handle:
                    self.status = "所选手柄暂时无法打开，正在等待重试"
                    return None
                self._handle, self._selection = handle, selection
                api.SDL_JoystickUpdate()
            handle = self._handle
            if not api.SDL_JoystickGetAttached(handle):
                self._close_handle()
                self._next_scan = 0.0
                self.status = "手柄已断开，等待重新连接"
                return None
            if api.SDL_JoystickInstanceID(handle) != selected["instance"]:
                self._close_handle()
                self._next_scan = 0.0
                self.status = "手柄连接正在更新…"
                return None
            state = {
                "buttons": tuple(bool(api.SDL_JoystickGetButton(handle, i))
                                 for i in range(self._count(api.SDL_JoystickNumButtons(handle)))),
                "hats": tuple(int(api.SDL_JoystickGetHat(handle, i))
                              for i in range(self._count(api.SDL_JoystickNumHats(handle)))),
                "axes": tuple(max(-1.0, min(1.0, api.SDL_JoystickGetAxis(handle, i) / 32767.0))
                              for i in range(self._count(api.SDL_JoystickNumAxes(handle)))),
                "connection": selected["instance"],
            }
            self.status = "已连接 · " + selected["name"]
            return state
        except Exception as exc:
            logger.warning("SDL joystick read failed: %s", type(exc).__name__)
            self._close_handle()
            self._next_scan = 0.0
            self.status = "手柄读取失败，正在等待重新连接"
            return None

    def release(self):
        """Release the selected handle while keeping enumeration available."""
        self._close_handle()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._initialized:
            try:
                self._close_handle()
            finally:
                self._quit_subsystem()
                self._initialized = False
        self._devices = {}

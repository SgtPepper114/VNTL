"""Offline native-runtime check for source and PyInstaller distributions."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def check_ocr(arguments: list[str]) -> int:
    """Usage: vntl --check-ocr IMAGE OUTPUT.json. Uses no game state or API."""
    if len(arguments) != 2:
        return 2
    image_path, output_path = map(Path, arguments)
    result = {"ok": False}
    try:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        from PIL import Image
        from local_ocr import MeikiReader
        app = QApplication.instance() or QApplication([])
        image = Image.open(image_path).convert("RGB")
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as executor:
            found = executor.submit(MeikiReader().recognize, image).result(timeout=60)
        result.update(ok=True, characters=len(found.text), confidence=found.confidence,
                      elapsed_ms=(time.perf_counter() - started) * 1000,
                      qt_platform=app.platformName(), provider="CPUExecutionProvider")
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1


def _check_controller_worker() -> dict:
    """Exercise a process-local SDL virtual pad on its owning worker thread."""
    import threading
    from controller_input import BindingLearner, BindingTrigger
    from controller_sdl import SDLReader

    started = time.perf_counter()
    reader = SDLReader()
    api = reader._api
    virtual_index = -1

    def require(success):
        if not success:
            raise RuntimeError("Controller diagnostic check failed")

    def attach_virtual():
        index = api.SDL_JoystickAttachVirtual(api.SDL_JOYSTICK_TYPE_GAMECONTROLLER, 2, 20, 1)
        require(index >= 0)
        return index

    def set_controls(handle, *, button=0, hat=0, axis=0):
        require(api.SDL_JoystickSetVirtualButton(handle, 19, button) == 0)
        require(api.SDL_JoystickSetVirtualHat(handle, 0, hat) == 0)
        require(api.SDL_JoystickSetVirtualAxis(handle, 0, axis) == 0)

    try:
        require(reader._initialized)
        checks = {
            "worker_thread": threading.current_thread() is not threading.main_thread(),
            "video_uninitialized": not bool(api.SDL_WasInit(api.SDL_INIT_VIDEO)),
            "background_input": api.SDL_GetHint(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS") == b"1",
        }
        virtual_index = attach_virtual()
        identity = reader._identity(api.SDL_JoystickGetDeviceGUID(virtual_index),
                                    api.SDL_JoystickPathForIndex(virtual_index))
        reader.list_devices()  # Enumeration opens no real controller handles.
        initial = reader.read(identity)
        require(initial is not None)
        handle = api.SDL_JoystickFromInstanceID(initial["connection"])
        require(bool(handle) and api.SDL_JoystickIsVirtual(virtual_index))
        device = {"id": identity, "name": "VNTL diagnostic virtual controller", "backend": "sdl"}
        learning = {}
        trigger_checks = {}
        for kind, control, index, direction in (
            ("button", {"button": 1}, 19, 0),
            ("hat", {"hat": api.SDL_HAT_RIGHT}, 0, api.SDL_HAT_RIGHT),
            ("axis", {"axis": 32767}, 0, 1),
        ):
            set_controls(handle)
            neutral = reader.read(identity)
            learner = BindingLearner(device, include_axes=kind == "axis", started=0.0)
            learner.update(neutral, 0.0)
            learner.update(neutral, 0.31)
            set_controls(handle, **control)
            active = reader.read(identity)
            learner.update(active, 0.4)
            set_controls(handle)
            neutral = reader.read(identity)
            binding = learner.update(neutral, 0.5)
            require(binding is not None)
            learning[kind] = (binding["kind"] == kind and binding["index"] == index
                              and binding["direction"] == direction)
            if kind == "button":
                checks["button_20"] = active["buttons"][19] is True
            elif kind == "hat":
                checks["hat_right"] = active["hats"][0] == api.SDL_HAT_RIGHT
            else:
                checks["axis_positive"] = active["axes"][0] == 1.0
            trigger = BindingTrigger()
            events = [trigger.update(binding, neutral, 1.0)]
            set_controls(handle, **control)
            events.append(trigger.update(binding, reader.read(identity), 1.1))
            events.append(trigger.update(binding, reader.read(identity), 1.9))
            set_controls(handle)
            events.append(trigger.update(binding, reader.read(identity), 2.0))
            set_controls(handle, **control)
            events.append(trigger.update(binding, reader.read(identity), 2.1))
            trigger_checks[kind] = events == [False, True, False, False, True]

        require(api.SDL_JoystickDetachVirtual(virtual_index) == 0)
        virtual_index = -1
        missing = reader.read(identity)
        checks["detach_returns_none"] = missing is None
        checks["detach_resets_trigger"] = not trigger.update(binding, missing, 3.0)

        virtual_index = attach_virtual()
        reader.list_devices()
        reconnected = reader.read(identity)
        require(reconnected is not None)
        checks["reconnect_new_connection"] = reconnected["connection"] != initial["connection"]
        handle = api.SDL_JoystickFromInstanceID(reconnected["connection"])
        require(bool(handle) and api.SDL_JoystickIsVirtual(virtual_index))
        set_controls(handle, axis=32767)
        checks["reconnect_hold_suppressed"] = not trigger.update(binding, reader.read(identity), 3.1)
        set_controls(handle)
        trigger.update(binding, reader.read(identity), 3.2)
        set_controls(handle, axis=32767)
        checks["reconnect_fresh_press"] = trigger.update(binding, reader.read(identity), 3.3)
        return {"ok": all(checks.values()) and all(learning.values()) and all(trigger_checks.values()),
                "checks": checks, "learning": learning, "single_press": trigger_checks,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "scope": "process-local SDL virtual controller"}
    finally:
        try:
            if virtual_index >= 0:
                api.SDL_JoystickDetachVirtual(virtual_index)
        finally:
            reader.close()


def check_controller(arguments: list[str]) -> int:
    """Usage: vntl --check-controller OUTPUT.json. Uses no user state or API."""
    if len(arguments) != 1:
        return 2
    output_path = Path(arguments[0])
    result = {"ok": False}
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vntl-controller-check") as executor:
            result = executor.submit(_check_controller_worker).result(timeout=15)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1

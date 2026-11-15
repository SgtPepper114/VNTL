"""Serializable controller bindings, shared by configuration, UI and readers."""
from __future__ import annotations


def legacy_binding(button="GR") -> dict:
    return {"backend": "switch2", "device": "switch2:auto",
            "name": "Switch 2 Pro USB", "kind": "button",
            "index": 1 if button == "GL" else 0, "direction": 0}


def normalize_binding(value) -> dict:
    if not isinstance(value, dict):
        return {}
    backend, device, kind = (value.get(key) for key in ("backend", "device", "kind"))
    index, direction = value.get("index"), value.get("direction", 0)
    if (backend not in ("sdl", "switch2") or not isinstance(device, str)
            or not device.startswith(backend + ":") or len(device) > 512
            or type(index) is not int or not 0 <= index <= 255
            or type(direction) is not int):
        return {}
    if kind == "button":
        if direction != 0 or (backend == "switch2" and index > 1):
            return {}
    elif backend == "sdl" and kind == "hat":
        if direction not in (1, 2, 4, 8):
            return {}
    elif backend == "sdl" and kind == "axis":
        if direction not in (-1, 1):
            return {}
    else:
        return {}
    name = value.get("name", "手柄")
    if not isinstance(name, str):
        name = "手柄"
    result = {"backend": backend, "device": device, "name": name[:160],
              "kind": kind, "index": index, "direction": direction}
    if kind == "axis":
        rest = value.get("rest", 0.0)
        if type(rest) not in (int, float) or not -1.0 <= rest <= 1.0:
            return {}
        result["rest"] = float(rest)
    return result


def binding_label(value) -> str:
    binding = normalize_binding(value)
    if not binding:
        return "待录入翻译按键"
    index, kind = binding["index"], binding["kind"]
    if binding["backend"] == "switch2":
        control = "GR · 右背键" if index == 0 else "GL · 左背键"
    elif kind == "button":
        control = f"按钮 {index + 1}"
    elif kind == "hat":
        control = f"方向键 {index + 1} · " + {1: "上", 2: "右", 4: "下", 8: "左"}[binding["direction"]]
    else:
        control = f"轴 {index + 1} · {'正向' if binding['direction'] > 0 else '负向'}"
    return f"{binding['name']} / {control}"

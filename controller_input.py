"""Device-independent press edges and deliberate single-control learning."""
from __future__ import annotations

from controller_binding import normalize_binding


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


def control_down(binding, state, was_down=False) -> bool:
    index, kind = binding["index"], binding["kind"]
    values = state.get({"button": "buttons", "hat": "hats", "axis": "axes"}[kind], ())
    if index >= len(values):
        return False
    if kind == "button":
        return bool(values[index])
    if kind == "hat":
        return bool(values[index] & binding["direction"])
    distance = (values[index] - binding["rest"]) * binding["direction"]
    return distance >= (0.30 if was_down else 0.65)


class BindingTrigger:
    def __init__(self):
        self.reset()

    def reset(self):
        self._edge = PressEdge()
        self._identity = None
        self._down = False

    def update(self, binding, state, now):
        if state is None:
            self.reset()
            return False
        identity = (binding, state.get("connection"))
        if identity != self._identity:
            self.reset()
            self._identity = identity
        self._down = control_down(binding, state, self._down)
        return self._edge.update(self._down, now)


class BindingLearner:
    """Settle at rest, then accept one press-and-release cycle within 20s."""
    def __init__(self, device, include_axes, started):
        self.device = device
        self.include_axes = include_axes
        self.started = started
        self.message = "请先松开所有按键，让摇杆和扳机静止。"
        self.expired = False
        self._connection = None
        self._rest = None
        self._quiet_since = None
        self._armed = False
        self._candidate = None
        self._candidate_at = None

    def update(self, state, now):
        if now - self.started >= 20:
            self.expired = True
            self.message = "录入已超时，请点击录入按键重试。"
            return None
        if state is None:
            self._connection = None
            self._rest = self._quiet_since = self._candidate = None
            self._armed = False
            self.message = "正在等待所选手柄连接…"
            return None
        if state.get("connection") != self._connection:
            self._connection = state.get("connection")
            self._rest = self._quiet_since = self._candidate = None
            self._armed = False
        buttons, hats, axes = (state.get(key, ()) for key in ("buttons", "hats", "axes"))
        if not self._armed:
            if any(buttons) or any(hats):
                self._quiet_since = None
                self.message = "请先松开所有按键。"
                return None
            if (self._quiet_since is None or self._rest is None
                    or (self.include_axes and (len(axes) != len(self._rest)
                        or any(abs(a - b) > 0.12 for a, b in zip(axes, self._rest))))):
                self._rest, self._quiet_since = tuple(axes), now
                return None
            if now - self._quiet_since >= 0.3:
                self._armed = True
                self.message = "请单独按下要绑定的按键，然后松开。"
            return None
        base = {"backend": self.device["backend"], "device": self.device["id"],
                "name": self.device["name"]}
        candidates = [dict(base, kind="button", index=i, direction=0)
                      for i, down in enumerate(buttons) if down]
        candidates += [dict(base, kind="hat", index=i, direction=mask)
                       for i, value in enumerate(hats) for mask in (1, 2, 4, 8) if value & mask]
        if self.include_axes:
            candidates += [dict(base, kind="axis", index=i, direction=1 if value > rest else -1, rest=rest)
                           for i, (value, rest) in enumerate(zip(axes, self._rest))
                           if abs(value - rest) >= 0.65]
        if self._candidate:
            others = [c for c in candidates if c != self._candidate]
            if others:
                self._candidate = None
                self._armed = False
                self._quiet_since = None
                self.message = "检测到多个输入，请松开后单独操作一个按键。"
            elif not control_down(self._candidate, state, True):
                if self._candidate["kind"] != "axis" or now - self._candidate_at >= 0.03:
                    self.message = "录入完成，点击保存后生效。"
                    return self._candidate
                self._candidate = None  # Reject a sub-30ms analog transient.
            return None
        if len(candidates) > 1:
            self._armed = False
            self._quiet_since = None
            self.message = "检测到多个输入，请松开后单独操作一个按键。"
        elif candidates:
            self._candidate = normalize_binding(candidates[0]) or None
            self._candidate_at = now
            self.message = "已检测到按键，请松开完成录入。"
        return None

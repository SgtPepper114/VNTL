from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from typing import Optional

logger = logging.getLogger("vntl.screenshot")

try:
    import mss
    from PIL import Image
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.warning(
        "mss/Pillow not installed — screenshot context disabled. "
        "Run: uv pip install mss Pillow"
    )

_COMPARE_SIZE          = (160, 90)  # thumbnail size for change detection (16:9)
_SEND_WIDTH            = 800        # max width for scene descriptions (px)
_TEXT_MAX_SIZE         = (1920, 1440)
_PIXEL_SENSITIVITY     = 10         # per-pixel noise floor for change counting (0–255)
_LAST_SCREENSHOT_PATH  = os.path.expanduser("~/.config/vntl/last_screenshot.jpg")

# ---------------------------------------------------------------------------
# Windows helpers: find the main window of a given PID
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as wt

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    # Explicit pointer-sized HWND signatures are required on 64-bit Windows.
    _user32.EnumWindows.argtypes = [_WNDENUMPROC, wt.LPARAM]
    _user32.EnumWindows.restype = wt.BOOL
    _user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wt.DWORD
    for _name in ("IsWindow", "IsWindowVisible", "IsIconic"):
        getattr(_user32, _name).argtypes = [wt.HWND]
        getattr(_user32, _name).restype = wt.BOOL
    _user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
    _user32.GetClientRect.restype = wt.BOOL
    _user32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(wt.POINT)]
    _user32.ClientToScreen.restype = wt.BOOL
    _user32.GetForegroundWindow.restype = wt.HWND
    _user32.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]
    _user32.SetWindowDisplayAffinity.restype = wt.BOOL
    _user32.GetWindowDisplayAffinity.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    _user32.GetWindowDisplayAffinity.restype = wt.BOOL

    def _find_game_window(pid: int) -> Optional[int]:
        """Return the HWND of the largest visible top-level window owned by pid."""
        found: list[tuple[int, int]] = []

        def cb(hwnd: int, _lparam: int) -> bool:
            lp = wt.DWORD(0)
            _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(lp))
            if lp.value == pid and _user32.IsWindowVisible(hwnd):
                r = wt.RECT()
                if _user32.GetClientRect(hwnd, ctypes.byref(r)):
                    width, height = r.right - r.left, r.bottom - r.top
                    if width > 0 and height > 0:
                        found.append((width * height, hwnd))
            return True

        if not _user32.EnumWindows(_WNDENUMPROC(cb), 0):
            return None
        return max(found)[1] if found else None

    def _window_is_minimized(hwnd: int) -> bool:
        return bool(_user32.IsIconic(hwnd))

    def _client_screen_rect(hwnd: int) -> Optional[tuple[int, int, int, int]]:
        """Return (left, top, width, height) of the client area in screen coords."""
        if not _user32.IsWindow(hwnd) or not _user32.IsWindowVisible(hwnd):
            return None
        r = wt.RECT()
        if not _user32.GetClientRect(hwnd, ctypes.byref(r)):
            return None
        pt = wt.POINT(0, 0)
        if not _user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None
        w = r.right - r.left
        h = r.bottom - r.top
        return (pt.x, pt.y, w, h) if w > 0 and h > 0 else None

else:
    def _find_game_window(pid: int) -> Optional[int]:  # type: ignore[misc]
        return None

    def _client_screen_rect(hwnd: int) -> Optional[tuple[int, int, int, int]]:  # type: ignore[misc]
        return None

    def _window_is_minimized(hwnd: int) -> bool:  # type: ignore[misc]
        return False


# ---------------------------------------------------------------------------
# ScreenshotService
# ---------------------------------------------------------------------------

class ScreenshotService:
    """
    Captures screenshots and detects scene changes via grayscale thumbnail diff.

    Call capture() each time a new dialogue line arrives.  Returns a
    base64-encoded JPEG when the scene has changed since the last sent
    screenshot, or None when the scene is unchanged (or libraries unavailable).

    Call set_pid(pid) to specify which process window to capture. The feature
    is inactive until a PID is set (is_attached == False).
    """

    def __init__(self) -> None:
        self._last_thumb: Optional[bytes] = None
        self._pid: Optional[int] = None
        self.threshold: float = 15.0
        self.cooldown: float = 1.0
        self._last_trigger_time: float = 0.0
        self.last_pct: float = 0.0
        self.last_triggered: bool = False
        self.generation = 0
        self.on_target_changed = None

    @property
    def is_attached(self) -> bool:
        return self._pid is not None

    def set_pid(self, pid: Optional[int]) -> None:
        if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0):
            raise ValueError("游戏进程 ID 必须是正整数。")
        changed = pid != self._pid
        if changed:
            self.generation += 1
            self._last_thumb = None
            self._last_trigger_time = 0.0
            self.last_pct = 0.0
            self.last_triggered = False
        self._pid = pid
        if changed and self.on_target_changed is not None:
            self.on_target_changed()

    @staticmethod
    def exclude_window(hwnd: int) -> bool:
        """Keep an owned VNTL window visible but out of desktop captures.

        Windows 10 2004+ supplies the pixels behind an excluded window. Older
        versions only produce black rectangles, so continuous capture refuses
        that path. Call on the Qt thread for each visible top-level window.
        """
        if sys.platform != "win32" or sys.getwindowsversion().build < 19041:
            return False
        current = wt.DWORD()
        if _user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(current)) and current.value == 0x11:
            return True
        return bool(_user32.SetWindowDisplayAffinity(hwnd, 0x11))

    @staticmethod
    def exclusion_error_text() -> str:
        if sys.platform != "win32":
            return "持续观察的窗口截图需要 Windows。"
        if sys.getwindowsversion().build < 19041:
            return "当前 Windows 版本需要更新到 Windows 10 2004 或更高版本以支持持续截图。"
        return (f"浮窗无法从截图中排除（Windows 错误 {ctypes.get_last_error()}）。"
                "请重启更新后的 VNTL；也可暂停观察后使用按键识屏。")

    def capture_frame(self):
        """Quiet continuous capture, paused while another application has focus.

        Uses the selected visible client area. Our windows must first be marked
        excluded on the GUI thread; this method never hides or moves a window.
        """
        if sys.platform == "win32":
            foreground = _user32.GetForegroundWindow()
            owner = wt.DWORD()
            _user32.GetWindowThreadProcessId(foreground, ctypes.byref(owner))
            # Our floating window may itself sit over another application. Only
            # a foreground game authorizes reading these screen pixels; owning
            # the foreground overlay does not prove the game is unobscured.
            if owner.value != self._pid:
                raise RuntimeError("观察已暂停：请将游戏切换到前台。")
        return self._capture_bound_image()

    @staticmethod
    def encode_frame(img) -> str:
        """Encode a frozen frame for visual context and uncertain OCR fallback."""
        img = img.copy()
        img.thumbnail(_TEXT_MAX_SIZE, Image.Resampling.LANCZOS)
        if img.width < 640 and img.height < 480:
            img = img.resize((img.width * 2, img.height * 2), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, subsampling=0)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _capture_bound_image(self):
        """Read only the selected game's visible client area, never the desktop.

        This uses screen pixels, so callers must hide their own overlay before
        capture and the player must keep the game visible and unobscured.
        """
        if not _AVAILABLE:
            raise RuntimeError("截图组件不可用，请重新安装项目依赖（mss 和 Pillow）。")
        if self._pid is None:
            raise RuntimeError("请先选择游戏或模拟器窗口，再读取屏幕。")
        hwnd = _find_game_window(self._pid)
        if hwnd is None:
            raise RuntimeError("找不到已选择的游戏窗口；游戏可能已关闭，请重新选择。")
        if _window_is_minimized(hwnd):
            raise RuntimeError("游戏窗口已最小化，请先恢复窗口并保持画面无遮挡。")
        bounds = _client_screen_rect(hwnd)
        if bounds is None:
            raise RuntimeError("无法读取游戏窗口区域，请恢复窗口后重新选择。")
        x, y, width, height = bounds
        if width <= 0 or height <= 0:
            raise RuntimeError("游戏窗口没有可读取的画面，请恢复窗口后重试。")
        region = {"left": x, "top": y, "width": width, "height": height}
        try:
            with mss.mss() as sct:
                raw = sct.grab(region)
                return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        except Exception as exc:
            logger.warning("Bound game capture failed: %s", type(exc).__name__)
            raise RuntimeError("游戏截图失败，请使用窗口模式并保持游戏画面可见后重试。") from exc

    def capture_for_text(self) -> str:
        """Return a detailed JPEG for OCR, or raise a user-readable error.

        Unlike scene thumbnails, text capture retains native detail up to
        1920x1440. Small PS1-size images are enlarged with nearest-neighbour
        sampling to preserve pixel glyphs. This method does not change scene
        detection state and does not persist the captured image to disk.
        """
        img = self._capture_bound_image()
        width, height = img.size
        scale = min(_TEXT_MAX_SIZE[0] / width, _TEXT_MAX_SIZE[1] / height)
        if width < 640 and height < 480:
            scale = min(scale, 2.0)
        else:
            scale = min(scale, 1.0)
        if scale != 1.0:
            img = img.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, subsampling=0)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def capture(self, force: bool = False) -> Optional[str]:
        """
        Take a screenshot of the bound game window only. Returns base64 JPEG
        if the scene changed, or None if unchanged/unavailable. An absent,
        closed, or minimized target never falls back to capturing a monitor.
        Pass force=True to bypass the scene-change threshold (used to retry a
        failed describe_scene call on the next dialogue line).
        """
        self.last_triggered = False
        try:
            img = self._capture_bound_image()
        except RuntimeError as exc:
            logger.warning("Scene screenshot unavailable: %s", exc)
            return None

        thumb = img.convert("L").resize(_COMPARE_SIZE).tobytes()
        pct = self._diff_pct(thumb)
        self._last_thumb = thumb  # always advance baseline — compare consecutive frames
        self.last_pct = pct
        now = time.monotonic()
        self.last_triggered = (
            pct > self.threshold
            and (now - self._last_trigger_time) >= self.cooldown
        )
        if not self.last_triggered and not force:
            return None

        w, h = img.size
        if w > _SEND_WIDTH:
            img = img.resize((_SEND_WIDTH, int(h * _SEND_WIDTH / w)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)

        os.makedirs(os.path.dirname(_LAST_SCREENSHOT_PATH), exist_ok=True)
        with open(_LAST_SCREENSHOT_PATH, "wb") as f:
            f.write(buf.getvalue())
        logger.debug(
            "Screenshot saved to %s (%d bytes).",
            _LAST_SCREENSHOT_PATH,
            buf.tell(),
        )
        return base64.b64encode(buf.getvalue()).decode()

    def reset_cooldown(self) -> None:
        """Reset the cooldown timer. Call after each line is fully processed."""
        self._last_trigger_time = time.monotonic()

    def _diff_pct(self, new_thumb: bytes) -> float:
        """Return % of pixels that changed by more than _PIXEL_SENSITIVITY."""
        if self._last_thumb is None or len(new_thumb) != len(self._last_thumb):
            return 100.0  # first frame always triggers
        changed = sum(1 for a, b in zip(new_thumb, self._last_thumb)
                      if abs(a - b) > _PIXEL_SENSITIVITY)
        return changed * 100.0 / len(new_thumb)

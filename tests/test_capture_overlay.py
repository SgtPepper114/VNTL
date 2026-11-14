"""Separate background alpha preserves text and Windows capture exclusion.

Unit tests use Qt's offscreen platform. The explicit --native-probe command
creates only our own nonactivating test windows and checks real MSS pixels.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

if "--native-probe" in sys.argv:
    os.environ["QT_QPA_PLATFORM"] = "windows"
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from runtime_bootstrap import preload_windows_runtime
preload_windows_runtime()
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QApplication, QWidget

import overlay
from config import Config
from overlay import OverlayWindow
from screenshot_service import ScreenshotService
from background_layer import FOREGROUND_KEY


def make_overlay(cfg):
    noop = lambda *args, **kwargs: None
    return OverlayWindow(cfg, noop, noop, noop, noop, noop, None, ScreenshotService())


class CaptureOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def window(self, native):
        cfg = Config()
        cfg.overlay.bg_color = [23, 27, 31, 173]
        cfg.overlay.jp_furigana_enabled = False
        with patch.object(overlay, "_NATIVE_WINDOW_OPACITY", native):
            window = make_overlay(cfg)
        self.addCleanup(window.close)
        return window, cfg

    def test_windows_opacity_applies_only_to_background_without_rewriting_config(self):
        window, cfg = self.window(True)
        self.assertFalse(window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))
        self.assertFalse(window.testAttribute(Qt.WidgetAttribute.WA_NoSystemBackground))
        self.assertEqual(window.windowOpacity(), 1.0)
        self.assertAlmostEqual(window._background_layer.windowOpacity(), 173 / 255, places=2)
        self.assertEqual(cfg.overlay.bg_color, [23, 27, 31, 173])

    def test_windows_appearance_preview_and_cancel_restore_native_opacity(self):
        window, cfg = self.window(True)
        before = copy.deepcopy(cfg.overlay)
        changed = copy.deepcopy(before)
        changed.bg_color = [30, 40, 50, 80]
        window.apply_appearance(changed)
        self.assertEqual(window.windowOpacity(), 1.0)
        self.assertAlmostEqual(window._background_layer.windowOpacity(), 80 / 255, places=2)
        self.assertEqual(cfg.overlay.bg_color, [30, 40, 50, 80])
        self.assertEqual(changed.bg_color, [30, 40, 50, 80])
        window.apply_appearance(before)
        self.assertEqual(window.windowOpacity(), 1.0)
        self.assertAlmostEqual(window._background_layer.windowOpacity(), 173 / 255, places=2)
        self.assertEqual(cfg.overlay.bg_color, before.bg_color)

    def test_windows_foreground_uses_opaque_keyed_matte(self):
        window, cfg = self.window(True)
        image = QImage(window.size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        window.render(image)
        for point in ((0, 40), (1, 50)):
            with self.subTest(point=point):
                color = image.pixelColor(*point)
                self.assertEqual(color.getRgb(), (*FOREGROUND_KEY, 255))
        self.assertEqual(cfg.overlay.bg_color[3], 173)
        self.assertEqual(image.pixelColor(0, 0).getRgb(), (34, 34, 45, 255))

    def test_zero_background_keeps_foreground_and_controls_visible(self):
        window, cfg = self.window(True)
        window.show()
        appearance = copy.deepcopy(cfg.overlay)
        appearance.bg_color[3] = 0
        before = (window._jp_label.styleSheet(), window._en_label.styleSheet())
        window.apply_appearance(appearance)
        self.assertEqual(window.windowOpacity(), 1.0)
        self.assertEqual(window._background_layer.windowOpacity(), 0.0)
        self.assertTrue(window._gear_btn.isVisible())
        self.assertEqual(before, (window._jp_label.styleSheet(), window._en_label.styleSheet()))

    def test_background_follows_visibility_geometry_and_never_takes_focus(self):
        window, cfg = self.window(True)
        window.show()
        window.setGeometry(40, 50, 540, 260)
        self.app.processEvents()
        layer = window._background_layer
        self.assertEqual(layer.geometry(), window.geometry())
        self.assertTrue(layer.isVisible())
        self.assertTrue(layer.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus)
        self.assertFalse(layer.windowFlags() & Qt.WindowType.WindowTransparentForInput)
        window.hide()
        self.assertFalse(layer.isVisible())
        window.show()
        self.assertTrue(layer.isVisible())
        window.close()
        self.assertFalse(layer.isVisible())

    def test_layer_attribute_failure_restores_opaque_style_and_stops_retrying(self):
        window, cfg = self.window(True)
        layer = window._background_layer
        layer._key_attempted = layer._key_ready = False
        api = Mock()
        api.GetWindowLongPtrW.return_value = 0x80
        api.SetLayeredWindowAttributes.return_value = 0
        layer._api = api
        self.assertFalse(layer.enable_foreground())
        self.assertEqual(api.SetWindowLongPtrW.call_args.args[-1], 0x80)
        self.assertFalse(layer.enable_foreground())
        api.SetLayeredWindowAttributes.assert_called_once()
        self.assertEqual(window.windowOpacity(), 1.0)

    def test_visible_backdrop_forwards_drag_and_context_input_to_overlay(self):
        window, cfg = self.window(True)
        event = Mock()
        for method in ("mousePressEvent", "mouseMoveEvent", "mouseReleaseEvent"):
            with patch.object(window, method) as handler:
                getattr(window._background_layer, method)(event)
                handler.assert_called_once_with(event)

    def test_other_platforms_keep_per_pixel_alpha_and_no_window_opacity(self):
        window, cfg = self.window(False)
        self.assertTrue(window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))
        self.assertEqual(window.windowOpacity(), 1.0)
        appearance = copy.deepcopy(cfg.overlay)
        appearance.bg_color[3] = 80
        window.apply_appearance(appearance)
        self.assertEqual(window.windowOpacity(), 1.0)
        self.assertEqual(cfg.overlay.bg_color[3], 80)


def native_capture_probe(destination: Path) -> dict:
    """Compare desktop pixels of our own background before/after exclusion."""
    if sys.platform != "win32":
        raise RuntimeError("The native capture probe is Windows-only")
    import asyncio
    import base64
    import ctypes
    import ctypes.wintypes as wt
    import io
    from types import SimpleNamespace
    import mss
    from PIL import Image
    import numpy as np
    from PyQt6.QtCore import QEvent, QObject
    from PyQt6.QtTest import QTest
    import main
    import screenshot_service
    from screenshot_service import _client_screen_rect

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = wt.HWND
    user32.GetWindowDisplayAffinity.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    user32.GetWindowDisplayAffinity.restype = wt.BOOL
    user32.IsWindowVisible.argtypes = [wt.HWND]
    user32.IsWindowVisible.restype = wt.BOOL
    original_foreground = int(user32.GetForegroundWindow() or 0)

    class Background(QWidget):
        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(29, 163, 83))
            painter.end()

    background = Background()
    background.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
    background.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    cfg = Config()
    cfg.overlay.jp_furigana_enabled = False
    cfg.overlay.bg_color = [20, 20, 25, 210]
    available = app.primaryScreen().availableGeometry()
    cfg.overlay.x, cfg.overlay.y = available.x() + 8, available.y() + 8
    cfg.overlay.width, cfg.overlay.height = 500, 200
    foreground_checks = []
    floating = None

    def settle():
        QTest.qWait(120)
        ctypes.WinDLL("dwmapi").DwmFlush()
        foreground_checks.append(original_foreground == int(user32.GetForegroundWindow() or 0))

    def capture(bounds):
        x, y, width, height = bounds
        with mss.mss() as screen:
            shot = screen.grab({"left": x, "top": y, "width": width, "height": height})
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def manual_capture_probe(baseline):
        """Exercise the production manual path without a post-exclusion wait."""
        windows = {"foreground": floating, "backdrop": floating._background_layer}
        events = {name: {"show": 0, "hide": 0} for name in windows}

        class VisibilityEvents(QObject):
            def eventFilter(self, watched, event):
                if event.type() in (QEvent.Type.Show, QEvent.Type.Hide):
                    name = next(name for name, window in windows.items() if window is watched)
                    events[name]["show" if event.type() == QEvent.Type.Show else "hide"] += 1
                return False

        watcher = VisibilityEvents()
        for window in windows.values():
            window.installEventFilter(watcher)
        initial_affinity = {}
        for name, window in windows.items():
            value = wt.DWORD()
            read = bool(user32.GetWindowDisplayAffinity(int(window.winId()), ctypes.byref(value)))
            initial_affinity[name] = {"read": read, "value": hex(value.value)}

        service = ScreenshotService()
        service.set_pid(os.getpid())
        facade = SimpleNamespace(activeModalWidget=lambda: None,
                                 topLevelWidgets=lambda: list(windows.values()))
        rows, frames = [], []

        async def capture_repeatedly():
            for attempt in range(5):
                # No sleep, processEvents or DwmFlush between the first affinity
                # update and MSS: capture_game controls the complete capture path.
                payload = await main.capture_game(service)
                jpeg = base64.b64decode(payload)
                frame = Image.open(io.BytesIO(jpeg)).convert("RGB")
                expected = baseline.resize(frame.size, Image.Resampling.NEAREST)
                error = np.abs(np.asarray(frame).astype(np.int16)
                               - np.asarray(expected).astype(np.int16))
                app.processEvents()
                foreground_unchanged = original_foreground == int(user32.GetForegroundWindow() or 0)
                foreground_checks.append(foreground_unchanged)
                rows.append({
                    "attempt": attempt + 1,
                    "max_channel_error": int(error.max()),
                    "pixels_outside_jpeg_tolerance": int(np.any(error > 3, axis=2).sum()),
                    "both_layers_visible": all(window.isVisible()
                                               and user32.IsWindowVisible(int(window.winId()))
                                               for window in windows.values()),
                    "foreground_unchanged": foreground_unchanged,
                })
                frames.append(("manual-first" if attempt == 0 else f"manual-repeat-{attempt}", jpeg))

        try:
            # Only our two overlay surfaces are excluded. Bind the real capture
            # service to our own synthetic background HWND, never a user game.
            with patch.object(main, "QApplication", facade), \
                    patch.object(screenshot_service, "_find_game_window", return_value=int(background.winId())), \
                    patch.object(service, "exclude_window", wraps=service.exclude_window) as exclude:
                asyncio.run(capture_repeatedly())
            expected_handles = [int(window.winId()) for window in windows.values()] * len(rows)
            result = {
                "initial_affinity": initial_affinity,
                "first_capture_started_unexcluded": all(row["read"] and row["value"] == "0x0"
                                                        for row in initial_affinity.values()),
                "both_layers_excluded_each_time": [call.args[0] for call in exclude.call_args_list] == expected_handles,
                "visibility_events": events,
                "no_show_or_hide_events": all(count == 0 for row in events.values() for count in row.values()),
                "captures": rows,
                "all_captures_match_background": all(row["pixels_outside_jpeg_tolerance"] == 0 for row in rows),
                "both_layers_remained_visible": all(row["both_layers_visible"] for row in rows),
                "foreground_unchanged": all(row["foreground_unchanged"] for row in rows),
                "notes": "Real capture_game/capture_for_text/MSS/JPEG; no post-exclusion settling; synthetic target only.",
            }
            result["passed"] = all(result[key] for key in (
                "first_capture_started_unexcluded", "both_layers_excluded_each_time",
                "no_show_or_hide_events", "all_captures_match_background",
                "both_layers_remained_visible", "foreground_unchanged"))
            return result, frames
        finally:
            for window in windows.values():
                window.removeEventFilter(watcher)

    try:
        background.setGeometry(cfg.overlay.x, cfg.overlay.y, cfg.overlay.width, cfg.overlay.height)
        background.show()
        settle()
        bounds = _client_screen_rect(int(background.winId()))
        baseline = capture(bounds)
        floating = make_overlay(cfg)
        floating.update_text("漢字と仮名", "VISIBLE TRANSLATION")
        floating.show()
        settle()
        alpha_frames = []
        for alpha in (0, 128, 255):
            appearance = copy.deepcopy(cfg.overlay)
            appearance.bg_color[3] = alpha
            floating.apply_appearance(appearance)
            settle()
            alpha_frames.append(capture(bounds))
        glyphs = np.asarray(alpha_frames[0]).min(axis=2) >= 220
        stable_glyphs = all(np.array_equal(np.asarray(frame)[glyphs], np.asarray(alpha_frames[0])[glyphs])
                           for frame in alpha_frames[1:])
        foreground_pixels = np.any(np.asarray(alpha_frames[0]) != np.asarray(baseline), axis=2)
        stable_foreground = all(np.array_equal(np.asarray(frame)[foreground_pixels],
                                               np.asarray(alpha_frames[0])[foreground_pixels])
                                for frame in alpha_frames[1:])
        appearance.bg_color[3] = 210
        floating.apply_appearance(appearance)
        settle()
        with_overlay = capture(bounds)
        manual_result, manual_frames = manual_capture_probe(baseline)
        hwnd = int(floating.winId())
        excluded = all(ScreenshotService.exclude_window(int(window.winId()))
                       for window in (floating, floating._background_layer))
        affinity = wt.DWORD()
        read_affinity = bool(user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)))
        settle()
        after_exclusion = capture(bounds)
        # The floating window stays shown throughout both captures.
        still_visible = bool(user32.IsWindowVisible(hwnd))
        points = [(round(baseline.width * x), round(baseline.height * y))
                  for x, y in ((0.1, 0.5), (0.5, 0.5), (0.9, 0.8))]
        samples = [{"point": list(point), "background": baseline.getpixel(point),
                    "covered": with_overlay.getpixel(point), "excluded": after_exclusion.getpixel(point)} for point in points]
        result = {
            "excluded": excluded, "affinity_read": read_affinity, "affinity": hex(affinity.value),
            "overlay_still_visible": still_visible, "foreground_unchanged": all(foreground_checks),
            "configured_color_unchanged": cfg.overlay.bg_color == [20, 20, 25, 210],
            "window_opacity": floating.windowOpacity(), "samples": samples,
            "background_opacity": floating._background_layer.windowOpacity(),
            "opaque_glyph_pixels": int(glyphs.sum()),
            "glyphs_unchanged_across_background_alpha": stable_glyphs,
            "foreground_pixels": int(foreground_pixels.sum()),
            "all_foreground_unchanged_across_background_alpha": stable_foreground,
            "zero_background_pixel_matches_underlying": alpha_frames[0].getpixel((0, baseline.height // 2)) == baseline.getpixel((0, baseline.height // 2)),
            "opaque_background_pixel": alpha_frames[2].getpixel((0, baseline.height // 2)),
            "pixels_restore_background": all(max(abs(a-b) for a,b in zip(row["background"], row["excluded"])) <= 2 for row in samples),
            "overlay_was_visible_in_control": any(row["covered"] != row["background"] for row in samples),
            "manual_capture": manual_result,
            "notes": "Only our synthetic windows were captured; no game input/focus changes. Exclusion was applied without hiding the overlay.",
        }
        destination.mkdir(parents=True, exist_ok=True)
        baseline.save(destination / "background.png")
        with_overlay.save(destination / "overlay-visible.png")
        after_exclusion.save(destination / "overlay-excluded.png")
        for alpha, frame in zip((0, 128, 255), alpha_frames):
            frame.save(destination / f"background-alpha-{alpha}.png")
        for name, jpeg in manual_frames:
            (destination / f"{name}.jpg").write_bytes(jpeg)
        (destination / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        if floating is not None:
            floating.hide()
            floating.deleteLater()
        background.hide()
        background.deleteLater()
        app.processEvents()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--native-probe"]:
        result = native_capture_probe(Path(sys.argv[2]))
        print(json.dumps(result, ensure_ascii=False))
        passed = all(result[name] for name in ("excluded", "overlay_still_visible", "foreground_unchanged",
                                               "configured_color_unchanged", "pixels_restore_background", "overlay_was_visible_in_control",
                                               "glyphs_unchanged_across_background_alpha", "all_foreground_unchanged_across_background_alpha",
                                               "zero_background_pixel_matches_underlying"))
        passed = passed and result["opaque_glyph_pixels"] > 100
        passed = passed and result["manual_capture"]["passed"]
        raise SystemExit(0 if passed else 1)
    unittest.main()

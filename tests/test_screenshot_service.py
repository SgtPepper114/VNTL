import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

import screenshot_service as module
from screenshot_service import ScreenshotService


class ScreenshotServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ScreenshotService()
        self.service.set_pid(123)
        for name, value in (
            ("_AVAILABLE", True),
            ("_find_game_window", lambda pid: 456),
            ("_window_is_minimized", lambda hwnd: False),
            ("_client_screen_rect", lambda hwnd: (100, 200, 320, 240)),
        ):
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        mss_patch = patch.object(module.mss, "mss")
        self.mss_factory = mss_patch.start()
        self.addCleanup(mss_patch.stop)

    def fake_frame(self, width=320, height=240):
        raw = SimpleNamespace(
            size=(width, height), bgra=b"\x10\x40\x90\xff" * (width * height)
        )
        self.grab = MagicMock(return_value=raw)
        # No monitors member: a successful test cannot depend on a desktop fallback.
        self.mss_factory.return_value.__enter__.return_value = SimpleNamespace(grab=self.grab)

    @unittest.skipUnless(module.sys.platform == "win32", "Windows foreground check")
    def test_continuous_capture_pauses_when_our_overlay_covers_another_app(self):
        native = MagicMock()
        native.GetWindowThreadProcessId.side_effect = lambda hwnd, owner: setattr(owner._obj, "value", module.os.getpid())
        with patch.object(module, "_user32", native):
            with self.assertRaisesRegex(RuntimeError, "切换到前台"):
                self.service.capture_frame()
        self.mss_factory.assert_not_called()

    @unittest.skipUnless(module.sys.platform == "win32", "Windows foreground check")
    def test_continuous_capture_reads_only_when_game_has_focus(self):
        self.fake_frame()
        native = MagicMock()
        native.GetWindowThreadProcessId.side_effect = lambda hwnd, owner: setattr(owner._obj, "value", 123)
        with patch.object(module, "_user32", native):
            self.assertEqual(self.service.capture_frame().size, (320, 240))

    def decode_image(self, encoded):
        return Image.open(io.BytesIO(base64.b64decode(encoded)))

    def assert_capture_rejected(self, message):
        with self.assertRaisesRegex(RuntimeError, message):
            self.service.capture_for_text()
        self.assertIsNone(self.service.capture(force=True))
        self.mss_factory.assert_not_called()

    def test_unselected_target_never_reads_desktop(self):
        self.service.set_pid(None)
        self.assert_capture_rejected("先选择")

    def test_closed_target_never_reads_desktop(self):
        with patch.object(module, "_find_game_window", return_value=None):
            self.assert_capture_rejected("找不到")

    def test_minimized_target_is_rejected(self):
        with patch.object(module, "_window_is_minimized", return_value=True):
            self.assert_capture_rejected("最小化")

    def test_failed_coordinate_lookup_never_reads_desktop(self):
        with patch.object(module, "_client_screen_rect", return_value=None):
            self.assert_capture_rejected("无法读取")

    def test_empty_client_area_is_rejected(self):
        with patch.object(module, "_client_screen_rect", return_value=(0, 0, 0, 200)):
            self.assert_capture_rejected("没有可读取")

    def test_missing_dependencies_have_actionable_error(self):
        with patch.object(module, "_AVAILABLE", False):
            self.assert_capture_rejected("截图组件不可用")

    def test_capture_failure_does_not_retry_another_region(self):
        self.fake_frame()
        self.grab.side_effect = OSError("test capture error")
        with self.assertRaisesRegex(RuntimeError, "游戏截图失败"):
            self.service.capture_for_text()
        self.grab.assert_called_once_with(
            {"left": 100, "top": 200, "width": 320, "height": 240}
        )

    def test_ps1_frame_reads_exact_client_and_preserves_pixel_detail(self):
        self.fake_frame()
        encoded = self.service.capture_for_text()
        self.grab.assert_called_once_with(
            {"left": 100, "top": 200, "width": 320, "height": 240}
        )
        with self.decode_image(encoded) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(img.size, (640, 480))
        self.assertIsNone(self.service._last_thumb)

    def test_native_hd_frame_is_not_enlarged(self):
        self.fake_frame(1280, 720)
        with patch.object(module, "_client_screen_rect", return_value=(-1280, 0, 1280, 720)):
            with self.decode_image(self.service.capture_for_text()) as img:
                self.assertEqual(img.size, (1280, 720))
        self.grab.assert_called_once_with(
            {"left": -1280, "top": 0, "width": 1280, "height": 720}
        )

    def test_large_frame_is_capped_without_changing_aspect_ratio(self):
        self.fake_frame(2560, 1440)
        with patch.object(module, "_client_screen_rect", return_value=(0, 0, 2560, 1440)):
            with self.decode_image(self.service.capture_for_text()) as img:
                self.assertEqual(img.size, (1920, 1080))

    def test_portrait_frame_is_capped_by_height(self):
        self.fake_frame(1080, 1920)
        with patch.object(module, "_client_screen_rect", return_value=(0, 0, 1080, 1920)):
            with self.decode_image(self.service.capture_for_text()) as img:
                self.assertEqual(img.size, (810, 1440))

    def test_scene_detection_still_skips_unchanged_frames(self):
        self.fake_frame()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, "_LAST_SCREENSHOT_PATH", str(Path(directory) / "scene.jpg")):
                self.assertIsInstance(self.service.capture(), str)
                self.assertIsNone(self.service.capture())

    def test_changing_game_resets_comparison_state(self):
        self.service._last_thumb = b"previous game"
        self.service._last_trigger_time = 5
        self.service.last_triggered = True
        self.service.set_pid(999)
        self.assertIsNone(self.service._last_thumb)
        self.assertEqual(self.service._last_trigger_time, 0)
        self.assertFalse(self.service.last_triggered)


if __name__ == "__main__":
    unittest.main()

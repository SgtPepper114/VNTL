import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import main
from config import Config
from context_manager import ContextManager


class InputLoopTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, kind="screen", text="", *, empty=False, failure=False, retry=False):
        context = ContextManager()
        if retry:
            context.add_line("誰？", "谁？")
        finished = asyncio.Event()
        overlay = Mock()
        overlay.set_busy.side_effect = lambda busy: finished.set() if not busy else None
        translator = Mock()
        translator.last_compact_error = False
        translator.read_screen = AsyncMock(return_value=("" if empty else "誰？", "昏暗的走廊。"))

        async def translate(jp, *, replace_last=False):
            if failure:
                raise RuntimeError("测试失败")
            if replace_last:
                context.pop_last_line_if_matches(jp)
            context.add_line(jp, "是谁？")
            return "是谁？"

        translator.translate = AsyncMock(side_effect=translate)
        cfg = Config(deepseek_api_key="test-placeholder")
        hooker = SimpleNamespace(is_attached=False)
        requests = asyncio.Queue(maxsize=1)
        main.enqueue(requests, kind, text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            fake_app = SimpleNamespace(clipboard=lambda: SimpleNamespace(text=lambda: ""))
            with patch.object(main, "QApplication", fake_app), \
                 patch.object(main, "capture_game", AsyncMock(return_value="fake-jpeg")), \
                 patch.object(main, "AUTOSAVE_PATH", path):
                task = asyncio.create_task(main.input_loop(translator, overlay, hooker, Mock(), cfg, context, requests))
                try:
                    await asyncio.wait_for(finished.wait(), 2)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                saved = path.read_text(encoding="utf-8") if path.exists() else ""
        return context, overlay, translator, saved

    async def test_screen_ocr_flows_into_translation_and_saved_context(self):
        context, overlay, translator, saved = await self.exercise()
        translator.read_screen.assert_awaited_once_with("fake-jpeg")
        translator.translate.assert_awaited_once_with("誰？", replace_last=False)
        self.assertEqual(context.history[0].scene_description, "昏暗的走廊。")
        self.assertIn("是谁？", saved)
        overlay.update_text.assert_called_once_with("誰？", "是谁？", replace_last=False)

    async def test_empty_image_does_not_translate_or_save_story(self):
        context, overlay, translator, saved = await self.exercise(empty=True)
        translator.translate.assert_not_awaited()
        self.assertEqual(context.history, [])
        self.assertEqual(saved, "")
        overlay.show_error_text.assert_called_once()

    async def test_failed_retry_preserves_previous_translation(self):
        context, overlay, translator, saved = await self.exercise("retry", "誰？", failure=True, retry=True)
        self.assertEqual([(l.jp, l.en) for l in context.history], [("誰？", "谁？")])
        self.assertEqual(saved, "")

    async def test_successful_retry_replaces_history(self):
        context, overlay, translator, saved = await self.exercise("retry", "誰？", retry=True)
        self.assertEqual([(l.jp, l.en) for l in context.history], [("誰？", "是谁？")])
        overlay.update_text.assert_called_once_with("誰？", "是谁？", replace_last=True)

    def test_repeated_shortcut_does_not_queue_many_requests(self):
        requests = asyncio.Queue(maxsize=1)
        for _ in range(50):
            main.enqueue(requests, "screen")
        self.assertEqual(requests.qsize(), 1)

    async def test_capture_does_not_hide_an_active_modal_dialog(self):
        fake_app = Mock()
        fake_app.activeModalWidget.return_value = object()
        with patch.object(main, "QApplication", fake_app):
            with self.assertRaisesRegex(RuntimeError, "先关闭设置"):
                await main.capture_game(Mock())
        fake_app.topLevelWidgets.assert_not_called()

    async def test_manual_capture_keeps_both_layers_visible_and_runs_off_gui_thread(self):
        foreground, backdrop, hidden = Mock(), Mock(), Mock()
        for window, hwnd in ((foreground, 101), (backdrop, 102)):
            window.isVisible.return_value = True
            window.winId.return_value = hwnd
        hidden.isVisible.return_value = False
        app = SimpleNamespace(activeModalWidget=lambda: None,
                              topLevelWidgets=lambda: [foreground, backdrop, hidden])
        screenshots = Mock()
        gui_thread = threading.get_ident()
        exclusion_threads = []
        capture_threads = []

        def exclude(hwnd):
            exclusion_threads.append(threading.get_ident())
            return True

        def capture():
            capture_threads.append(threading.get_ident())
            return "jpeg"

        screenshots.exclude_window.side_effect = exclude
        screenshots.capture_for_text.side_effect = capture
        with patch.object(main, "QApplication", app):
            for _ in range(3):
                self.assertEqual(await main.capture_game(screenshots), "jpeg")
        self.assertEqual([call.args[0] for call in screenshots.exclude_window.call_args_list],
                         [101, 102] * 3)
        self.assertTrue(all(t == gui_thread for t in exclusion_threads))
        self.assertTrue(all(t != gui_thread for t in capture_threads))
        hidden.winId.assert_not_called()
        screenshots.capture_frame.assert_not_called()
        for window in (foreground, backdrop, hidden):
            window.hide.assert_not_called()
            window.show.assert_not_called()

    async def test_manual_capture_stops_if_backdrop_cannot_be_excluded(self):
        foreground, backdrop = Mock(), Mock()
        for window, hwnd in ((foreground, 101), (backdrop, 102)):
            window.isVisible.return_value = True
            window.winId.return_value = hwnd
        app = SimpleNamespace(activeModalWidget=lambda: None,
                              topLevelWidgets=lambda: [foreground, backdrop])
        screenshots = Mock()
        screenshots.exclude_window.side_effect = [True, False]
        screenshots.exclusion_error_text.return_value = "浮窗无法从截图中排除（Windows 错误 8）"
        with patch.object(main, "QApplication", app):
            with self.assertRaisesRegex(RuntimeError, "Windows 错误 8"):
                await main.capture_game(screenshots)
        screenshots.capture_for_text.assert_not_called()
        for window in (foreground, backdrop):
            window.hide.assert_not_called()
            window.show.assert_not_called()

    async def test_manual_capture_failure_preserves_window_visibility(self):
        visible, hidden = Mock(), Mock()
        visible.isVisible.return_value = True
        visible.winId.return_value = 101
        hidden.isVisible.return_value = False
        app = SimpleNamespace(activeModalWidget=lambda: None,
                              topLevelWidgets=lambda: [visible, hidden])
        screenshots = Mock()
        screenshots.exclude_window.return_value = True
        screenshots.capture_for_text.side_effect = RuntimeError("游戏窗口已最小化")
        with patch.object(main, "QApplication", app):
            with self.assertRaisesRegex(RuntimeError, "游戏窗口已最小化"):
                await main.capture_game(screenshots)
        for window in (visible, hidden):
            window.hide.assert_not_called()
            window.show.assert_not_called()

    def test_clear_requests_removes_pending_previous_route(self):
        requests = asyncio.Queue(maxsize=1)
        main.enqueue(requests, "retry", "旧路线台词")
        main.clear_requests(requests)
        self.assertTrue(requests.empty())

    async def test_clear_during_screen_read_discards_old_route_before_translation(self):
        context = ContextManager()
        cfg = Config(deepseek_api_key="test-placeholder")
        finished = asyncio.Event()
        overlay = Mock()
        overlay.set_busy.side_effect = lambda busy: finished.set() if not busy else None
        async def read_screen(image):
            context.clear()
            context.add_line("NEW_ROUTE_JP", "新路线")
            return "OLD_ROUTE_JP", "旧场景"
        translator = Mock()
        translator.read_screen = AsyncMock(side_effect=read_screen)
        translator.translate = AsyncMock(return_value="旧路线")
        requests = asyncio.Queue(maxsize=1)
        main.enqueue(requests, "screen")
        fake_app = SimpleNamespace(clipboard=lambda: SimpleNamespace(text=lambda: ""))
        with patch.object(main, "QApplication", fake_app), \
             patch.object(main, "capture_game", AsyncMock(return_value="image")):
            task = asyncio.create_task(main.input_loop(
                translator, overlay, SimpleNamespace(is_attached=False), Mock(), cfg, context, requests, Mock()))
            await asyncio.wait_for(finished.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        translator.translate.assert_not_awaited()
        overlay.update_text.assert_not_called()
        self.assertEqual([line.jp for line in context.history], ["NEW_ROUTE_JP"])
        self.assertIsNone(context.current_scene_description)

    async def test_continuous_capture_keeps_own_windows_visible(self):
        window = Mock()
        window.isVisible.return_value = True
        window.winId.return_value = 123
        app = SimpleNamespace(activeModalWidget=lambda: None, topLevelWidgets=lambda: [window])
        screenshots = Mock()
        screenshots.exclude_window.return_value = True
        screenshots.capture_frame.return_value = "frame"
        with patch.object(main, "QApplication", app):
            self.assertEqual(await main.capture_observation(screenshots), "frame")
        screenshots.exclude_window.assert_called_once_with(123)
        window.hide.assert_not_called()
        window.show.assert_not_called()

    async def test_capture_exclusion_failure_reports_actual_cause(self):
        window = Mock()
        window.isVisible.return_value = True
        window.winId.return_value = 123
        app = SimpleNamespace(activeModalWidget=lambda: None, topLevelWidgets=lambda: [window])
        screenshots = Mock()
        screenshots.exclude_window.return_value = False
        screenshots.exclusion_error_text.return_value = "浮窗无法从截图中排除（Windows 错误 8）"
        with patch.object(main, "QApplication", app):
            with self.assertRaisesRegex(RuntimeError, "Windows 错误 8"):
                await main.capture_observation(screenshots)
        screenshots.capture_frame.assert_not_called()
        window.hide.assert_not_called()


if __name__ == "__main__":
    unittest.main()

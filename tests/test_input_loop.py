import asyncio
import tempfile
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

    def test_clear_requests_removes_pending_previous_route(self):
        requests = asyncio.Queue(maxsize=1)
        main.enqueue(requests, "retry", "旧路线台词")
        main.clear_requests(requests)
        self.assertTrue(requests.empty())


if __name__ == "__main__":
    unittest.main()

"""Deterministic scheduling checks: no network, game input, or real saved memory."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import main
from config import Config
from context_manager import ContextManager
from summary_worker import BackgroundSummarizer


class BackgroundSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_threshold_starts_after_current_translation_and_next_translation_does_not_wait(self):
        context = ContextManager(summarize_at_tokens=1, min_recent_lines=1)
        context.add_line("一行目", "第一行")
        started, release, next_translated, committed = (asyncio.Event() for _ in range(4))
        overlay = Mock()
        translator = Mock()
        translator.last_compact_error = False

        async def translate(jp, *, replace_last=False):
            context.add_line(jp, "译文：" + jp)
            if jp == "第三行":
                next_translated.set()
            return "译文：" + jp

        async def summarize():
            snapshot = context.prepare_summarization()
            started.set()
            await release.wait()
            return context.apply_summarization("旧剧情摘要", snapshot)

        translator.translate = AsyncMock(side_effect=translate)
        translator._summarize_context = AsyncMock(side_effect=summarize)
        cfg = Config(deepseek_api_key="test-placeholder")
        requests = asyncio.Queue(maxsize=1)
        main.enqueue(requests, "text", "第二行")
        fake_app = SimpleNamespace(clipboard=lambda: SimpleNamespace(text=lambda: ""))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(main, "QApplication", fake_app), \
             patch.object(main, "AUTOSAVE_PATH", Path(directory) / "memory.json"):
            worker = main.make_summary_worker(translator, overlay, cfg, context)
            original_commit = worker._on_committed

            def on_commit():
                original_commit()
                # Stop further passes so assertions isolate the first snapshot.
                context.update_thresholds(100000, 1)
                committed.set()

            worker._on_committed = on_commit
            task = asyncio.create_task(main.input_loop(
                translator, overlay, SimpleNamespace(is_attached=False), Mock(),
                cfg, context, requests, worker,
            ))
            try:
                # No second input is needed to start this summary.
                await asyncio.wait_for(started.wait(), 2)
                self.assertEqual(overlay.set_busy.call_args.args, (False,))
                self.assertEqual(overlay.update_text.call_args.args[:2], ("第二行", "译文：第二行"))
                self.assertTrue(worker.is_running)
                main.enqueue(requests, "text", "第三行")
                await asyncio.wait_for(next_translated.wait(), 2)
                self.assertFalse(release.is_set())
                await asyncio.sleep(0)
                self.assertEqual(overlay.update_text.call_count, 2)
                self.assertEqual(overlay.update_text.call_args.args[:2], ("第三行", "译文：第三行"))
                translator._summarize_context.assert_awaited_once()
                self.assertTrue(all("摘要" not in str(call) for call in overlay.show_loading.call_args_list))
                release.set()
                await asyncio.wait_for(committed.wait(), 2)
                self.assertEqual([line.jp for line in context.history], ["第二行", "第三行"])
                self.assertEqual(overlay.update_text.call_count, 2)
                saved = json.loads(main.AUTOSAVE_PATH.read_text(encoding="utf-8"))
                self.assertEqual(saved["summary"], "旧剧情摘要")
                self.assertEqual([line["jp"] for line in saved["history"]], ["第二行", "第三行"])
                overlay.show_error_text.assert_not_called()
                overlay.restore_display.assert_not_called()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await worker.aclose()

    async def test_manual_summary_is_background_and_does_not_touch_subtitles_or_busy_state(self):
        context = ContextManager(summarize_at_tokens=100000, min_recent_lines=1)
        context.add_line("古い", "旧")
        context.add_line("新しい", "新")
        started = asyncio.Event()

        async def summarize():
            started.set()
            await asyncio.Event().wait()

        translator = Mock(_summarize_context=AsyncMock(side_effect=summarize))
        overlay = Mock()
        requests = asyncio.Queue(maxsize=1)
        main.enqueue(requests, "compact")
        fake_app = SimpleNamespace(clipboard=lambda: SimpleNamespace(text=lambda: ""))
        with patch.object(main, "QApplication", fake_app):
            task = asyncio.create_task(main.input_loop(
                translator, overlay, SimpleNamespace(is_attached=False), Mock(),
                Config(deepseek_api_key="test-placeholder"), context, requests,
            ))
            try:
                await asyncio.wait_for(started.wait(), 2)
                overlay.set_busy.assert_not_called()
                overlay.show_loading.assert_not_called()
                overlay.update_text.assert_not_called()
                overlay.restore_display.assert_not_called()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_single_flight_failure_has_no_automatic_retry_loop(self):
        context = ContextManager(summarize_at_tokens=1, min_recent_lines=1)
        context.add_line("古い", "旧")
        context.add_line("新しい", "新")
        started, release = asyncio.Event(), asyncio.Event()

        async def fail():
            started.set()
            await release.wait()
            return False

        translator = Mock(_summarize_context=AsyncMock(side_effect=fail))
        on_commit = Mock()
        worker = BackgroundSummarizer(translator, context, on_commit)
        before = context.to_dict()
        try:
            self.assertTrue(worker.request())
            await asyncio.wait_for(started.wait(), 2)
            for _ in range(10):
                self.assertFalse(worker.request(force=True))
            task = worker._task
            release.set()
            await asyncio.wait_for(task, 2)
            await asyncio.sleep(0)
            translator._summarize_context.assert_awaited_once()
            on_commit.assert_not_called()
            self.assertEqual(context.to_dict(), before)
            self.assertFalse(worker.is_running)
            self.assertTrue(worker.request())  # Explicit/new-line request can retry.
            await worker._task
            self.assertEqual(translator._summarize_context.await_count, 2)
        finally:
            await worker.aclose()

    async def test_cancellation_does_not_clear_the_replacement_task(self):
        context = ContextManager(summarize_at_tokens=1, min_recent_lines=1)
        context.add_line("古い", "旧")
        context.add_line("新しい", "新")
        calls = []

        async def blocked():
            calls.append(asyncio.current_task())
            await asyncio.Event().wait()

        worker = BackgroundSummarizer(Mock(_summarize_context=AsyncMock(side_effect=blocked)), context, Mock())
        try:
            worker.request()
            await asyncio.sleep(0)
            old_task = worker._task
            worker.cancel()
            self.assertTrue(worker.request())
            new_task = worker._task
            await asyncio.sleep(0)
            self.assertTrue(old_task.cancelled())
            self.assertIs(worker._task, new_task)
            self.assertTrue(worker.is_running)
        finally:
            await worker.aclose()
        self.assertFalse(worker.is_running)
        self.assertFalse(worker.request(force=True))

    async def test_ready_check_failure_never_calls_model_or_changes_memory(self):
        context = ContextManager(summarize_at_tokens=1, min_recent_lines=1)
        context.add_line("古い", "旧")
        context.add_line("新しい", "新")
        translator = Mock(_summarize_context=AsyncMock())
        worker = BackgroundSummarizer(translator, context, Mock(),
                                     check_ready=Mock(side_effect=RuntimeError("missing test key")))
        try:
            worker.request()
            await worker._task
            translator._summarize_context.assert_not_awaited()
            self.assertEqual(len(context.history), 2)
        finally:
            await worker.aclose()


if __name__ == "__main__":
    unittest.main()

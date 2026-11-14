import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

from config import Config
from context_manager import ContextManager
from local_ocr import OCRResult
from observation_service import ObservationService


class GameViewportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = 100.0
        # Patch only the service clock; asyncio's deadlines use real time.
        clock = SimpleNamespace(monotonic=lambda: self.now, perf_counter=time.perf_counter)
        self.clock_patch = patch("observation_service.time", clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.cfg = Config(scene_observation_enabled=False)
        self.context = ContextManager()
        self.target = 1
        self.translator = SimpleNamespace(locate_game_viewport=AsyncMock(return_value=(0, .1, 1, .9)))
        self.service = ObservationService(
            self.cfg, self.context, self.translator, capture=AsyncMock(), target_key=lambda: self.target,
            on_display=Mock(), on_memory=Mock(), on_status=Mock(), reader=Mock())
        self.image = Image.new("RGB", (100, 100), "green")
        self.image.paste("red", (0, 0, 100, 10))
        self.image.paste("blue", (0, 90, 100, 100))

    async def asyncTearDown(self):
        await self.service.aclose()

    async def prepare(self, image=None):
        return await self.service._prepare_game_frame(
            self.image if image is None else image, self.service._epoch, self.target)

    async def finish_retry(self, image=None):
        await asyncio.wait_for(self.service._viewport_task, 2)
        return await self.prepare(image)

    async def test_calibrates_once_then_crops_locally(self):
        first = await self.prepare()
        second = await self.prepare()
        self.assertEqual(first.size, (100, 80))
        self.assertEqual(second.getpixel((50, 79)), (0, 128, 0))
        self.translator.locate_game_viewport.assert_awaited_once()
        self.assertFalse(self.context.history)

    async def test_disabled_calibration_preserves_full_window(self):
        self.cfg.auto_game_viewport = False
        self.assertIs(await self.prepare(), self.image)
        self.translator.locate_game_viewport.assert_not_called()

    async def test_invalid_model_answer_uses_full_window_until_scheduled_retry(self):
        self.translator.locate_game_viewport.return_value = None
        self.assertIs(await self.prepare(), self.image)
        self.now += 4.99
        self.assertIs(await self.prepare(), self.image)
        self.translator.locate_game_viewport.assert_awaited_once()
        self.assertTrue(self.service._viewport_fallback)
        self.service._set_status("观察中")
        self.assertIn("使用完整窗口", self.service.on_status.call_args.args[0])
        self.assertIn("稍后自动重试", self.service.on_status.call_args.args[0])
        self.assertEqual(self.service._viewport_retry_at, 105)

    async def test_failures_retry_with_capped_backoff_and_no_per_frame_calls(self):
        self.translator.locate_game_viewport.return_value = None
        await self.prepare()
        for delay in (15, 30, 60, 60, 60):
            self.now = self.service._viewport_retry_at
            attempts = self.translator.locate_game_viewport.await_count
            self.assertIs(await self.prepare(), self.image)
            await self.finish_retry()
            self.assertEqual(self.translator.locate_game_viewport.await_count, attempts + 1)
            self.assertEqual(self.service._viewport_retry_at, self.now + delay)
            for _ in range(10):
                self.assertIs(await self.prepare(), self.image)
            self.assertEqual(self.translator.locate_game_viewport.await_count, attempts + 1)

    async def test_manual_retry_wakes_sampler_without_parallel_requests(self):
        self.translator.locate_game_viewport.return_value = None
        await self.prepare()
        ready, release = asyncio.Event(), asyncio.Event()

        async def pending(*_):
            ready.set()
            await release.wait()
            return (0, .1, 1, .9)

        self.translator.locate_game_viewport.side_effect = pending
        self.service._wake.clear()
        self.service.request_display(retry=True)
        self.assertTrue(self.service._wake.is_set())
        self.assertEqual(self.service._viewport_retry_at, self.now)
        self.assertIs(await self.prepare(), self.image)
        await asyncio.wait_for(ready.wait(), 2)
        task = self.service._viewport_task
        for _ in range(10):
            self.now += 100
            self.service.request_display(retry=True)
            self.assertIs(await self.prepare(), self.image)
        self.assertIs(self.service._viewport_task, task)
        self.assertEqual(self.translator.locate_game_viewport.await_count, 2)
        self.assertIn("后台标定中", self.service.on_status.call_args.args[0])
        release.set()
        self.assertEqual((await self.finish_retry()).size, (100, 80))

    async def test_translation_key_can_advance_retry_before_retry_button_exists(self):
        self.translator.locate_game_viewport.return_value = None
        await self.prepare()
        self.assertIsNone(self.service._current)
        self.assertEqual(self.service._viewport_retry_at, self.now + 5)
        self.service.request_display()
        self.assertEqual(self.service._viewport_retry_at, self.now)
        self.assertTrue(self.service._requested)
        await self.prepare()
        await self.finish_retry()
        self.assertEqual(self.translator.locate_game_viewport.await_count, 2)
        self.assertTrue(self.service._requested)

    async def test_failed_retry_logs_only_exception_class_and_remains_scheduled(self):
        self.translator.locate_game_viewport.return_value = None
        await self.prepare()
        self.translator.locate_game_viewport.side_effect = RuntimeError("PRIVATE_TOKEN private://image")
        self.now = self.service._viewport_retry_at
        with self.assertLogs("observation_service", level="WARNING") as logs:
            await self.prepare()
            await self.finish_retry()
        self.assertIn("RuntimeError", logs.output[0])
        self.assertNotIn("PRIVATE_", str(logs.output))
        self.assertNotIn("private://", str(logs.output))
        self.assertEqual(self.service._viewport_retry_at, self.now + 15)

    async def test_timed_out_retry_is_cancelled_and_rescheduled(self):
        self.translator.locate_game_viewport.return_value = None
        await self.prepare()
        cancelled = asyncio.Event()

        async def pending(*_):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.translator.locate_game_viewport.side_effect = pending
        self.now = self.service._viewport_retry_at
        with patch("observation_service._VIEWPORT_TIMEOUT", .01):
            with self.assertLogs("observation_service", level="WARNING"):
                await self.prepare()
                await self.finish_retry()
        self.assertTrue(cancelled.is_set())
        self.assertTrue(self.service._viewport_fallback)
        self.assertEqual(self.service._viewport_retry_at, self.now + 15)

    async def test_delayed_success_invalidates_full_window_state_and_crops_latest_frame(self):
        self.translator.locate_game_viewport.return_value = None
        await self.prepare()
        old_epoch = self.service._epoch
        line = self.context.observe("以前の画面")
        old_ref = (line.observation_id, line.revision)
        self.service._current = old_ref
        self.service._cache = (old_ref, "旧译文")
        self.service._valid = self.service._requested = True
        self.service._last_ocr = OCRResult("以前の画面", .95)
        self.service._last_frame = b"old-fingerprint"
        self.service._suppressed_keys.add("old-ocr-artifact")
        self.service.stability.observe("以前の画面", 0)
        self.service.stability.observe("以前の画面", 1)
        old_translation = self.service._translation_task = self.service._spawn(asyncio.Event().wait())
        await asyncio.sleep(0)
        self.translator.locate_game_viewport.return_value = (0, .1, 1, .9)
        self.now = self.service._viewport_retry_at
        await self.prepare()
        latest = Image.new("RGB", self.image.size, "yellow")
        result = await self.finish_retry(latest)
        await asyncio.sleep(0)
        self.assertEqual(result.size, (100, 80))
        self.assertEqual(result.getpixel((50, 50)), (255, 255, 0))
        self.assertNotEqual(self.service._epoch, old_epoch)
        self.assertIsNone(self.service._cache)
        self.assertIsNone(self.service._current)
        self.assertIsNone(self.service._last_ocr)
        self.assertIsNone(self.service._last_frame)
        self.assertFalse(self.service.stability.is_stable)
        self.assertFalse(self.service._suppressed_keys)
        self.assertTrue(old_translation.cancelled())
        self.assertTrue(self.service._requested)
        self.assertFalse(self.service._valid)
        self.assertFalse(self.service._viewport_fallback)
        self.assertIsNone(self.service._viewport_retry_at)
        self.service.on_display.assert_not_called()

    async def test_background_retry_does_not_block_local_ocr(self):
        self.translator.locate_game_viewport.return_value = None
        await self.prepare()
        ready, release, observed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def pending(*_):
            ready.set()
            await release.wait()
            return (0, .1, 1, .9)

        self.translator.locate_game_viewport.side_effect = pending
        self.now = self.service._viewport_retry_at
        self.service.capture.return_value = self.image
        self.service.reader.recognize.return_value = OCRResult("", 0)
        self.service.on_status.side_effect = lambda value: observed.set() if "暂无台词" in value else None
        self.service.start()
        await asyncio.wait_for(ready.wait(), 2)
        await asyncio.wait_for(observed.wait(), 2)
        self.assertFalse(self.service._viewport_task.done())
        self.assertEqual(self.service.reader.recognize.call_args.args[0].size, (100, 100))
        release.set()
        await asyncio.wait_for(self.service._viewport_task, 2)
        observed.clear()
        self.service._wake.set()
        await asyncio.wait_for(observed.wait(), 2)
        self.assertEqual(self.service.reader.recognize.call_count, 2)
        self.assertEqual(self.service.reader.recognize.call_args.args[0].size, (100, 80))
        self.assertFalse(self.service._viewport_fallback)

    async def test_reset_cancels_old_calibration_and_discards_answer(self):
        ready = asyncio.Event()
        async def pending(*_):
            ready.set()
            await asyncio.Event().wait()
        self.translator.locate_game_viewport.side_effect = pending
        work = asyncio.create_task(self.prepare())
        await asyncio.wait_for(ready.wait(), 2)
        self.target = 2
        self.service.reset()
        self.assertIsNone(await asyncio.wait_for(work, 2))
        self.assertIsNone(self.service._viewport_size)
        self.assertIsNone(self.service._viewport_rect)

    async def start_pending_retry(self):
        self.translator.locate_game_viewport.return_value = None
        await self.prepare()
        ready = asyncio.Event()

        async def pending(*_):
            ready.set()
            await asyncio.Event().wait()

        self.translator.locate_game_viewport.side_effect = pending
        self.now = self.service._viewport_retry_at
        await self.prepare()
        await asyncio.wait_for(ready.wait(), 2)
        return self.service._viewport_task

    async def test_target_change_cancels_pending_retry_and_discards_its_geometry(self):
        old = await self.start_pending_retry()
        self.target = 2
        self.service.request_display()
        await asyncio.gather(old, return_exceptions=True)
        self.assertTrue(old.cancelled())
        self.assertIsNone(self.service._viewport_size)
        self.assertIsNone(self.service._viewport_rect)
        self.assertIsNone(self.service._viewport_task)
        self.assertIsNone(self.service._viewport_retry_at)
        self.translator.locate_game_viewport.side_effect = None
        self.translator.locate_game_viewport.return_value = (0, 0, .5, 1)
        self.assertEqual((await self.prepare()).size, (50, 100))
        self.assertEqual(self.translator.locate_game_viewport.await_count, 3)

    async def test_config_change_cancels_pending_retry_and_disabled_observation_does_not_retry(self):
        old = await self.start_pending_retry()
        self.cfg.observation_enabled = False
        self.service.configure()
        await asyncio.gather(old, return_exceptions=True)
        self.assertTrue(old.cancelled())
        self.now += 1000
        self.assertIsNone(await self.prepare())
        self.service.request_display(retry=True)
        self.assertIsNone(self.service._viewport_task)
        self.assertIsNone(self.service._viewport_retry_at)
        self.assertEqual(self.translator.locate_game_viewport.await_count, 2)

    async def test_resize_cancels_pending_retry_and_recalibrates_new_size(self):
        old = await self.start_pending_retry()
        self.service._requested = True
        self.translator.locate_game_viewport.side_effect = None
        self.translator.locate_game_viewport.return_value = (0, 0, .5, 1)
        resized = self.image.resize((200, 100))
        self.assertEqual((await self.prepare(resized)).size, (100, 100))
        await asyncio.gather(old, return_exceptions=True)
        self.assertTrue(old.cancelled())
        self.assertEqual(self.service._viewport_size, (200, 100))
        self.assertEqual(self.service._viewport_rect, (0, 0, 100, 100))
        self.assertTrue(self.service._requested)
        self.assertEqual(self.translator.locate_game_viewport.await_count, 3)

    async def test_shutdown_cancels_pending_background_calibration(self):
        old = await self.start_pending_retry()
        await self.service.aclose()
        self.assertTrue(old.cancelled())
        self.now += 1000
        self.assertIsNone(await self.prepare())
        self.assertEqual(self.translator.locate_game_viewport.await_count, 2)

    async def test_worker_receives_canvas_then_user_subregion(self):
        self.cfg.ocr_region_top_percent = 25
        self.cfg.ocr_region_bottom_percent = 75
        self.service.configure()
        self.service.capture.return_value = self.image
        done = asyncio.Event()
        def recognize(crop):
            self.assertEqual(crop.size, (100, 40))
            self.assertEqual(crop.getpixel((50, 39)), (0, 128, 0))
            return OCRResult("", 0)
        self.service.reader.recognize.side_effect = recognize
        self.service.on_status.side_effect = lambda value: done.set() if "暂无台词" in value else None
        self.service.start()
        await asyncio.wait_for(done.wait(), 2)
        self.translator.locate_game_viewport.assert_awaited_once()

    async def test_resize_recalibrates_and_preserves_pending_manual_request(self):
        await self.prepare()
        self.service._requested = True
        resized = self.image.resize((200, 100))
        self.service.capture.return_value = resized
        self.service.reader.recognize.return_value = OCRResult("", 0)
        done = asyncio.Event()
        self.service.on_status.side_effect = lambda value: done.set() if "暂无台词" in value else None
        self.service.start()
        await asyncio.wait_for(done.wait(), 2)
        self.assertEqual(self.translator.locate_game_viewport.await_count, 2)
        self.assertEqual(self.service._viewport_size, (200, 100))
        self.assertTrue(self.service._requested)


if __name__ == "__main__":
    unittest.main()

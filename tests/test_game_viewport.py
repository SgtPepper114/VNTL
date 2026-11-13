import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from PIL import Image

from config import Config
from context_manager import ContextManager
from local_ocr import OCRResult
from observation_service import ObservationService


class GameViewportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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

    async def prepare(self):
        return await self.service._prepare_game_frame(self.image, self.service._epoch, self.target)

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

    async def test_invalid_model_answer_falls_back_once_per_size(self):
        self.translator.locate_game_viewport.return_value = None
        self.assertIs(await self.prepare(), self.image)
        self.assertIs(await self.prepare(), self.image)
        self.translator.locate_game_viewport.assert_awaited_once()
        self.assertTrue(self.service._viewport_fallback)
        self.service._set_status("观察中")
        self.assertIn("使用完整窗口", self.service.on_status.call_args.args[0])

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

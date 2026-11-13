import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from PIL import Image

from config import Config
from context_manager import ContextManager
from local_ocr import OCRResult
from observation_service import ObservationService, SceneGate


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg = Config(scene_observation_enabled=False)
        self.context = ContextManager()
        self.target = 1
        self.display = Mock()
        self.memory = Mock()
        self.translator = SimpleNamespace(
            translate_observation=AsyncMock(return_value="中文译文"),
            read_screen=AsyncMock(return_value=("校正された台詞", "教室")),
            describe_scene=AsyncMock(return_value="窗边的教室"),
        )
        self.service = ObservationService(
            self.cfg, self.context, self.translator,
            capture=AsyncMock(), target_key=lambda: self.target,
            on_display=self.display, on_memory=self.memory, on_status=Mock(), reader=Mock(),
        )
        self.image = Image.new("RGB", (320, 240), "black")

    async def asyncTearDown(self):
        await self.service.aclose()

    def frame(self, text, now, *, uncertain=False, image=None):
        self.service.process_frame(image or self.image, OCRResult(text, .95, (), uncertain), now)

    def stable(self, text="誰かいる？", now=0, **kwargs):
        self.frame(text, now, **kwargs)
        self.frame(text, now + .61, **kwargs)

    async def finish(self):
        for _ in range(10):
            await asyncio.sleep(0)

    async def test_manual_records_and_pretranslates_silently_then_reveals_cache(self):
        self.stable()
        self.assertEqual(self.context.history[0].jp, "誰かいる？")
        self.assertEqual(self.context.history[0].en, "")
        self.memory.assert_called()
        await self.finish()
        self.display.assert_not_called()
        self.assertEqual(self.context.history[0].en, "中文译文")
        self.service.request_display()
        self.display.assert_called_once_with("誰かいる？", "中文译文", False)
        self.assertEqual(self.translator.translate_observation.await_count, 1)

    async def test_auto_displays_once_and_append_replaces_same_record(self):
        self.cfg.translation_mode = "auto"
        self.stable("夜中に音がした。")
        await self.finish()
        first_id = self.context.history[0].observation_id
        self.frame("夜中に音がした。", 1)
        self.frame("夜中に音がした。", 1.2)
        self.assertEqual(self.display.call_count, 1)
        self.stable("夜中に音がした。誰かいる？", 2)
        await self.finish()
        self.assertEqual(len(self.context.history), 1)
        self.assertEqual(self.context.history[0].observation_id, first_id)
        self.assertEqual(self.context.history[0].revision, 2)
        self.assertTrue(self.display.call_args.args[2])
        self.assertEqual(self.display.call_count, 2)

    async def test_manual_revealed_page_updates_when_more_text_arrives(self):
        self.stable("待って。")
        await self.finish()
        self.service.request_display()
        self.stable("待って。まだ誰かいる。", 2)
        await self.finish()
        self.assertEqual(self.display.call_count, 2)
        self.assertTrue(self.display.call_args.args[2])

    async def test_dirty_text_cannot_show_previous_cached_page(self):
        self.stable()
        await self.finish()
        self.frame("先生", 1)
        self.service.request_display()
        self.display.assert_not_called()
        self.frame("先生", 1.61)
        await self.finish()
        self.display.assert_called_once_with("先生", "中文译文", False)

    async def test_clear_screen_invalidates_cache_and_repeated_phrase_is_new_page(self):
        self.stable()
        await self.finish()
        first_id = self.context.history[0].observation_id
        self.frame("", 1)
        self.service.request_display()
        self.display.assert_not_called()
        self.stable(now=2)
        await self.finish()
        self.assertNotEqual(self.context.history[-1].observation_id, first_id)
        self.assertEqual(self.display.call_count, 1)

    async def test_target_or_context_change_rejects_cache_before_next_poll(self):
        self.stable()
        await self.finish()
        self.target = 2
        self.service.request_display()
        self.display.assert_not_called()
        self.stable(now=2)
        await self.finish()
        self.display.reset_mock()
        self.context.clear()
        self.service.request_display()
        self.display.assert_not_called()

    async def test_ocr_jitter_back_to_earlier_text_recovers_same_page(self):
        a, b = "夜中二時ころ音楽室", "夜中二時ごろ音楽室"
        self.stable(a)
        await self.finish()
        self.stable(b, 2)
        await self.finish()
        self.stable(a, 4)
        await self.finish()
        self.service.request_display()
        self.display.assert_called_once_with(a, "中文译文", False)
        self.assertEqual(len(self.context.history), 1)

    async def test_mode_switch_reuses_cache(self):
        self.stable()
        await self.finish()
        self.cfg.translation_mode = "auto"
        self.service.configure()
        self.display.assert_called_once()
        self.assertEqual(self.translator.translate_observation.await_count, 1)

    async def test_pending_manual_request_waits_and_only_reveals_current_revision(self):
        gate = asyncio.Event()
        async def slow(*ref):
            await gate.wait()
            return "稍后的译文"
        self.translator.translate_observation.side_effect = slow
        self.stable()
        await self.finish()
        self.service.request_display()
        self.frame("", 1)
        gate.set()
        await self.finish()
        self.display.assert_not_called()

    async def test_network_error_can_be_retried_by_translation_key(self):
        self.translator.translate_observation.side_effect = RuntimeError("offline")
        self.stable()
        await self.finish()
        self.translator.translate_observation.side_effect = None
        self.service.request_display()
        await self.finish()
        self.display.assert_called_once()
        self.assertEqual(self.translator.translate_observation.await_count, 2)

    async def test_uncertain_ocr_is_checked_and_keeps_display_valid_on_next_frame(self):
        self.stable("校正ざれた台詞", uncertain=True)
        await self.finish()
        self.frame("校正ざれた台詞", 1, uncertain=True)
        self.service.request_display()
        self.display.assert_called_once_with("校正された台詞", "中文译文", False)
        self.assertEqual(self.translator.read_screen.await_count, 1)

    async def test_vision_blank_suppresses_local_noise_until_text_changes(self):
        self.translator.read_screen.return_value = ("", "無人の教室")
        self.stable("UI誤検出", uncertain=True)
        await self.finish()
        self.service.request_display()
        self.frame("UI誤検出", 1, uncertain=True)
        self.frame("UI誤検出", 2, uncertain=True)
        self.display.assert_not_called()
        self.translator.translate_observation.assert_not_awaited()
        self.assertTrue(all(not line.jp for line in self.context.history))

    async def test_scene_rate_limit_and_failed_placeholder_cleanup(self):
        self.cfg.scene_observation_enabled = True
        self.translator.describe_scene.side_effect = RuntimeError("offline")
        self.stable("", 0)
        await self.finish()
        self.assertEqual(self.context.history, [])
        self.stable("", 2, image=Image.new("RGB", (320, 240), "white"))
        await self.finish()
        self.assertEqual(self.translator.describe_scene.await_count, 1)
        self.frame("", 12)
        await self.finish()
        self.assertEqual(self.translator.describe_scene.await_count, 2)
        self.assertEqual(self.context.history, [])

    async def test_current_cache_survives_compaction_and_later_scenes_keep_summarizing(self):
        self.context.update_thresholds(1, 1)
        self.context.add_line("古い台詞", "旧台词")
        self.stable()
        await self.finish()
        self.context.observe("", "廊下")
        snapshot = self.context.prepare_summarization()
        self.assertEqual(snapshot.count, 2)
        self.context.apply_summarization("先前的情节", snapshot)
        self.service.request_display()
        self.display.assert_called_once()
        self.context.observe("", "操场")
        self.assertTrue(self.context.needs_summarization())
        self.service.request_display(retry=True)
        await self.finish()
        self.assertEqual(self.translator.translate_observation.await_count, 2)

    async def test_other_input_does_not_get_overwritten_in_backlog(self):
        self.stable()
        await self.finish()
        self.service.request_display()
        self.service.external_display()
        self.service.request_display()
        self.assertFalse(self.display.call_args.args[2])

    async def test_old_scene_cannot_overwrite_a_newer_visual_revision(self):
        ready = asyncio.Event()
        async def delayed(*args):
            await ready.wait()
            return "OLDER_SCENE_A"
        self.translator.describe_scene.side_effect = delayed
        line = self.context.observe("最初の台詞")
        original = (line.observation_id, line.revision)
        pending = self.context.observe("")
        task = asyncio.create_task(self.service._describe(
            (pending.observation_id, pending.revision), self.image, line.jp, original))
        await self.finish()
        revised = self.context.observe("最初の台詞。その後。", "NEWER_SCENE_B", replace_id=line.observation_id)
        ready.set()
        await task
        self.assertEqual(revised.scene_description, "NEWER_SCENE_B")
        self.assertIsNone(self.context.get_observation(pending.observation_id))

    def test_scene_gate_masks_dialogue_and_detects_background(self):
        gate = SceneGate()
        black = Image.new("RGB", (320, 240), "black")
        white = Image.new("RGB", (320, 240), "white")
        gate.submitted(black, ((0, .6, 1, 1),), 0)
        bottom = black.copy()
        bottom.paste(white.crop((0, 144, 320, 240)), (0, 144))
        self.assertFalse(gate.changed(bottom, (), 5))
        self.assertTrue(gate.changed(white, (), 5))


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

from config import Config
from context_manager import ContextManager
from local_ocr import OCRResult
from observation_service import ObservationService, SceneGate


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg = Config(scene_observation_enabled=False, auto_game_viewport=False)
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

    def frame(self, text, now, *, uncertain=False, image=None, boxes=(), confidence=.95):
        self.service.process_frame(image or self.image, OCRResult(text, confidence, boxes, uncertain), now)

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
        self.translator.read_screen.side_effect = lambda *_: (self.service.stability.current_text, "")
        self.stable(a)
        await self.finish()
        self.stable(b, 2)
        await self.finish()
        self.stable(a, 4)
        await self.finish()
        self.service.request_display()
        self.display.assert_called_once()
        self.assertIn(self.display.call_args.args[0], (a, b))
        self.assertEqual(self.display.call_args.args[1:], ("中文译文", False))
        self.assertEqual(len(self.context.history), 1)

    async def test_mode_switch_reuses_cache(self):
        self.stable()
        await self.finish()
        self.cfg.translation_mode = "auto"
        self.service.configure()
        self.display.assert_called_once()
        self.assertEqual(self.translator.translate_observation.await_count, 1)

    async def test_blinking_tail_keeps_pretranslation_and_manual_cache_available(self):
        text = "ミカ：あんた、キホ先生を疑うわけ？"
        self.translator.read_screen.return_value = (text, "")
        for i in range(15):
            bright = i % 2 == 0
            self.frame(text if bright else text + "「", i * .2,
                       boxes=((10, 100, 280 if bright else 310, 130),),
                       confidence=.97 if bright else .90)
            await self.finish()
        self.assertTrue(self.service._valid)
        self.assertIsNotNone(self.service._cache)
        self.assertEqual(self.translator.translate_observation.await_count, 1)
        self.assertEqual(len(self.context.history), 1)
        self.service.request_display()
        self.display.assert_called_once_with(text, "中文译文", False)
        self.assertNotIn("等待文字稳定", self.service.on_status.call_args.args[0])
        self.frame("", 3)
        self.assertFalse(self.service._valid)
        self.assertIsNone(self.service._cache)

    async def test_configured_bottom_boundary_excludes_status_bar_from_ocr(self):
        self.cfg.ocr_region_top_percent = 60
        self.cfg.ocr_region_bottom_percent = 96
        self.service.configure()
        seen = asyncio.Event()
        image = Image.new("RGB", (100, 100), "black")
        image.paste("red", (0, 96, 100, 100))
        self.service.capture = AsyncMock(return_value=image)

        def recognize(crop):
            self.assertEqual(crop.size, (100, 36))
            self.assertEqual(crop.getpixel((50, 35)), (0, 0, 0))
            return OCRResult("", 0)

        self.service.reader.recognize.side_effect = recognize
        self.service.on_status.side_effect = lambda text: seen.set() if "暂无台词" in text else None
        self.service.start()
        await asyncio.wait_for(seen.wait(), 2)
        self.service.reader.recognize.assert_called_once()

    async def test_changing_bottom_boundary_discards_prepared_cache(self):
        self.stable()
        await self.finish()
        self.assertIsNotNone(self.service._cache)
        self.cfg.ocr_region_bottom_percent = 96
        self.service.configure()
        self.assertIsNone(self.service._cache)
        self.service.request_display()
        self.display.assert_not_called()

    async def test_excluded_footer_does_not_trigger_scene_observation(self):
        self.cfg.scene_observation_enabled = True
        self.cfg.ocr_region_bottom_percent = 95
        self.service.configure()
        self.stable("", 0)
        await self.finish()
        self.assertEqual(self.translator.describe_scene.await_count, 1)
        footer = self.image.copy()
        footer.paste("white", (0, 228, 320, 240))
        self.frame("", 12, image=footer)
        await self.finish()
        self.assertEqual(self.translator.describe_scene.await_count, 1)

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

    async def test_visual_confirmation_reuses_same_page_translation_after_new_icon(self):
        text = "誰か音楽室にいるような気がする。"
        self.translator.read_screen.return_value = (text, "教室")
        self.stable(text)
        await self.finish()
        self.stable("!\n" + text, 2, uncertain=True)
        await self.finish()
        self.service.request_display()
        self.translator.translate_observation.assert_awaited_once()
        self.translator.read_screen.assert_awaited_once()
        self.display.assert_called_once_with(text, "中文译文", False)
        self.assertEqual(len(self.context.history), 1)
        self.assertEqual(self.context.history[0].en, "中文译文")
        self.assertEqual(self.service.metrics["verified_cache_reuses"], 1)

    async def test_visual_confirmation_of_real_addition_requires_new_translation(self):
        text = "誰か音楽室にいるような気がする。"
        self.stable(text)
        await self.finish()
        changed = text + "待って。"
        self.translator.read_screen.return_value = (changed, "")
        self.stable(changed, 2, uncertain=True)
        await self.finish()
        self.assertEqual(self.translator.translate_observation.await_count, 2)
        self.assertEqual(self.context.history[-1].jp, changed)

    async def test_visual_confirmation_after_clear_uses_fresh_context_translation(self):
        text = "誰か音楽室にいるような気がする。"
        self.stable(text)
        await self.finish()
        self.frame("", 1)
        self.translator.read_screen.return_value = (text, "")
        self.stable(text, 2, uncertain=True)
        await self.finish()
        self.assertEqual(self.translator.translate_observation.await_count, 2)
        self.assertEqual(len(self.context.history), 2)

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

    async def test_manual_request_preserves_unbound_and_empty_screen_reasons(self):
        self.target = None
        self.service.request_display()
        self.assertIn("选择游戏窗口", self.service.on_status.call_args.args[0])
        self.assertNotIn("完成预译", self.service.on_status.call_args.args[0])
        self.target = 1
        self.service.reset()
        self.frame("", 0)
        reason = self.service.not_ready_reason
        self.service.request_display()
        self.assertEqual(self.service.on_status.call_args.args[0], reason)
        self.assertIn("暂无台词", reason)
        self.translator.translate_observation.assert_not_awaited()

    async def test_manual_request_keeps_capture_block_reason_and_wakes_serial_retry(self):
        self.cfg.ocr_poll_ms = 1000
        failed, retried = asyncio.Event(), asyncio.Event()
        reason = "持续观察需要 Windows 10 2004 或更新版本的窗口截图支持。"
        async def capture():
            if not failed.is_set():
                failed.set()
                raise RuntimeError(reason)
            retried.set()
            return self.image
        self.service.capture = capture
        self.service.reader.recognize.return_value = OCRResult("", .95, ())
        self.service.start()
        await asyncio.wait_for(failed.wait(), 1)
        await asyncio.sleep(0)
        self.assertEqual(self.service.not_ready_reason, reason)
        self.assertGreater(self.service._ocr_failed_until, 0)
        self.service.request_display()
        self.assertEqual(self.service.on_status.call_args.args[0], reason)
        # Less than either the one-second poll or two-second failure backoff.
        await asyncio.wait_for(retried.wait(), .3)
        self.assertEqual(self.service._ocr_failed_until, 0)

    async def test_capture_block_keeps_cache_hidden_and_success_clears_reason(self):
        self.stable()
        await self.finish()
        cached = self.service._cache
        self.service._valid = False
        reason = "观察已暂停：请将游戏切换到前台。"
        self.service._local_failure("capture", RuntimeError(reason))
        self.service.request_display()
        self.display.assert_not_called()
        self.assertEqual(self.service.on_status.call_args.args[0], reason)
        self.assertEqual(self.service._cache, cached)
        self.frame("誰かいる？", 2)
        self.assertIsNone(self.service._blocked_reason)
        self.display.assert_called_once()

    def test_local_failure_logging_is_throttled_and_does_not_echo_secrets(self):
        secret = "sk-example-secret https://example.invalid?token=secret"
        with patch("observation_service.time.monotonic", side_effect=[100.0, 101.0, 131.0]):
            with self.assertLogs("observation_service", level="WARNING") as logs:
                for _ in range(3):
                    self.service._local_failure("ocr", RuntimeError(secret))
        self.assertEqual(len(logs.output), 2)
        for message in logs.output:
            self.assertIn("stage=ocr", message)
            self.assertIn("error=RuntimeError", message)
            self.assertNotIn("sk-example", message)
            self.assertNotIn("token=", message)
        self.assertNotIn(secret, self.service.not_ready_reason)
        self.service.request_display()
        self.assertIn("本地文字识别", self.service.on_status.call_args.args[0])

    def test_known_numeric_capture_error_is_visible_but_extra_payload_is_not(self):
        reason = ("浮窗无法从截图中排除（Windows 错误 87）。"
                  "请重启更新后的 VNTL；也可暂停观察后使用按键识屏。")
        with self.assertLogs("observation_service", level="WARNING") as logs:
            self.service._local_failure("capture", RuntimeError(reason))
        self.assertEqual(self.service.not_ready_reason, reason)
        self.assertIn("Windows 错误 87", logs.output[0])
        self.service.reset()
        with self.assertLogs("observation_service", level="WARNING") as logs:
            self.service._local_failure("capture", RuntimeError(reason + " token=secret"))
        self.assertNotIn("token=", logs.output[0])
        self.assertEqual(self.service.not_ready_reason, "游戏截图暂缓（RuntimeError），正在重试…")

    def test_reset_forces_status_refresh_after_window_picker_updates_label(self):
        reason = "观察已暂停：请将游戏切换到前台。"
        self.service._local_failure("capture", RuntimeError(reason))
        self.service.reset()
        self.service.on_status("游戏窗口已选择，等待画面稳定")
        self.service._local_failure("capture", RuntimeError(reason))
        self.assertEqual(self.service.on_status.call_args.args[0], reason)

    async def test_failed_translation_reason_is_not_replaced_by_stable_poll(self):
        self.translator.translate_observation.side_effect = RuntimeError("offline")
        self.stable()
        await self.finish()
        reason = self.service.not_ready_reason
        self.frame("誰かいる？", 1.3)
        self.assertEqual(self.service.not_ready_reason, reason)
        self.assertIn("再按翻译键重试", reason)

    async def test_run_preserves_failed_translation_readiness_across_capture_recovery(self):
        self.cfg.ocr_stable_ms = 0
        self.cfg.ocr_poll_ms = 10
        self.service.configure()
        self.translator.translate_observation.side_effect = RuntimeError("offline")
        self.service.reader.recognize.return_value = OCRResult("誰かいる？", .95, ())
        failed, polled, blocked, recovered = (asyncio.Event() for _ in range(4))
        capture_error = "观察已暂停：请将游戏切换到前台。"
        fail_capture = False

        def status(text):
            if "再按翻译键重试" in text:
                failed.set()

        async def capture():
            nonlocal fail_capture
            if fail_capture:
                fail_capture = False
                blocked.set()
                raise RuntimeError(capture_error)
            if failed.is_set():
                polled.set()
            if blocked.is_set():
                recovered.set()
            return self.image

        self.service.on_status.side_effect = status
        self.service.capture = capture
        self.service.start()
        await asyncio.wait_for(failed.wait(), 1)
        reason = self.service.not_ready_reason
        await asyncio.wait_for(polled.wait(), 1)
        self.assertEqual(self.service.not_ready_reason, reason)
        self.assertIn("再按翻译键重试", reason)

        fail_capture = True
        self.service._wake.set()
        await asyncio.wait_for(blocked.wait(), 1)
        self.assertEqual(self.service.not_ready_reason, capture_error)
        self.service.request_display()
        self.assertEqual(self.service.on_status.call_args.args[0], capture_error)
        await asyncio.wait_for(recovered.wait(), .3)
        self.assertEqual(self.service.not_ready_reason, reason)
        self.assertEqual(self.service.on_status.call_args.args[0], reason)
        self.assertEqual(self.translator.translate_observation.await_count, 1)
        self.assertEqual(self.service.reader.recognize.call_count, 1)
        self.display.assert_not_called()

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

    async def test_empty_vision_suppresses_raw_variant_as_well_as_canonical_text(self):
        canonical = "廊下に誰かいるような気がする。"
        raw_variant = canonical[:-1] + "「"
        boxes = ((20, 170, 290, 198),)
        self.translator.read_screen.return_value = ("", "")

        # A high-confidence candidate remains canonical when the last frame
        # contains a lower-confidence trailing glyph. Vision sees that last frame.
        self.frame(canonical, 0, boxes=boxes, confidence=.99, uncertain=True)
        self.frame(raw_variant, .61, boxes=boxes, confidence=.55, uncertain=True)
        self.assertEqual(self.context.history[-1].jp, canonical)
        self.assertNotEqual(self.context.history[-1].jp, raw_variant)
        await self.finish()
        self.assertEqual(self.translator.read_screen.await_count, 1)

        # The same screen keeps alternating between the two observed OCR forms.
        # A successful visual blank decision should suppress the whole excursion.
        for index, text in enumerate((raw_variant, canonical, raw_variant, canonical)):
            self.frame(text, 1.4 + index * .7, boxes=boxes,
                       confidence=.55 if text == raw_variant else .99, uncertain=True)
            await self.finish()
        self.service.request_display()
        await self.finish()
        self.assertEqual(self.translator.read_screen.await_count, 1)
        self.translator.translate_observation.assert_not_awaited()
        self.display.assert_not_called()
        self.assertFalse(self.context.history)


if __name__ == "__main__":
    unittest.main()

"""Await-boundary races for source observations and legacy translation calls."""
import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, Mock, patch

from config import Config
from context_manager import ContextManager
from translator import Translator


class ObservationTranslationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = Config()
        self.context = ContextManager(min_recent_lines=1)
        with patch.object(Translator, "_build_client", return_value=object()):
            self.translator = Translator(self.cfg, self.context)
        self.translator._log_call = Mock()

    async def test_pure_result_keeps_fixed_target_messages_while_context_moves_forward(self):
        old = self.context.observe("既読", "最初の場面")
        target = self.context.observe("ミカ：待って！")
        started, finish = asyncio.Event(), asyncio.Event()
        captured = []
        async def pending(*args, **kwargs):
            captured.append(copy.deepcopy((args, kwargs)))
            started.set()
            await finish.wait()
            return "ミカ：等等！"
        self.translator._call = AsyncMock(side_effect=pending)
        task = asyncio.create_task(self.translator.translate_observation(target.observation_id, 1))
        await started.wait()
        self.context.observe("未来の台詞", "未来の場面")
        self.context.attach_scene(target.observation_id, 1, "后来补到的当前场景")
        self.cfg.game_notes = "后来修改的偏好"
        before = copy.deepcopy(self.context.to_dict())
        finish.set()
        self.assertEqual(await task, "ミカ：等等！")
        self.assertEqual(self.context.to_dict(), before)
        request = str(captured[0])
        self.assertIn("既読", request)
        self.assertIn("最初の場面", request)
        for absent in ("未来の台詞", "未来の場面", "后来补到", "后来修改"):
            self.assertNotIn(absent, request)
        self.assertEqual(target.en, "")
        self.assertTrue(self.context.commit_translation(target.observation_id, 1, "ミカ：等等！"))
        self.assertEqual(old.en, "")

    async def test_empty_retry_reuses_frozen_profile_prompt_messages_and_parameters(self):
        self.cfg.game_notes = "保留游戏人名"
        self.cfg.translator_system_prompt = "最初的翻译提示词"
        target = self.context.observe("名前", "最初的场景")
        captured = []
        async def respond(*args, **kwargs):
            captured.append(copy.deepcopy((args, kwargs)))
            if len(captured) == 1:
                self.cfg.game_notes = "后来的偏好"
                self.cfg.translator_system_prompt = "后来的提示词"
                self.cfg.translator_model = "another-model"
                self.cfg.translator_max_tokens = 19
                self.cfg.translator_google_thinking_level = "high"
                self.context.observe("新的原文", replace_id=target.observation_id)
                return "   "
            return "名字"
        self.translator._call = AsyncMock(side_effect=respond)
        self.assertEqual(await self.translator.translate_observation(target.observation_id, 1), "名字")
        self.assertEqual(captured[0], captured[1])
        self.assertNotIn("后来的", str(captured[1]))
        self.assertFalse(self.context.commit_translation(target.observation_id, 1, "名字"))
        self.assertEqual(self.context.history[0].en, "")

    async def test_scene_only_and_missing_revision_never_call_the_provider(self):
        self.translator._call = AsyncMock()
        scene = self.context.observe("", "校舍")
        self.assertEqual(await self.translator.translate_observation(scene.observation_id, 1), "")
        with self.assertRaises(ValueError):
            await self.translator.translate_observation(scene.observation_id, 2)
        with self.assertRaises(ValueError):
            await self.translator.translate_observation("missing", 1)
        self.translator._call.assert_not_awaited()

    async def test_legacy_result_cannot_commit_after_clear_load_or_invalidate(self):
        for action in ("clear", "load", "invalidate"):
            with self.subTest(action=action):
                self.context.clear()
                self.context.add_line("既読", "已读")
                self.context.current_scene_description = "旧路线的待处理场景"
                saved = copy.deepcopy(self.context.to_dict())
                started, finish = asyncio.Event(), asyncio.Event()
                async def pending(*args, **kwargs):
                    started.set()
                    await finish.wait()
                    return "过期译文"
                self.translator._call = AsyncMock(side_effect=pending)
                task = asyncio.create_task(self.translator.translate("古い結果"))
                await started.wait()
                if action == "clear":
                    self.context.clear()
                elif action == "load":
                    self.context.load_from_dict(saved)
                else:
                    self.context.invalidate_observations()
                before = copy.deepcopy(self.context.to_dict())
                finish.set()
                self.assertEqual(await task, "")
                self.assertEqual(self.context.to_dict(), before)

    async def test_legacy_await_cannot_consume_a_newer_pending_scene(self):
        self.context.current_scene_description = "这句的场景"
        started, finish = asyncio.Event(), asyncio.Event()
        async def pending(*args, **kwargs):
            started.set()
            await finish.wait()
            return "这句的译文"
        self.translator._call = AsyncMock(side_effect=pending)
        task = asyncio.create_task(self.translator.translate("この台詞"))
        await started.wait()
        self.context.current_scene_description = "下一句的场景"
        finish.set()
        await task
        self.assertEqual(self.context.history[0].scene_description, "这句的场景")
        self.assertEqual(self.context.current_scene_description, "下一句的场景")

    async def test_legacy_retry_updates_original_id_after_new_observations_arrive(self):
        original = self.context.observe("同じ台詞", "旧场景")
        self.context.commit_translation(original.observation_id, 1, "旧译文")
        started, finish = asyncio.Event(), asyncio.Event()
        async def pending(*args, **kwargs):
            started.set()
            await finish.wait()
            return "新译文"
        self.translator._call = AsyncMock(side_effect=pending)
        task = asyncio.create_task(self.translator.translate(original.jp, replace_last=True))
        await started.wait()
        later = self.context.observe("之后的观察")
        finish.set()
        await task
        self.assertEqual(self.context.history, [original, later])
        self.assertEqual(original.en, "新译文")
        self.assertEqual(later.en, "")

    async def test_pending_translation_cancel_keeps_untranslated_source(self):
        target = self.context.observe("取消しても原文は残る")
        started = asyncio.Event()
        async def pending(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()
        self.translator._call = AsyncMock(side_effect=pending)
        task = asyncio.create_task(self.translator.translate_observation(target.observation_id, 1))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.context.history, [target])
        self.assertEqual(target.en, "")

    async def test_summary_of_untranslated_observations_survives_concurrent_recent_translation(self):
        self.context.observe("先に観測した台詞", "最初の場面")
        target = self.context.observe("最近の台詞")
        started, finish = asyncio.Event(), asyncio.Event()
        captured = []
        async def pending(*args, **kwargs):
            captured.append(args)
            started.set()
            await finish.wait()
            return "原文と场景に基づく摘要"
        self.translator._call = AsyncMock(side_effect=pending)
        task = asyncio.create_task(self.translator._summarize_context())
        await started.wait()
        self.context.commit_translation(target.observation_id, 1, "最近的译文")
        self.context.observe("新增的原文")
        finish.set()
        self.assertTrue(await task)
        self.assertIn("[JP] 先に観測した台詞", captured[0][3][0]["content"])
        self.assertNotIn("[ZH]", captured[0][3][0]["content"])
        self.assertEqual(self.context.history[0], target)
        self.assertEqual(target.en, "最近的译文")


if __name__ == "__main__":
    unittest.main()

"""Offline summary requests commit only the captured context and current settings."""

import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, Mock, patch

from config import Config
from context_manager import ContextManager
from translator import Translator


class SummaryTranslationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.game_title = "测试作品"
        self.cfg.game_notes = "仅游戏内角色名保留日语。"
        self.context = ContextManager(summarize_at_tokens=1, min_recent_lines=2)
        self.context.load_from_dict({"summary": "此前已确认的剧情。"})
        self.context.current_scene_description = "旧教室。"
        for index in range(4):
            self.context.add_line(f"旧原文{index}", f"旧译文{index}")
        with patch.object(Translator, "_build_client", return_value=object()):
            self.translator = Translator(self.cfg, self.context)
        self.translator._call = AsyncMock(return_value="  更新后的已读摘要。\n")
        self.translator._log_call = Mock()

    async def start_pending_request(self):
        started = asyncio.Event()
        response = asyncio.get_running_loop().create_future()

        async def pending_call(*args, **kwargs):
            started.set()
            return await response

        self.translator._call = AsyncMock(side_effect=pending_call)
        task = asyncio.create_task(self.translator._summarize_context())
        await asyncio.wait_for(started.wait(), timeout=1)
        return task, response

    async def test_success_normalizes_and_logs_the_actual_request(self):
        self.cfg.summarizer_system_prompt = "摘要目标{max_summary_tokens}；旧输入{current_context_tokens}。"
        original_tokens = self.context.estimated_tokens
        recent = copy.deepcopy(self.context.history[-2:])

        self.assertTrue(await self.translator._summarize_context())

        self.assertEqual(self.context.summary, "更新后的已读摘要。")
        self.assertEqual(self.context.history, recent)
        self.assertFalse(self.translator.last_compact_error)
        self.translator._call.assert_awaited_once()
        request = self.translator._call.await_args
        self.assertIn(f"旧输入{original_tokens}。", request.args[2])
        message = request.args[3][0]["content"]
        self.assertIn("此前已确认的剧情。", message)
        self.assertIn("[Scene: 旧教室。]", message)
        self.assertIn("[JP] 旧原文0\n[ZH] 旧译文0", message)
        self.assertIn("[JP] 旧原文1\n[ZH] 旧译文1", message)
        self.assertNotIn("旧原文2", message)
        self.translator._log_call.assert_called_once_with(
            "SUMMARIZE", request.args[0], request.args[1], request.args[2],
            request.args[3], "更新后的已读摘要。",
        )

    async def test_no_eligible_prefix_makes_no_api_call(self):
        self.context.clear()
        self.context.add_line("一行。", "一行。")
        before = copy.deepcopy(self.context.to_dict())
        self.translator.last_compact_error = True

        self.assertFalse(await self.translator._summarize_context())

        self.assertEqual(self.context.to_dict(), before)
        self.translator._call.assert_not_awaited()
        self.translator._log_call.assert_not_called()
        self.assertFalse(self.translator.last_compact_error)

    async def test_empty_or_whitespace_response_preserves_context_and_reports_failure(self):
        before = copy.deepcopy(self.context.to_dict())
        for response in ("", " \n\t "):
            with self.subTest(response=response):
                self.translator._call = AsyncMock(return_value=response)
                with self.assertLogs("translator", level="WARNING"):
                    self.assertFalse(await self.translator._summarize_context())
                self.assertTrue(self.translator.last_compact_error)
                self.assertEqual(self.context.to_dict(), before)
                self.translator._call.assert_awaited_once()

    async def test_new_translations_and_pending_scene_survive_background_commit(self):
        self.cfg.summarizer_system_prompt = "快照输入{current_context_tokens}。"
        tokens_before = self.context.estimated_tokens
        task, response = await self.start_pending_request()
        request_before = copy.deepcopy(self.translator._call.await_args.args[3])
        self.context.current_scene_description = "新走廊。"
        self.context.add_line("新しい会話。", "新的对白。")
        self.context.current_scene_description = "尚未消费的新场景。"
        response.set_result("合并旧记忆。")

        self.assertTrue(await task)

        self.assertEqual([line.jp for line in self.context.history], ["旧原文2", "旧原文3", "新しい会話。"])
        self.assertEqual(self.context.history[-1].scene_description, "新走廊。")
        self.assertEqual(self.context.current_scene_description, "尚未消费的新场景。")
        self.assertEqual(self.translator._call.await_args.args[3], request_before)
        self.assertIn(f"快照输入{tokens_before}。", self.translator._call.await_args.args[2])
        self.assertNotIn("新しい会話。", request_before[0]["content"])

    async def test_retry_of_retained_tail_survives_background_commit(self):
        task, response = await self.start_pending_request()
        self.assertTrue(self.context.pop_last_line_if_matches("旧原文3"))
        self.context.add_line("旧原文3", "重译后的内容。")
        response.set_result("旧部分摘要。")

        self.assertTrue(await task)
        self.assertEqual([line.en for line in self.context.history], ["旧译文2", "重译后的内容。"])

    async def test_clear_or_reload_discards_old_response_without_error(self):
        original = copy.deepcopy(self.context.to_dict())
        for change in ("clear", "reload_same_data"):
            with self.subTest(change=change):
                self.context.load_from_dict(original)
                task, response = await self.start_pending_request()
                if change == "clear":
                    self.context.clear()
                else:
                    self.context.load_from_dict(original)
                expected = copy.deepcopy(self.context.to_dict())
                response.set_result("属于旧会话的摘要。")

                self.assertFalse(await task)
                self.assertEqual(self.context.to_dict(), expected)
                self.assertFalse(self.translator.last_compact_error)

    async def test_edited_old_prefix_or_base_summary_discards_result(self):
        original = copy.deepcopy(self.context.to_dict())
        for change in ("prefix", "summary"):
            with self.subTest(change=change):
                self.context.load_from_dict(original)
                task, response = await self.start_pending_request()
                if change == "prefix":
                    self.context.history[0].en = "已人工更正。"
                else:
                    self.context.summary = "新的已确认摘要。"
                expected = copy.deepcopy(self.context.to_dict())
                response.set_result("过时摘要。")

                self.assertFalse(await task)
                self.assertEqual(self.context.to_dict(), expected)
                self.assertFalse(self.translator.last_compact_error)

    async def test_changed_game_or_summary_settings_discard_result(self):
        changes = {
            "game_title": "另一个游戏", "game_notes": "新的称名约定。",
            "summarizer_provider": "google", "summarizer_model": "another-model",
            "summarizer_system_prompt": "新的摘要要求。", "summary_max_tokens": 123,
            "summarizer_max_tokens": 321, "summarizer_ollama_thinking": False,
            "summarizer_google_thinking_level": "high",
        }
        before = copy.deepcopy(self.context.to_dict())
        for field, value in changes.items():
            with self.subTest(field=field):
                old_value = getattr(self.cfg, field)
                if value == old_value:
                    value = not old_value
                task, response = await self.start_pending_request()
                setattr(self.cfg, field, value)
                response.set_result("旧设置生成的摘要。")

                self.assertFalse(await task)
                self.assertEqual(self.context.to_dict(), before)
                self.assertFalse(self.translator.last_compact_error)
                setattr(self.cfg, field, old_value)

    async def test_api_failure_preserves_context_and_returns_false(self):
        before = copy.deepcopy(self.context.to_dict())
        self.translator._call = AsyncMock(side_effect=RuntimeError("mock request failed"))
        with self.assertLogs("translator", level="ERROR"):
            self.assertFalse(await self.translator._summarize_context())
        self.assertEqual(self.context.to_dict(), before)
        self.assertTrue(self.translator.last_compact_error)
        self.translator._call.assert_awaited_once()

    async def test_cancellation_propagates_without_overwriting_newer_error_state(self):
        before = copy.deepcopy(self.context.to_dict())
        old_task, _ = await self.start_pending_request()
        self.translator._call = AsyncMock(side_effect=RuntimeError("new request failed"))
        with self.assertLogs("translator", level="ERROR"):
            self.assertFalse(await self.translator._summarize_context())
        self.assertTrue(self.translator.last_compact_error)
        old_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await old_task
        self.assertTrue(self.translator.last_compact_error)
        self.assertEqual(self.context.to_dict(), before)

    async def test_old_request_failure_does_not_override_newer_success(self):
        old_task, old_response = await self.start_pending_request()
        self.translator._call = AsyncMock(return_value="新任务提交的摘要。")
        self.assertTrue(await self.translator._summarize_context())
        after_success = copy.deepcopy(self.context.to_dict())
        old_response.set_exception(RuntimeError("old request failed"))
        with self.assertLogs("translator", level="ERROR"):
            self.assertFalse(await old_task)
        self.assertFalse(self.translator.last_compact_error)
        self.assertEqual(self.context.to_dict(), after_success)


if __name__ == "__main__":
    unittest.main()

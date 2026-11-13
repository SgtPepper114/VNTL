"""Offline checks for Chinese translation memory and vision OCR responses."""

import json
import unittest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from config import Config
from context_manager import ContextManager
from translator import Translator


class TranslationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = Config()
        self.context = ContextManager(min_recent_lines=1)
        with patch.object(Translator, "_build_client", return_value=object()):
            self.translator = Translator(self.cfg, self.context)
        self.translator._log_call = Mock()

    async def test_chinese_translation_uses_memory_and_persists_unicode(self):
        self.context.load_from_dict({"summary": "已知人物：木村。地点：学校。"})
        self.context.add_line("木村：待って。", "木村：等等。")
        self.context.current_scene_description = "木村站在走廊。"
        self.translator._call = AsyncMock(return_value="木村：这里是什么地方？")

        result = await self.translator.translate("木村：ここはどこ？")

        self.assertEqual(result, "木村：这里是什么地方？")
        args = self.translator._call.await_args.args
        self.assertIn("简体中文", args[2])
        self.assertIn("不转为罗马字", args[2])
        self.assertIn("剧透", args[2])
        messages = args[3]
        self.assertEqual(messages[-2]["content"], "木村：等等。")
        self.assertIn("[TRANSLATE THIS:]\n木村：ここはどこ？", messages[-1]["content"])
        self.assertIn("木村站在走廊。", messages[-1]["content"])

        saved = json.loads(json.dumps(self.context.to_dict(), ensure_ascii=False))
        self.assertIn("en", saved["history"][-1])  # Existing saved contexts stay compatible.
        restored = ContextManager()
        restored.current_scene_description = "另一条路线的过时场景"
        restored.load_from_dict(saved)
        self.assertIsNone(restored.current_scene_description)
        self.assertEqual(restored.history[-1].en, result)
        self.assertEqual(restored.history[-1].scene_description, "木村站在走廊。")
        self.assertEqual(restored.summary, self.context.summary)
        self.assertEqual(restored.estimated_tokens, self.context.estimated_tokens)

    async def test_screen_read_uses_descriptor_and_only_ten_recent_lines(self):
        self.cfg.descriptor_provider = "google"
        self.cfg.descriptor_model = "mock-vision"
        self.context.summary = "此前已确认的信息"
        for number in range(13):
            self.context.add_line(f"jp-{number:02}", f"中文-{number:02}")
        initial_memory = self.context.to_dict()
        self.translator._call = AsyncMock(return_value=json.dumps({
            "japanese": "木村：そこは[判读不明]だ。\n行く\n戻る",
            "scene": "昏暗走廊里有一名学生。",
        }, ensure_ascii=False))

        result = await self.translator.read_screen("mock-base64-jpeg")

        self.assertEqual(result, ("木村：そこは[判读不明]だ。\n行く\n戻る", "昏暗走廊里有一名学生。"))
        args = self.translator._call.await_args.args
        self.assertEqual(args[:2], ("google", "mock-vision"))
        self.assertIn("不得猜测缺失部分", args[2])
        content = args[3][0]["content"]
        self.assertEqual(content[0]["image_url"]["url"], "data:image/jpeg;base64,mock-base64-jpeg")
        text = content[1]["text"]
        self.assertIn("此前已确认的信息", text)
        self.assertNotIn("jp-00", text)
        self.assertNotIn("jp-02", text)
        self.assertIn("jp-03", text)
        self.assertIn("jp-12", text)
        self.assertIn("[ZH] 中文-12", text)
        self.assertEqual(self.context.to_dict(), initial_memory)

    async def test_cancelled_retry_keeps_committed_history(self):
        self.context.current_scene_description = "走廊。"
        self.context.add_line("誰？", "谁？")
        before = self.context.to_dict()
        started = asyncio.Event()

        async def pending(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        self.translator._call = AsyncMock(side_effect=pending)
        task = asyncio.create_task(self.translator.translate("誰？", replace_last=True))
        await started.wait()
        self.assertEqual(self.context.to_dict(), before)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.context.to_dict(), before)

    async def test_retry_replaces_only_after_success_and_preserves_scene(self):
        self.context.current_scene_description = "走廊。"
        self.context.add_line("誰？", "谁？")
        self.translator._call = AsyncMock(return_value="是谁？")
        await self.translator.translate("誰？", replace_last=True)
        self.assertEqual(len(self.context.history), 1)
        self.assertEqual(self.context.history[0].en, "是谁？")
        self.assertEqual(self.context.history[0].scene_description, "走廊。")
        messages = self.translator._call.await_args.args[3]
        self.assertEqual(len(messages), 1)
        self.assertIn("走廊。", messages[0]["content"])

    async def test_anthropic_screen_image_and_empty_dialogue(self):
        self.cfg.descriptor_provider = "anthropic"
        self.translator._call = AsyncMock(return_value='{"japanese":"","scene":"黑屏。"}')
        self.assertEqual(await self.translator.read_screen("jpeg"), ("", "黑屏。"))
        image = self.translator._call.await_args.args[3][0]["content"][0]
        self.assertEqual(image["source"]["data"], "jpeg")
        self.assertEqual(image["source"]["media_type"], "image/jpeg")
        self.assertEqual(self.context.history, [])

    async def test_empty_image_never_calls_provider(self):
        self.translator._call = AsyncMock()
        with self.assertRaises(ValueError):
            await self.translator.read_screen("")
        self.translator._call.assert_not_awaited()

    async def test_invalid_ocr_response_does_not_change_memory(self):
        self.context.add_line("既読", "已读")
        self.translator._call = AsyncMock(return_value="无法确定画面。")
        with self.assertRaises(ValueError):
            await self.translator.read_screen("jpeg")
        self.assertEqual(len(self.context.history), 1)
        self.assertEqual(self.context.history[0].en, "已读")

    async def test_summarization_keeps_chinese_mapping_and_recent_lines(self):
        self.context.add_line("学校", "学校")
        self.context.add_line("走ろう。", "跑吧。")
        self.translator._call = AsyncMock(return_value="人物正在学校，刚刚提议奔跑。")
        await self.translator._summarize_context()
        args = self.translator._call.await_args.args
        self.assertIn("日译中", args[2])
        self.assertNotIn("{max_summary_tokens}", args[2])
        self.assertIn("[JP] 学校\n[ZH] 学校", args[3][0]["content"])
        self.assertNotIn("[EN]", args[3][0]["content"])
        self.assertEqual(self.context.summary, "人物正在学校，刚刚提议奔跑。")
        self.assertEqual([line.en for line in self.context.history], ["跑吧。"])


class ScreenResponseTests(unittest.TestCase):
    def test_preserves_newlines_quotes_and_unknown_glyph_markers(self):
        payload = json.dumps({
            "japanese": '「行く？」\n1. はい\n2. [判读不明] "いいえ"',
            "scene": "选择画面。",
        }, ensure_ascii=False)
        self.assertEqual(Translator._parse_screen_result(payload)[0], json.loads(payload)["japanese"])

    def test_accepts_one_json_fence(self):
        self.assertEqual(Translator._parse_screen_result(
            '```json\n{"japanese":"待って。","scene":"教室。"}\n```'
        ), ("待って。", "教室。"))

    def test_rejects_missing_fields_extra_fields_wrong_types_and_prose(self):
        invalid = [
            "", "[]", "null", '{"japanese":"はい"}',
            '{"japanese":null,"scene":""}', '{"japanese":1,"scene":""}',
            '{"japanese":"はい","scene":[]}',
            '{"japanese":"はい","scene":"","translation":"是"}',
            '{"japanese":"はい","japanese":"いいえ","scene":""}',
            '结果如下：{"japanese":"はい","scene":""}',
            '{"japanese":"はい","scene":""}\n后面还有内容',
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                Translator._parse_screen_result(payload)


class ContextTests(unittest.TestCase):
    def test_chinese_context_is_not_underestimated_as_english(self):
        chinese = ContextManager(summarize_at_tokens=200, min_recent_lines=1)
        english = ContextManager(summarize_at_tokens=200, min_recent_lines=1)
        for _ in range(2):
            chinese.add_line("学校に戻る。", "学校" * 40)
            english.add_line("学校に戻る。", "ab" * 40)
        self.assertTrue(chinese.needs_summarization())
        self.assertFalse(english.needs_summarization())
        self.assertGreater(chinese.estimated_tokens, english.estimated_tokens * 2)

    def test_legacy_context_can_load_and_retry_preserves_scene(self):
        memory = ContextManager()
        memory.load_from_dict({
            "summary": "已有中文记忆。",
            "history": [{"jp": "ここだ。", "en": "就是这里。", "scene_description": "校门口。"}],
        })
        tokens_with_line = memory.estimated_tokens
        self.assertTrue(memory.pop_last_line_if_matches("ここだ。"))
        self.assertEqual(memory.current_scene_description, "校门口。")
        self.assertLess(memory.estimated_tokens, tokens_with_line)
        memory.add_line("ここだ。", "就在这里。")
        self.assertEqual(memory.history[0].scene_description, "校门口。")


if __name__ == "__main__":
    unittest.main()

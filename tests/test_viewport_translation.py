"""Offline geometry requests: one image, strict bounds, and unchanged memory."""
import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from config import Config
from context_manager import ContextManager
from translator import Translator, _VIEWPORT_SYSTEM


class ViewportParserTests(unittest.TestCase):
    def test_accepts_full_frame_crop_and_one_json_fence(self):
        self.assertEqual(Translator._parse_viewport_result('{"viewport":[0,0,1,1]}'),
                         (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(Translator._parse_viewport_result(
            '```json\n{"viewport":[0.05,0.1,0.95,0.9]}\n```'), (.05, .1, .95, .9))
        self.assertEqual(Translator._parse_viewport_result('{"viewport":[0.2,0,0.6,0.5]}'),
                         (.2, 0.0, .6, .5))
        self.assertIsNone(Translator._parse_viewport_result('{"viewport":null}'))

    def test_rejects_invalid_structure_numbers_and_small_rectangles(self):
        invalid = [
            '', 'not JSON', 'null', '[0,0,1,1]', '{}', '{"rectangle":[0,0,1,1]}',
            '{"viewport":[0,0,1,1],"confidence":1}',
            '{"viewport":[0,0,1,1],"viewport":[0,0,0.5,0.5]}',
            '{"viewport":[0,0,1]}', '{"viewport":[0,0,1,1,1]}',
            '{"viewport":{"left":0,"top":0,"right":1,"bottom":1}}',
            '{"viewport":[false,0,1,1]}', '{"viewport":[0,"0",1,1]}',
            '{"viewport":[0,0,null,1]}', '{"viewport":[0,0,NaN,1]}',
            '{"viewport":[0,0,Infinity,1]}', '{"viewport":[0,0,-Infinity,1]}',
            '{"viewport":[0,0,1e9999,1]}', '{"viewport":[-0.01,0,1,1]}',
            '{"viewport":[0,0,1.01,1]}', '{"viewport":[0.8,0,0.2,1]}',
            '{"viewport":[0,0.8,1,0.2]}', '{"viewport":[0,0,0,1]}',
            '{"viewport":[0,0,0.19,1]}', '{"viewport":[0,0,0.4,0.4]}',
            'Here: {"viewport":[0,0,1,1]}', '{"viewport":[0,0,1,1]} trailing',
            '```python\n{"viewport":[0,0,1,1]}\n```',
            '__import__("os").system("invalid")', None,
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                Translator._parse_viewport_result(payload)


class ViewportTranslationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.game_title = "PRIVATE_GAME_TITLE"
        self.cfg.game_notes = "PRIVATE_GAME_NOTES"
        self.cfg.descriptor_system_prompt = "PRIVATE_CUSTOM_SCENE_PROMPT"
        self.context = ContextManager()
        self.context.summary = "PRIVATE_STORY_SUMMARY"
        self.context.observe("PRIVATE_JAPANESE", "PRIVATE_SCENE")
        self.context.current_scene_description = "PRIVATE_CURRENT_SCENE"
        with patch.object(Translator, "_build_client", return_value=object()):
            self.translator = Translator(self.cfg, self.context)
        self.translator._log_call = Mock()

    async def test_one_image_fixed_prompt_and_no_story_or_settings_in_request(self):
        self.cfg.descriptor_provider = "google"
        self.cfg.descriptor_model = "mock-vision"
        self.cfg.descriptor_max_tokens = 4096
        self.translator._call = AsyncMock(return_value='{"viewport":[0,0.05,1,0.95]}')
        before = copy.deepcopy(self.context.to_dict())
        epoch = self.context.epoch
        result = await self.translator.locate_game_viewport("fixture-jpeg")
        self.assertEqual(result, (0.0, .05, 1.0, .95))
        args, kwargs = self.translator._call.await_args
        self.assertEqual(args[:3], ("google", "mock-vision", _VIEWPORT_SYSTEM))
        self.assertEqual(kwargs["max_tokens"], 128)
        self.assertEqual(kwargs["reasoning_effort"], self.cfg.descriptor_google_thinking_level)
        self.assertEqual(len(args[3]), 1)
        self.assertEqual(args[3][0]["role"], "user")
        content = args[3][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["image_url"]["url"], "data:image/jpeg;base64,fixture-jpeg")
        self.assertEqual(content[1]["type"], "text")
        self.assertNotIn("PRIVATE_", str(args))
        self.translator._log_call.assert_not_called()
        self.assertEqual(self.context.to_dict(), before)
        self.assertEqual(self.context.epoch, epoch)
        self.assertEqual(self.context.current_scene_description, "PRIVATE_CURRENT_SCENE")

    async def test_existing_provider_adapters_and_response_extraction(self):
        response_text = '```json\n{"viewport":[0,0,1,1]}\n```'
        for provider in ("anthropic", "openai", "google", "deepseek", "ollama"):
            with self.subTest(provider=provider):
                self.cfg.descriptor_provider = provider
                self.cfg.descriptor_model = "vision-fixture"
                create = AsyncMock()
                if provider == "anthropic":
                    create.return_value = SimpleNamespace(
                        content=[SimpleNamespace(type="text", text=response_text)], stop_reason="end_turn")
                    client = SimpleNamespace(messages=SimpleNamespace(create=create))
                else:
                    create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                        message=SimpleNamespace(content=response_text), finish_reason="stop")])
                    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                self.translator._clients[provider] = client
                self.assertEqual(await self.translator.locate_game_viewport("fixture-jpeg"), (0, 0, 1, 1))
                request = create.await_args.kwargs
                self.assertEqual(request["model"], "vision-fixture")
                self.assertEqual(request["max_completion_tokens" if provider == "openai" else "max_tokens"], 128)
                messages = request["messages"]
                self.assertEqual(len(messages), 1 if provider == "anthropic" else 2)
                image = messages[-1]["content"][0]
                if provider == "anthropic":
                    self.assertEqual(image["source"]["data"], "fixture-jpeg")
                else:
                    self.assertEqual(image["image_url"]["url"], "data:image/jpeg;base64,fixture-jpeg")
                self.assertNotIn("PRIVATE_", str(request))
        self.translator._log_call.assert_not_called()

    async def test_empty_image_and_unrecognized_frame_return_none(self):
        self.translator._call = AsyncMock(return_value='{"viewport":null}')
        for image in (None, "", "  "):
            self.assertIsNone(await self.translator.locate_game_viewport(image))
        self.translator._call.assert_not_awaited()
        self.assertIsNone(await self.translator.locate_game_viewport("fixture-jpeg"))
        self.translator._call.assert_awaited_once()

    async def test_invalid_result_and_provider_error_only_log_exception_class(self):
        before = copy.deepcopy(self.context.to_dict())
        for outcome in ('{"viewport":[0,0,NaN,1]} PRIVATE_PAYLOAD',
                        RuntimeError("PRIVATE_TOKEN https://private.invalid?token=PRIVATE_SECRET")):
            self.translator._call = AsyncMock()
            if isinstance(outcome, Exception):
                self.translator._call.side_effect = outcome
            else:
                self.translator._call.return_value = outcome
            with self.assertLogs("translator", level="WARNING") as logs:
                self.assertIsNone(await self.translator.locate_game_viewport("PRIVATE_SCREENSHOT"))
            self.assertEqual(len(logs.output), 1)
            self.assertNotIn("PRIVATE_", logs.output[0])
            self.assertNotIn("http", logs.output[0])
            self.assertIn("RuntimeError" if isinstance(outcome, Exception) else "ValueError", logs.output[0])
        self.assertEqual(self.context.to_dict(), before)
        self.translator._log_call.assert_not_called()

    async def test_cancellation_propagates_without_mutating_context(self):
        started = asyncio.Event()
        async def pending(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()
        self.translator._call = AsyncMock(side_effect=pending)
        before = copy.deepcopy(self.context.to_dict())
        task = asyncio.create_task(self.translator.locate_game_viewport("fixture-jpeg"))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.context.to_dict(), before)
        self.translator._log_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()

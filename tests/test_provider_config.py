"""Provider request checks use mocks; config tests never read the user's file."""

import json
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from config import Config, ROLE_MODEL_OPTIONS, ROLE_PROVIDER_DEFAULTS, load_config, save_config
from context_manager import ContextManager
from translator import Translator


def response_with_text(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason="stop"
    )])


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "配置.json"
        self.config_patch = patch("config.CONFIG_PATH", str(self.path))
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def write_config(self, data):
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_new_and_empty_config_use_deepseek_with_blank_keys(self):
        for data in (None, {}):
            if data is not None:
                self.write_config(data)
            cfg = load_config()
            self.assertEqual(cfg, Config())
            for role in ("translator", "descriptor", "summarizer"):
                self.assertEqual(getattr(cfg, f"{role}_provider"), "deepseek")
                self.assertEqual(getattr(cfg, f"{role}_model"), "deepseek-flash")
                self.assertEqual(getattr(cfg, f"{role}_system_prompt"), "")
            for provider in ("anthropic", "openai", "google", "deepseek"):
                self.assertEqual(getattr(cfg, f"{provider}_api_key"), "")

    def test_utf8_round_trip_preserves_existing_role_choices_and_fonts(self):
        cfg = Config()
        cfg.translator_provider = "google"
        cfg.translator_model = "custom-gemini-model"
        cfg.descriptor_provider = "anthropic"
        cfg.descriptor_model = "claude-sonnet-4-6"
        cfg.deepseek_api_key = "test-only-placeholder"
        cfg.translator_system_prompt = "请保留日语人名中的汉字。"
        cfg.translator_google_thinking_level = "high"
        cfg.overlay.en_font_family = "微软雅黑"
        save_config(cfg)
        raw = self.path.read_text(encoding="utf-8")
        self.assertIn("请保留日语人名中的汉字。", raw)
        self.assertEqual(load_config(), cfg)

    def test_legacy_backend_migration_keeps_model_and_key(self):
        self.write_config({"backend": "claude", "model": "legacy-model", "api_key": "test-key"})
        cfg = load_config()
        for role in ("translator", "descriptor", "summarizer"):
            self.assertEqual(getattr(cfg, f"{role}_provider"), "anthropic")
            self.assertEqual(getattr(cfg, f"{role}_model"), "legacy-model")
        self.assertEqual(cfg.anthropic_api_key, "test-key")

    def test_existing_per_role_choice_wins_over_legacy_and_defaults(self):
        self.write_config({
            "backend": "ollama", "ollama_model": "old-local-model",
            "translator_provider": "google", "translator_model": "saved-google-model",
            "descriptor_provider": "deepseek", "descriptor_model": "deepseek-flash",
            "translator_google_thinking_level": "high",
            "descriptor_google_thinking_level": "invalid",
        })
        cfg = load_config()
        self.assertEqual((cfg.translator_provider, cfg.translator_model), ("google", "saved-google-model"))
        self.assertEqual((cfg.descriptor_provider, cfg.descriptor_model), ("deepseek", "deepseek-flash"))
        self.assertEqual((cfg.summarizer_provider, cfg.summarizer_model), ("ollama", "old-local-model"))
        self.assertEqual(cfg.translator_google_thinking_level, "high")
        self.assertEqual(cfg.descriptor_google_thinking_level, "low")

    def test_presets_keep_legacy_models_and_exclude_nonvision_deepseek_from_ocr(self):
        for role in ("translator", "descriptor", "summarizer"):
            for provider, model in ROLE_PROVIDER_DEFAULTS[role].items():
                self.assertIn(model, ROLE_MODEL_OPTIONS[role][provider])
            self.assertIn("gemini-3.8-flash", ROLE_MODEL_OPTIONS[role]["google"])
            self.assertIn("claude-sonnet-5-5", ROLE_MODEL_OPTIONS[role]["anthropic"])
            self.assertIn("claude-sonnet-4-6", ROLE_MODEL_OPTIONS[role]["anthropic"])
        self.assertNotIn("deepseek-v4-pro", ROLE_MODEL_OPTIONS["descriptor"]["deepseek"])


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = Config()
        self.context = ContextManager(min_recent_lines=1)
        self.create = AsyncMock(return_value=response_with_text("测试译文"))
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        with patch.object(Translator, "_build_client", return_value=self.client):
            self.translator = Translator(self.cfg, self.context)
        self.translator._clients = {p: self.client for p in ("google", "deepseek", "openai", "ollama")}
        self.translator._log_call = Mock()

    async def test_deepseek_default_translation_disables_thinking(self):
        await self.translator.translate("待って。")
        kwargs = self.create.await_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-flash")
        self.assertEqual(kwargs["max_tokens"], self.cfg.translator_max_tokens)
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("max_completion_tokens", kwargs)

    async def test_deepseek_ocr_sends_jpeg_image_and_validates_result(self):
        self.create.return_value = response_with_text('{"japanese":"待って。","scene":"学校。"}')
        result = await self.translator.read_screen("test-jpeg")
        self.assertEqual(result, ("待って。", "学校。"))
        kwargs = self.create.await_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(kwargs["messages"][-1]["content"][0]["image_url"]["url"],
                         "data:image/jpeg;base64,test-jpeg")

    async def test_google_thinking_level_is_routed_per_role(self):
        for role in ("translator", "descriptor", "summarizer"):
            setattr(self.cfg, f"{role}_provider", "google")
            setattr(self.cfg, f"{role}_model", "gemini-3.8-flash")
        self.cfg.translator_google_thinking_level = "high"
        await self.translator.translate("待って。")
        self.assertEqual(self.create.await_args.kwargs["reasoning_effort"], "high")
        self.create.return_value = response_with_text('{"japanese":"はい。","scene":"教室。"}')
        await self.translator.read_screen("jpeg")
        self.assertEqual(self.create.await_args.kwargs["reasoning_effort"], "low")
        self.create.return_value = response_with_text("教室里有学生。")
        await self.translator.describe_scene("jpeg", "はい。")
        self.assertEqual(self.create.await_args.kwargs["reasoning_effort"], "low")
        self.context.add_line("はい。", "好的。")
        await self.translator._summarize_context()
        self.assertEqual(self.create.await_args.kwargs["reasoning_effort"], "low")
        self.assertNotIn("extra_body", self.create.await_args.kwargs)

    async def test_openai_and_ollama_keep_provider_specific_parameters(self):
        await self.translator._call("openai", "test-model", "system", [], 256,
                                   reasoning_effort="medium")
        self.assertEqual(self.create.await_args.kwargs["max_completion_tokens"], 256)
        self.assertNotIn("reasoning_effort", self.create.await_args.kwargs)
        await self.translator._call("ollama", "test-model", "system", [], 256, think=False)
        self.assertEqual(self.create.await_args.kwargs["extra_body"], {"think": False})

    async def test_anthropic_text_blocks_follow_thinking_blocks(self):
        create = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking="private"),
                     SimpleNamespace(type="text", text="第一句。"),
                     SimpleNamespace(type="text", text="第二句。")],
            stop_reason="end_turn",
        ))
        self.translator._clients["anthropic"] = SimpleNamespace(messages=SimpleNamespace(create=create))
        result = await self.translator._call("anthropic", "claude-sonnet-5-5", "system", [], 256)
        self.assertEqual(result, "第一句。\n第二句。")

    async def test_deepseek_client_uses_its_own_key_and_official_endpoint(self):
        self.cfg.deepseek_api_key = "test-deepseek-key"
        self.cfg.openai_api_key = "test-openai-key"
        with patch("openai.AsyncOpenAI") as create_client:
            self.translator._build_client("deepseek")
        create_client.assert_called_once()
        self.assertEqual(create_client.call_args.kwargs["api_key"], "test-deepseek-key")
        self.assertEqual(create_client.call_args.kwargs["base_url"], "https://api.deepseek.com")


class SettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from PyQt6.QtWidgets import QWidget
        from settings import SettingsDialog

        class OverlayStub(QWidget):
            def apply_appearance(self, appearance):
                pass

        self.overlay = OverlayStub()
        self.cfg = Config()
        self.cfg.translator_provider = "google"
        self.cfg.translator_model = "my-saved-model"
        self.cfg.translator_google_thinking_level = "high"
        self.saved = Mock()
        before = copy.deepcopy(self.cfg)
        self.dialog = SettingsDialog(self.cfg, self.overlay, self.saved)
        self.addCleanup(self.dialog.close)
        self.addCleanup(self.overlay.close)
        self.assertEqual(self.cfg, before)

    def test_existing_settings_and_default_prompts_survive_dialog_save(self):
        self.dialog._deepseek_key_edit.setText("test-ui-placeholder")
        self.dialog._on_save()
        self.saved.assert_called_once_with(self.cfg)
        self.assertEqual(self.cfg.deepseek_api_key, "test-ui-placeholder")
        self.assertEqual(self.cfg.translator_model, "my-saved-model")
        self.assertEqual(self.cfg.translator_google_thinking_level, "high")
        self.assertEqual(self.cfg.translator_system_prompt, "")
        self.assertEqual(self.cfg.descriptor_system_prompt, "")
        self.assertEqual(self.cfg.summarizer_system_prompt, "")

    def test_provider_switch_and_scrolling_keep_controls_usable(self):
        from PyQt6.QtWidgets import QScrollArea, QTabWidget
        from settings import _PROVIDER_KEY_MAP
        combo = self.dialog._role_combos["translator"]
        deepseek_display = next(k for k, v in _PROVIDER_KEY_MAP.items() if v == "deepseek")
        combo.setCurrentText(deepseek_display)
        self.assertEqual(self.dialog._role_model_combos["translator"].currentText(), "deepseek-flash")
        self.assertFalse(self.dialog._role_google_thinking_combos["translator"].isEnabled())
        google_display = next(k for k, v in _PROVIDER_KEY_MAP.items() if v == "google")
        combo.setCurrentText(google_display)
        self.assertTrue(self.dialog._role_google_thinking_combos["translator"].isEnabled())
        self.assertEqual(self.dialog._role_model_combos["translator"].currentText(), "gemini-3.8-flash")
        tabs = self.dialog.findChild(QTabWidget)
        self.dialog.show()
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            self.app.processEvents()
            self.assertIsInstance(tabs.widget(index), QScrollArea)
            self.assertLessEqual(self.dialog.height(), 900)


if __name__ == "__main__":
    unittest.main()

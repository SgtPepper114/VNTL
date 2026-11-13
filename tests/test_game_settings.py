"""Game context tests use temporary configuration and an offscreen dialog only."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
from config import Config, GAME_NOTES_MAX_LENGTH, GAME_TITLE_MAX_LENGTH, load_config, save_config


class GameConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "配置.json"
        patcher = patch.object(config, "CONFIG_PATH", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_config(self, data):
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_new_and_legacy_configuration_do_not_assume_any_game(self):
        for data in (None, {}, {"backend": "claude", "model": "legacy-model"}):
            with self.subTest(data=data):
                if data is not None:
                    self.write_config(data)
                cfg = load_config()
                self.assertEqual(cfg.game_title, "")
                self.assertEqual(cfg.game_notes, "")
                if data and "model" in data:
                    self.assertEqual(cfg.translator_model, "legacy-model")

    def test_unicode_context_round_trip_trims_edges_but_preserves_line_breaks(self):
        cfg = Config(game_title="  测试作品  ", game_notes="\n  第一章：学校\n既定译法：校長 → 校长。  \n")
        save_config(cfg)
        loaded = load_config()
        self.assertEqual(loaded.game_title, "测试作品")
        self.assertEqual(loaded.game_notes, "第一章：学校\n既定译法：校長 → 校长。")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["game_title"], loaded.game_title)
        self.assertEqual(raw["game_notes"], loaded.game_notes)
        for provider in ("anthropic", "openai", "google", "deepseek"):
            self.assertEqual(raw[provider + "_api_key"], "")

    def test_non_strings_are_empty_without_resetting_other_settings(self):
        for value in (None, True, False, 0, 123, [], ["title"], {}, {"title": "test"}):
            with self.subTest(value=value):
                self.write_config({"game_title": value, "game_notes": value,
                                   "controller_button": "GL"})
                cfg = load_config()
                self.assertEqual(cfg.game_title, "")
                self.assertEqual(cfg.game_notes, "")
                self.assertEqual(cfg.controller_button, "GL")

    def test_loading_context_limits_length_after_trimming(self):
        self.write_config({
            "game_title": "  " + "名" * (GAME_TITLE_MAX_LENGTH + 10) + "  ",
            "game_notes": "\n" + "注" * (GAME_NOTES_MAX_LENGTH + 10) + "\n",
        })
        cfg = load_config()
        self.assertEqual(cfg.game_title, "名" * GAME_TITLE_MAX_LENGTH)
        self.assertEqual(cfg.game_notes, "注" * GAME_NOTES_MAX_LENGTH)

    def test_saving_context_never_persists_oversized_values(self):
        cfg = Config(game_title="名" * (GAME_TITLE_MAX_LENGTH + 1),
                     game_notes="注" * (GAME_NOTES_MAX_LENGTH + 1))
        save_config(cfg)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(raw["game_title"]), GAME_TITLE_MAX_LENGTH)
        self.assertEqual(len(raw["game_notes"]), GAME_NOTES_MAX_LENGTH)


class GameSettingsTests(unittest.TestCase):
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

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.json"
        patcher = patch.object(config, "CONFIG_PATH", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cfg = Config(game_title="原作品", game_notes="原篇章\n原术语")
        save_config(self.cfg)
        self.overlay = OverlayStub()
        self.saved = Mock(side_effect=save_config)
        before = copy.deepcopy(self.cfg)
        self.dialog = SettingsDialog(self.cfg, self.overlay, self.saved)
        self.addCleanup(self.overlay.close)
        self.addCleanup(self.dialog.close)
        self.assertEqual(self.cfg, before)

    def test_first_tab_displays_current_game_and_save_applies_both_fields(self):
        from PyQt6.QtWidgets import QTabWidget
        tabs = self.dialog.findChild(QTabWidget)
        self.assertEqual(tabs.tabText(0), "游戏")
        self.assertEqual(tabs.currentIndex(), 0)
        self.assertEqual(self.dialog._game_title_edit.text(), "原作品")
        self.assertEqual(self.dialog._game_notes_edit.toPlainText(), "原篇章\n原术语")

        self.dialog._game_title_edit.setText("  新作品  ")
        self.dialog._game_notes_edit.setPlainText("\n第二章\n新版术语  \n")
        self.assertEqual(self.cfg.game_title, "原作品")
        self.assertEqual(self.cfg.game_notes, "原篇章\n原术语")
        self.saved.assert_not_called()
        self.dialog._on_save()

        self.saved.assert_called_once_with(self.cfg)
        self.assertEqual(self.cfg.game_title, "新作品")
        self.assertEqual(self.cfg.game_notes, "第二章\n新版术语")
        self.assertEqual(load_config().game_title, "新作品")
        self.assertEqual(load_config().game_notes, "第二章\n新版术语")

    def test_cancel_discards_game_edits_and_keeps_saved_file(self):
        before = copy.deepcopy(self.cfg)
        file_before = self.path.read_bytes()
        self.dialog._game_title_edit.setText("不应保存")
        self.dialog._game_notes_edit.setPlainText("不应保存的说明")
        self.dialog._on_cancel()
        self.saved.assert_not_called()
        self.assertEqual(self.cfg, before)
        self.assertEqual(self.path.read_bytes(), file_before)

    def test_ui_limits_match_configuration_limits(self):
        self.assertEqual(self.dialog._game_title_edit.maxLength(), GAME_TITLE_MAX_LENGTH)
        self.dialog._game_title_edit.setText("名" * (GAME_TITLE_MAX_LENGTH + 20))
        self.dialog._game_notes_edit.setPlainText("注" * (GAME_NOTES_MAX_LENGTH + 20))
        self.assertEqual(len(self.dialog._game_title_edit.text()), GAME_TITLE_MAX_LENGTH)
        self.assertEqual(len(self.dialog._game_notes_edit.toPlainText()), GAME_NOTES_MAX_LENGTH)
        self.assertEqual(self.dialog._game_notes_count_label.text(),
                         f"{GAME_NOTES_MAX_LENGTH} / {GAME_NOTES_MAX_LENGTH} 字")
        self.dialog._on_save()
        self.assertEqual(len(load_config().game_title), GAME_TITLE_MAX_LENGTH)
        self.assertEqual(len(load_config().game_notes), GAME_NOTES_MAX_LENGTH)

    def test_both_fields_can_be_cleared(self):
        self.dialog._game_title_edit.setText("   ")
        self.dialog._game_notes_edit.setPlainText("\n  \n")
        self.dialog._on_save()
        self.assertEqual(self.cfg.game_title, "")
        self.assertEqual(self.cfg.game_notes, "")


if __name__ == "__main__":
    unittest.main()

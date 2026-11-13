"""Furigana preferences use temporary configuration and an offscreen dialog."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
from config import Config, load_config, save_config


class FuriganaConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "config.json"
        patcher = patch.object(config, "CONFIG_PATH", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_new_and_existing_configurations_enable_furigana_by_default(self):
        for data in (None, {}, {"overlay": {"jp_font_size": 18}}):
            with self.subTest(data=data):
                if data is not None:
                    self.path.write_text(json.dumps(data), encoding="utf-8")
                self.assertIs(load_config().overlay.jp_furigana_enabled, True)

    def test_round_trip_preserves_both_states_and_existing_settings(self):
        cfg = Config(game_title="黄昏症候群 探索篇", controller_button="GL")
        cfg.overlay.jp_font_size = 18
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                cfg.overlay.jp_furigana_enabled = enabled
                save_config(cfg)
                restored = load_config()
                self.assertIs(restored.overlay.jp_furigana_enabled, enabled)
                self.assertEqual(restored.overlay.jp_font_size, 18)
                self.assertEqual(restored.game_title, cfg.game_title)
                self.assertEqual(restored.controller_button, "GL")

    def test_invalid_enable_values_do_not_reset_other_configuration(self):
        for invalid in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(value=invalid):
                self.path.write_text(json.dumps({
                    "overlay": {"jp_furigana_enabled": invalid, "jp_font_size": 22},
                    "controller_button": "GL",
                }), encoding="utf-8")
                restored = load_config()
                self.assertIs(restored.overlay.jp_furigana_enabled, True)
                self.assertEqual(restored.overlay.jp_font_size, 22)
                self.assertEqual(restored.controller_button, "GL")


class FuriganaSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from PyQt6.QtWidgets import QWidget
        from settings import SettingsDialog

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "config.json"
        patcher = patch.object(config, "CONFIG_PATH", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cfg = Config()
        save_config(self.cfg)
        self.overlay = QWidget()
        self.overlay.apply_appearance = Mock()
        self.saved = Mock(side_effect=save_config)
        self.dialog = SettingsDialog(self.cfg, self.overlay, self.saved)
        self.addCleanup(self.overlay.close)
        self.addCleanup(self.dialog.close)

    def test_toggle_previews_current_subtitle_without_saving_until_save(self):
        from PyQt6.QtWidgets import QTabWidget
        tabs = self.dialog.findChild(QTabWidget)
        appearance = tabs.widget(next(
            i for i in range(tabs.count()) if tabs.tabText(i) == "外观"
        ))
        self.assertTrue(appearance.isAncestorOf(self.dialog._jp_furigana_check))
        self.assertTrue(self.dialog._jp_furigana_check.isChecked())
        self.overlay.apply_appearance.assert_not_called()
        self.dialog._jp_furigana_check.setChecked(False)
        self.assertFalse(self.cfg.overlay.jp_furigana_enabled)
        self.overlay.apply_appearance.assert_called_once_with(self.cfg.overlay)
        self.saved.assert_not_called()
        self.assertTrue(load_config().overlay.jp_furigana_enabled)
        self.dialog._on_save()
        self.saved.assert_called_once_with(self.cfg)
        self.assertFalse(load_config().overlay.jp_furigana_enabled)

    def test_cancel_reverts_preview_without_touching_saved_file(self):
        before = copy.deepcopy(self.cfg.overlay)
        file_before = self.path.read_bytes()
        self.dialog._jp_furigana_check.setChecked(False)
        self.dialog._on_cancel()
        self.assertEqual(self.cfg.overlay, before)
        self.overlay.apply_appearance.assert_called_with(before)
        self.saved.assert_not_called()
        self.assertEqual(self.path.read_bytes(), file_before)

    def test_escape_reverts_preview_without_saving(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        self.dialog.show()
        self.dialog._jp_furigana_check.setChecked(False)
        QTest.keyClick(self.dialog, Qt.Key.Key_Escape)
        self.assertTrue(self.cfg.overlay.jp_furigana_enabled)
        self.saved.assert_not_called()

    def test_window_close_reverts_preview_without_saving(self):
        self.dialog.show()
        self.dialog._jp_furigana_check.setChecked(False)
        self.dialog.close()
        self.assertTrue(self.cfg.overlay.jp_furigana_enabled)
        self.saved.assert_not_called()

    def test_reopened_dialog_reflects_saved_disabled_state(self):
        from settings import SettingsDialog
        self.dialog._jp_furigana_check.setChecked(False)
        self.dialog._on_save()
        reopened = SettingsDialog(load_config(), self.overlay, self.saved)
        self.addCleanup(reopened.close)
        self.assertFalse(reopened._jp_furigana_check.isChecked())


if __name__ == "__main__":
    unittest.main()

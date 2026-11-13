"""Controller configuration tests; no HID device, network, or user config is used."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
from config import Config, load_config, save_config


class ControllerConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.json"
        patcher = patch.object(config, "CONFIG_PATH", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_config(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_missing_settings_enable_right_back_button_by_default(self):
        for data in (None, {}):
            with self.subTest(data=data):
                if data is not None:
                    self.write_config(data)
                cfg = load_config()
                self.assertTrue(cfg.controller_enabled)
                self.assertEqual(cfg.controller_button, "GR")

    def test_round_trip_keeps_enable_state_and_selected_button(self):
        for enabled, button in ((True, "GR"), (True, "GL"), (False, "GL")):
            with self.subTest(enabled=enabled, button=button):
                save_config(Config(controller_enabled=enabled, controller_button=button))
                restored = load_config()
                self.assertIs(restored.controller_enabled, enabled)
                self.assertEqual(restored.controller_button, button)
                for provider in ("anthropic", "openai", "google", "deepseek"):
                    self.assertEqual(getattr(restored, provider + "_api_key"), "")

    def test_invalid_buttons_use_only_right_back_button_fallback(self):
        for invalid in ("gr", "gl", "A", " GR ", "", None, 1, True, [], {}, ["GL"]):
            with self.subTest(invalid=invalid):
                self.write_config({"controller_button": invalid, "controller_enabled": False})
                cfg = load_config()
                self.assertEqual(cfg.controller_button, "GR")
                self.assertFalse(cfg.controller_enabled)

    def test_non_boolean_enable_values_do_not_become_truthy_settings(self):
        for invalid in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(invalid=invalid):
                self.write_config({"controller_enabled": invalid, "controller_button": "GL"})
                cfg = load_config()
                self.assertIs(cfg.controller_enabled, True)
                self.assertEqual(cfg.controller_button, "GL")


class ControllerSettingsTests(unittest.TestCase):
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
        self.cfg = Config(controller_enabled=False, controller_button="GL")
        save_config(self.cfg)
        self.overlay = OverlayStub()
        self.overlay.controller_status_fn = Mock(return_value="未连接手柄。")
        self.saved = Mock(side_effect=save_config)
        self.dialog = SettingsDialog(self.cfg, self.overlay, self.saved)
        self.addCleanup(self.overlay.close)
        self.addCleanup(self.dialog.close)

    def choose(self, button):
        combo = self.dialog._controller_button_combo
        combo.setCurrentIndex(combo.findData(button))

    def test_tab_reflects_saved_mapping_and_saves_only_on_request(self):
        from PyQt6.QtWidgets import QTabWidget
        tabs = self.dialog.findChild(QTabWidget)
        self.assertIn("手柄", [tabs.tabText(index) for index in range(tabs.count())])
        self.assertFalse(self.dialog._controller_enabled_check.isChecked())
        self.assertEqual(self.dialog._controller_button_combo.currentData(), "GL")
        self.assertFalse(self.dialog._controller_button_combo.isEnabled())

        self.dialog._controller_enabled_check.setChecked(True)
        self.choose("GR")
        self.assertTrue(self.dialog._controller_button_combo.isEnabled())
        self.assertFalse(self.cfg.controller_enabled)
        self.assertEqual(self.cfg.controller_button, "GL")
        self.saved.assert_not_called()

        self.dialog._on_save()
        self.saved.assert_called_once_with(self.cfg)
        self.assertTrue(self.cfg.controller_enabled)
        self.assertEqual(self.cfg.controller_button, "GR")
        self.assertTrue(load_config().controller_enabled)
        self.assertEqual(load_config().controller_button, "GR")
        self.assertFalse(self.dialog._controller_status_timer.isActive())

    def test_cancel_preserves_previous_configuration_and_file(self):
        before = copy.deepcopy(self.cfg)
        file_before = self.path.read_bytes()
        self.dialog._controller_enabled_check.setChecked(True)
        self.choose("GR")
        self.dialog._on_cancel()
        self.saved.assert_not_called()
        self.assertEqual(self.cfg, before)
        self.assertEqual(self.path.read_bytes(), file_before)
        self.assertFalse(self.dialog._controller_status_timer.isActive())

    def test_save_can_disable_monitoring_without_losing_selected_button(self):
        self.dialog._controller_enabled_check.setChecked(True)
        self.choose("GR")
        self.dialog._controller_enabled_check.setChecked(False)
        self.dialog._on_save()
        self.assertFalse(load_config().controller_enabled)
        self.assertEqual(load_config().controller_button, "GR")

    def test_connection_status_refreshes_and_handles_unavailable_callback(self):
        self.assertEqual(self.dialog._controller_status_label.text(), "未连接手柄。")
        self.assertEqual(self.dialog._controller_status_timer.interval(), 1000)
        self.assertIs(self.dialog._controller_status_timer.parent(), self.dialog)
        self.overlay.controller_status_fn.return_value = "已连接 Switch2 Pro，GR 可用。"
        self.dialog._refresh_controller_status()
        self.assertEqual(self.dialog._controller_status_label.text(), "已连接 Switch2 Pro，GR 可用。")
        self.overlay.controller_status_fn.side_effect = RuntimeError("test device removed")
        self.dialog._refresh_controller_status()
        self.assertIn("暂时无法读取", self.dialog._controller_status_label.text())
        self.overlay.controller_status_fn = None
        self.dialog._refresh_controller_status()
        self.assertIn("暂不可用", self.dialog._controller_status_label.text())


if __name__ == "__main__":
    unittest.main()

"""Controller configuration and recording UI, with a signal-based fake service."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QObject, pyqtSignal

import config
from config import Config, load_config, save_config
from controller_binding import legacy_binding


def binding(kind="button", index=4, direction=0):
    value = {"backend": "sdl", "device": "sdl:example-guid:path-hash", "name": "Example Gamepad",
             "kind": kind, "index": index, "direction": direction}
    if kind == "axis":
        value["rest"] = -1.0
    return value


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

    def test_new_settings_wait_for_device_binding(self):
        for data in (None, {}):
            with self.subTest(data=data):
                if data is not None:
                    self.write_config(data)
                cfg = load_config()
                self.assertTrue(cfg.controller_enabled)
                self.assertEqual(cfg.controller_binding, {})

    def test_legacy_rear_button_is_migrated_and_explicit_clear_stays_clear(self):
        for button in ("GR", "GL"):
            self.write_config({"controller_enabled": False, "controller_button": button})
            cfg = load_config()
            self.assertFalse(cfg.controller_enabled)
            self.assertEqual(cfg.controller_binding, legacy_binding(button))
            cfg.controller_binding = {}
            save_config(cfg)
            self.assertEqual(load_config().controller_binding, {})

    def test_round_trip_preserves_binding_and_enable_state(self):
        bindings = (binding(), binding("hat", 0, 8), binding("axis", 2, 1), legacy_binding("GL"), {})
        for enabled in (True, False):
            for value in bindings:
                with self.subTest(enabled=enabled, binding=value):
                    save_config(Config(controller_enabled=enabled, controller_binding=value))
                    restored = load_config()
                    self.assertIs(restored.controller_enabled, enabled)
                    self.assertEqual(restored.controller_binding, value)
                    for provider in ("anthropic", "openai", "google", "deepseek"):
                        self.assertEqual(getattr(restored, provider + "_api_key"), "")

    def test_invalid_bindings_stay_unbound(self):
        invalid_values = (None, True, [], "GR", {**binding(), "kind": "unknown"},
                          {**binding(), "index": True}, {**binding(), "device": "switch2:auto"},
                          {**binding("axis", 1, 1), "rest": float("nan")})
        for value in invalid_values:
            with self.subTest(value=value):
                self.write_config({"controller_binding": value, "controller_button": "GR"})
                self.assertEqual(load_config().controller_binding, {})

    def test_non_boolean_enable_values_do_not_become_truthy_settings(self):
        for invalid in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(invalid=invalid):
                self.write_config({"controller_enabled": invalid, "controller_button": "GL"})
                cfg = load_config()
                self.assertIs(cfg.controller_enabled, True)
                self.assertEqual(cfg.controller_binding, legacy_binding("GL"))


class FakeController(QObject):
    binding_learned = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.status = "测试手柄已连接"
        self.learning = False
        self.learning_message = ""
        self.devices = Mock(return_value=[{"id": binding()["device"], "name": "Example Gamepad", "backend": "sdl"}])
        self.start_learning = Mock(side_effect=self._start)
        self.cancel_learning = Mock(side_effect=self._cancel)

    def _start(self, device, include_axes=False):
        self.learning = True
        self.learning_message = "请松开所有按键"

    def _cancel(self):
        self.learning = False


class ControllerSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        self.cfg = Config(controller_enabled=False, controller_button="GL",
                          controller_binding=legacy_binding("GL"))
        save_config(self.cfg)
        self.overlay = OverlayStub()
        self.service = FakeController()
        self.overlay.controller_service = self.service
        self.overlay.controller_status_fn = Mock(return_value="未连接手柄。")
        self.saved = Mock(side_effect=save_config)
        self.dialog = SettingsDialog(self.cfg, self.overlay, self.saved)
        self.addCleanup(self.overlay.close)
        self.addCleanup(self.dialog.close)

    def start_recording(self, axes=False):
        self.dialog._controller_enabled_check.setChecked(True)
        combo = self.dialog._controller_device_combo
        combo.setCurrentIndex(combo.findData(binding()["device"]))
        self.dialog._controller_axes_check.setChecked(axes)
        self.dialog._start_controller_learning()

    def finish_recording(self, value=None):
        self.service.learning = False
        self.service.binding_learned.emit(value or binding())

    def test_recording_and_saving_apply_generic_binding_only_on_save(self):
        from PyQt6.QtWidgets import QTabWidget
        tabs = self.dialog.findChild(QTabWidget)
        self.assertIn("手柄", [tabs.tabText(index) for index in range(tabs.count())])
        self.assertFalse(self.dialog._controller_enabled_check.isChecked())
        self.assertIn("GL", self.dialog._controller_binding_label.text())
        self.assertEqual(self.dialog._controller_device_combo.currentData(), "switch2:auto")
        self.assertFalse(self.dialog._controller_learn_btn.isEnabled())
        self.start_recording()
        self.service.start_learning.assert_called_once_with(binding()["device"], include_axes=False)
        self.assertFalse(self.dialog._controller_device_combo.isEnabled())
        self.finish_recording()
        self.assertIn("按钮 5", self.dialog._controller_binding_label.text())
        self.assertEqual(self.cfg.controller_binding, legacy_binding("GL"))
        self.saved.assert_not_called()
        self.dialog._on_save()
        self.saved.assert_called_once_with(self.cfg)
        self.assertTrue(load_config().controller_enabled)
        self.assertEqual(load_config().controller_binding, binding())
        self.assertFalse(self.dialog._controller_status_timer.isActive())
        self.assertFalse(self.dialog._controller_signal_connected)

    def test_cancel_preserves_configuration_file_and_disconnects_recording(self):
        before, file_before = copy.deepcopy(self.cfg), self.path.read_bytes()
        self.start_recording()
        self.finish_recording()
        self.dialog._on_cancel()
        self.saved.assert_not_called()
        self.assertEqual(self.cfg, before)
        self.assertEqual(self.path.read_bytes(), file_before)
        self.assertFalse(self.dialog._controller_status_timer.isActive())
        self.assertFalse(self.dialog._controller_signal_connected)
        draft = copy.deepcopy(self.dialog._controller_binding_draft)
        self.service.binding_learned.emit(binding("hat", 0, 1))
        self.assertEqual(self.dialog._controller_binding_draft, draft)

    def test_explicit_cancel_and_wrong_device_discard_pending_results(self):
        self.start_recording()
        self.service.binding_learned.emit({**binding(), "device": "sdl:another"})
        self.assertTrue(self.dialog._controller_learning)
        self.dialog._cancel_controller_learning()
        self.finish_recording()
        self.assertEqual(self.dialog._controller_binding_draft, legacy_binding("GL"))

    def test_axes_are_opt_in_and_success_signal_survives_status_timer_race(self):
        self.start_recording(axes=True)
        self.service.start_learning.assert_called_once_with(binding()["device"], include_axes=True)
        self.service.learning = False
        self.service.learning_message = "录入完成"
        self.dialog._refresh_controller_status()
        self.assertIn("录入完成", self.dialog._controller_learning_label.text())
        self.finish_recording(binding("axis", 2, 1))
        self.dialog._on_save()
        self.assertEqual(load_config().controller_binding, binding("axis", 2, 1))

    def test_save_can_disable_monitoring_without_losing_recorded_button(self):
        self.start_recording()
        self.finish_recording()
        self.dialog._controller_enabled_check.setChecked(False)
        self.dialog._on_save()
        self.assertFalse(load_config().controller_enabled)
        self.assertEqual(load_config().controller_binding, binding())

    def test_clear_binding_persists_and_legacy_field_does_not_restore_it(self):
        self.dialog._clear_controller_binding()
        self.assertIn("待录入", self.dialog._controller_binding_label.text())
        self.dialog._on_save()
        self.assertEqual(load_config().controller_binding, {})

    def test_refresh_preserves_offline_selected_device(self):
        self.start_recording()
        self.finish_recording()
        self.service.devices.return_value = []
        self.dialog._refresh_controller_devices()
        self.assertEqual(self.dialog._controller_device_combo.currentData(), binding()["device"])
        self.assertIn("待连接", self.dialog._controller_device_combo.currentText())
        self.assertEqual(self.dialog._controller_binding_draft, binding())

    def test_recording_timeout_reenables_controls_and_keeps_existing_binding(self):
        self.start_recording()
        self.service.learning = False
        self.service.learning_message = "录入已超时，请重新录入"
        self.dialog._refresh_controller_status()
        self.assertTrue(self.dialog._controller_learn_btn.isEnabled())
        self.assertFalse(self.dialog._controller_cancel_learn_btn.isEnabled())
        self.assertIn("超时", self.dialog._controller_learning_label.text())
        self.assertEqual(self.dialog._controller_binding_draft, legacy_binding("GL"))

    def test_save_during_recording_cancels_it_before_callback(self):
        self.start_recording()
        def save(cfg):
            self.assertFalse(self.service.learning)
            self.service.binding_learned.emit(binding())
            save_config(cfg)
        self.saved.side_effect = save
        self.dialog._on_save()
        self.assertEqual(load_config().controller_binding, legacy_binding("GL"))

    def test_window_close_cancels_active_recording_and_ignores_late_result(self):
        self.start_recording()
        self.dialog.close()
        self.assertFalse(self.service.learning)
        self.assertFalse(self.dialog._controller_signal_connected)
        self.assertFalse(self.dialog._controller_status_timer.isActive())
        self.finish_recording()
        self.assertEqual(self.dialog._controller_binding_draft, legacy_binding("GL"))
        self.saved.assert_not_called()

    def test_connection_status_refreshes_and_handles_unavailable_callback(self):
        self.assertEqual(self.dialog._controller_status_label.text(), "未连接手柄。")
        self.assertEqual(self.dialog._controller_status_timer.interval(), 1000)
        self.overlay.controller_status_fn.return_value = "测试手柄已连接"
        self.dialog._refresh_controller_status()
        self.assertEqual(self.dialog._controller_status_label.text(), "测试手柄已连接")
        self.overlay.controller_status_fn.side_effect = RuntimeError("device removed")
        self.dialog._refresh_controller_status()
        self.assertIn("暂时无法读取", self.dialog._controller_status_label.text())
        self.overlay.controller_status_fn = None
        self.dialog._refresh_controller_status()
        self.assertEqual(self.dialog._controller_status_label.text(), self.service.status)

    def test_absent_service_keeps_settings_usable(self):
        from settings import SettingsDialog
        self.dialog.close()
        del self.overlay.controller_service
        self.overlay.controller_status_fn = None
        dialog = SettingsDialog(self.cfg, self.overlay, self.saved)
        self.addCleanup(dialog.close)
        self.assertFalse(dialog._controller_learn_btn.isEnabled())
        self.assertIn("暂不可用", dialog._controller_status_label.text())
        dialog._on_cancel()


if __name__ == "__main__":
    unittest.main()

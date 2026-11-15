"""Async settings ownership and modal behavior, using isolated offscreen widgets."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QTimer, Qt
from PyQt6.QtWidgets import QApplication

import config
from config import Config, load_config, save_config
from overlay import OverlayWindow
from screenshot_service import ScreenshotService
from settings import SettingsDialog
from tests.test_controller_settings import FakeController, binding


class SettingsLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.json"
        patcher = patch.object(config, "CONFIG_PATH", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cfg = Config(controller_enabled=True, controller_binding=binding())
        self.cfg.overlay.jp_furigana_enabled = False
        self.saved = Mock(side_effect=save_config)
        noop = lambda *args, **kwargs: None
        self.overlay = OverlayWindow(self.cfg, self.saved, noop, noop, noop, noop,
                                     None, ScreenshotService())
        self.service = FakeController()
        self.overlay.controller_service = self.service
        self.overlay.controller_status_fn = lambda: self.service.status
        self.addCleanup(self.dispose_windows)
        self.overlay.show()
        self.app.processEvents()

    @staticmethod
    def delete_pending():
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def dispose_windows(self):
        dialog = self.overlay._settings_dialog
        if dialog is not None:
            dialog.close()
        self.overlay.close()
        self.overlay.deleteLater()
        self.delete_pending()
        self.app.processEvents()

    def open_settings(self):
        # Regressing to exec() fails immediately instead of blocking the suite.
        with patch.object(SettingsDialog, "exec", side_effect=AssertionError("nested event loop")) as nested:
            self.overlay._open_settings()
            nested.assert_not_called()
        dialog = self.overlay._settings_dialog
        self.assertIsInstance(dialog, SettingsDialog)
        self.assertTrue(dialog.isVisible())
        self.assertIs(self.app.activeModalWidget(), dialog)
        return dialog

    def test_open_returns_with_application_modal_dialog_and_outer_timer_runs(self):
        callbacks = []
        QTimer.singleShot(0, lambda: callbacks.append("outer loop resumed"))
        dialog = self.open_settings()
        self.assertEqual(dialog.windowModality(), Qt.WindowModality.ApplicationModal)
        self.assertTrue(dialog.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose))
        self.assertEqual(callbacks, [])
        self.app.processEvents()
        self.assertEqual(callbacks, ["outer loop resumed"])

    def test_repeated_open_reuses_one_dialog_and_preserves_its_draft(self):
        with patch("settings.SettingsDialog", wraps=SettingsDialog) as factory:
            self.overlay._open_settings()
            dialog = self.overlay._settings_dialog
            dialog._game_title_edit.setText("尚未保存的游戏名")
            for _ in range(5):
                self.overlay._open_settings()
            factory.assert_called_once()
        self.assertIs(self.overlay._settings_dialog, dialog)
        self.assertEqual(dialog._game_title_edit.text(), "尚未保存的游戏名")
        self.assertEqual(self.cfg.game_title, "")

    def assert_finished_cleanly(self, dialog):
        self.assertIsNone(self.overlay._settings_dialog)
        self.assertIsNone(self.app.activeModalWidget())
        self.assertFalse(dialog._controller_status_timer.isActive())
        self.assertFalse(dialog._controller_signal_connected)
        self.assertFalse(self.service.learning)

    def test_save_persists_and_reopen_loads_current_values(self):
        dialog = self.open_settings()
        dialog._game_title_edit.setText("保存后的游戏名")
        dialog._start_controller_learning()
        self.assertTrue(self.service.learning)
        dialog._on_save()
        self.assert_finished_cleanly(dialog)
        self.saved.assert_called_once_with(self.cfg)
        self.assertEqual(load_config().game_title, "保存后的游戏名")
        reopened = self.open_settings()
        self.assertIsNot(reopened, dialog)
        self.assertEqual(reopened._game_title_edit.text(), "保存后的游戏名")
        self.assertTrue(reopened._controller_status_timer.isActive())
        self.delete_pending()
        self.assertTrue(sip.isdeleted(dialog))
        self.assertIs(self.overlay._settings_dialog, reopened)

    def test_cancel_reverts_preview_and_reopen_keeps_saved_configuration(self):
        dialog = self.open_settings()
        before = copy.deepcopy(self.cfg)
        dialog._game_title_edit.setText("取消的游戏名")
        dialog._on_opacity_changed(17)
        dialog._start_controller_learning()
        dialog._on_cancel()
        self.assert_finished_cleanly(dialog)
        self.assertEqual(self.cfg, before)
        self.saved.assert_not_called()
        reopened = self.open_settings()
        self.assertEqual(reopened._game_title_edit.text(), before.game_title)
        self.delete_pending()
        self.assertTrue(sip.isdeleted(dialog))
        self.assertIs(self.overlay._settings_dialog, reopened)

    def test_window_close_stops_recording_and_can_open_fresh_dialog(self):
        dialog = self.open_settings()
        dialog._start_controller_learning()
        dialog.close()
        self.assert_finished_cleanly(dialog)
        self.saved.assert_not_called()
        reopened = self.open_settings()
        self.assertIsNot(reopened, dialog)
        self.assertFalse(reopened._controller_learning)
        self.assertTrue(reopened._controller_signal_connected)
        self.delete_pending()
        self.assertTrue(sip.isdeleted(dialog))
        self.assertIs(self.overlay._settings_dialog, reopened)

    def test_external_destruction_clears_reference_without_finished_signal(self):
        dialog = self.open_settings()
        finished = Mock()
        dialog.finished.connect(finished)
        dialog.deleteLater()
        self.delete_pending()
        finished.assert_not_called()
        self.assertTrue(sip.isdeleted(dialog))
        self.assertIsNone(self.overlay._settings_dialog)
        self.assertIsNot(self.open_settings(), dialog)


if __name__ == "__main__":
    unittest.main()

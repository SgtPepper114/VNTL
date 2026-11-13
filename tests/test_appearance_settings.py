"""Background alpha controls share live preview and reversible settings state."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
from config import Config, load_config, save_config


class AppearanceSettingsTests(unittest.TestCase):
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
        self.cfg.overlay.bg_color = [20, 30, 40, 210]
        self.cfg.overlay.jp_color = [220, 230, 240, 245]
        self.cfg.overlay.en_color = [170, 180, 190, 200]
        save_config(self.cfg)
        self.original = copy.deepcopy(self.cfg.overlay)
        self.overlay = QWidget()
        self.overlay.apply_appearance = Mock()
        self.saved = Mock(side_effect=save_config)
        self.dialog = SettingsDialog(self.cfg, self.overlay, self.saved)
        self.addCleanup(self.overlay.close)
        self.addCleanup(self.dialog.close)

    def choose_background(self, rgba):
        from PyQt6.QtGui import QColor
        with patch("settings.QColorDialog.getColor", return_value=QColor(*rgba)):
            self.dialog._bg_btn.click()

    def test_slider_supports_fully_transparent_and_opaque_background_only(self):
        self.assertEqual(self.dialog._opacity_slider.minimum(), 0)
        self.assertEqual(self.dialog._opacity_slider.maximum(), 100)
        for value, alpha in ((0, 0), (100, 255)):
            with self.subTest(value=value):
                self.overlay.apply_appearance.reset_mock()
                self.dialog._opacity_slider.setValue(value)
                self.assertEqual(self.cfg.overlay.bg_color, [20, 30, 40, alpha])
                self.assertEqual(self.dialog._opacity_lbl.text(), f"{value}%")
                self.assertEqual(self.cfg.overlay.jp_color, self.original.jp_color)
                self.assertEqual(self.cfg.overlay.en_color, self.original.en_color)
                self.overlay.apply_appearance.assert_called_once_with(self.cfg.overlay)
        self.saved.assert_not_called()
        self.assertEqual(load_config().overlay, self.original)

    def test_color_picker_syncs_slider_without_quantizing_alpha_or_duplicate_preview(self):
        self.choose_background([80, 90, 100, 129])
        self.assertEqual(self.cfg.overlay.bg_color, [80, 90, 100, 129])
        self.assertEqual(self.dialog._opacity_slider.value(), 51)
        self.assertEqual(self.dialog._opacity_lbl.text(), "51%")
        self.assertIn("rgba(80,90,100,129)", self.dialog._bg_btn.styleSheet())
        self.overlay.apply_appearance.assert_called_once_with(self.cfg.overlay)
        self.assertEqual(self.cfg.overlay.jp_color, self.original.jp_color)
        self.assertEqual(self.cfg.overlay.en_color, self.original.en_color)

        # The temporary signal block must end before the next user adjustment.
        self.overlay.apply_appearance.reset_mock()
        self.dialog._opacity_slider.setValue(52)
        self.assertEqual(self.cfg.overlay.bg_color, [80, 90, 100, 133])
        self.overlay.apply_appearance.assert_called_once_with(self.cfg.overlay)

    def test_cancelled_color_picker_leaves_preview_and_controls_unchanged(self):
        from PyQt6.QtGui import QColor
        value = self.dialog._opacity_slider.value()
        with patch("settings.QColorDialog.getColor", return_value=QColor()):
            self.dialog._bg_btn.click()
        self.assertEqual(self.cfg.overlay, self.original)
        self.assertEqual(self.dialog._opacity_slider.value(), value)
        self.assertEqual(self.dialog._opacity_lbl.text(), f"{value}%")
        self.overlay.apply_appearance.assert_not_called()

    def test_save_preserves_exact_picker_alpha_and_reopens_at_matching_percentage(self):
        from settings import SettingsDialog
        self.choose_background([80, 90, 100, 129])
        self.assertEqual(load_config().overlay, self.original)
        self.dialog._on_save()
        self.saved.assert_called_once_with(self.cfg)
        restored = load_config()
        self.assertEqual(restored.overlay.bg_color, [80, 90, 100, 129])
        reopened = SettingsDialog(restored, self.overlay, self.saved)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened._opacity_slider.value(), 51)
        self.assertEqual(reopened._opacity_lbl.text(), "51%")

    def test_slider_percentage_stays_consistent_after_save_and_reopen(self):
        from settings import SettingsDialog
        self.dialog._opacity_slider.setValue(50)
        self.dialog._on_save()
        reopened = SettingsDialog(load_config(), self.overlay, self.saved)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened._opacity_slider.value(), 50)
        self.assertEqual(reopened._opacity_lbl.text(), "50%")

    def test_cancel_restores_background_and_live_preview_without_saving(self):
        file_before = self.path.read_bytes()
        self.choose_background([80, 90, 100, 129])
        self.dialog._opacity_slider.setValue(0)
        self.dialog._on_cancel()
        self.assertEqual(self.cfg.overlay, self.original)
        self.overlay.apply_appearance.assert_called_with(self.original)
        self.saved.assert_not_called()
        self.assertEqual(self.path.read_bytes(), file_before)


if __name__ == "__main__":
    unittest.main()

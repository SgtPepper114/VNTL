"""Observation preferences and UI isolation; no capture, model or user config."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import config
from config import Config, load_config, save_config


class ObservationConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "config.json"
        patcher = patch.object(config, "CONFIG_PATH", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, **data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_legacy_configuration_gets_manual_observation_defaults(self):
        self.write(game_title="原游戏", screenshot_enabled=True,
                   context_summarize_tokens=12000)
        cfg = load_config()
        self.assertTrue(cfg.observation_enabled)
        self.assertEqual(cfg.translation_mode, "manual")
        self.assertEqual((cfg.ocr_stable_ms, cfg.ocr_poll_ms, cfg.ocr_region_top_percent),
                         (600, 200, 0))
        self.assertTrue(cfg.scene_observation_enabled)
        self.assertEqual(cfg.scene_observation_interval, 10.0)
        self.assertTrue(cfg.screenshot_enabled)
        self.assertEqual(cfg.context_summarize_tokens, 12000)
        self.assertEqual(cfg.game_title, "原游戏")

    def test_round_trip_preserves_independent_scene_and_legacy_switches(self):
        cfg = Config(observation_enabled=False, translation_mode="auto",
                     ocr_stable_ms=850, ocr_poll_ms=350, ocr_region_top_percent=60,
                     scene_observation_enabled=False, scene_observation_interval=12.5,
                     screenshot_enabled=True, controller_button="GL")
        save_config(cfg)
        self.assertEqual(load_config(), cfg)

    def test_invalid_values_fall_back_without_losing_other_fields(self):
        for invalid in (None, True, False, "bad", [], {}, float("nan"), float("inf")):
            with self.subTest(value=invalid):
                self.write(ocr_stable_ms=invalid, ocr_poll_ms=invalid,
                           ocr_region_top_percent=invalid,
                           scene_observation_interval=invalid,
                           translation_mode=invalid, game_title="应保留")
                cfg = load_config()
                self.assertEqual((cfg.ocr_stable_ms, cfg.ocr_poll_ms,
                                  cfg.ocr_region_top_percent, cfg.scene_observation_interval),
                                 (600, 200, 0, 10.0))
                self.assertEqual(cfg.translation_mode, "manual")
                self.assertEqual(cfg.game_title, "应保留")
        for invalid in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(boolean=invalid):
                self.write(observation_enabled=invalid, scene_observation_enabled=invalid)
                self.assertTrue(load_config().observation_enabled)
                self.assertTrue(load_config().scene_observation_enabled)

    def test_numeric_limits_clamp_each_field_independently(self):
        for number, expected in ((-1, (200, 100, 0, 3.0)),
                                 (9999, (2000, 1000, 90, 120.0))):
            with self.subTest(number=number):
                self.write(ocr_stable_ms=number, ocr_poll_ms=number,
                           ocr_region_top_percent=number, scene_observation_interval=number,
                           controller_button="GL")
                cfg = load_config()
                self.assertEqual((cfg.ocr_stable_ms, cfg.ocr_poll_ms,
                                  cfg.ocr_region_top_percent, cfg.scene_observation_interval),
                                 expected)
                self.assertEqual(cfg.controller_button, "GL")


class ObservationSettingsTests(unittest.TestCase):
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
        self.overlay.update_observation_controls = Mock()
        self.saved = Mock(side_effect=save_config)
        self.dialog = SettingsDialog(self.cfg, self.overlay, self.saved)
        self.addCleanup(self.overlay.close)
        self.addCleanup(self.dialog.close)

    def edit(self):
        self.dialog._translation_mode_combo.setCurrentIndex(
            self.dialog._translation_mode_combo.findData("auto"))
        self.dialog._ocr_stable_spin.setValue(800)
        self.dialog._ocr_poll_spin.setValue(300)
        self.dialog._ocr_region_top_spin.setValue(60)
        self.dialog._scene_observation_interval_spin.setValue(15.5)
        self.dialog._scene_observation_check.setChecked(False)
        self.dialog._observation_enabled_check.setChecked(False)

    def test_save_applies_every_observation_setting_and_refreshes_controls(self):
        from PyQt6.QtWidgets import QTabWidget
        tabs = self.dialog.findChild(QTabWidget)
        self.assertEqual(tabs.tabText(0), "游戏")
        self.assertEqual(tabs.tabText(1), "识屏")
        before = copy.deepcopy(self.cfg)
        self.edit()
        self.assertEqual(self.cfg, before)
        self.saved.assert_not_called()
        self.dialog._on_save()
        cfg = load_config()
        self.assertFalse(cfg.observation_enabled)
        self.assertEqual(cfg.translation_mode, "auto")
        self.assertEqual((cfg.ocr_stable_ms, cfg.ocr_poll_ms, cfg.ocr_region_top_percent),
                         (800, 300, 60))
        self.assertFalse(cfg.scene_observation_enabled)
        self.assertEqual(cfg.scene_observation_interval, 15.5)
        self.saved.assert_called_once_with(self.cfg)
        self.overlay.update_observation_controls.assert_called_once_with()

    def test_cancel_does_not_change_configuration_or_saved_file(self):
        before = copy.deepcopy(self.cfg)
        file_before = self.path.read_bytes()
        self.edit()
        self.dialog.reject()
        self.assertEqual(self.cfg, before)
        self.assertEqual(self.path.read_bytes(), file_before)
        self.saved.assert_not_called()
        self.overlay.update_observation_controls.assert_not_called()

    def test_scene_interval_requires_both_scene_and_observation_switches(self):
        interval = self.dialog._scene_observation_interval_spin
        self.assertTrue(interval.isEnabled())
        self.assertTrue(self.dialog._scene_thresh_slider.isEnabled())
        self.dialog._scene_observation_check.setChecked(False)
        self.assertFalse(interval.isEnabled())
        self.assertFalse(self.dialog._scene_thresh_slider.isEnabled())
        self.dialog._observation_enabled_check.setChecked(False)
        self.dialog._scene_observation_check.setChecked(True)
        self.assertFalse(interval.isEnabled())
        self.dialog._observation_enabled_check.setChecked(True)
        self.assertTrue(interval.isEnabled())
        self.assertTrue(self.dialog._scene_thresh_slider.isEnabled())

    def test_scene_threshold_belongs_to_observation_and_preserves_legacy_cooldown(self):
        from PyQt6.QtWidgets import QTabWidget
        from settings import _slider_to_threshold
        tabs = self.dialog.findChild(QTabWidget)
        observation = tabs.widget(next(i for i in range(tabs.count())
                                       if tabs.tabText(i) == "识屏"))
        self.assertTrue(observation.isAncestorOf(self.dialog._scene_thresh_slider))
        self.assertFalse(hasattr(self.dialog, "_scene_cooldown_spin"))
        self.cfg.scene_change_cooldown = 3.5
        self.dialog._scene_thresh_slider.setValue(65)
        self.assertEqual(self.cfg.scene_change_threshold, 5.0)
        self.dialog._on_save()
        restored = load_config()
        self.assertAlmostEqual(restored.scene_change_threshold, _slider_to_threshold(65))
        self.assertEqual(restored.scene_change_cooldown, 3.5)


class ObservationOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from overlay import OverlayWindow
        self.cfg = Config()
        self.cfg.overlay.jp_furigana_enabled = False
        self.saved = Mock()
        self.requested = Mock()
        self.overlay = OverlayWindow(self.cfg, self.saved, Mock(), Mock(), Mock(),
                                     Mock(), Mock(), Mock(), request_screen_fn=lambda: self.requested())
        self.addCleanup(self.overlay.close)

    def test_background_status_keeps_subtitles_backlog_and_capture_available(self):
        self.overlay.update_text("学校", "学校")
        for status in ("正在观察", "正在预译", "场景观察失败", "缓存已就绪"):
            self.overlay.set_observation_status(status)
            self.assertEqual(self.overlay._jp_label.text(), "学校")
            self.assertEqual(self.overlay._en_label.text(), "学校")
            self.assertEqual(self.overlay._backlog, [("学校", "学校")])
            self.assertEqual(self.overlay._observation_status_label.text(), status)
            self.assertFalse(self.overlay.is_busy)
            self.assertTrue(self.overlay._capture_btn.isEnabled())
        self.overlay._capture_btn.click()
        self.requested.assert_called_once_with()

    def test_quick_controls_save_configuration_without_triggering_capture(self):
        self.overlay._translation_mode_btn.click()
        self.assertEqual(self.cfg.translation_mode, "auto")
        self.assertEqual(self.overlay._translation_mode_btn.text(), "自动显示")
        self.overlay._observation_btn.click()
        self.assertFalse(self.cfg.observation_enabled)
        self.assertIn("暂停", self.overlay._observation_status_label.text())
        self.overlay._observation_btn.click()
        self.assertTrue(self.cfg.observation_enabled)
        self.assertEqual(self.saved.call_count, 3)
        self.saved.assert_called_with(self.cfg)
        self.requested.assert_not_called()

    def test_background_status_does_not_clear_a_foreground_busy_operation(self):
        self.overlay.set_busy(True)
        self.overlay.set_observation_status("后台预译完成")
        self.assertTrue(self.overlay.is_busy)
        self.assertFalse(self.overlay._capture_btn.isEnabled())
        self.assertFalse(self.overlay._translation_mode_btn.isEnabled())
        self.overlay.set_busy(False)
        self.assertTrue(self.overlay._capture_btn.isEnabled())

    def test_minimum_width_controls_do_not_overlap_and_status_stays_below_subtitles(self):
        self.overlay.resize(480, 150)
        self.overlay.set_observation_status("后台状态" * 100)
        self.overlay.show()
        for _ in range(5):
            self.app.processEvents()
        controls = [self.overlay._capture_btn, self.overlay._window_btn,
                    self.overlay._translation_mode_btn, self.overlay._observation_btn,
                    self.overlay._retry_btn, self.overlay._context_btn,
                    self.overlay._backlog_btn, self.overlay._gear_btn, self.overlay._exit_btn]
        for left, right in zip(controls, controls[1:]):
            self.assertLess(left.geometry().right(), right.geometry().left())
        self.assertEqual(self.overlay.width(), 480)
        self.assertGreater(self.overlay._observation_status_label.geometry().top(),
                           self.overlay._subtitle_scroll.geometry().bottom())

    def test_context_viewer_distinguishes_pending_text_and_scene_only_records(self):
        from context_manager import DialogueLine
        from overlay import ContextViewerWindow
        viewer = ContextViewerWindow(self.cfg.overlay)
        self.addCleanup(viewer.close)
        lines = [DialogueLine("学校", ""), DialogueLine("", "", "进入走廊")]
        before = copy.deepcopy(lines)
        viewer.set_history(lines)
        first, second = viewer._history_entries
        self.assertEqual(first[1].text(), "学校")
        self.assertEqual(first[2].text(), "原文待译")
        self.assertEqual(second[0].text(), "场景观察：进入走廊")
        self.assertTrue(second[1].isHidden())
        self.assertIn("场景观察", second[2].text())
        self.assertEqual(lines, before)


if __name__ == "__main__":
    unittest.main()

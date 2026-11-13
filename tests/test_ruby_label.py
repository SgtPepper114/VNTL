"""Exercise actual offscreen layout and the existing subtitle/history paths."""
import copy
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest

from config import Config
from context_manager import DialogueLine
from furigana import RubySpan
from ruby_label import RubyLabel


class RubyLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_label(self, text="教室には誰もいない。", enabled=True):
        label = RubyLabel(text, furigana_enabled=enabled)
        label.setFont(QFont("MS Gothic", 15))
        self.addCleanup(label.close)
        return label

    def test_current_text_toggles_without_changing_source_or_requiring_translation(self):
        label = self.make_label()
        original = label.text()
        height_with_ruby = label.heightForWidth(500)
        self.assertTrue(label._has_ruby())
        label.setFuriganaEnabled(False)
        self.assertFalse(label._has_ruby())
        self.assertLess(label.heightForWidth(500), height_with_ruby)
        self.assertEqual(label.text(), original)
        label.setFuriganaEnabled(True)
        self.assertTrue(label._has_ruby())
        self.assertEqual(label.text(), original)

    def test_off_switch_does_not_call_dictionary(self):
        with patch("ruby_label.annotate", side_effect=AssertionError("disabled")):
            label = self.make_label(enabled=False)
            label.setText("新しい教室")
            label.clear()
        self.assertEqual(label.text(), "")

    def test_wrapping_keeps_all_text_and_ruby_inside_computed_bounds(self):
        label = self.make_label("教室には誰もいない。\n学校の廊下を歩いていた。" * 4)
        for width in (45, 140, 480, 900):
            with self.subTest(width=width):
                cells, height = label._layout(width)
                self.assertEqual("".join(c.text for c in cells), label.text().replace("\n", ""))
                self.assertTrue(all(c.rect.right() <= width + 0.01 for c in cells))
                self.assertTrue(all(c.rect.bottom() <= height for c in cells))
                self.assertTrue(all(c.ruby_height > 0 for c in cells if c.reading))
                label.resize(width, height)
                self.assertFalse(label.grab().isNull())
        self.assertGreater(label.heightForWidth(140), label.heightForWidth(900))

    def test_markup_is_literal_and_copy_contains_only_original(self):
        source = '<b>学校</b> & 「教室」'
        label = self.make_label(source)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setSelection(0, len(source))
        self.assertEqual(label.textFormat(), Qt.TextFormat.PlainText)
        self.assertEqual(label.selectedText(), source)

    def test_copy_and_word_selection_preserve_supplementary_unicode(self):
        source = "😀学校へ行く𠮷。"
        label = self.make_label(source)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                      Qt.TextInteractionFlag.TextSelectableByKeyboard)
        label.resize(500, label.heightForWidth(500))
        label.show()
        self.app.processEvents()
        QTest.keyClick(label, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(label.selectedText(), source)
        with patch("ruby_label.QApplication") as app:
            QTest.keyClick(label, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
            app.clipboard().setText.assert_called_once_with(source)
        word = next(c for c in label._layout(500)[0] if c.text == "学校")
        QTest.mouseDClick(label, Qt.MouseButton.LeftButton, pos=word.rect.center().toPoint())
        self.assertEqual(label.selectedText(), "学校")
        self.assertEqual(label._selection_start(), 1)
        label._select_span(source.index("𠮷"), 1)
        self.assertEqual(label.selectedText(), "𠮷")

    def test_small_overlay_scrolls_to_every_line_with_ruby_and_large_fonts(self):
        from overlay import OverlayWindow
        cfg = Config()
        overlay = OverlayWindow(cfg, Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        self.addCleanup(overlay.close)
        overlay.show()
        source = "教室には誰もいない。\n学校の廊下を歩いていた。" * 8
        overlay.update_text(source, "教室里没有人。\n她沿着学校的走廊往前走。")
        overlay.resize(480, 150)
        for _ in range(8):
            self.app.processEvents()
        for enabled, size in ((True, 10), (False, 10), (True, 24)):
            with self.subTest(enabled=enabled, font_size=size):
                appearance = copy.deepcopy(cfg.overlay)
                appearance.jp_furigana_enabled = enabled
                appearance.jp_font_size = size
                overlay.apply_appearance(appearance)
                for _ in range(8):
                    self.app.processEvents()
                label = overlay._jp_label
                self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
                scroll = overlay._subtitle_scroll
                scrollbar = scroll.verticalScrollBar()
                self.assertGreater(scrollbar.maximum(), 0)
                scrollbar.setValue(scrollbar.maximum())
                self.app.processEvents()
                bottom = overlay._en_label.mapTo(scroll.viewport(), overlay._en_label.rect().bottomLeft()).y()
                self.assertLess(bottom, scroll.viewport().height())
                self.assertEqual(label.text(), source)
        overlay.update_text("学校", "学校")
        for _ in range(8):
            self.app.processEvents()
        self.assertEqual(overlay._subtitle_scroll.verticalScrollBar().value(), 0)

    def test_long_reading_wrap_and_font_changes_update_geometry(self):
        with patch("ruby_label.annotate", return_value=(RubySpan("一", "いっしゅんかん"),)):
            label = self.make_label("一")
        cells, _ = label._layout(30)
        self.assertEqual(cells[0].rect.width(), 30)
        before = label.heightForWidth(300)
        label.setFont(QFont("MS Gothic", 30))
        self.assertGreater(label.heightForWidth(300), before)

    def test_overlay_backlog_and_memory_share_toggle_without_mutating_content(self):
        from overlay import OverlayWindow, BacklogWindow, ContextViewerWindow
        cfg = Config()
        overlay = OverlayWindow(cfg, Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        self.addCleanup(overlay.close)
        source, translation = "この教室には誰もいない。", "这间教室里没有人。"
        overlay.update_text(source, translation)
        overlay._backlog_win = BacklogWindow(cfg.overlay)
        self.addCleanup(overlay._backlog_win.close)
        overlay._backlog_win.append_entry(source, translation)
        overlay._context_viewer = ContextViewerWindow(cfg.overlay)
        self.addCleanup(overlay._context_viewer.close)
        line = DialogueLine(jp=source, en=translation)
        overlay._context_viewer.set_history([line])
        labels = (overlay._jp_label, overlay._backlog_win._entries[0][0],
                  overlay._context_viewer._history_entries[0][1])
        self.assertTrue(all(label._has_ruby() for label in labels))
        appearance = copy.deepcopy(cfg.overlay)
        appearance.jp_furigana_enabled = False
        overlay.apply_appearance(appearance)
        self.assertTrue(all(not label._has_ruby() for label in labels))
        self.assertTrue(all(label.text() == source for label in labels))
        self.assertEqual(overlay._en_label.text(), translation)
        self.assertEqual(overlay._backlog, [(source, translation)])
        self.assertEqual(line.jp, source)
        appearance.jp_furigana_enabled = True
        overlay.apply_appearance(appearance)
        self.assertTrue(all(label._has_ruby() for label in labels))
        overlay.restore_display()
        self.assertEqual(overlay._jp_label.text(), source)
        overlay.show_error_text("学校", "识屏失败")
        self.assertTrue(overlay._jp_label._has_ruby())
        self.assertEqual(overlay._jp_label.text(), "学校")


if __name__ == "__main__":
    unittest.main()

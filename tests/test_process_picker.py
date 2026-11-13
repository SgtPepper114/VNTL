"""Process picker checks use a fake catalog and offscreen Qt, never OS processes."""

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QTreeWidget

from overlay import ProcessPickerDialog
from process_catalog import ProcessEntry


class ProcessPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.entries = [
            ProcessEntry(pid=9002, name="helper.exe", window_titles=()),
            ProcessEntry(pid=4101, name="duckstation-qt.exe", window_titles=(
                "黄昏症候群 探索篇", "Controller Settings",
            )),
            ProcessEntry(pid=8003, name="agent.exe", window_titles=()),
            ProcessEntry(pid=4102, name="duckstation-qt.exe", window_titles=("另一个游戏",)),
        ]
        patcher = patch("overlay.list_process_entries", return_value=self.entries)
        self.catalog = patcher.start()
        self.addCleanup(patcher.stop)
        self.attach = Mock(return_value=True)
        self.dialog = ProcessPickerDialog(self.attach)
        self.addCleanup(self.dialog.close)
        self.tree = self.dialog.findChild(QTreeWidget)
        self.assertIsNotNone(self.tree)

    def groups(self):
        return [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]

    def rows(self):
        return [group.child(i) for group in self.groups()
                for i in range(group.childCount())]

    def row(self, pid):
        return next(row for row in self.rows()
                    if row.data(0, Qt.ItemDataRole.UserRole) == pid)

    def visible_pids(self):
        # Collapsed children still match the filter. Ignore presentation collapse.
        return {row.data(0, Qt.ItemDataRole.UserRole) for row in self.rows()
                if not row.isHidden() and not row.parent().isHidden()}

    def select(self, pid):
        row = self.row(pid)
        self.tree.setCurrentItem(row)
        row.setSelected(True)
        return row

    def test_applications_come_first_with_background_collapsed(self):
        apps, background = self.groups()
        self.assertTrue(apps.text(0).startswith("应用"))
        self.assertTrue(background.text(0).startswith("后台进程"))
        self.assertEqual(
            {apps.child(i).data(0, Qt.ItemDataRole.UserRole) for i in range(apps.childCount())},
            {4101, 4102},
        )
        self.assertEqual(
            {background.child(i).data(0, Qt.ItemDataRole.UserRole)
             for i in range(background.childCount())},
            {8003, 9002},
        )
        self.assertTrue(apps.isExpanded())
        self.assertFalse(background.isExpanded())
        self.assertFalse(self.dialog._attach_btn.isEnabled())
        self.attach.assert_not_called()

    def test_process_rows_show_window_title_executable_and_pid(self):
        game = self.row(4101)
        self.assertIn("黄昏症候群 探索篇", game.text(0))
        self.assertEqual(game.text(1), "duckstation-qt.exe")
        self.assertEqual(game.text(2), "4101")
        self.assertEqual(self.row(4102).text(1), game.text(1))
        self.assertNotEqual(self.row(4102).text(2), game.text(2))

    def test_filter_matches_title_other_window_executable_or_pid(self):
        for query, expected in (
            ("黄昏症候群", {4101}),
            ("  CONTROLLER SETTINGS  ", {4101}),
            ("DUCKSTATION-QT", {4101, 4102}),
            ("4102", {4102}),
            ("9002", {9002}),
            ("does-not-exist", set()),
        ):
            with self.subTest(query=query):
                self.dialog._filter.setText(query)
                self.assertEqual(self.visible_pids(), expected)
                self.attach.assert_not_called()
        self.dialog._filter.clear()
        self.assertEqual(self.visible_pids(), {4101, 4102, 8003, 9002})

    def test_search_expands_background_matches(self):
        self.dialog._filter.setText("helper.exe")
        helper = self.row(9002)
        self.assertTrue(helper.parent().isExpanded())
        self.assertFalse(helper.isHidden())
        self.assertFalse(helper.parent().isHidden())

    def test_groups_cannot_be_selected_or_attached(self):
        for group in self.groups():
            self.assertFalse(group.flags() & Qt.ItemFlag.ItemIsSelectable)
            self.tree.setCurrentItem(group)
            self.dialog._do_attach()
            self.tree.itemActivated.emit(group, 0)
            self.assertFalse(self.dialog._attach_btn.isEnabled())
            self.attach.assert_not_called()

    def test_selection_button_attaches_exact_pid_for_duplicate_executable(self):
        self.select(4102)
        self.assertTrue(self.dialog._attach_btn.isEnabled())
        self.dialog._attach_btn.click()
        self.attach.assert_called_once_with(4102)
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Accepted)

    def test_double_click_attaches_exact_pid_once(self):
        self.dialog.show()
        game = self.select(4101)
        self.app.processEvents()
        position = self.tree.visualItemRect(game).center()
        QTest.mouseClick(self.tree.viewport(), Qt.MouseButton.LeftButton, pos=position)
        QTest.mouseDClick(self.tree.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.attach.assert_called_once_with(4101)
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Accepted)

    def test_enter_attaches_selected_process_once(self):
        self.dialog.show()
        self.select(4102)
        self.tree.setFocus()
        self.app.processEvents()
        QTest.keyClick(self.tree, Qt.Key.Key_Return)
        self.attach.assert_called_once_with(4102)
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Accepted)

    def test_refresh_keeps_existing_selection_and_current_filter(self):
        self.dialog._filter.setText("duckstation")
        self.select(4102)
        self.catalog.return_value = list(reversed(self.entries))
        self.dialog._populate()
        self.assertEqual(self.dialog._filter.text(), "duckstation")
        self.assertEqual(self.visible_pids(), {4101, 4102})
        current = self.tree.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(current.data(0, Qt.ItemDataRole.UserRole), 4102)
        self.assertTrue(self.dialog._attach_btn.isEnabled())
        self.attach.assert_not_called()

    def test_disappeared_selection_cannot_attach_after_refresh(self):
        self.select(4101)
        self.catalog.return_value = [entry for entry in self.entries if entry.pid != 4101]
        self.dialog._populate()
        self.assertFalse(self.dialog._attach_btn.isEnabled())
        self.dialog._do_attach()
        self.attach.assert_not_called()
        self.assertNotIn(4101, self.visible_pids())

    def test_hidden_selection_cannot_attach_after_filter(self):
        self.select(4101)
        self.dialog._filter.setText("helper.exe")
        self.assertFalse(self.dialog._attach_btn.isEnabled())
        self.dialog._do_attach()
        self.attach.assert_not_called()

    def test_failed_attachment_keeps_dialog_open_and_shows_error(self):
        self.dialog.show()
        self.attach.return_value = False
        self.select(4102)
        with patch("overlay.QMessageBox.warning") as warning:
            self.dialog._attach_btn.click()
        self.attach.assert_called_once_with(4102)
        self.assertTrue(self.dialog.isVisible())
        self.assertNotEqual(self.dialog.result(), QDialog.DialogCode.Accepted)
        warning.assert_called_once()
        self.assertIn("4102", warning.call_args.args[2])


if __name__ == "__main__":
    unittest.main()

"""Window grouping tests never activate windows or open target processes."""
import ctypes
import ctypes.wintypes as wt
from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import process_catalog as module
from process_catalog import ProcessEntry, list_process_entries


class FakeWindows:
    def __init__(self, windows, *, enumeration_ok=True, dwm_supported=True):
        self.windows = windows
        self.enumeration_ok = enumeration_ok
        self.get_cloaked = self._get_cloaked if dwm_supported else None

    def enum_windows(self, callback, lparam):
        for hwnd in self.windows:
            if not callback(hwnd, lparam):
                return False
        return self.enumeration_ok

    def is_visible(self, hwnd):
        return self.windows[hwnd].get("visible", True)

    def get_style(self, hwnd, _index):
        return self.windows[hwnd].get("style", 0)

    def title_length(self, hwnd):
        if self.windows[hwnd].get("disappeared"):
            raise OSError("window closed")
        return len(self.windows[hwnd].get("title", ""))

    def get_title(self, hwnd, buffer, size):
        title = self.windows[hwnd].get("title", "")[:size-1]
        buffer.value = title
        return len(title)

    def get_pid(self, hwnd, output):
        ctypes.cast(output, ctypes.POINTER(wt.DWORD)).contents.value = self.windows[hwnd]["pid"]
        return 1

    def _get_cloaked(self, hwnd, _attribute, output, _size):
        ctypes.cast(output, ctypes.POINTER(wt.DWORD)).contents.value = self.windows[hwnd].get("cloaked", 0)
        return self.windows[hwnd].get("dwm_error", 0)


class ProcessCatalogTests(unittest.TestCase):
    def collect(self, windows, processes, **kwargs):
        with patch.object(module.sys, "platform", "win32"), \
             patch.object(module, "_window_api", return_value=FakeWindows(windows, **kwargs)), \
             patch.object(module, "list_processes", return_value=processes):
            return list_process_entries()

    def test_groups_unicode_and_minimized_windows_first_and_deduplicates_titles(self):
        windows = {
            0x100000001: {"pid": 20, "title": "トワイライトシンドローム ～探索編～"},
            0x100000002: {"pid": 20, "title": "コントローラー設定", "minimized": True},
            0x100000003: {"pid": 20, "title": "トワイライトシンドローム ～探索編～"},
            0x100000004: {"pid": 10, "title": "最小化中的游戏", "minimized": True},
        }
        processes = [(30, "aaa-background.exe"), (20, "DuckStation.exe"), (10, "Game.exe")]
        entries = self.collect(windows, processes)
        self.assertEqual([entry.pid for entry in entries], [20, 10, 30])
        self.assertEqual(entries[0].window_titles, ("コントローラー設定", "トワイライトシンドローム ～探索編～"))
        self.assertTrue(entries[1].is_application)
        self.assertFalse(entries[2].is_application)
        self.assertEqual(self.collect(dict(reversed(list(windows.items()))), list(reversed(processes))), entries)

    def test_excludes_tools_hidden_cloaked_and_untitled_without_dropping_processes(self):
        windows = {
            1: {"pid": 1, "title": "Tool", "style": 0x80},
            2: {"pid": 2, "title": "Hidden", "visible": False},
            3: {"pid": 3, "title": "Other desktop", "cloaked": 2},
            4: {"pid": 4, "title": "   "},
            5: {"pid": 5, "title": "Closing", "disappeared": True},
            6: {"pid": 6, "title": "Visible"},
        }
        entries = self.collect(windows, [(pid, f"process{pid}.exe") for pid in range(1, 7)])
        self.assertEqual(len(entries), 6)
        self.assertEqual([entry.pid for entry in entries if entry.is_application], [6])

    def test_unavailable_or_failed_dwm_query_does_not_hide_valid_windows(self):
        windows = {1: {"pid": 1, "title": "Window", "cloaked": 2, "dwm_error": -1}}
        for supported in (True, False):
            with self.subTest(dwm_supported=supported):
                entries = self.collect(windows, [(1, "game.exe")], dwm_supported=supported)
                self.assertTrue(entries[0].is_application)

    def test_failed_enumeration_discards_partial_grouping_not_processes(self):
        entries = self.collect({1: {"pid": 2, "title": "Partial"}},
                               [(2, "b.exe"), (1, "a.exe")], enumeration_ok=False)
        self.assertEqual(entries, [ProcessEntry(1, "a.exe"), ProcessEntry(2, "b.exe")])

    def test_api_failure_and_non_windows_keep_background_snapshot(self):
        processes = [(2, "b.exe"), (1, "a.exe")]
        with patch.object(module, "list_processes", return_value=processes):
            with patch.object(module.sys, "platform", "win32"), \
                 patch.object(module, "_window_api", side_effect=OSError("unavailable")):
                self.assertEqual(list_process_entries(), [ProcessEntry(1, "a.exe"), ProcessEntry(2, "b.exe")])
            with patch.object(module.sys, "platform", "linux"), \
                 patch.object(module, "_window_api") as api:
                self.assertEqual(list_process_entries(), [ProcessEntry(1, "a.exe"), ProcessEntry(2, "b.exe")])
                api.assert_not_called()

    def test_entry_is_immutable(self):
        entry = ProcessEntry(1, "game.exe", ("Game",))
        with self.assertRaises(FrozenInstanceError):
            entry.pid = 2


if __name__ == "__main__":
    unittest.main()

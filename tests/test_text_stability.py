import unittest
from dataclasses import FrozenInstanceError

from text_stability import StableText, TextStabilizer


class TextStabilityTests(unittest.TestCase):
    def test_waits_until_final_observed_text_is_quiet(self):
        tracker = TextStabilizer(0.6)
        self.assertIsNone(tracker.observe("教", 0.0))
        self.assertIsNone(tracker.observe("教室", 0.4))
        self.assertIsNone(tracker.observe("教室へ", 0.8))
        self.assertIsNone(tracker.observe("教室へ", 1.3))
        self.assertFalse(tracker.is_stable)
        self.assertEqual(tracker.observe("教室へ", 1.4), StableText("教室へ", False, 0))
        self.assertTrue(tracker.is_stable)
        self.assertIsNone(tracker.observe("教室へ", 8.0))

    def test_normalization_only_affects_comparison_and_keeps_latest_original(self):
        tracker = TextStabilizer()
        tracker.observe("ミカ：１２段\nなの？", 10.0)
        latest = "ミカ:12段　なの？"
        self.assertEqual(tracker.observe(latest, 10.6), StableText(latest, False, 0))
        self.assertEqual(tracker.current_text, latest)
        self.assertIsNone(tracker.observe("ミカ：１２段\nなの？", 11.0))
        self.assertTrue(tracker.is_stable)

    def test_append_is_full_current_page_not_only_suffix(self):
        tracker = TextStabilizer()
        tracker.observe("誰もいない。", 0.0)
        first = tracker.observe("誰もいない。", 0.6)
        self.assertFalse(first.append)
        self.assertIsNone(tracker.observe("誰もいない。\nでも声がする。", 1.0))
        self.assertFalse(tracker.is_stable)
        event = tracker.observe("誰もいない。\nでも声がする。", 1.6)
        self.assertEqual(event, StableText("誰もいない。\nでも声がする。", True, 0))

    def test_blank_immediately_invalidates_current_without_repeated_page_advances(self):
        tracker = TextStabilizer()
        tracker.observe("ミカ：何？", 0.0)
        tracker.observe("ミカ：何？", 0.6)
        self.assertIsNone(tracker.observe(" \n\t　", 0.7))
        self.assertEqual(tracker.current_text, "")
        self.assertFalse(tracker.is_stable)
        self.assertEqual(tracker.page, 1)
        tracker.observe("", 1.0)
        tracker.observe("", 2.0)
        self.assertEqual(tracker.page, 1)
        tracker.observe("ミカ：何？", 3.0)
        self.assertEqual(tracker.observe("ミカ：何？", 3.6), StableText("ミカ：何？", False, 1))

    def test_clear_during_typing_discards_unemitted_candidate(self):
        tracker = TextStabilizer()
        tracker.observe("古い文字", 0.0)
        tracker.observe("", 0.3)
        self.assertIsNone(tracker.observe("", 0.7))
        tracker.observe("新", 0.8)
        self.assertEqual(tracker.observe("新", 1.4), StableText("新", False, 1))

    def test_obvious_replacement_creates_page_and_waits_again(self):
        tracker = TextStabilizer()
        tracker.observe("屋上には誰もいなかった。", 0.0)
        tracker.observe("屋上には誰もいなかった。", 0.6)
        self.assertIsNone(tracker.observe("1. 教室を調べる\n2. 廊下に戻る", 1.0))
        self.assertEqual(tracker.page, 1)
        event = tracker.observe("1. 教室を調べる\n2. 廊下に戻る", 1.6)
        self.assertFalse(event.append)
        self.assertEqual(event.page, 1)

    def test_explicit_page_change_can_repeat_identical_sentence(self):
        tracker = TextStabilizer()
        tracker.observe("……", 0.0)
        tracker.observe("……", 0.6)
        self.assertIsNone(tracker.observe("……", 1.0, page_changed=True))
        self.assertEqual(tracker.observe("……", 1.6), StableText("……", False, 1))

    def test_explicit_change_and_blank_only_advance_once(self):
        tracker = TextStabilizer()
        tracker.observe("何かいる。", 0.0)
        tracker.observe("", 0.3, page_changed=True)
        self.assertEqual(tracker.page, 1)

    def test_small_ocr_correction_stays_on_same_page_but_restarts_quiet_time(self):
        tracker = TextStabilizer()
        tracker.observe("旧核舎に入った。", 0.0)
        tracker.observe("旧校舎に入った。", 0.5)
        self.assertIsNone(tracker.observe("旧校舎に入った。", 0.6))
        self.assertEqual(tracker.observe("旧校舎に入った。", 1.1), StableText("旧校舎に入った。", False, 0))

    def test_transient_shrink_cannot_repeat_already_emitted_content(self):
        tracker = TextStabilizer()
        tracker.observe("夜の教室", 0.0)
        tracker.observe("夜の教室", 0.6)
        tracker.observe("夜の", 0.7)
        tracker.observe("夜の教室", 0.8)
        self.assertIsNone(tracker.observe("夜の教室", 1.4))
        self.assertTrue(tracker.is_stable)
        self.assertEqual(tracker.page, 0)

    def test_clock_rewind_restarts_debounce(self):
        tracker = TextStabilizer()
        tracker.observe("時間", 10.0)
        self.assertIsNone(tracker.observe("時間", 1.0))
        self.assertEqual(tracker.observe("時間", 1.6), StableText("時間", False, 0))

    def test_reset_forgets_candidate_and_emitted_pages(self):
        tracker = TextStabilizer()
        tracker.observe("同じ文", 0.0)
        tracker.observe("同じ文", 0.6)
        tracker.reset()
        self.assertEqual((tracker.page, tracker.current_text, tracker.is_stable), (0, "", False))
        tracker.observe("同じ文", 1.0)
        self.assertEqual(tracker.observe("同じ文", 1.6), StableText("同じ文", False, 0))

    def test_zero_wait_and_invalid_times(self):
        tracker = TextStabilizer(0)
        event = tracker.observe("即時", 0.0)
        self.assertEqual(event, StableText("即時", False, 0))
        with self.assertRaises(FrozenInstanceError):
            event.text = "変更"
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TextStabilizer(value)
        with self.assertRaises(ValueError):
            tracker.observe("時刻", float("nan"))


if __name__ == "__main__":
    unittest.main()

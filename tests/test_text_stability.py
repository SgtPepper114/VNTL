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
        self.assertIsNone(tracker.observe("即時", 0.0))
        event = tracker.observe("即時", 0.01)
        self.assertEqual(event, StableText("即時", False, 0))
        with self.assertRaises(FrozenInstanceError):
            event.text = "変更"
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TextStabilizer(value)
        with self.assertRaises(ValueError):
            tracker.observe("時刻", float("nan"))

    def test_blinking_tail_keeps_one_high_confidence_original(self):
        tracker = TextStabilizer()
        text = "ミカ：１２時に旧校舎へ行こう。"
        events = []
        for index in range(20):
            extra = "「" if index % 2 else ""
            event = tracker.observe(text + extra, index * .2,
                                    boxes=((10, 10, 240 + 30 * bool(extra), 40),),
                                    confidence=.87 if not extra else .84)
            if event:
                events.append(event)
            if events:
                self.assertTrue(tracker.is_stable)
                self.assertEqual(tracker.current_text, text)
        self.assertEqual(events, [StableText(text, False, 0, "fuzzy")])
        self.assertEqual(tracker.emitted_text, text)

    def test_cached_tail_excursion_stays_usable_and_new_single_character_can_append(self):
        tracker = TextStabilizer()
        text = "今夜は旧校舎の音楽室へ行こう"
        tracker.observe(text, 0)
        tracker.observe(text, .6)
        self.assertIsNone(tracker.observe(text + "「", .8))
        self.assertTrue(tracker.is_stable)
        self.assertEqual(tracker.current_text, text)
        tracker.observe(text, 1.0)
        self.assertIsNone(tracker.observe(text + "「", 2.0))
        self.assertIsNone(tracker.observe(text + "「", 2.8))
        self.assertEqual(tracker.current_text, text)
        self.assertIsNone(tracker.observe(text + "よ", 3.0))
        self.assertEqual(tracker.observe(text + "よ", 3.6), StableText(text + "よ", True, 0, "fuzzy"))

    def test_lower_confidence_small_addition_requests_visual_verification(self):
        tracker = TextStabilizer()
        text = "今夜は旧校舎の音楽室へ行こう"
        tracker.observe(text, 0, confidence=.99)
        self.assertIsNone(tracker.observe(text + "よ", .2, confidence=.8))
        self.assertIsNone(tracker.observe(text + "よ", .6, confidence=.8))
        self.assertEqual(tracker.observe(text + "よ", .8, confidence=.8), StableText(text, False, 0, "fuzzy"))

    def test_long_dark_cursor_phase_keeps_the_earlier_high_confidence_frame(self):
        tracker = TextStabilizer()
        text = "ミカ：１２時に旧校舎へ行こう。"
        tracker.observe(text, 0, confidence=.872)
        for now in (.2, .4, .6):
            self.assertIsNone(tracker.observe(text + "「", now, confidence=.869))
        event = tracker.observe(text + "「", .8, confidence=.869)
        self.assertEqual(event, StableText(text, False, 0, "fuzzy"))
        self.assertIsNone(tracker.observe(text + "「", 1.4, confidence=.869))
        self.assertEqual(tracker.current_text, text)

    def test_monotonic_single_character_growth_waits_for_the_final_text(self):
        tracker = TextStabilizer()
        text = "今夜は旧校舎の音楽室へ行こう"
        for index, length in enumerate(range(10, len(text) + 1)):
            self.assertIsNone(tracker.observe(text[:length], index * .2))
        self.assertEqual(tracker.observe(text, index * .2 + .6), StableText(text, False, 0))

    def test_sorted_line_boxes_allow_short_tail_noise_without_repeated_events(self):
        tracker = TextStabilizer()
        boxes = ((10, 10, 80, 30), (10, 40, 80, 60))
        events = []
        variants = ("テストあ", "テストぃ", "テストぅ")
        for index in range(12):
            event = tracker.observe(variants[index % 3], index * .2,
                                    boxes=boxes if index % 2 else tuple(reversed(boxes)))
            if event:
                events.append(event)
            if events:
                self.assertTrue(tracker.is_stable)
                self.assertEqual(tracker.current_text, events[0].text)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reason, "boxes")

    def test_new_line_resets_geometry_confirmation(self):
        tracker = TextStabilizer()
        one_line = ((10, 10, 80, 30),)
        two_lines = one_line + ((10, 40, 80, 60),)
        tracker.observe("テストあ", 0, boxes=one_line)
        tracker.observe("テストぃ", .4, boxes=one_line)
        self.assertIsNone(tracker.observe("テストぅ", .6, boxes=two_lines))
        self.assertIsNone(tracker.observe("テストえ", 1.0, boxes=two_lines))
        event = tracker.observe("テストお", 1.2, boxes=two_lines)
        self.assertEqual(event.reason, "boxes")

    def test_four_second_timeout_retains_most_complete_original(self):
        tracker = TextStabilizer()
        text = "一二三四五六七八九十百千万円冊頁巻章節項"
        for index, length in enumerate((5, 8, 11, 14)):
            self.assertIsNone(tracker.observe(text[:length], index))
        event = tracker.observe(text[:17], 4)
        self.assertEqual(event, StableText(text[:17], False, 0, "timeout"))

    def test_same_length_new_page_needs_fresh_confirmation(self):
        tracker = TextStabilizer()
        tracker.observe("誰もいない教室", 0)
        tracker.observe("誰もいない教室", .6)
        self.assertIsNone(tracker.observe("遠く響く足音だ", .8))
        self.assertFalse(tracker.is_stable)
        self.assertEqual(tracker.emitted_text, "")
        self.assertEqual(tracker.observe("遠く響く足音だ", 1.4), StableText("遠く響く足音だ", False, 1))

    def test_large_relocation_does_not_reuse_cached_stability_on_first_frame(self):
        tracker = TextStabilizer()
        tracker.observe("同じ台詞", 0, boxes=((10, 10, 90, 40),))
        tracker.observe("同じ台詞", .6, boxes=((10, 10, 90, 40),))
        self.assertIsNone(tracker.observe("同じ台詞", .8, boxes=((10, 100, 90, 130),)))
        self.assertFalse(tracker.is_stable)
        self.assertIsNone(tracker.observe("同じ台詞", 1.4, boxes=((10, 100, 90, 130),)))
        self.assertTrue(tracker.is_stable)

    def test_return_to_cached_page_discards_an_abandoned_growth_timeout(self):
        tracker = TextStabilizer()
        text = "夜中の学校で誰かが静かに歩いていた。"
        tracker.observe(text, 0)
        tracker.observe(text, .6)
        tracker.observe(text + "窓の外にも何かがいる。", .8)
        self.assertFalse(tracker.is_stable)
        self.assertIsNone(tracker.observe(text, 1.0))
        self.assertTrue(tracker.is_stable)
        new_text = text + "しかし、遠くで音が鳴った。"
        self.assertIsNone(tracker.observe(new_text, 4.9))
        self.assertFalse(tracker.is_stable)
        event = tracker.observe(new_text, 5.5)
        self.assertEqual(event, StableText(new_text, True, 0))

    def test_confirmed_alternating_line_layouts_keep_the_cached_page_usable(self):
        tracker = TextStabilizer()
        text = "ミカ：今夜は旧校舎の音楽室へ行こう。"
        dialogue = ((10, 100, 450, 140),)
        with_extra = ((460, 10, 480, 30),) + dialogue
        tracker.observe(text, 0, boxes=dialogue)
        self.assertIsNotNone(tracker.observe(text, .6, boxes=dialogue))
        self.assertIsNone(tracker.observe("!\n" + text, .8, boxes=with_extra))
        self.assertFalse(tracker.is_stable)
        self.assertIsNotNone(tracker.observe("!\n" + text, 1.4, boxes=with_extra))
        representative = tracker.current_text
        for index in range(10):
            has_extra = index % 2
            event = tracker.observe("!\n" + text if has_extra else text, 1.6 + index * .2,
                                    boxes=tuple(reversed(with_extra)) if has_extra else dialogue)
            self.assertIsNone(event)
            self.assertTrue(tracker.is_stable)
            self.assertEqual(tracker.current_text, representative)

    def test_return_from_unconfirmed_layout_recovers_cached_page_and_reset_forgets_layouts(self):
        tracker = TextStabilizer()
        text = "ミカ：今夜は旧校舎の音楽室へ行こう。"
        dialogue = ((10, 100, 450, 140),)
        with_extra = ((460, 10, 480, 30),) + dialogue
        tracker.observe(text, 0, boxes=dialogue)
        tracker.observe(text, .6, boxes=dialogue)
        tracker.observe("!\n" + text, .8, boxes=with_extra)
        self.assertFalse(tracker.is_stable)
        self.assertIsNone(tracker.observe(text, 1.0, boxes=dialogue))
        self.assertTrue(tracker.is_stable)
        tracker.observe("", 1.2)
        self.assertIsNone(tracker.observe(text, 1.4, boxes=dialogue))
        self.assertFalse(tracker.is_stable)
        self.assertEqual(tracker.emitted_text, "")

    def test_confirmation_remembers_layouts_of_observed_text_variants(self):
        tracker = TextStabilizer()
        text = "ミカ：今夜は旧校舎の音楽室へ行こう。"
        clean = ((10, 100, 450, 140),)
        wide = ((10, 100, 490, 140),)
        with_extra = ((500, 10, 520, 30),) + wide
        tracker.observe(text + "[", 0, boxes=wide, confidence=.8)
        tracker.observe(text, .2, boxes=clean, confidence=.9)
        first = tracker.observe(text, .8, boxes=clean, confidence=.9)
        self.assertEqual(first.text, text)
        tracker.observe("!\n" + text + "[", 1.0, boxes=with_extra)
        self.assertIsNotNone(tracker.observe("!\n" + text + "[", 1.6, boxes=with_extra))
        self.assertIsNone(tracker.observe(text + "[", 1.8, boxes=wide))
        self.assertTrue(tracker.is_stable)


if __name__ == "__main__":
    unittest.main()

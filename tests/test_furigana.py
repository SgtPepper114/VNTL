import importlib.util
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import furigana
from furigana import RubySpan, annotate


class Morpheme:
    def __init__(self, text, start, reading, oov=False):
        self.text = text
        self.start = start
        self.reading = reading
        self.oov = oov

    def surface(self):
        return self.text

    def begin(self):
        return self.start

    def end(self):
        return self.start + len(self.text)

    def reading_form(self):
        return self.reading

    def is_oov(self):
        return self.oov


class FuriganaTests(unittest.TestCase):
    def setUp(self):
        furigana._annotate_cached.cache_clear()
        self.engine = Mock()
        self.get_engine = patch.object(furigana, "_get_tokenizer", return_value=self.engine)
        self.get_engine.start()
        self.addCleanup(self.get_engine.stop)
        self.addCleanup(furigana._annotate_cached.cache_clear)

    def parse(self, text, items):
        self.engine.tokenize.return_value = [Morpheme(*item) for item in items]
        spans = annotate(text)
        self.assertEqual("".join(span.text for span in spans), text)
        return spans

    def test_compound_uses_whole_reading_not_per_kanji_guess(self):
        self.assertEqual(self.parse("黄昏症候群", [("黄昏症候群", 0, "タソガレショウコウグン")]),
                         (RubySpan("黄昏症候群", "たそがれしょうこうぐん"),))

    def test_prefix_and_suffix_okurigana_stay_plain(self):
        self.assertEqual(self.parse("お母さん", [("お母さん", 0, "オカアサン")]),
                         (RubySpan("お"), RubySpan("母", "かあ"), RubySpan("さん")))
        self.assertEqual(self.parse("食べる", [("食べる", 0, "タベル")]),
                         (RubySpan("食", "た"), RubySpan("べる")))

    def test_internal_kana_aligns_only_when_unique(self):
        self.assertEqual(self.parse("取り扱い", [("取り扱い", 0, "トリアツカイ")]),
                         (RubySpan("取", "と"), RubySpan("り"), RubySpan("扱", "あつか"), RubySpan("い")))

    def test_ambiguous_internal_kana_keeps_whole_word(self):
        self.assertEqual(self.parse("甲あ乙", [("甲あ乙", 0, "カアアキ")]),
                         (RubySpan("甲あ乙", "かああき"),))

    def test_no_per_kanji_split_for_irregular_reading(self):
        self.assertEqual(self.parse("大人", [("大人", 0, "オトナ")]),
                         (RubySpan("大人", "おとな"),))

    def test_preserves_whitespace_newlines_emoji_and_literal_markup(self):
        text = ' \t<b>\U0001f47b学校</b>&\r\n来た。  '
        spans = self.parse(text, [("学校", text.index("学校"), "ガッコウ"),
                                  ("来た", text.index("来た"), "キタ")])
        self.assertIn(RubySpan("学校", "がっこう"), spans)
        self.assertEqual(spans[0].text, ' \t<b>\U0001f47b')
        self.assertEqual(spans[-1].text, "た。  ")

    def test_oov_invalid_reading_and_nonkanji_never_get_ruby(self):
        spans = self.parse("未知 既知 カナ", [("未知", 0, "ミチ", True),
                                              ("既知", 3, "?漢字"),
                                              ("カナ", 6, "カナ")])
        self.assertEqual(spans, (RubySpan("未知 既知 カナ"),))

    def test_numeric_or_punctuation_token_keeps_plain(self):
        self.assertEqual(self.parse("3人", [("3人", 0, "サンニン")]), (RubySpan("3人"),))

    def test_empty_and_kana_only_need_no_tokenizer(self):
        self.assertEqual(annotate(""), ())
        self.assertEqual(annotate("かな <b>&\n"), (RubySpan("かな <b>&\n"),))
        self.engine.tokenize.assert_not_called()

    def test_invalid_offsets_fall_back_without_losing_text(self):
        with self.assertLogs("furigana", "WARNING"):
            self.assertEqual(self.parse("原文", [("別文", 0, "ベツブン")]), (RubySpan("原文"),))

    def test_parser_failure_falls_back_and_is_cached(self):
        self.engine.tokenize.side_effect = RuntimeError("test error")
        with self.assertLogs("furigana", "WARNING"):
            self.assertEqual(annotate("原文"), (RubySpan("原文"),))
        self.assertEqual(annotate("原文"), (RubySpan("原文"),))
        self.engine.tokenize.assert_called_once()

    def test_missing_dictionary_falls_back(self):
        with patch.object(furigana, "_get_tokenizer", return_value=None):
            self.assertEqual(annotate("原文"), (RubySpan("原文"),))

    def test_repeated_line_is_cached_and_cache_has_bound(self):
        self.engine.tokenize.return_value = [Morpheme("学校", 0, "ガッコウ")]
        first = annotate("学校")
        self.assertIs(annotate("学校"), first)
        self.engine.tokenize.assert_called_once()
        self.assertEqual(furigana._annotate_cached.cache_parameters()["maxsize"], 256)
        with self.assertRaises(FrozenInstanceError):
            first[0].reading = "invalid"

    def test_oversized_input_does_not_initialize_or_fill_cache(self):
        text = "漢" * (furigana._MAX_TEXT_LENGTH + 1)
        self.assertEqual(annotate(text), (RubySpan(text),))
        self.engine.tokenize.assert_not_called()
        self.assertEqual(furigana._annotate_cached.cache_info().currsize, 0)


@unittest.skipUnless(
    importlib.util.find_spec("sudachipy") is not None
    and importlib.util.find_spec("sudachidict_core") is not None,
    "Real Sudachi integration requires the project's installed tokenizer and dictionary",
)
class FuriganaIntegrationTests(unittest.TestCase):
    """Real local readings for pinned SudachiPy 0.6.11 / core 20260428."""

    def setUp(self):
        furigana._annotate_cached.cache_clear()
        self.addCleanup(furigana._annotate_cached.cache_clear)

    def assertAnnotated(self, text, expected):
        spans = annotate(text)
        self.assertEqual("".join(span.text for span in spans), text)
        self.assertEqual(spans, expected)

    def test_real_dictionary_today_and_compound_readings(self):
        self.assertAnnotated("今日は学校。", (
            RubySpan("今日", "きょう"), RubySpan("は"),
            RubySpan("学校", "がっこう"), RubySpan("。"),
        ))
        self.assertAnnotated("黄昏症候群", (
            RubySpan("黄昏", "たそがれ"), RubySpan("症候群", "しょうこうぐん"),
        ))

    def test_real_dictionary_okurigana_and_internal_kana(self):
        cases = (
            ("お母さんは食べる。", (
                RubySpan("お"), RubySpan("母", "かあ"), RubySpan("さんは"),
                RubySpan("食", "た"), RubySpan("べる。"),
            )),
            ("取り扱いを教える。", (
                RubySpan("取", "と"), RubySpan("り"), RubySpan("扱", "あつか"),
                RubySpan("いを"), RubySpan("教", "おし"), RubySpan("える。"),
            )),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertAnnotated(text, expected)

    def test_real_dictionary_reading_of_sei_changes_with_word(self):
        cases = (
            ("生きる。", (RubySpan("生", "い"), RubySpan("きる。"))),
            ("生ビール。", (RubySpan("生", "なま"), RubySpan("ビール。"))),
            ("生まれる。", (RubySpan("生", "う"), RubySpan("まれる。"))),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertAnnotated(text, expected)

    def test_real_offsets_preserve_emoji_literal_markup_and_newlines(self):
        text = ' \t🦉<b>学校</b>&\r\n「来た？」\n\n  '
        self.assertAnnotated(text, (
            RubySpan(' \t🦉<b>'), RubySpan("学校", "がっこう"),
            RubySpan('</b>&\r\n「'), RubySpan("来", "き"),
            RubySpan('た？」\n\n  '),
        ))

    def test_real_unknown_kanji_remain_plain_without_hiding_known_words(self):
        # This rare-character sequence is OOV in the pinned dictionary. Its
        # surface fallback must never be presented as a kana pronunciation.
        unknown = "𠮷𠮷𠮷"
        analyzer = furigana._get_tokenizer()
        self.assertTrue(all(word.is_oov() for word in analyzer.tokenize(unknown)))
        self.assertAnnotated(unknown + " 🦉学校", (
            RubySpan(unknown + " 🦉"), RubySpan("学校", "がっこう"),
        ))

    def test_real_multiline_subtitle_preserves_offsets_across_all_lines(self):
        text = "今日は学校へ行った。\r\n" * 200
        spans = annotate(text)
        self.assertEqual("".join(span.text for span in spans), text)
        ruby = [(span.text, span.reading) for span in spans if span.reading]
        self.assertEqual(ruby, [("今日", "きょう"), ("学校", "がっこう"), ("行", "い")] * 200)


if __name__ == "__main__":
    unittest.main()

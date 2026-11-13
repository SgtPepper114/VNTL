"""Local, conservative ruby readings; the original subtitle is never rewritten."""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RubySpan:
    text: str
    reading: str = ""


_KANJI_CHARS = "\u3005\u3006\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f"
_KANJI = re.compile(f"[{_KANJI_CHARS}]")
_RUNS = re.compile(f"[{_KANJI_CHARS}]+|[\u3041-\u3096\u309d\u309e\u30a1-\u30fa\u30fd\u30fe\u30fc]+")
_HIRAGANA = re.compile(r"[\u3041-\u3096\u309d\u309e\u30fc]+\Z")
_UNINITIALIZED = object()
_tokenizer = _UNINITIALIZED
_dictionary = None
_split_mode = None
_lock = threading.RLock()
_MAX_TEXT_LENGTH = 8192
_MAX_ALIGNMENT_LENGTH = 128


def _hiragana(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        chr(ord(char) - 0x60)
        if "\u30a1" <= char <= "\u30f6" or char in "\u30fd\u30fe"
        else char
        for char in text
    )


def _get_tokenizer():
    """Create one dictionary/tokenizer, only when a kanji subtitle needs it."""
    global _tokenizer, _dictionary, _split_mode
    if _tokenizer is _UNINITIALIZED:
        try:
            from sudachipy import Dictionary, SplitMode

            _dictionary = Dictionary(dict="core")
            _tokenizer = _dictionary.create()
            _split_mode = SplitMode.B
        except Exception as exc:
            _tokenizer = None
            logger.warning("Furigana dictionary unavailable; showing original text (%s)", type(exc).__name__)
    return _tokenizer


def _aligned_runs(groups: tuple[str, ...], reading: str):
    """Return a unique kana-anchor alignment, never a guessed per-kanji split."""
    if len(reading) > _MAX_ALIGNMENT_LENGTH or len(groups) > 32:
        return None

    @lru_cache(maxsize=None)
    def walk(index: int, offset: int):
        if index == len(groups):
            return ((),) if offset == len(reading) else ()
        group = groups[index]
        results = []
        if _KANJI.fullmatch(group[0]):
            # Every complete kanji run receives at least one kana. Stop after
            # two solutions: an ambiguous alignment must use whole-word ruby.
            for end in range(offset + 1, len(reading) + 1):
                for tail in walk(index + 1, end):
                    results.append((RubySpan(group, reading[offset:end]),) + tail)
                    if len(results) == 2:
                        return tuple(results)
        else:
            kana = _hiragana(group)
            if reading.startswith(kana, offset):
                results.extend(
                    (RubySpan(group),) + tail
                    for tail in walk(index + 1, offset + len(kana))
                )
        return tuple(results)

    solutions = walk(0, 0)
    return solutions[0] if len(solutions) == 1 else None


def _word_spans(surface: str, reading: str) -> tuple[RubySpan, ...]:
    reading = _hiragana(reading)
    if not _KANJI.search(surface) or not _HIRAGANA.fullmatch(reading):
        return (RubySpan(surface),)
    groups = list(_RUNS.findall(surface))
    # Do not put a pronunciation over punctuation, Latin text or numeric OCR
    # fragments accidentally grouped into a token by the analyzer.
    if "".join(groups) != surface:
        return (RubySpan(surface),)

    prefix = []
    suffix = []
    if groups and not _KANJI.search(groups[0]):
        kana = _hiragana(groups[0])
        if reading.startswith(kana) and len(reading) > len(kana):
            prefix.append(RubySpan(groups.pop(0)))
            reading = reading[len(kana):]
    if groups and not _KANJI.search(groups[-1]):
        kana = _hiragana(groups[-1])
        if reading.endswith(kana) and len(reading) > len(kana):
            suffix.append(RubySpan(groups.pop()))
            reading = reading[:-len(kana)]

    aligned = _aligned_runs(tuple(groups), reading)
    if aligned is None:
        # A word-level reading remains valid when its internal kana anchors
        # are ambiguous. Preserve its compound as one span instead of assigning
        # invented readings to the individual kanji.
        aligned = (RubySpan("".join(groups), reading),)
    return tuple(prefix) + aligned + tuple(suffix)


def _coalesce(spans: list[RubySpan]) -> tuple[RubySpan, ...]:
    result = []
    for span in spans:
        if not span.text:
            continue
        if result and not result[-1].reading and not span.reading:
            result[-1] = RubySpan(result[-1].text + span.text)
        else:
            result.append(span)
    return tuple(result)


@lru_cache(maxsize=256)
def _annotate_cached(text: str) -> tuple[RubySpan, ...]:
    # Sudachi's mutable tokenizer is shared, so initialization and all parsing
    # happen under the same lock, including calls from different UI workers.
    with _lock:
        analyzer = _get_tokenizer()
        if analyzer is None:
            return (RubySpan(text),)
        try:
            spans = []
            cursor = 0
            for morpheme in analyzer.tokenize(text, _split_mode):
                start, end = morpheme.begin(), morpheme.end()
                surface = morpheme.surface()
                if not (cursor <= start < end <= len(text)) or text[start:end] != surface:
                    raise ValueError("Tokenizer offsets do not match original subtitle")
                if start > cursor:
                    spans.append(RubySpan(text[cursor:start]))
                if morpheme.is_oov():
                    spans.append(RubySpan(surface))
                else:
                    spans.extend(_word_spans(surface, morpheme.reading_form()))
                cursor = end
            if cursor < len(text):
                spans.append(RubySpan(text[cursor:]))
            return _coalesce(spans)
        except Exception as exc:
            logger.warning("Furigana analysis failed; showing original text (%s)", type(exc).__name__)
            return (RubySpan(text),)


def annotate(text: str) -> tuple[RubySpan, ...]:
    """Return immutable ruby spans which reconstruct *text* exactly.

    Readings are local dictionary estimates, not verified character-name
    readings. Unknown words stay plain. No network request is performed.
    """
    if not text:
        return ()
    if len(text) > _MAX_TEXT_LENGTH or not _KANJI.search(text):
        return (RubySpan(text),)
    return _annotate_cached(text)

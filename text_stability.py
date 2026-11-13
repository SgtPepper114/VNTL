"""Track stable OCR observations without rewriting the displayed Japanese text."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class StableText:
    text: str
    append: bool
    page: int


def _comparison_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def _same_page(previous: str, current: str) -> bool:
    # A shortened OCR result or one corrected glyph does not prove a page turn.
    if previous.startswith(current) or current.startswith(previous):
        return True
    if min(len(previous), len(current)) < 4:
        return False
    return SequenceMatcher(None, previous, current, autojunk=False).ratio() >= 0.75


class TextStabilizer:
    """Single-threaded, monotonic-time debounce with page-local deduplication.

    ``is_stable`` describes the latest observation, including one already emitted.
    ``append`` means this complete text extends the last emitted text on this page;
    the event always contains the entire current text, not just its new suffix.
    Empty observations invalidate the current candidate immediately.
    """

    def __init__(self, stable_seconds: float = 0.6):
        if not math.isfinite(stable_seconds) or stable_seconds < 0:
            raise ValueError("stable_seconds must be finite and nonnegative")
        self.stable_seconds = float(stable_seconds)
        self.reset()

    def reset(self) -> None:
        self.current_text = ""
        self.is_stable = False
        self.page = 0
        self._current_key = ""
        self._candidate_since: float | None = None
        self._last_now: float | None = None
        self._last_emitted_key = ""
        self._emitted_keys: set[str] = set()

    def _new_page(self) -> None:
        self.page += 1
        self._current_key = ""
        self._candidate_since = None
        self._last_emitted_key = ""
        self._emitted_keys.clear()
        self.is_stable = False

    def observe(self, text: str, now: float, *, page_changed: bool = False) -> StableText | None:
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        key = _comparison_text(text)
        clock_rewound = self._last_now is not None and now < self._last_now
        self._last_now = now
        if page_changed:
            self._new_page()
        elif self._current_key and (not key or not _same_page(self._current_key, key)):
            self._new_page()

        self.current_text = text if key else ""
        if not key:
            self._current_key = ""
            self._candidate_since = None
            self.is_stable = False
            return None

        if key != self._current_key or clock_rewound:
            self._current_key = key
            self._candidate_since = now
            self.is_stable = False
        assert self._candidate_since is not None
        self.is_stable = now - self._candidate_since + 1e-9 >= self.stable_seconds
        if not self.is_stable or key in self._emitted_keys:
            return None

        append = bool(self._last_emitted_key and key.startswith(self._last_emitted_key))
        self._last_emitted_key = key
        self._emitted_keys.add(key)
        return StableText(text=self.current_text, append=append, page=self.page)

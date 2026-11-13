"""Tolerate OCR frame noise while retaining complete original Japanese text.

This independent implementation follows GameSentenceMiner's documented strategy:
fuzzy text, per-line geometry, and bounded waiting time. Algorithm reference:
https://github.com/bpwhelan/GameSentenceMiner/blob/main/GameSentenceMiner/ocr/gsm_ocr.py
"""
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
    reason: str = "text"


def _comparison_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def _similar(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous == current:
        return True
    longest = max(len(previous), len(current))
    return (abs(len(previous) - len(current)) <= max(2, int(longest * .10))
            and SequenceMatcher(None, previous, current, autojunk=False).ratio() >= .90)


def _same_page(previous: str, current: str) -> bool:
    if previous.startswith(current) or current.startswith(previous):
        return True
    if min(len(previous), len(current)) < 4:
        return False
    return SequenceMatcher(None, previous, current, autojunk=False).ratio() >= .75


def _ordered_boxes(boxes):
    try:
        result = tuple(tuple(float(value) for value in box) for box in boxes)
        if any(len(box) != 4 or not all(math.isfinite(value) for value in box)
               or box[2] <= box[0] or box[3] <= box[1] for box in result):
            return ()
        return tuple(sorted(result, key=lambda box: (box[1], box[0], box[3], box[2])))
    except (TypeError, ValueError, OverflowError):
        return ()


class TextStabilizer:
    """A page-local state machine called on one thread with monotonic time.

    current_text is the selected original observation, not the latest noisy
    frame. emitted_text is the last emitted source on this page. is_stable says
    whether that representative remains usable. Each trigger needs two frames.
    Text and geometry also honor stable_seconds; a four-second fallback bounds
    the initial wait. Empty text immediately invalidates the current page.
    """

    def __init__(self, stable_seconds: float = 0.6):
        if not math.isfinite(stable_seconds) or stable_seconds < 0:
            raise ValueError("stable_seconds must be finite and nonnegative")
        self.stable_seconds = float(stable_seconds)
        self.max_pending_seconds = 4.0
        self.reset()

    def reset(self) -> None:
        self.page = 0
        self._last_now = None
        self._clear_page()

    def _clear_page(self):
        self.current_text = self.emitted_text = ""
        self.is_stable = False
        self._emitted_key = ""
        self._emitted_boxes = ()
        self._confirmed_layouts = []
        self._known_variants = set()
        self._boxes = ()
        self._boxes_since = None
        self._box_frames = 0
        self._pending_variant = ""
        self._variant_since = None
        self._variant_frames = 0
        self._reconfirm = False
        self._clear_candidate()

    def _clear_candidate(self):
        self._candidate_text = self._candidate_key = ""
        self._candidate_confidence = -1.0
        self._candidate_since = self._first_seen = None
        self._candidate_frames = self._observation_count = 0
        self._highwater = 0
        self._candidate_variants = set()
        self._candidate_layouts = []
        self._exact_key = ""
        self._exact_since = None
        self._exact_frames = 0

    def _new_page(self):
        self.page += 1
        self._clear_page()

    def _track_geometry(self, boxes, now):
        previous = self._boxes
        close = bool(boxes and previous and len(boxes) == len(previous)
                     and all(abs(a - b) <= 3 for old, new in zip(previous, boxes)
                             for a, b in zip(old, new)))
        if close:
            self._box_frames += 1
        else:
            self._boxes_since = now if boxes else None
            self._box_frames = 1 if boxes else 0
        self._boxes = boxes
        # A new line or substantial relocation needs fresh confirmation. The
        # right edge alone is excluded: a blinking end marker changes it.
        return bool(previous and boxes and (
            len(previous) != len(boxes) or any(
                max(abs(old[0] - new[0]), abs(old[1] - new[1])) > max(12, (old[3] - old[1]) / 2)
                for old, new in zip(previous, boxes))))

    def _remember_variant(self, key):
        if len(self._known_variants) < 64:
            self._known_variants.add(key)

    def _layout_confirmed(self, boxes):
        return bool(boxes and any(
            len(previous) == len(boxes)
            and all(abs(a - b) <= 3 for old, new in zip(previous, boxes)
                    for a, b in zip(old, new))
            for previous in self._confirmed_layouts))

    def _remember_layout(self, boxes):
        if boxes and not self._layout_confirmed(boxes):
            self._confirmed_layouts.append(boxes)
            if len(self._confirmed_layouts) > 16:
                self._confirmed_layouts.pop(0)

    def _emit(self, text, reason):
        key = _comparison_text(text)
        previous = self._emitted_key
        self.current_text = self.emitted_text = text
        self._emitted_key = key
        self._emitted_boxes = self._boxes
        for variant, layout in self._candidate_layouts:
            if _same_page(key, variant):
                self._remember_layout(layout)
        self._remember_layout(self._boxes)
        self.is_stable = True
        self._reconfirm = False
        self._pending_variant = ""
        for variant in self._candidate_variants:
            if _same_page(key, variant):
                self._remember_variant(variant)
        self._remember_variant(key)
        self._clear_candidate()
        if key == previous:
            return None
        return StableText(text, bool(previous and key.startswith(previous)), self.page, reason)

    def observe(self, text: str, now: float, *, page_changed: bool = False,
                boxes=(), confidence: float = 1.0) -> StableText | None:
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        key = _comparison_text(text)
        previous_key = self._candidate_key or self._emitted_key
        rewound = self._last_now is not None and now < self._last_now
        self._last_now = now
        if page_changed or (previous_key and (not key or not _same_page(previous_key, key))):
            self._new_page()
        if not key:
            self.current_text = ""
            self.is_stable = False
            return None
        if rewound:
            self._clear_candidate()
            self._boxes = ()
            self._reconfirm = bool(self._emitted_key)
        layout_changed = self._track_geometry(_ordered_boxes(boxes), now)
        if layout_changed and self._emitted_key:
            self._clear_candidate()
            # UI elements can add/remove a separate detection box without
            # changing the dialogue. Confirm a new layout once, then allow
            # already-confirmed layouts to alternate on this same page.
            self._reconfirm = not self._layout_confirmed(self._boxes)

        known = key == self._emitted_key or key in self._known_variants
        geometry_matches = bool(self._boxes and self._emitted_boxes
                                and len(self._boxes) == len(self._emitted_boxes)
                                and all(abs(a - b) <= 3
                                        for old, new in zip(self._emitted_boxes, self._boxes)
                                        for a, b in zip(old, new)))
        if (self._emitted_key and not self._reconfirm
                and (known or _similar(self._emitted_key, key) or geometry_matches)):
            # Keep an emitted representative usable through small OCR noise.
            # A persistent new variant can still become a real small addition.
            # Returning to a known variant identifies a short excursion as noise.
            self._clear_candidate()
            self.current_text = self.emitted_text
            self.is_stable = True
            if known:
                self._remember_layout(self._boxes)
                if self._pending_variant:
                    self._remember_variant(self._pending_variant)
                self._pending_variant = ""
                return None
            if key != self._pending_variant:
                self._pending_variant = key
                self._variant_since, self._variant_frames = now, 1
                return None
            self._variant_frames += 1
            if self._variant_frames >= 2 and now - self._variant_since + 1e-9 >= self.stable_seconds:
                return self._emit(text, "fuzzy")
            return None

        self.is_stable = False
        if key != self._exact_key:
            self._exact_key, self._exact_since, self._exact_frames = key, now, 1
        else:
            self._exact_frames += 1
        try:
            score = float(confidence)
            score = max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0
        except (TypeError, ValueError, OverflowError):
            score = 0.0
        if self._first_seen is None:
            self._first_seen = now
        self._observation_count += 1
        compatible = _similar(self._candidate_key, key)
        growing = len(key) > self._highwater
        if not compatible or growing:
            self._candidate_since, self._candidate_frames = now, 1
        else:
            self._candidate_frames += 1
        self._highwater = max(self._highwater, len(key))
        if len(self._candidate_variants) < 64:
            self._candidate_variants.add(key)
        if self._boxes and (key, self._boxes) not in self._candidate_layouts:
            self._candidate_layouts.append((key, self._boxes))
            if len(self._candidate_layouts) > 16:
                self._candidate_layouts.pop(0)
        unrelated_variant = (self._candidate_key and not compatible
                             and not self._candidate_key.startswith(key)
                             and not key.startswith(self._candidate_key))
        small_variant = compatible and abs(len(key) - len(self._candidate_key)) <= 2
        prefer_candidate = ((score > self._candidate_confidence
                             or (score == self._candidate_confidence and len(key) >= len(self._candidate_key)))
                            if small_variant else (len(key) >= len(self._candidate_key)))
        if not self._candidate_key or unrelated_variant or prefer_candidate:
            self._candidate_text, self._candidate_key, self._candidate_confidence = text, key, score
        elif key == self._candidate_key:
            self._candidate_text = text
        self.current_text = self._candidate_text

        text_ready = (self._candidate_frames >= 2
                      and now - self._candidate_since + 1e-9 >= self.stable_seconds)
        boxes_ready = (self._box_frames >= 2 and self._boxes_since is not None
                       and now - self._boxes_since + 1e-9 >= self.stable_seconds)
        timeout = self._observation_count >= 2 and now - self._first_seen >= self.max_pending_seconds
        if text_ready or boxes_ready or timeout:
            exact_ready = (self._exact_frames >= 2
                           and now - self._exact_since + 1e-9 >= self.stable_seconds)
            if text_ready and exact_ready and self._candidate_key == key:
                return self._emit(self._candidate_text, "text")
            return self._emit(self._candidate_text, "fuzzy" if text_ready else "boxes" if boxes_ready else "timeout")
        return None

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DialogueLine:
    jp: str
    en: str  # Legacy storage key; contains the Simplified Chinese translation.
    scene_description: str | None = None  # Non-None = first line of a new scene


@dataclass(frozen=True)
class SummaryLine:
    """Value copy of a dialogue line; in-flight requests cannot follow edits."""

    jp: str
    en: str
    scene_description: str | None = None


@dataclass(frozen=True)
class SummarySnapshot:
    generation: int
    base_summary: str | None
    prefix: tuple[SummaryLine, ...]
    estimated_tokens: int

    @property
    def count(self) -> int:
        return len(self.prefix)


class ContextManager:
    """
    Maintains a rolling conversation history of translated VN dialogue.

    Works similarly to Claude Code's compact context:
    - Keeps all recent lines verbatim as multi-turn messages.
    - When estimated token usage crosses SUMMARIZE_AT_TOKENS, the oldest
      lines can be summarized into a compact narrative summary.
    - The summary (plus the most recent MIN_RECENT_LINES lines) is then
      used as the context for subsequent translations.
    - This lets the translator preserve character voices, pronouns, and
      plot context across an entire VN session with no hard cutoff.
    """

    # Conservative mixed-script estimate for JP/ZH. Unlike English, CJK text
    # should not be divided by four; model tokenizers and rare kanji vary.

    def __init__(
        self,
        summarize_at_tokens: int = 20_000,
        min_recent_lines: int = 15,
    ) -> None:
        self._summarize_at_tokens = summarize_at_tokens
        self._min_recent_lines = min_recent_lines
        self._summary_generation = 0
        self.history: list[DialogueLine] = []
        self.summary: str | None = None
        self._estimated_tokens: int = 0
        self.current_scene_description: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_line(self, jp: str, en: str) -> None:
        """Record a translated dialogue line and update token estimates.

        If current_scene_description is set, it is stored on this line (marking
        it as the first line of a new scene) and then cleared.
        """
        line = DialogueLine(jp=jp, en=en, scene_description=self.current_scene_description)
        self.current_scene_description = None  # consumed — subsequent lines in same scene get None
        self.history.append(line)
        self._estimated_tokens += self._estimate_line_tokens(line)

    def clear(self) -> None:
        """Discard all history and the summary, resetting to a blank session."""
        self._summary_generation += 1
        self.history = []
        self.summary = None
        self._estimated_tokens = 0
        self.current_scene_description = None

    def update_thresholds(self, summarize_at_tokens: int, min_recent_lines: int) -> None:
        """Update thresholds and invalidate work using old retention settings."""
        if (summarize_at_tokens, min_recent_lines) != (self._summarize_at_tokens, self._min_recent_lines):
            self._summary_generation += 1
            self._summarize_at_tokens, self._min_recent_lines = summarize_at_tokens, min_recent_lines

    def pop_last_line_if_matches(self, jp: str) -> bool:
        """Remove the last history entry if its JP text matches jp.

        Called before a retry so the old translation is replaced, not duplicated.
        If the popped line carried a scene description (it was the first line of a
        new scene), restore it to current_scene_description so the retry translation
        still sees the correct visual context.
        Returns True when the entry was removed.
        """
        if self.history and self.history[-1].jp == jp:
            removed = self.history.pop()
            self._estimated_tokens -= self._estimate_line_tokens(removed)
            if removed.scene_description is not None and self.current_scene_description is None:
                self.current_scene_description = removed.scene_description
            return True
        return False

    def needs_summarization(self) -> bool:
        return (
            self._estimated_tokens >= self._summarize_at_tokens
            and len(self.history) > self._min_recent_lines
        )

    def get_messages(self, current_jp: str) -> list[dict]:
        """
        Build provider-independent translation messages.

        Structure:
          [optional summary injection as user+assistant pair]
          [history lines as user (JP) / assistant (ZH) pairs]
          [current JP line as final user message]
        """
        messages: list[dict] = []

        if self.summary:
            messages.append({
                "role": "user",
                "content": f"[Context summary of earlier dialogue]\n{self.summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "我会参考已读剧情，将后续日语翻译为简体中文，不补写未知剧情。",
            })

        for line in self.history:
            if line.scene_description is not None:
                user_content = f"[Scene: {line.scene_description}]\n{line.jp}"
            else:
                user_content = line.jp
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": line.en})

        if self.current_scene_description:
            content = f"[Scene: {self.current_scene_description}]\n[TRANSLATE THIS:]\n{current_jp}"
        else:
            content = f"[TRANSLATE THIS:]\n{current_jp}"
        messages.append({"role": "user", "content": content})
        return messages

    def prepare_summarization(self) -> SummarySnapshot | None:
        """Freeze only the older prefix; the latest retained lines stay live.

        This also serves explicit/manual compaction below the automatic token
        threshold. No state is changed until a matching result is committed.
        """
        count = len(self.history) - self._min_recent_lines
        if count <= 0:
            return None
        prefix = tuple(self._summary_line(line) for line in self.history[:count])
        return SummarySnapshot(self._summary_generation, self.summary, prefix, self.estimated_tokens)

    def snapshot_for_summarization(self) -> SummarySnapshot | None:
        """Alias for callers that explicitly name the frozen request snapshot."""
        return self.prepare_summarization()

    def apply_summarization(self, new_summary: str, snapshot: SummarySnapshot | None = None) -> bool:
        """Commit exactly the snapshotted prefix, preserving new/retried tail lines.

        A clear/load, earlier compaction, changed settings, edited prefix or
        changed base summary makes an old API result stale. Rejected results
        leave the complete current context untouched.
        """
        if not isinstance(new_summary, str) or not new_summary.strip():
            return False
        if snapshot is None:
            snapshot = self.prepare_summarization()
        if snapshot is None or snapshot.count == 0:
            return False
        if (snapshot.generation != self._summary_generation
                or snapshot.base_summary != self.summary
                or len(self.history) - snapshot.count < self._min_recent_lines
                or tuple(self._summary_line(line) for line in self.history[:snapshot.count]) != snapshot.prefix):
            return False
        lines_to_keep = self.history[snapshot.count:]

        logger.info(
            "Compacting context: summarized %d lines, keeping %d verbatim.",
            snapshot.count,
            len(lines_to_keep),
        )

        self.summary = new_summary.strip()
        self.history = lines_to_keep
        self._summary_generation += 1

        # Recalculate token estimate from scratch
        self._estimated_tokens = self._estimate_summary_tokens(self.summary)
        for line in self.history:
            self._estimated_tokens += self._estimate_line_tokens(line)
        return True

    def build_summarization_messages(self, snapshot: SummarySnapshot | None = None) -> list[dict]:
        """
        Build a messages list asking the summarizer to compress the current context.
        Returns the messages to send (without the system prompt).
        """
        if snapshot is None:
            snapshot = self.prepare_summarization()
        if snapshot is None:
            return []
        parts = []
        for line in snapshot.prefix:
            if line.scene_description is not None:
                parts.append(f"[Scene: {line.scene_description}]")
            parts.append(f"[JP] {line.jp}\n[ZH] {line.en}")
        history_text = "\n".join(parts)

        if snapshot.base_summary:
            prompt = (
                f"此前摘要：\n\n{snapshot.base_summary}\n\n"
                f"需要纳入的新对话：\n\n{history_text}\n\n"
                f"请更新中文摘要，仅记录已经读到的内容。"
            )
        else:
            prompt = f"需要整理的已读对话：\n\n{history_text}"

        return [{"role": "user", "content": prompt}]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize history and summary to a plain dict (JSON-safe)."""
        return {
            "summary": self.summary,
            "history": [
                {"jp": line.jp, "en": line.en, "scene_description": line.scene_description}
                for line in self.history
            ],
        }

    def load_from_dict(self, data: dict) -> None:
        """Replace current state with data previously produced by to_dict()."""
        history = [
            DialogueLine(jp=d["jp"], en=d["en"], scene_description=d.get("scene_description"))
            for d in data.get("history", [])
        ]
        summary = data.get("summary")
        tokens = sum(self._estimate_line_tokens(line) for line in history)
        if summary:
            tokens += self._estimate_summary_tokens(summary)
        self._summary_generation += 1
        self.current_scene_description = None
        self.history, self.summary, self._estimated_tokens = history, summary, tokens

    @property
    def summarize_at_tokens(self) -> int:
        return self._summarize_at_tokens

    @property
    def min_recent_lines(self) -> int:
        return self._min_recent_lines

    @property
    def estimated_tokens(self) -> int:
        return self._estimated_tokens

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_line(line: DialogueLine) -> SummaryLine:
        return SummaryLine(line.jp, line.en, line.scene_description)

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        # Count all non-ASCII characters conservatively, including kana, kanji
        # and Chinese punctuation. This also handles mixed CJK/Latin names.
        non_ascii = sum(ord(char) > 127 for char in text)
        return math.ceil(non_ascii * 1.5 + (len(text) - non_ascii) / 4)

    def _estimate_line_tokens(self, line: DialogueLine | SummaryLine) -> int:
        jp_tokens = self._estimate_text_tokens(line.jp)
        zh_tokens = self._estimate_text_tokens(line.en)
        scene_tokens = (
            self._estimate_text_tokens(line.scene_description) + 4
            if line.scene_description is not None else 0
        )
        return jp_tokens + zh_tokens + scene_tokens + 8  # +8 for message overhead

    def _estimate_summary_tokens(self, summary: str) -> int:
        return self._estimate_text_tokens(summary) + 16

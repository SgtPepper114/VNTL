from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class DialogueLine:
    jp: str
    en: str = ""  # Legacy key; empty until a Chinese translation is committed.
    scene_description: str | None = None  # Non-None = first line of a new scene
    observation_id: str = field(default_factory=lambda: str(uuid4()))
    revision: int = 1


@dataclass(frozen=True)
class SummaryLine:
    """Value copy of a dialogue line; in-flight requests cannot follow edits."""

    jp: str
    en: str
    scene_description: str | None = None
    observation_id: str = ""
    revision: int = 1


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
    Maintains observed VN source/scene records and their optional translations.

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
        self._observation_epoch = 0
        self.history: list[DialogueLine] = []
        self.summary: str | None = None
        self._summary_scene_description: str | None = None
        self._estimated_tokens: int = 0
        self.current_scene_description: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def epoch(self) -> int:
        """Session fence for async callers; ordinary history updates preserve it."""
        return self._observation_epoch

    def invalidate_observations(self) -> None:
        """Invalidate in-flight work while keeping already committed memory."""
        self._observation_epoch += 1
        self._summary_generation += 1
        self.current_scene_description = None

    def observe(self, jp: str, scene_description: str | None = None, *,
                replace_id: str | None = None) -> DialogueLine:
        """Store source/scene before translation, or revise the same observed page.

        Revisions replace the value at its existing position and clear its old
        translation. A missing scene on revision preserves the page's scene.
        Empty JP is valid for scene-only observations and pending scene reads.
        """
        if not isinstance(jp, str) or (scene_description is not None and not isinstance(scene_description, str)):
            raise TypeError("Observation text and scene must be strings")
        if replace_id is None:
            line = DialogueLine(jp=jp, scene_description=scene_description)
            self.history.append(line)
        else:
            old = self.get_observation(replace_id)
            if old is None:
                raise ValueError("Observation no longer exists")
            line = replace(old, jp=jp, en="", revision=old.revision + 1,
                           scene_description=scene_description if scene_description is not None else old.scene_description)
            self.history[self.history.index(old)] = line
        self._recalculate_tokens()
        return line

    def get_observation(self, observation_id: str, revision: int | None = None) -> DialogueLine | None:
        return next((line for line in self.history
                     if line.observation_id == observation_id
                     and (revision is None or line.revision == revision)), None)

    def commit_translation(self, observation_id: str, revision: int, zh: str) -> bool:
        """Attach a result only to the exact retained source revision.

        Callers also compare ``epoch`` across awaits: reloading a saved route
        deliberately preserves its observation IDs and revisions.
        """
        line = self.get_observation(observation_id, revision)
        if line is None or not line.jp.strip() or not isinstance(zh, str) or not zh.strip():
            return False
        line.en = zh
        self._recalculate_tokens()
        return True

    def attach_scene(self, observation_id: str, revision: int, scene: str) -> bool:
        line = self.get_observation(observation_id, revision)
        if line is None or not isinstance(scene, str) or not scene.strip():
            return False
        line.scene_description = scene
        self._recalculate_tokens()
        return True

    def remove_empty_observation(self, observation_id: str, revision: int) -> bool:
        """Remove a failed pending scene read only while its record is empty."""
        line = self.get_observation(observation_id, revision)
        if line is None or self._has_content(line):
            return False
        self.history.remove(line)
        self._recalculate_tokens()
        return True

    def add_line(self, jp: str, en: str) -> None:
        """Record a translated dialogue line and update token estimates.

        If current_scene_description is set, it is stored on this line (marking
        it as the first line of a new scene) and then cleared.
        """
        line = DialogueLine(jp=jp, en=en, scene_description=self.current_scene_description)
        self.current_scene_description = None  # consumed — subsequent lines in same scene get None
        self.history.append(line)
        self._recalculate_tokens()

    def clear(self) -> None:
        """Discard all history and the summary, resetting to a blank session."""
        self.invalidate_observations()
        self.history = []
        self.summary = None
        self._summary_scene_description = None
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
            self.estimated_tokens >= self._summarize_at_tokens
            and self._summarizable_count() > 0
        )

    def get_messages(self, current_jp: str) -> list[dict]:
        """
        Build provider-independent translation messages.

        Structure:
          [optional summary injection as user+assistant pair]
          [history lines as user (JP) / assistant (ZH) pairs]
          [current JP line as final user message]
        """
        scene = self.current_scene_description or self._latest_scene(self.history)
        return self._build_messages(self.history, current_jp, scene)

    def get_observation_messages(self, observation_id: str, revision: int) -> list[dict]:
        """Freeze chronological context ending at one source revision, never later.

        A target compacted out of recent history cannot be translated from this
        API: its old source and place in the summary are no longer available.
        """
        target = self.get_observation(observation_id, revision)
        if target is None:
            raise ValueError("Observation no longer exists or has changed")
        index = self.history.index(target)
        earlier = self.history[:index]
        scene = target.scene_description or self._latest_scene(earlier)
        return self._build_messages(earlier, target.jp, scene)

    def _latest_scene(self, history: list[DialogueLine]) -> str | None:
        return next((line.scene_description for line in reversed(history) if line.scene_description),
                    self._summary_scene_description)

    def _build_messages(self, history: list[DialogueLine], current_jp: str,
                        scene: str | None) -> list[dict]:
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

        for line in history:
            if line.scene_description:
                user_content = f"[Scene: {line.scene_description}]\n{line.jp}"
            else:
                user_content = line.jp
            if user_content.strip():
                messages.append({"role": "user", "content": user_content})
                if line.jp.strip() and line.en.strip():
                    messages.append({"role": "assistant", "content": line.en})

        if scene:
            content = f"[Scene: {scene}]\n[TRANSLATE THIS:]\n{current_jp}"
        else:
            content = f"[TRANSLATE THIS:]\n{current_jp}"
        messages.append({"role": "user", "content": content})
        return messages

    def prepare_summarization(self) -> SummarySnapshot | None:
        """Freeze only the older prefix; the latest retained lines stay live.

        This also serves explicit/manual compaction below the automatic token
        threshold. No state is changed until a matching result is committed.
        """
        count = self._summarizable_count()
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
                or snapshot.count > self._summarizable_count()
                or tuple(self._summary_line(line) for line in self.history[:snapshot.count]) != snapshot.prefix):
            return False
        lines_to_keep = self.history[snapshot.count:]

        logger.info(
            "Compacting context: summarized %d lines, keeping %d verbatim.",
            snapshot.count,
            len(lines_to_keep),
        )

        self.summary = new_summary.strip()
        self._summary_scene_description = next(
            (line.scene_description for line in reversed(snapshot.prefix) if line.scene_description),
            self._summary_scene_description)
        self.history = lines_to_keep
        self._summary_generation += 1

        # Recalculate token estimate from scratch
        self._recalculate_tokens()
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
            if line.scene_description:
                parts.append(f"[Scene: {line.scene_description}]")
            if line.jp.strip():
                parts.append(f"[JP] {line.jp}")
                if line.en.strip():
                    parts.append(f"[ZH] {line.en}")
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
        result = {
            "summary": self.summary,
            "history": [
                {"jp": line.jp, "en": line.en, "scene_description": line.scene_description,
                 "observation_id": line.observation_id, "revision": line.revision}
                for line in self.history if self._has_content(line)
            ],
        }
        if self._summary_scene_description:
            result["summary_scene_description"] = self._summary_scene_description
        return result

    def load_from_dict(self, data: dict) -> None:
        """Replace current state with data previously produced by to_dict()."""
        history = []
        seen_ids = set()
        for item in data.get("history", []):
            jp, en, scene = item["jp"], item.get("en", ""), item.get("scene_description")
            if (not isinstance(jp, str) or not isinstance(en, str)
                    or (scene is not None and not isinstance(scene, str))):
                raise ValueError("Invalid saved observation text")
            observation_id = item.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id or observation_id in seen_ids:
                observation_id = str(uuid4())
            revision = item.get("revision", 1)
            if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
                raise ValueError("Invalid saved observation revision")
            seen_ids.add(observation_id)
            line = DialogueLine(jp, en, scene, observation_id, revision)
            # A saved empty placeholder has no request left to complete after
            # restart. Preserve meaningful source/scene, discard transient slots.
            if self._has_content(line):
                history.append(line)
        summary = data.get("summary")
        if summary is not None and not isinstance(summary, str):
            raise ValueError("Invalid saved summary")
        summary_scene = data.get("summary_scene_description")
        if summary_scene is not None and not isinstance(summary_scene, str):
            raise ValueError("Invalid saved summary scene")
        tokens = sum(self._estimate_line_tokens(line) for line in history)
        if summary:
            tokens += self._estimate_summary_tokens(summary)
        self.invalidate_observations()
        self.current_scene_description = None
        self.history, self.summary, self._estimated_tokens = history, summary, tokens
        self._summary_scene_description = summary_scene
        self._recalculate_tokens()

    @property
    def summarize_at_tokens(self) -> int:
        return self._summarize_at_tokens

    @property
    def min_recent_lines(self) -> int:
        return self._min_recent_lines

    @property
    def estimated_tokens(self) -> int:
        # Public history remains mutable for legacy editors. Recompute so their
        # edits and direct summary changes cannot leave the trigger stale.
        self._recalculate_tokens()
        return self._estimated_tokens

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_line(line: DialogueLine) -> SummaryLine:
        return SummaryLine(line.jp, line.en, line.scene_description, line.observation_id, line.revision)

    def _recalculate_tokens(self) -> None:
        self._estimated_tokens = sum(self._estimate_line_tokens(line) for line in self.history)
        if self.summary:
            self._estimated_tokens += self._estimate_summary_tokens(self.summary)
        if self._summary_scene_description:
            self._estimated_tokens += self._estimate_text_tokens(self._summary_scene_description) + 4

    @staticmethod
    def _has_content(line: DialogueLine | SummaryLine) -> bool:
        return bool(line.jp.strip() or line.en.strip() or (line.scene_description or "").strip())

    def _summarizable_count(self) -> int:
        """Retain recent complete records and keep pending scene reads alive.

        A blank observation is an in-flight scene placeholder, not a completed
        record. Never compact across it before attach_scene/removal resolves it.
        """
        complete = [index for index, line in enumerate(self.history) if self._has_content(line)]
        keep = max(0, self._min_recent_lines)
        if len(complete) <= keep:
            return 0
        count = complete[-keep] if keep else complete[-1] + 1
        pending = next((index for index, line in enumerate(self.history) if not self._has_content(line)), count)
        return min(count, pending)

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        # Count all non-ASCII characters conservatively, including kana, kanji
        # and Chinese punctuation. This also handles mixed CJK/Latin names.
        non_ascii = sum(ord(char) > 127 for char in text)
        return math.ceil(non_ascii * 1.5 + (len(text) - non_ascii) / 4)

    def _estimate_line_tokens(self, line: DialogueLine | SummaryLine) -> int:
        if not self._has_content(line):
            return 0
        jp_tokens = self._estimate_text_tokens(line.jp)
        zh_tokens = self._estimate_text_tokens(line.en)
        scene_tokens = (
            self._estimate_text_tokens(line.scene_description) + 4
            if line.scene_description is not None else 0
        )
        return jp_tokens + zh_tokens + scene_tokens + 8  # +8 for message overhead

    def _estimate_summary_tokens(self, summary: str) -> int:
        return self._estimate_text_tokens(summary) + 16

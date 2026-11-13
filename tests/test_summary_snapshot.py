"""Context compaction snapshots preserve dialogue arriving during a model call."""
import copy
from dataclasses import FrozenInstanceError
import unittest

from context_manager import ContextManager


class SummarySnapshotTests(unittest.TestCase):
    def make_context(self, *, count=5, keep=2, threshold=1):
        context = ContextManager(summarize_at_tokens=threshold, min_recent_lines=keep)
        context.load_from_dict({"summary": "此前已读摘要。"})
        for index in range(count):
            if index == 0:
                context.current_scene_description = "旧场景：音乐教室。"
            context.add_line(f"日文-{index}", f"译文-{index}")
        return context

    def test_snapshot_freezes_prefix_summary_and_estimate_but_not_live_tail(self):
        context = self.make_context()
        before = copy.deepcopy(context.to_dict())
        tokens = context.estimated_tokens
        snapshot = context.prepare_summarization()
        self.assertEqual(snapshot.count, 3)
        self.assertEqual([line.jp for line in snapshot.prefix], ["日文-0", "日文-1", "日文-2"])
        self.assertEqual(snapshot.base_summary, "此前已读摘要。")
        self.assertEqual(snapshot.estimated_tokens, tokens)
        self.assertEqual(context.to_dict(), before)
        self.assertEqual(context.snapshot_for_summarization(), snapshot)
        with self.assertRaises(FrozenInstanceError):
            snapshot.base_summary = "改动"
        with self.assertRaises(FrozenInstanceError):
            snapshot.prefix[0].jp = "改动"

        original_request = context.build_summarization_messages(snapshot)
        context.add_line("后来新增日文", "后来新增译文")
        context.summary = "后来变动的摘要"
        context.history[0].en = "后来更改的旧译文"
        self.assertEqual(context.build_summarization_messages(snapshot), original_request)
        text = original_request[0]["content"]
        self.assertIn("此前已读摘要。", text)
        self.assertIn("[Scene: 旧场景：音乐教室。]", text)
        self.assertIn("[JP] 日文-0\n[ZH] 译文-0", text)
        self.assertNotIn("日文-3", text)
        self.assertNotIn("后来", text)
        self.assertEqual(snapshot.estimated_tokens, tokens)

    def test_commit_preserves_new_lines_retried_tail_and_pending_scene(self):
        context = self.make_context()
        snapshot = context.prepare_summarization()
        retained = context.history[3]
        context.pop_last_line_if_matches("日文-4")
        context.add_line("日文-4", "修改后的近期译文")
        retried = context.history[-1]
        context.add_line("刚刚新增的日文", "刚刚新增的译文")
        appended = context.history[-1]
        context.current_scene_description = "等待下一句使用的新场景。"

        self.assertTrue(context.apply_summarization("  新的有效摘要。\n", snapshot))
        self.assertEqual(context.summary, "新的有效摘要。")
        self.assertEqual(len(context.history), 3)
        self.assertIs(context.history[0], retained)
        self.assertIs(context.history[1], retried)
        self.assertIs(context.history[2], appended)
        self.assertEqual(context.current_scene_description, "等待下一句使用的新场景。")
        restored = ContextManager()
        restored.load_from_dict(context.to_dict())
        self.assertEqual(context.estimated_tokens, restored.estimated_tokens)

    def test_clear_and_load_invalidate_even_identical_restored_content(self):
        for clear_first in (False, True):
            with self.subTest(clear_first=clear_first):
                context = self.make_context()
                snapshot = context.prepare_summarization()
                same_content = copy.deepcopy(context.to_dict())
                if clear_first:
                    context.clear()
                context.load_from_dict(same_content)
                before = copy.deepcopy(context.to_dict())
                self.assertFalse(context.apply_summarization("过期结果", snapshot))
                self.assertEqual(context.to_dict(), before)

    def test_clear_alone_rejects_inflight_result(self):
        context = self.make_context()
        snapshot = context.prepare_summarization()
        context.clear()
        self.assertFalse(context.apply_summarization("不能恢复旧路线", snapshot))
        self.assertEqual(context.to_dict(), {"summary": None, "history": []})
        self.assertEqual(context.estimated_tokens, 0)

    def test_prefix_edits_or_reordering_reject_without_mutating_current_memory(self):
        for change in ("jp", "en", "scene_description", "order"):
            with self.subTest(change=change):
                context = self.make_context()
                snapshot = context.prepare_summarization()
                if change == "order":
                    context.history[0], context.history[1] = context.history[1], context.history[0]
                else:
                    setattr(context.history[0], change, "用户已经修改的内容")
                before = copy.deepcopy(context.to_dict())
                self.assertFalse(context.apply_summarization("基于旧原文的结果", snapshot))
                self.assertEqual(context.to_dict(), before)

    def test_changed_base_summary_rejects_without_clobbering_new_summary(self):
        context = self.make_context()
        snapshot = context.prepare_summarization()
        context.summary = "用户已经修订的摘要。"
        before = copy.deepcopy(context.to_dict())
        self.assertFalse(context.apply_summarization("过期摘要", snapshot))
        self.assertEqual(context.to_dict(), before)

    def test_successful_commit_invalidates_other_results_from_same_generation(self):
        context = self.make_context()
        first = context.prepare_summarization()
        second = context.prepare_summarization()
        self.assertTrue(context.apply_summarization("较早完成的有效摘要", first))
        for index in range(3):
            context.add_line(f"新增{index}", f"新增译文{index}")
        before = copy.deepcopy(context.to_dict())
        self.assertFalse(context.apply_summarization("较晚返回的过期摘要", second))
        self.assertEqual(context.to_dict(), before)

    def test_retention_or_threshold_changes_invalidate_inflight_work(self):
        for threshold, keep in ((1, 4), (2, 2)):
            with self.subTest(threshold=threshold, keep=keep):
                context = self.make_context()
                snapshot = context.prepare_summarization()
                context.update_thresholds(threshold, keep)
                before = copy.deepcopy(context.to_dict())
                self.assertFalse(context.apply_summarization("旧设置下的结果", snapshot))
                self.assertEqual(context.to_dict(), before)
        context = self.make_context()
        snapshot = context.prepare_summarization()
        context.update_thresholds(1, 2)
        self.assertTrue(context.apply_summarization("设置未变的有效结果", snapshot))

    def test_missing_recent_tail_never_compacts_below_retention(self):
        context = self.make_context()
        snapshot = context.prepare_summarization()
        self.assertTrue(context.pop_last_line_if_matches("日文-4"))
        before = copy.deepcopy(context.to_dict())
        self.assertFalse(context.apply_summarization("结果", snapshot))
        self.assertEqual(context.to_dict(), before)

    def test_empty_result_is_rejected_without_losing_any_dialogue(self):
        for result in ("", " \n\t", None):
            with self.subTest(result=result):
                context = self.make_context()
                snapshot = context.prepare_summarization()
                before = copy.deepcopy(context.to_dict())
                tokens = context.estimated_tokens
                self.assertFalse(context.apply_summarization(result, snapshot))
                self.assertEqual(context.to_dict(), before)
                self.assertEqual(context.estimated_tokens, tokens)
                self.assertTrue(context.apply_summarization("稍后成功的结果", snapshot))

    def test_no_older_prefix_yields_no_request_or_commit(self):
        context = self.make_context(count=2, keep=2)
        before = copy.deepcopy(context.to_dict())
        self.assertFalse(context.needs_summarization())
        self.assertIsNone(context.prepare_summarization())
        self.assertEqual(context.build_summarization_messages(), [])
        self.assertFalse(context.apply_summarization("不应替换的摘要"))
        self.assertEqual(context.to_dict(), before)

    def test_manual_snapshot_and_legacy_commit_work_below_token_threshold(self):
        context = self.make_context(threshold=1_000_000)
        self.assertFalse(context.needs_summarization())
        self.assertIsNotNone(context.prepare_summarization())
        self.assertIn("[JP] 日文-0", context.build_summarization_messages()[0]["content"])
        self.assertTrue(context.apply_summarization("手动整理的摘要"))
        self.assertEqual([line.jp for line in context.history], ["日文-3", "日文-4"])

    def test_automatic_trigger_remains_token_based(self):
        context = ContextManager(summarize_at_tokens=1000, min_recent_lines=1)
        context.add_line("長文" * 500, "长句" * 500)
        self.assertGreater(context.estimated_tokens, 1000)
        self.assertFalse(context.needs_summarization())  # Retention still protects the sole line.
        context.add_line("はい", "是")
        self.assertTrue(context.needs_summarization())
        context.update_thresholds(context.estimated_tokens + 1, 1)
        self.assertFalse(context.needs_summarization())
        context.update_thresholds(context.estimated_tokens, 1)
        self.assertTrue(context.needs_summarization())


if __name__ == "__main__":
    unittest.main()

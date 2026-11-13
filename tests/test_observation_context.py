"""Observed source revisions, scene placeholders, and compaction migration."""
import copy
import json
import unittest
from uuid import UUID

from context_manager import ContextManager


class ObservationContextTests(unittest.TestCase):
    def test_observation_is_stored_before_translation_and_revision_replaces_in_place(self):
        context = ContextManager()
        first = context.observe("  ミカ：待っ", "走廊。")
        self.assertEqual(str(UUID(first.observation_id)), first.observation_id)
        self.assertEqual(first.revision, 1)
        self.assertEqual(first.en, "")
        self.assertGreater(context.estimated_tokens, 0)
        self.assertTrue(context.commit_translation(first.observation_id, 1, "旧译文"))
        later = context.observe("後の台詞")
        updated = context.observe("  ミカ：待って！\n", replace_id=first.observation_id)
        self.assertIsNot(updated, first)
        self.assertEqual([line.observation_id for line in context.history], [first.observation_id, later.observation_id])
        self.assertEqual((updated.revision, updated.en, updated.scene_description), (2, "", "走廊。"))
        self.assertEqual(updated.jp, "  ミカ：待って！\n")
        self.assertIsNone(context.get_observation(first.observation_id, 1))
        self.assertFalse(context.commit_translation(first.observation_id, 1, "过期译文"))
        self.assertFalse(context.attach_scene(first.observation_id, 1, "过期场景"))
        self.assertTrue(context.commit_translation(first.observation_id, 2, "等等！"))
        self.assertEqual(later.en, "")

    def test_exact_commit_and_scene_update_preserve_ids_and_recalculate_tokens(self):
        context = ContextManager()
        line = context.observe("学校へ")
        original_tokens = context.estimated_tokens
        self.assertTrue(context.commit_translation(line.observation_id, 1, "回到学校" * 10))
        self.assertGreater(context.estimated_tokens, original_tokens)
        self.assertTrue(context.attach_scene(line.observation_id, 1, "夜晚的校门。"))
        self.assertEqual(line.revision, 1)
        self.assertFalse(context.commit_translation(line.observation_id, 1, "  \n"))
        self.assertFalse(context.commit_translation("missing", 1, "中文"))
        before = context.to_dict()
        with self.assertRaises(ValueError):
            context.observe("追加", replace_id="missing")
        self.assertEqual(context.to_dict(), before)

    def test_target_messages_exclude_its_old_answer_future_and_pending_global_scene(self):
        context = ContextManager()
        context.summary = "更早的剧情。"
        context.add_line("以前", "之前")
        context.observe("", "此前看到的音乐室。")
        context.observe("まだ訳していない")
        target = context.observe("今の台詞")
        context.commit_translation(target.observation_id, 1, "目标的旧译文")
        context.observe("未来台词", "未来场景")
        context.current_scene_description = "更未来的临时场景"
        messages = context.get_observation_messages(target.observation_id, 1)
        text = json.dumps(messages, ensure_ascii=False)
        for expected in ("更早的剧情", "以前", "之前", "まだ訳していない", "此前看到的音乐室"):
            self.assertIn(expected, text)
        for forbidden in ("目标的旧译文", "未来台词", "未来场景", "更未来"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(messages[-1]["content"], "[Scene: 此前看到的音乐室。]\n[TRANSLATE THIS:]\n今の台詞")
        self.assertTrue(all(message["content"].strip() for message in messages))
        self.assertFalse(any(message["role"] == "assistant" and not message["content"].strip() for message in messages))
        context.observe("今の台詞の続き", replace_id=target.observation_id)
        with self.assertRaises(ValueError):
            context.get_observation_messages(target.observation_id, 1)

    def test_scene_placeholder_has_no_tokens_or_messages_and_only_empty_record_can_be_removed(self):
        context = ContextManager()
        pending = context.observe("")
        self.assertEqual(context.estimated_tokens, 0)
        self.assertEqual(context.get_messages("はい"), [{"role": "user", "content": "[TRANSLATE THIS:]\nはい"}])
        self.assertTrue(context.remove_empty_observation(pending.observation_id, 1))
        self.assertFalse(context.remove_empty_observation(pending.observation_id, 1))
        scene = context.observe("")
        self.assertTrue(context.attach_scene(scene.observation_id, 1, "階段。"))
        self.assertFalse(context.remove_empty_observation(scene.observation_id, 1))
        self.assertFalse(context.commit_translation(scene.observation_id, 1, "凭空出现的译文"))
        self.assertIn("階段", context.get_messages("行こう")[-1]["content"])

    def test_pending_scene_blocks_compaction_across_it_until_resolved(self):
        context = ContextManager(summarize_at_tokens=1, min_recent_lines=1)
        pending = context.observe("")
        context.observe("先に読めた台詞")
        context.observe("次の台詞")
        self.assertFalse(context.needs_summarization())
        self.assertIsNone(context.prepare_summarization())
        context.attach_scene(pending.observation_id, 1, "台词出现前的场景。")
        snapshot = context.prepare_summarization()
        self.assertEqual(snapshot.count, 2)
        self.assertTrue(context.needs_summarization())
        context.apply_summarization("最初的场景和台词。", snapshot)
        self.assertEqual([line.jp for line in context.history], ["次の台詞"])

    def test_pending_slots_are_not_persisted_or_resurrected_after_restart(self):
        context = ContextManager(min_recent_lines=1)
        context.observe("")
        source = context.observe("観測済みの原文")
        context.observe("", "已经观察到的场景")
        saved = context.to_dict()
        self.assertEqual(len(saved["history"]), 2)
        self.assertEqual(saved["history"][0]["observation_id"], source.observation_id)
        saved["history"].insert(0, {"jp": "", "en": "", "scene_description": None})
        restored = ContextManager(min_recent_lines=1)
        restored.load_from_dict(saved)
        self.assertEqual(len(restored.history), 2)
        self.assertEqual(restored.prepare_summarization().count, 1)

    def test_untranslated_source_and_scene_only_records_are_summarized_by_tokens(self):
        context = ContextManager(summarize_at_tokens=50, min_recent_lines=2)
        context.observe("原文" * 50, "最初的场景")
        scene = context.observe("", "第二个场景")
        recent = context.observe("最新台词")
        self.assertTrue(context.needs_summarization())
        snapshot = context.prepare_summarization()
        request = context.build_summarization_messages(snapshot)[0]["content"]
        self.assertIn("[JP] 原文", request)
        self.assertNotIn("[ZH]", request)
        self.assertNotIn("最新台词", request)
        self.assertTrue(context.apply_summarization("已有证据的中文摘要", snapshot))
        self.assertEqual(context.history, [scene, recent])
        # Scene-only records use the same bounded recent-record retention.
        for index in range(5):
            context.observe("", f"后来的场景{index}")
        snapshot = context.prepare_summarization()
        self.assertTrue(context.apply_summarization("进一步观察的场景", snapshot))
        self.assertEqual(len(context.history), 2)
        self.assertEqual([line.scene_description for line in context.history], ["后来的场景3", "后来的场景4"])

    def test_summary_scene_survives_compaction_and_roundtrip_without_future_context(self):
        context = ContextManager(min_recent_lines=1)
        context.observe("", "先前确认的楼梯")
        target = context.observe("ここ？")
        self.assertTrue(context.apply_summarization("场景已纳入剧情摘要"))
        restored = ContextManager()
        restored.load_from_dict(context.to_dict())
        restored.observe("未来文字", "后来的教室")
        last = restored.get_observation_messages(target.observation_id, 1)[-1]["content"]
        self.assertIn("先前确认的楼梯", last)
        self.assertNotIn("后来的教室", last)

    def test_summary_snapshot_rejects_source_revision_translation_or_scene_changed_prefix(self):
        for update in ("revision", "translation", "scene", "identical_reinsert"):
            with self.subTest(update=update):
                context = ContextManager(min_recent_lines=1)
                old = context.observe("古い台詞", "旧场景")
                context.observe("新しい台詞")
                snapshot = context.prepare_summarization()
                if update == "revision":
                    context.observe(old.jp, replace_id=old.observation_id)
                elif update == "translation":
                    context.commit_translation(old.observation_id, 1, "迟来的译文")
                elif update == "scene":
                    context.attach_scene(old.observation_id, 1, "修正后的场景")
                else:
                    from context_manager import DialogueLine
                    context.history[0] = DialogueLine(old.jp, old.en, old.scene_description)
                before = copy.deepcopy(context.to_dict())
                self.assertFalse(context.apply_summarization("已经过期", snapshot))
                self.assertEqual(context.to_dict(), before)

    def test_summary_keeps_translated_recent_tail_and_appended_observations(self):
        context = ContextManager(min_recent_lines=1)
        context.observe("古い")
        recent = context.observe("最近")
        snapshot = context.prepare_summarization()
        context.commit_translation(recent.observation_id, 1, "最近的译文")
        appended = context.observe("さらに新しい", "新的场景")
        self.assertTrue(context.apply_summarization("旧的原文摘要", snapshot))
        self.assertEqual(context.history, [recent, appended])
        self.assertEqual(recent.en, "最近的译文")

    def test_epoch_only_moves_for_clear_load_and_explicit_invalidation(self):
        context = ContextManager(min_recent_lines=1)
        epoch = context.epoch
        old = context.observe("最初")
        context.observe("次")
        context.commit_translation(old.observation_id, 1, "最初")
        context.attach_scene(old.observation_id, 1, "场景")
        context.update_thresholds(1, 1)
        context.apply_summarization("摘要")
        self.assertEqual(context.epoch, epoch)
        saved = context.to_dict()
        context.invalidate_observations()
        self.assertEqual(context.epoch, epoch + 1)
        self.assertEqual(context.to_dict(), saved)
        context.load_from_dict(saved)
        self.assertEqual(context.epoch, epoch + 2)
        context.clear()
        self.assertEqual(context.epoch, epoch + 3)

    def test_old_save_migrates_and_new_save_keeps_identity_revision_and_untranslated_source(self):
        context = ContextManager()
        context.load_from_dict({"summary": "旧摘要", "history": [
            {"jp": "旧台詞", "en": "旧译文", "scene_description": "旧场景"},
            {"jp": "新台詞"}, {"jp": "", "scene_description": "仅有场景"},
        ]})
        self.assertEqual(context.history[0].en, "旧译文")
        for line in context.history:
            UUID(line.observation_id)
            self.assertEqual(line.revision, 1)
        revised = context.observe("新台詞の続き", replace_id=context.history[1].observation_id)
        saved = json.loads(json.dumps(context.to_dict(), ensure_ascii=False))
        restored = ContextManager()
        restored.load_from_dict(saved)
        self.assertEqual(restored.to_dict(), saved)
        self.assertIsNotNone(restored.get_observation(revised.observation_id, 2))
        self.assertEqual(restored.estimated_tokens, context.estimated_tokens)

    def test_invalid_load_is_atomic_and_duplicate_saved_ids_are_disambiguated(self):
        context = ContextManager()
        line = context.observe("保留")
        before, epoch = context.to_dict(), context.epoch
        for change in ({"revision": 0}, {"revision": True}, {"jp": None}, {"en": 4}):
            payload = copy.deepcopy(before)
            payload["history"][0].update(change)
            with self.assertRaises(ValueError):
                context.load_from_dict(payload)
            self.assertEqual(context.to_dict(), before)
            self.assertEqual(context.epoch, epoch)
        payload = copy.deepcopy(before)
        payload["history"].append(copy.deepcopy(payload["history"][0]))
        context.load_from_dict(payload)
        self.assertEqual(context.history[0].observation_id, line.observation_id)
        self.assertNotEqual(context.history[0].observation_id, context.history[1].observation_id)


if __name__ == "__main__":
    unittest.main()

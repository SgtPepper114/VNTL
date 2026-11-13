"""Continuous local OCR, independent story memory and on-demand subtitle display.

All state commits happen on the asyncio/Qt thread. A single dedicated worker
owns MeikiOCR; model inference never blocks Qt. Network work uses frozen frames
and observation revisions, and stale results cannot reveal an earlier page.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from local_ocr import MeikiReader, OCRResult
from screenshot_service import ScreenshotService
from text_stability import TextStabilizer

logger = logging.getLogger(__name__)

_LOCAL_ERROR_LOG_INTERVAL = 30.0
# These messages originate in our capture code. Unknown exception text can
# contain signed URLs or credentials, so it is never shown or logged here.
_SAFE_CAPTURE_REASONS = frozenset({
    "持续观察的窗口截图需要 Windows。",
    "当前 Windows 版本需要更新到 Windows 10 2004 或更高版本以支持持续截图。",
    "设置窗口打开时观察暂停，关闭后自动继续。",
    "持续观察需要 Windows 10 2004 或更新版本的窗口截图支持。",
    "观察已暂停：请将游戏切换到前台。",
    "截图组件不可用，请重新安装项目依赖（mss 和 Pillow）。",
    "请先选择游戏或模拟器窗口，再读取屏幕。",
    "找不到已选择的游戏窗口；游戏可能已关闭，请重新选择。",
    "游戏窗口已最小化，请先恢复窗口并保持画面无遮挡。",
    "无法读取游戏窗口区域，请恢复窗口后重新选择。",
    "游戏窗口没有可读取的画面，请恢复窗口后重试。",
    "游戏截图失败，请使用窗口模式并保持游戏画面可见后重试。",
})


def text_key(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


class SceneGate:
    """Compare against the last submitted scene, masking recognized dialogue."""

    def __init__(self):
        self.reference = None
        self.reference_boxes = ()
        self.last_attempt = float("-inf")

    @staticmethod
    def thumbnail(image):
        return np.asarray(image.convert("L").resize((160, 90)), dtype=np.int16)

    def changed(self, image, boxes, threshold):
        if self.reference is None:
            return True
        current = self.thumbnail(image)
        mask = Image.new("L", (160, 90), 255)
        draw = ImageDraw.Draw(mask)
        for x1, y1, x2, y2 in (*self.reference_boxes, *boxes):
            draw.rectangle((x1 * 160 - 2, y1 * 90 - 2, x2 * 160 + 2, y2 * 90 + 2), fill=0)
        visible = np.asarray(mask) > 0
        count = int(visible.sum())
        return count > 0 and float(((np.abs(current - self.reference) > 18) & visible).sum()) * 100 / count >= threshold

    def submitted(self, image, boxes, now):
        self.reference = self.thumbnail(image)
        self.reference_boxes = boxes
        self.last_attempt = now


class ObservationService:
    def __init__(self, cfg, context, translator, *, capture, target_key,
                 on_display, on_memory, on_status, reader=None, can_display=lambda: True):
        self.cfg, self.context, self.translator = cfg, context, translator
        self.capture, self.target_key = capture, target_key
        self.on_display, self.on_memory, self.on_status = on_display, on_memory, on_status
        self.can_display = can_display
        self.reader = reader or MeikiReader()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vntl-meiki")
        self._tasks = set()
        self._translation_task = self._scene_task = None
        self._run_task = None
        self._closed = False
        self._status = None
        self._blocked_reason = None
        self._not_ready_reason = None
        self._local_error_logs = {}
        self._wake = asyncio.Event()
        self._signature = None
        self._target = target_key()
        self._epoch = context.epoch
        self._requested = False
        self._current = None  # (observation ID, revision)
        self._current_page = None
        self._current_text = ""
        self._local_text = ""
        self._valid = False
        self._cache = None  # only the current observation needs a display cache
        self._displayed = None
        self._displayed_page = None
        self._last_reveal = None
        self._last_raw_change = 0.0
        self._last_raw_key = None
        self._last_frame = None
        self._last_ocr = None
        self._suppressed_key = None
        self._suppressed_keys = set()
        self._page_raw_keys = set()
        self._page_raw_page = None
        self._vision_ref = None
        self._ocr_failed_until = 0.0
        self._viewport_size = None
        self._viewport_rect = None
        self._viewport_fallback = False
        self.metrics = {"frames": 0, "ocr_calls": 0, "observations": 0,
                        "translation_calls": 0, "scene_calls": 0, "vision_fallbacks": 0}
        self.configure()

    def _set_status(self, text):
        if self._viewport_fallback:
            text += " · 游戏画面标定未完成，使用完整窗口"
        if text != self._status:
            self._status = text
            self.on_status(text)

    @property
    def not_ready_reason(self):
        """Current readiness, independent of the last text rendered in the UI."""
        return self._blocked_reason or self._not_ready_reason

    def _local_failure(self, stage, exc):
        error = type(exc).__name__
        reason = None
        if stage == "capture" and type(exc) is RuntimeError:
            message = str(exc)
            if message in _SAFE_CAPTURE_REASONS:
                reason = message
            else:
                match = re.fullmatch(
                    r"浮窗无法从截图中排除（Windows 错误 ([0-9]{1,10})）。"
                    r"请重启更新后的 VNTL；也可暂停观察后使用按键识屏。", message)
                if match:
                    reason = (f"浮窗无法从截图中排除（Windows 错误 {int(match[1])}）。"
                              "请重启更新后的 VNTL；也可暂停观察后使用按键识屏。")
        if reason is None:
            label = {"capture": "游戏截图", "ocr": "本地文字识别", "processing": "观察结果处理"}[stage]
            reason = f"{label}暂缓（{error}），正在重试…"
        # A capture/OCR failure temporarily blocks the current page, but does
        # not replace its translation state. Restore that state on recovery.
        self._blocked_reason = reason
        self._set_status(reason)
        now = time.monotonic()
        key = (stage, error, reason)
        if now - self._local_error_logs.get(key, float("-inf")) >= _LOCAL_ERROR_LOG_INTERVAL:
            logger.warning("Local observation blocked: stage=%s error=%s status=%s", stage, error, reason)
            self._local_error_logs[key] = now

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def configure(self):
        # Mode switches and appearance edits reuse a ready translation. Model,
        # game, crop and prompt changes require fresh observations.
        fields = ("observation_enabled", "auto_game_viewport", "ocr_stable_ms", "ocr_region_top_percent", "ocr_region_bottom_percent",
                  "game_title", "game_notes", "translator_provider", "translator_model",
                  "translator_system_prompt", "descriptor_provider", "descriptor_model",
                  "descriptor_system_prompt", "translator_max_tokens", "descriptor_max_tokens",
                  "translator_google_thinking_level", "translator_ollama_thinking",
                  "descriptor_google_thinking_level", "descriptor_ollama_thinking",
                  "anthropic_api_key", "openai_api_key", "google_api_key", "deepseek_api_key",
                  "ollama_base_url", "scene_observation_enabled")
        signature = tuple(getattr(self.cfg, name, None) for name in fields)
        if signature != self._signature:
            self.reset()
            self._signature = signature
        if self.cfg.translation_mode == "auto":
            self._reveal()

    def reset(self):
        for task in tuple(self._tasks):
            task.cancel()
        for line in list(self.context.history):
            self.context.remove_empty_observation(line.observation_id, line.revision)
        self.context.invalidate_observations()
        self._epoch = self.context.epoch
        self._target = self.target_key()
        self.stability = TextStabilizer(self.cfg.ocr_stable_ms / 1000)
        self.scene_gate = SceneGate()
        self._current = self._cache = self._displayed = None
        self._current_page = None
        self._current_text = ""
        self._local_text = ""
        self._last_reveal = None
        self._displayed_page = None
        self._valid = self._requested = False
        self._last_raw_key = self._last_frame = self._last_ocr = None
        self._suppressed_key = self._vision_ref = None
        self._suppressed_keys.clear()
        self._page_raw_keys.clear()
        self._page_raw_page = None
        self._translation_task = self._scene_task = None
        self._blocked_reason = None
        self._status = None
        self._not_ready_reason = ("选择游戏窗口后开始本地观察" if self._target is None
                                  else "等待读取当前画面…")
        self._ocr_failed_until = 0.0
        self._viewport_size = self._viewport_rect = None
        self._viewport_fallback = False
        self._wake.set()

    def start(self):
        if self._run_task is None:
            self._run_task = asyncio.create_task(self.run())

    def request_display(self, *, retry=False):
        if not self._live(self._epoch):
            self.reset()
            self._target = self.target_key()
        self._requested = True
        if not self.cfg.observation_enabled:
            self._set_status("观察已暂停 · 按键可识屏翻译")
            return
        if self._target is None:
            self._not_ready_reason = "选择游戏窗口后开始本地观察"
            self._set_status(self._not_ready_reason)
            return
        if self._blocked_reason:
            # Wake the existing serial sampler instead of creating a second
            # capture/OCR task. Repeated key presses collapse into this event.
            self._ocr_failed_until = 0.0
            self._wake.set()
            self._set_status(self._blocked_reason)
            return
        pending = self._translation_task and not self._translation_task.done()
        if self._current and self._valid and (retry or (not self._cache and not pending)):
            self._cache = None
            if self.context.get_observation(*self._current) is None:
                line = self.context.observe(self._current_text)
                self._current = (line.observation_id, line.revision)
                self.on_memory()
            self._start_translation(self._current)
        if not self._reveal():
            self._set_status(self._not_ready_reason or "等待当前台词稳定并完成预译…")

    def external_display(self):
        """Another input source owns the last visible/backlog entry now."""
        self._displayed = None
        self._displayed_page = None

    def _reveal(self):
        if (not self._valid or not self._current or not self._cache or not self.can_display()
                or not self._live(self._epoch)):
            return False
        ref, translated = self._cache
        if ref != self._current:
            return False
        same_page = self._displayed_page == self._current_page and self._displayed is not None
        if not (self._requested or self.cfg.translation_mode == "auto" or same_page):
            return False
        reveal = (ref, translated)
        if self._requested or self._last_reveal != reveal:
            self.on_display(self._current_text, translated, same_page)
            self._last_reveal = reveal
        self._displayed = ref[0]
        self._displayed_page = self._current_page
        self._requested = False
        self._set_status("台词已显示 · 持续观察中")
        return True

    def _live(self, epoch, ref=None):
        return (not self._closed and self.cfg.observation_enabled
                and self.context.epoch == epoch == self._epoch
                and self.target_key() == self._target
                and (ref is None or self.context.get_observation(*ref) is not None))

    async def _prepare_game_frame(self, image, epoch, target):
        """Calibrate once per target/size; all later frames crop locally."""
        if not self.cfg.auto_game_viewport:
            return image
        if self._viewport_size == image.size:
            return image.crop(self._viewport_rect) if self._viewport_rect else image
        self._valid = False
        self._not_ready_reason = "正在识别游戏画面范围…"
        self._set_status(self._not_ready_reason)
        bounds = None
        try:
            request = self._spawn(self.translator.locate_game_viewport(
                ScreenshotService.encode_frame(image)))
            bounds = await asyncio.wait_for(request, 12)
        except asyncio.CancelledError:
            # Configuration/target reset cancels the old child request. App
            # shutdown cancels the sampler itself and must propagate normally.
            if self._closed or asyncio.current_task().cancelling():
                raise
            return None
        except Exception as exc:
            logger.warning("Game viewport calibration failed: %s", type(exc).__name__)
        if not self._live(epoch) or target != self.target_key():
            return None
        self._viewport_size = image.size
        self._viewport_rect = None
        if bounds is not None:
            left, top, right, bottom = bounds
            self._viewport_rect = (round(left * image.width), round(top * image.height),
                                   round(right * image.width), round(bottom * image.height))
        self._viewport_fallback = self._viewport_rect is None
        self._not_ready_reason = "等待读取当前画面…"
        self._set_status(self._not_ready_reason)
        self.metrics["viewport_calibrations"] = self.metrics.get("viewport_calibrations", 0) + 1
        return image.crop(self._viewport_rect) if self._viewport_rect else image

    async def run(self):
        while not self._closed:
            started = time.monotonic()
            stage = "capture"
            try:
                if self.context.epoch != self._epoch or self.target_key() != self._target:
                    self.reset()
                    self._target = self.target_key()
                if not self.cfg.observation_enabled:
                    self._blocked_reason = None
                    self._not_ready_reason = "观察已暂停 · 按键可识屏翻译"
                    self._set_status(self._not_ready_reason)
                elif self._target is None:
                    self._valid = False
                    self._blocked_reason = None
                    self._not_ready_reason = "选择游戏窗口后开始本地观察"
                    self._set_status(self._not_ready_reason)
                elif started >= self._ocr_failed_until:
                    epoch, target = self._epoch, self._target
                    image = await self.capture()
                    if not self._live(epoch) or target != self.target_key():
                        continue
                    self._blocked_reason = None
                    if (self.cfg.auto_game_viewport and self._viewport_size is not None
                            and self._viewport_size != image.size):
                        requested = self._requested
                        self.reset()
                        self._requested = requested
                        epoch, target = self._epoch, self._target
                    image = await self._prepare_game_frame(image, epoch, target)
                    if image is None:
                        continue
                    self._blocked_reason = None
                    self.metrics["frames"] += 1
                    top = round(image.height * self.cfg.ocr_region_top_percent / 100)
                    bottom = round(image.height * self.cfg.ocr_region_bottom_percent / 100)
                    cropped = image.crop((0, top, image.width, bottom))
                    # Exact downsampled fingerprint suppresses repeated inference
                    # on paused pages; stable time still advances every poll.
                    fingerprint = cropped.resize((320, 180)).tobytes()
                    if fingerprint != self._last_frame or self._last_ocr is None:
                        stage = "ocr"
                        self._valid = False
                        if self._last_ocr is None:
                            self._not_ready_reason = "正在加载 MeikiOCR 并读取当前画面…"
                            self._set_status(self._not_ready_reason)
                        before = time.perf_counter()
                        result = await asyncio.get_running_loop().run_in_executor(
                            self._executor, self.reader.recognize, cropped)
                        self.metrics["last_ocr_ms"] = (time.perf_counter() - before) * 1000
                        self.metrics["ocr_calls"] += 1
                        if epoch != self._epoch or target != self.target_key():
                            continue
                        self._last_ocr = result
                        self._last_frame = fingerprint
                    if epoch == self.context.epoch and target == self.target_key():
                        stage = "processing"
                        self.process_frame(image, self._last_ocr, time.monotonic(), top=top)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._valid = False
                self._local_failure(stage, exc)
                self._ocr_failed_until = time.monotonic() + 2.0
            delay = max(.01, self.cfg.ocr_poll_ms / 1000 - (time.monotonic() - started))
            try:
                await asyncio.wait_for(self._wake.wait(), delay)
            except asyncio.TimeoutError:
                pass
            finally:
                self._wake.clear()

    def process_frame(self, image, result: OCRResult, now: float, *, top=0):
        """Pure input boundary used by the live loop and recorded-video tests."""
        self._blocked_reason = None
        key = text_key(result.text)
        if self._suppressed_key is not None:
            if key == self._suppressed_key or key in self._suppressed_keys:
                result = OCRResult("", result.confidence, result.boxes)
                key = ""
            else:
                self._suppressed_key = None
                self._suppressed_keys.clear()
        if key != self._last_raw_key:
            self._last_raw_key, self._last_raw_change = key, now
        event = self.stability.observe(result.text, now, boxes=result.boxes,
                                       confidence=result.confidence)
        if self.stability.page != self._page_raw_page:
            self._page_raw_keys.clear()
            self._page_raw_page = self.stability.page
        if key and len(self._page_raw_keys) < 64:
            self._page_raw_keys.add(key)
        # The stabilizer owns page identity and tolerates OCR jitter. Comparing
        # the raw string again here would invalidate a good cache on every blink.
        key = text_key(self.stability.current_text)
        self._valid = bool(self._current and self.stability.is_stable
                           and self.stability.page == self._current_page)
        if not key:
            self._valid = False
            self._current = self._cache = None
        if event is not None:
            replace_id = None
            previous_translation = None
            if (self._cache and self._current == self._cache[0]
                    and event.page == self._current_page):
                previous_translation = (self._current_text, self._cache[1])
            if (self._current and event.page == self._current_page
                    and self.context.get_observation(*self._current) is not None):
                replace_id = self._current[0]
            line = self.context.observe(event.text, replace_id=replace_id)
            self._current = (line.observation_id, line.revision)
            self._current_page, self._current_text = event.page, event.text
            self._local_text = event.text
            self._valid, self._cache = True, None
            self.metrics["observations"] += 1
            self.metrics["last_stability_reason"] = event.reason
            self.on_memory()
            if self._translation_task and not self._translation_task.done():
                self._translation_task.cancel()
            if result.uncertain or event.reason in ("fuzzy", "boxes", "timeout"):
                self._not_ready_reason = "正在校验当前画面文字…"
                self._vision_ref = self._current
                self._translation_task = self._spawn(self._resolve_uncertain(
                    self._current, image.copy(), previous_translation=previous_translation))
            else:
                self._not_ready_reason = "台词已稳定 · 正在后台预译…"
                self._start_translation(self._current)
        elif self._valid:
            self._valid = True
            if self._cache:
                self._not_ready_reason = None
            elif self._translation_task and not self._translation_task.done():
                self._not_ready_reason = ("正在校验当前画面文字…" if self._vision_ref is not None
                                          else "台词已稳定 · 正在后台预译…")
            elif self._not_ready_reason:
                self._set_status(self._not_ready_reason)
            if self._cache and not self._reveal():
                self._set_status("译文已缓存 · 按背键 / Ctrl+Alt+T 显示")
        elif key:
            self._not_ready_reason = "正在等待文字稳定…"
            self._set_status(self._not_ready_reason)
        else:
            self._not_ready_reason = "观察中 · 当前画面暂无台词"
            self._set_status(self._not_ready_reason)

        # Scene descriptions wait for stable dialogue/blank text too. Character
        # boxes mask the dialogue so typewriter pixels do not count as scenes.
        quiet = self.stability.is_stable or (not key and now - self._last_raw_change >= self.cfg.ocr_stable_ms / 1000)
        if self.cfg.scene_observation_enabled and quiet and self._vision_ref is None:
            boxes = tuple((x1 / image.width, (y1 + top) / image.height,
                           x2 / image.width, (y2 + top) / image.height)
                          for x1, y1, x2, y2 in result.boxes)
            # An explicitly selected dialogue band is also excluded as a whole.
            if top:
                boxes += ((0, top / image.height, 1, 1),)
            elif self.cfg.ocr_region_bottom_percent < 100:
                boxes += ((0, self.cfg.ocr_region_bottom_percent / 100, 1, 1),)
            if (not self._scene_task or self._scene_task.done()) and (
                    now - self.scene_gate.last_attempt >= self.cfg.scene_observation_interval
                    and self.scene_gate.changed(image, boxes, self.cfg.scene_change_threshold)):
                self.scene_gate.submitted(image, boxes, now)
                # The placeholder fixes chronology while the network is pending.
                # Message builders omit an empty record until its scene arrives.
                line = self.context.observe("")
                ref = (line.observation_id, line.revision)
                dialogue_ref = self._current if event is not None else None
                self._scene_task = self._spawn(self._describe(ref, image.copy(), self.stability.current_text, dialogue_ref))

    def _start_translation(self, ref):
        if self._translation_task and not self._translation_task.done():
            self._translation_task.cancel()
        self._not_ready_reason = "台词已稳定 · 正在后台预译…"
        self._translation_task = self._spawn(self._translate(ref))

    async def _translate(self, ref):
        epoch = self._epoch
        try:
            if self._current == ref:
                self._not_ready_reason = "台词已稳定 · 正在后台预译…"
            self.metrics["translation_calls"] += 1
            before = time.perf_counter()
            translated = await asyncio.wait_for(self.translator.translate_observation(*ref), 45)
            if not translated.strip():
                raise RuntimeError("Empty translation")
            if self._live(epoch) and translated.strip():
                if self.context.commit_translation(*ref, translated):
                    self.on_memory()
                self.metrics["last_translation_ms"] = (time.perf_counter() - before) * 1000
                if self._current == ref:
                    self._not_ready_reason = None
                    self._cache = (ref, translated)
                    if not self._reveal():
                        self._set_status("译文已缓存 · 按背键 / Ctrl+Alt+T 显示")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Background translation failed: %s", type(exc).__name__)
            if self._live(epoch, ref) and self._current == ref:
                self._not_ready_reason = "预译暂未完成 · 再按翻译键重试，模型连接可在设置中检查"
                self._set_status(self._not_ready_reason)

    async def _resolve_uncertain(self, ref, image, *, previous_translation=None):
        epoch = self._epoch
        original_ref = ref
        try:
            self.metrics["vision_fallbacks"] += 1
            jp, scene = await asyncio.wait_for(
                self.translator.read_screen(ScreenshotService.encode_frame(image)), 45)
            if not self._live(epoch) or self._current != ref:
                return
            self.scene_gate.submitted(image, (), time.monotonic())
            if jp.strip():
                retained = self.context.get_observation(*ref) is not None
                line = self.context.observe(jp, scene or None, replace_id=ref[0] if retained else None)
                ref = (line.observation_id, line.revision)
                self._current = ref
                # Track the local text separately: corrected text is presented,
                # while local stabilization/dedup continues on the image OCR.
                self._current_text = jp
                # A new local glyph may be a HUD/icon rather than new dialogue.
                # Reuse only a same-page cache after vision confirms identical
                # source text; a real addition or a page change still translates.
                if previous_translation and text_key(jp) == text_key(previous_translation[0]):
                    translated = previous_translation[1]
                    self.context.commit_translation(*ref, translated)
                    self._cache = (ref, translated)
                    self._not_ready_reason = None
                    self.metrics["verified_cache_reuses"] = self.metrics.get("verified_cache_reuses", 0) + 1
                    self.on_memory()
                    if not self._reveal():
                        self._set_status("译文已缓存 · 按背键 / Ctrl+Alt+T 显示")
                    return
                self.on_memory()
            else:
                # Vision may identify local OCR as a cursor/menu artefact. Keep
                # any useful scene, and suppress that exact OCR key until change.
                self._suppressed_key = text_key(self._local_text)
                self._suppressed_keys = set(self._page_raw_keys)
                retained = self.context.get_observation(*ref) is not None
                line = self.context.observe("", scene or None, replace_id=ref[0] if retained else None)
                if not scene:
                    self.context.remove_empty_observation(line.observation_id, line.revision)
                self._current = self._cache = None
                self._valid = False
                self._not_ready_reason = "画面校验完成 · 当前暂无台词"
                self.stability.reset()
                self._page_raw_keys.clear()
                self._page_raw_page = None
                self.on_memory()
                self._set_status(self._not_ready_reason)
                return
            await self._translate(ref)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Visual OCR verification failed: %s", type(exc).__name__)
            if self._live(epoch, ref):
                await self._translate(ref)
        finally:
            if self._vision_ref == original_ref:
                self._vision_ref = None

    async def _describe(self, ref, image, text, dialogue_ref=None):
        epoch = self._epoch
        try:
            self.metrics["scene_calls"] += 1
            scene = await asyncio.wait_for(
                self.translator.describe_scene(ScreenshotService.encode_frame(image), text), 45)
            if not scene.strip():
                raise RuntimeError("Empty scene description")
            if self._live(epoch, ref) and scene.strip():
                dialogue = self.context.get_observation(dialogue_ref[0]) if dialogue_ref else None
                if dialogue is not None and dialogue.revision != dialogue_ref[1]:
                    # A revised page may already carry a newer visual check.
                    # Preserve it and allow a fresh scene read after cooldown.
                    self.scene_gate.reference = None
                    return
                if dialogue is not None:
                    self.context.attach_scene(dialogue.observation_id, dialogue.revision, scene)
                    self.context.remove_empty_observation(*ref)
                    self.on_memory()
                elif self.context.attach_scene(*ref, scene):
                    self.on_memory()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Scene observation failed: %s", type(exc).__name__)
            if self._live(epoch, ref):
                self.scene_gate.reference = None  # retry after the rate limit
        finally:
            if epoch == self.context.epoch and self.context.remove_empty_observation(*ref):
                self.on_memory()

    def stop(self):
        self._closed = True
        for line in list(self.context.history):
            self.context.remove_empty_observation(line.observation_id, line.revision)
        if self._run_task:
            self._run_task.cancel()
        for task in tuple(self._tasks):
            task.cancel()

    async def aclose(self):
        self.stop()
        pending = list(self._tasks)
        if self._run_task:
            pending.append(self._run_task)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._executor.shutdown(wait=False, cancel_futures=True)

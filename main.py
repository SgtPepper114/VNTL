from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from runtime_bootstrap import preload_windows_runtime

preload_windows_runtime()

from PyQt6.QtCore import QLockFile
from PyQt6.QtWidgets import QApplication, QMessageBox
import qasync

from config import load_config, save_config
from context_manager import ContextManager
from translator import Translator
from overlay import OverlayWindow
from hooker import HookerService
from hotkeys import GlobalHotkeys
from controller import ControllerService
from fonts import initialize_fonts
from screenshot_service import ScreenshotService
from summary_worker import BackgroundSummarizer
from observation_service import ObservationService

DATA_DIR = Path.home() / ".config" / "vntl"
AUTOSAVE_PATH = DATA_DIR / "contexts" / "自动保存.json"
LEGACY_AUTOSAVE_PATH = DATA_DIR / "contexts" / "黄昏症候群_自动保存.json"
logger = logging.getLogger("vntl")


def _setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(DATA_DIR / "vntl.log", maxBytes=2 * 1024 * 1024,
                                  backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def save_context_file(context: ContextManager, path: str | Path) -> None:
    """Atomic save so interruption cannot leave a half-written story memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(context.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_provider_key(cfg, role: str) -> None:
    provider = getattr(cfg, role + "_provider")
    if provider != "ollama" and not getattr(cfg, provider + "_api_key", ""):
        raise RuntimeError("请先打开设置，填写所选模型供应商的 API Key。")


def friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status == 401 or "Authentication" in name:
        return "API 身份验证失败，请检查设置中的密钥。"
    if status == 402:
        return "API 余额不足，请检查模型账户余额。"
    if status == 429 or "RateLimit" in name:
        return "API 请求过于频繁或额度不足，请稍后重试。"
    if "Timeout" in name:
        return "模型响应超时，请重试。"
    if "Connection" in name:
        return "无法连接模型服务，请检查网络。"
    if status is not None:
        return f"模型接口返回 HTTP {status}，请检查模型名称和服务状态。"
    if type(exc) in (RuntimeError, ValueError) and str(exc) and len(str(exc)) < 180:
        return str(exc)
    return f"处理失败（{name}），请重试。"


async def capture_game(screenshots: ScreenshotService) -> str:
    """Keep our own floating windows out of the visible-screen capture."""
    if QApplication.activeModalWidget() is not None:
        raise RuntimeError("请先关闭设置或文件选择窗口，再识屏翻译。")
    visible = [w for w in QApplication.topLevelWidgets() if w.isVisible()]
    try:
        for window in visible:
            window.hide()
        await asyncio.sleep(0.15)
        return screenshots.capture_for_text()
    finally:
        for window in visible:
            window.show()


async def capture_observation(screenshots: ScreenshotService):
    """Keep Qt responsive and avoid hiding/showing windows in a capture loop."""
    if QApplication.activeModalWidget() is not None:
        raise RuntimeError("设置窗口打开时观察暂停，关闭后自动继续。")
    for window in QApplication.topLevelWidgets():
        if window.isVisible() and not screenshots.exclude_window(int(window.winId())):
            raise RuntimeError(screenshots.exclusion_error_text())
    return await asyncio.to_thread(screenshots.capture_frame)


def make_summary_worker(translator, overlay, cfg, context):
    def on_committed():
        save_context_file(context, AUTOSAVE_PATH)
        overlay.update_context(context.summary, context.history)

    return BackgroundSummarizer(
        translator, context, on_committed,
        check_ready=lambda: check_provider_key(cfg, "summarizer"),
    )


async def input_loop(translator, overlay, hooker, screenshots, cfg, context, requests,
                     summaries=None, observer=None):
    owns_worker = summaries is None
    summaries = summaries or make_summary_worker(translator, overlay, cfg, context)
    try:
        summaries.request()  # Loaded memory may already meet the token threshold.
        await _process_inputs(translator, overlay, hooker, screenshots, cfg, context, requests, summaries, observer)
    finally:
        if owns_worker:
            await summaries.aclose()


async def _process_inputs(translator, overlay, hooker, screenshots, cfg, context, requests, summaries, observer=None):
    clipboard = QApplication.clipboard()
    last_clipboard = clipboard.text().strip()
    while True:
        command, text = "", ""
        try:
            command, text = requests.get_nowait()
        except asyncio.QueueEmpty:
            if hooker.is_attached:
                try:
                    text = hooker.text_queue.get_nowait()
                    await asyncio.sleep(cfg.line_batch_window_ms / 1000)
                    items = [text]
                    while not hooker.text_queue.empty():
                        items.append(hooker.text_queue.get_nowait())
                    text = "\n".join(items)
                    command = "text"
                except asyncio.QueueEmpty:
                    pass
            if not command and cfg.clipboard_enabled:
                current = clipboard.text().strip()
                if current and current != last_clipboard:
                    command, text = "text", current
                last_clipboard = current
        if not command:
            await asyncio.sleep(0.1)
            continue

        if command == "compact":
            summaries.request(force=True)
            continue

        if command == "screen" and observer is not None and cfg.observation_enabled:
            observer.request_display()
            continue

        overlay.set_busy(True)
        operation_epoch = context.epoch
        operation_target = getattr(screenshots, "generation", None)

        def operation_current():
            return (context.epoch == operation_epoch
                    and getattr(screenshots, "generation", None) == operation_target)

        try:
            check_provider_key(cfg, "translator")
            if command == "screen":
                check_provider_key(cfg, "descriptor")
                image = await capture_game(screenshots)
                if not operation_current():
                    continue
                overlay.show()
                overlay.show_loading("正在识别日文与画面…")
                text, scene = await translator.read_screen(image)
                if not operation_current():
                    continue
                if not text.strip():
                    overlay.show_error_text("", "当前画面没有识别到台词。请等待文字显示完整后重试。")
                    continue
                context.current_scene_description = scene or None
            elif cfg.screenshot_enabled and screenshots.is_attached and command != "retry":
                check_provider_key(cfg, "descriptor")
                image = await capture_game(screenshots)
                if not operation_current():
                    continue
                overlay.show_loading("正在读取场景…")
                scene = await translator.describe_scene(image, text)
                if not operation_current():
                    continue
                context.current_scene_description = scene

            text = text.strip()
            if not text:
                continue
            # Repeating the same visible line replaces the previous translation.
            old_line = context.history[-1] if context.history and context.history[-1].jp == text else None
            replaced = old_line is not None
            overlay.show_loading()
            translated = await translator.translate(text, replace_last=replaced)
            if not operation_current():
                continue
            if not translated:
                raise RuntimeError("模型没有返回译文，请重试。")
            if observer is not None:
                observer.external_display()
            overlay.update_text(text, translated, replace_last=replaced)
            overlay.update_context(context.summary, context.history)
            save_context_file(context, AUTOSAVE_PATH)
            overlay.show_retry_button(lambda t=text: enqueue(requests, "retry", t))
            summaries.request()
        except Exception as exc:
            if not operation_current():
                continue
            logger.warning("Translation operation failed: %s (HTTP %s)",
                           type(exc).__name__, getattr(exc, "status_code", None))
            overlay.show()
            overlay.show_error_text(text or "", friendly_error(exc))
            if text:
                overlay.show_retry_button(lambda t=text: enqueue(requests, "retry", t), is_error=True)
        finally:
            overlay.set_busy(False)


def enqueue(requests, command: str, text: str = "") -> None:
    # One pending command is enough; key repeat must not create a paid backlog.
    if requests.empty():
        requests.put_nowait((command, text))


def clear_requests(requests):
    while not requests.empty():
        requests.get_nowait()


async def main() -> None:
    app = QApplication.instance()
    app.setQuitOnLastWindowClosed(False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    instance_lock = QLockFile(str(DATA_DIR / "vntl.lock"))
    if not instance_lock.tryLock(0):
        QMessageBox.information(None, "VNTL 已启动", "程序已经在运行。可按 Ctrl+Alt+H 显示浮窗。")
        return
    cfg = load_config()
    initialize_fonts(app, cfg)
    for provider, env_name in [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"),
                               ("google", "GOOGLE_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")]:
        if hasattr(cfg, provider + "_api_key") and not getattr(cfg, provider + "_api_key"):
            setattr(cfg, provider + "_api_key", os.environ.get(env_name, ""))
    if not cfg.google_api_key:
        cfg.google_api_key = os.environ.get("GEMINI_API_KEY", "")

    context = ContextManager(cfg.context_summarize_tokens, cfg.context_min_recent_lines)
    restore_path = AUTOSAVE_PATH if AUTOSAVE_PATH.exists() else LEGACY_AUTOSAVE_PATH
    if restore_path.exists():
        try:
            context.load_from_dict(json.loads(restore_path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError, OSError):
            logger.warning("Could not load autosaved context; starting a new session.")
    translator = Translator(cfg, context)
    requests = asyncio.Queue(maxsize=1)
    hooker = HookerService()
    hooker.set_debounce(cfg.debounce_ms)
    hooker.start_server()
    screenshots = ScreenshotService()
    screenshots.threshold = cfg.scene_change_threshold
    screenshots.cooldown = cfg.scene_change_cooldown
    def request_screen():
        if app.activeModalWidget() is None and (cfg.observation_enabled or not overlay.is_busy):
            enqueue(requests, "screen")

    def request_controller_screen():
        if cfg.controller_enabled and app.activeModalWidget() is None:
            logger.info("Controller translation requested: %s", cfg.controller_button)
            request_screen()

    def on_save(updated):
        summaries.cancel()
        save_config(updated)
        translator.rebuild_clients()
        context.update_thresholds(updated.context_summarize_tokens, updated.context_min_recent_lines)
        overlay.update_game_title()
        overlay.set_screenshot_enabled(updated.screenshot_enabled)
        overlay.update_observation_controls()
        controller.configure(updated.controller_enabled, updated.controller_button)
        observer.configure()
        summaries.request()

    def load_context(path):
        summaries.cancel()
        context.load_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        observer.reset()
        clear_requests(requests)
        save_context_file(context, AUTOSAVE_PATH)
        summaries.request()

    def clear_context():
        summaries.cancel()
        context.clear()
        observer.reset()
        clear_requests(requests)
        save_context_file(context, AUTOSAVE_PATH)

    overlay = OverlayWindow(
        cfg, on_save, lambda path: save_context_file(context, path), load_context,
        clear_context, lambda: enqueue(requests, "compact"), hooker, screenshots,
        get_context_fn=lambda: (context.summary, context.history),
        request_screen_fn=request_screen,
    )
    summaries = make_summary_worker(translator, overlay, cfg, context)

    def memory_changed():
        save_context_file(context, AUTOSAVE_PATH)
        overlay.update_context(context.summary, context.history)
        summaries.request()

    def display_observation(jp, zh, replace):
        overlay.show()
        overlay.update_text(jp, zh, replace_last=replace)
        overlay.show_retry_button(lambda t=jp: observer.request_display(retry=True)
                                  if cfg.observation_enabled else enqueue(requests, "retry", t))

    observer = ObservationService(
        cfg, context, translator,
        capture=lambda: capture_observation(screenshots),
        target_key=lambda: screenshots.generation if screenshots.is_attached else None,
        on_display=display_observation, on_memory=memory_changed,
        on_status=overlay.set_observation_status,
        can_display=lambda: app.activeModalWidget() is None and not overlay.is_busy,
    )

    def target_changed():
        observer.reset()
        clear_requests(requests)
        summaries.cancel()
        summaries.request()

    screenshots.on_target_changed = target_changed
    overlay.update_context(context.summary, context.history)
    overlay.show()
    hotkeys = GlobalHotkeys(app, request_screen, overlay.toggle_visible)
    controller = ControllerService(app, request_controller_screen,
                                   enabled=cfg.controller_enabled, button=cfg.controller_button)
    overlay.controller_status_fn = lambda: controller.status
    if hotkeys.failed:
        overlay.show_error_text("", "快捷键被占用：" + "、".join(hotkeys.failed) + "。可使用浮窗按钮。")

    def on_exit():
        observer.stop()
        summaries.stop()
        overlay.save_geometry_to_config()
        save_config(cfg)
        save_context_file(context, AUTOSAVE_PATH)
        hotkeys.close()
        controller.close()
        hooker.detach()
        instance_lock.unlock()

    app.aboutToQuit.connect(on_exit)
    logger.info("VNTL 日译中启动，模型 %s/%s", cfg.translator_provider, cfg.translator_model)
    try:
        observer.start()
        await input_loop(translator, overlay, hooker, screenshots, cfg, context, requests, summaries, observer)
    finally:
        await observer.aclose()
        await summaries.aclose()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check-ocr"]:
        from diagnostics import check_ocr
        raise SystemExit(check_ocr(sys.argv[2:]))
    _setup_logging()
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        try:
            loop.run_until_complete(main())
        except RuntimeError as exc:
            if "Event loop stopped before Future completed" not in str(exc):
                logger.exception("Startup failed")
                raise

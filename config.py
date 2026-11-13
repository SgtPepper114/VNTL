import json
import math
import os
from dataclasses import dataclass, field, asdict

CONFIG_PATH = os.path.expanduser("~/.config/vntl/config.json")
GAME_TITLE_MAX_LENGTH = 200
GAME_NOTES_MAX_LENGTH = 4000


def normalize_game_text(value: object, max_length: int) -> str:
    """Keep user-provided game context as bounded plain text."""
    return value.strip()[:max_length].rstrip() if isinstance(value, str) else ""


def _bounded_number(value: object, default: int | float, minimum, maximum):
    """Reject malformed settings locally, without discarding the rest of the file."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    number = max(minimum, min(maximum, number))
    return int(number) if isinstance(default, int) else float(number)

ROLE_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "translator": {
        "anthropic": "claude-sonnet-5-5",
        "openai":    "gpt-5.2",
        "google":    "gemini-3.8-flash",
        "deepseek":  "deepseek-flash",
        "ollama":    "qwen3:14b",
    },
    "descriptor": {
        "anthropic": "claude-haiku-4-5",
        "openai":    "gpt-4o-mini",
        "google":    "gemini-3.8-flash",
        "deepseek":  "deepseek-flash",
        "ollama":    "qwen2.5vl:7b",
    },
    "summarizer": {
        "anthropic": "claude-haiku-4-5",
        "openai":    "gpt-5-mini",
        "google":    "gemini-3.8-flash",
        "deepseek":  "deepseek-flash",
        "ollama":    "qwen3:14b",
    },
}

ROLE_MODEL_OPTIONS: dict[str, dict[str, list[str]]] = {
    "translator": {
        "anthropic": ["claude-sonnet-5-5", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-6"],
        "openai":    ["gpt-5.2", "gpt-5-mini", "gpt-4o"],
        "google":    ["gemini-3.8-flash", "gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-2.5-flash"],
        "deepseek":  ["deepseek-flash", "deepseek-v4-pro"],
        "ollama":    ["qwen3:14b", "mistral-small3.2:24b", "gemma3:12b"],
    },
    "descriptor": {
        "anthropic": ["claude-sonnet-5-5", "claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6"],
        "openai":    ["gpt-4o-mini", "gpt-4o", "gpt-5.2"],
        "google":    ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-pro"],
        "deepseek":  ["deepseek-flash"],
        "ollama":    ["qwen2.5vl:7b", "gemma3:4b", "moondream"],
    },
    "summarizer": {
        "anthropic": ["claude-sonnet-5-5", "claude-haiku-4-5", "claude-sonnet-4-6", "claude-sonnet-4-5"],
        "openai":    ["gpt-5-mini", "gpt-5.2", "gpt-5-nano"],
        "google":    ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-pro-preview"],
        "deepseek":  ["deepseek-flash", "deepseek-v4-pro"],
        "ollama":    ["qwen3:14b", "gpt-oss:20b", "gemma3:12b"],
    },
}


@dataclass
class OverlayConfig:
    x: int = 100
    y: int = 100
    width: int = 700
    height: int = 150
    bg_color: list = field(default_factory=lambda: [20, 20, 25, 210])
    jp_color: list = field(default_factory=lambda: [200, 200, 200, 180])
    en_color: list = field(default_factory=lambda: [255, 255, 255, 230])
    jp_font_family: str = "Yu Gothic"
    jp_font_size:   int = 10
    jp_furigana_enabled: bool = True
    en_font_family: str = "Microsoft YaHei UI"
    en_font_size:   int = 13


@dataclass
class Config:
    # API keys — shared per provider
    anthropic_api_key: str = ""
    openai_api_key:    str = ""
    google_api_key:    str = ""
    deepseek_api_key:  str = ""
    ollama_base_url:   str = "http://localhost:11434/v1"

    # Per-role provider + model (anthropic | openai | google | deepseek | ollama).
    translator_provider: str = "deepseek"
    translator_model:    str = "deepseek-flash"
    descriptor_provider: str = "deepseek"
    descriptor_model:    str = "deepseek-flash"
    summarizer_provider: str = "deepseek"
    summarizer_model:    str = "deepseek-flash"

    # Per-role Ollama thinking mode (only applies when provider == "ollama")
    translator_ollama_thinking: bool = False
    descriptor_ollama_thinking:  bool = False
    summarizer_ollama_thinking:  bool = False

    # Gemini's OpenAI-compatible endpoint maps these to its thinking level.
    translator_google_thinking_level: str = "medium"
    descriptor_google_thinking_level: str = "low"
    summarizer_google_thinking_level: str = "low"

    screenshot_enabled: bool = False
    clipboard_enabled:  bool = False
    # Continuous observation starts only after a game window is selected.
    observation_enabled: bool = True
    translation_mode: str = "manual"  # manual | auto; observation runs in both.
    ocr_stable_ms: int = 600
    ocr_poll_ms: int = 200
    ocr_region_top_percent: int = 0
    scene_observation_enabled: bool = True
    scene_observation_interval: float = 10.0
    game_title: str = ""
    game_notes: str = ""
    controller_enabled: bool = True
    controller_button: str = "GR"
    context_summarize_tokens: int = 10_000
    context_min_recent_lines: int = 30
    summary_max_tokens:       int = 5_000
    line_batch_window_ms:     int = 200
    debounce_ms:              int = 150

    # Per-role max output tokens
    translator_max_tokens: int = 2048
    descriptor_max_tokens: int = 2048
    summarizer_max_tokens: int = 8192
    # Screenshot scene-change sensitivity (% of pixels changed, stored as float)
    scene_change_threshold: float = 5.0
    # Seconds to suppress re-triggering after a scene change (absorbs transition animations)
    scene_change_cooldown: float = 1.0

    # Editable system prompts — empty string means "use built-in default"
    translator_system_prompt: str = ""
    descriptor_system_prompt: str = ""
    summarizer_system_prompt: str = ""

    overlay: OverlayConfig = field(default_factory=OverlayConfig)


def load_config() -> Config:
    if not os.path.exists(CONFIG_PATH):
        return Config()
    try:
        with open(CONFIG_PATH, encoding="utf-8-sig") as f:
            data = json.load(f)
        cfg = Config()
        cfg.game_title = normalize_game_text(data.get("game_title", ""), GAME_TITLE_MAX_LENGTH)
        cfg.game_notes = normalize_game_text(data.get("game_notes", ""), GAME_NOTES_MAX_LENGTH)

        # --- Migration: read legacy single-backend fields ---
        legacy_api_key      = data.get("api_key", "")
        legacy_model        = data.get("model", "claude-sonnet-4-6")
        legacy_backend      = data.get("backend", "claude")   # "claude" | "ollama"
        legacy_ollama_model = data.get("ollama_model", "qwen3:14b")

        _backend_map = {"claude": "anthropic", "ollama": "ollama"}
        legacy_provider = _backend_map.get(legacy_backend, "anthropic")
        legacy_role_model = legacy_ollama_model if legacy_backend == "ollama" else legacy_model
        has_legacy_backend = any(key in data for key in ("backend", "model", "api_key", "ollama_model"))

        # --- New fields (legacy values used only when new keys are absent) ---
        cfg.anthropic_api_key = data.get("anthropic_api_key", legacy_api_key)
        cfg.openai_api_key    = data.get("openai_api_key", "")
        cfg.google_api_key    = data.get("google_api_key", "")
        cfg.deepseek_api_key  = data.get("deepseek_api_key", "")
        cfg.ollama_base_url   = data.get("ollama_base_url", "http://localhost:11434/v1")

        for role in ("translator", "descriptor", "summarizer"):
            provider_attr, model_attr = f"{role}_provider", f"{role}_model"
            default_provider = legacy_provider if has_legacy_backend else getattr(cfg, provider_attr)
            provider = data.get(provider_attr, default_provider)
            default_model = (
                legacy_role_model if has_legacy_backend and provider == legacy_provider
                else ROLE_PROVIDER_DEFAULTS.get(role, {}).get(provider, getattr(cfg, model_attr))
            )
            setattr(cfg, provider_attr, provider)
            setattr(cfg, model_attr, data.get(model_attr, default_model))
            thinking_attr = f"{role}_google_thinking_level"
            level = data.get(thinking_attr, getattr(cfg, thinking_attr))
            if level in ("low", "medium", "high"):
                setattr(cfg, thinking_attr, level)

        cfg.translator_ollama_thinking = data.get("translator_ollama_thinking", False)
        cfg.descriptor_ollama_thinking  = data.get("descriptor_ollama_thinking",  False)
        cfg.summarizer_ollama_thinking  = data.get("summarizer_ollama_thinking",  False)

        cfg.screenshot_enabled        = data.get("screenshot_enabled", False)
        cfg.clipboard_enabled         = data.get("clipboard_enabled", False)
        for name in ("observation_enabled", "scene_observation_enabled"):
            value = data.get(name, getattr(cfg, name))
            if isinstance(value, bool):
                setattr(cfg, name, value)
        mode = data.get("translation_mode", cfg.translation_mode)
        if mode in ("manual", "auto"):
            cfg.translation_mode = mode
        for name, minimum, maximum in (
            ("ocr_stable_ms", 200, 2000),
            ("ocr_poll_ms", 100, 1000),
            ("ocr_region_top_percent", 0, 90),
            ("scene_observation_interval", 3.0, 120.0),
        ):
            default = getattr(cfg, name)
            setattr(cfg, name, _bounded_number(data.get(name, default), default, minimum, maximum))
        controller_enabled = data.get("controller_enabled", True)
        cfg.controller_enabled = controller_enabled if isinstance(controller_enabled, bool) else True
        controller_button = data.get("controller_button", "GR")
        cfg.controller_button = controller_button if controller_button in ("GR", "GL") else "GR"
        cfg.context_summarize_tokens  = data.get("context_summarize_tokens", cfg.context_summarize_tokens)
        cfg.context_min_recent_lines  = data.get("context_min_recent_lines", 30)
        cfg.summary_max_tokens        = data.get("summary_max_tokens", 5_000)
        cfg.line_batch_window_ms      = data.get("line_batch_window_ms", 200)
        cfg.debounce_ms               = data.get("debounce_ms", 150)

        cfg.translator_max_tokens  = data.get("translator_max_tokens", 2048)
        cfg.descriptor_max_tokens  = data.get("descriptor_max_tokens", 2048)
        cfg.summarizer_max_tokens  = data.get("summarizer_max_tokens", cfg.summarizer_max_tokens)
        cfg.scene_change_threshold = float(data.get("scene_change_threshold", cfg.scene_change_threshold))
        cfg.scene_change_cooldown  = float(data.get("scene_change_cooldown", 1.0))

        cfg.translator_system_prompt = data.get("translator_system_prompt", "")
        cfg.descriptor_system_prompt = data.get("descriptor_system_prompt", "")
        cfg.summarizer_system_prompt = data.get("summarizer_system_prompt", "")

        if "overlay" in data:
            ov = data["overlay"]
            furigana_enabled = ov.get("jp_furigana_enabled", True)
            cfg.overlay = OverlayConfig(
                x=ov.get("x", 100),
                y=ov.get("y", 100),
                width=ov.get("width", 700),
                height=ov.get("height", 150),
                bg_color=ov.get("bg_color", [20, 20, 25, 210]),
                jp_color=ov.get("jp_color", [200, 200, 200, 180]),
                en_color=ov.get("en_color", [255, 255, 255, 230]),
                jp_font_family=ov.get("jp_font_family", "Yu Gothic"),
                jp_font_size=ov.get("jp_font_size", 10),
                jp_furigana_enabled=furigana_enabled if isinstance(furigana_enabled, bool) else True,
                en_font_family=ov.get("en_font_family", "Microsoft YaHei UI"),
                en_font_size=ov.get("en_font_size", 13),
            )
        return cfg
    except Exception:
        return Config()


def save_config(cfg: Config) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    data = asdict(cfg)
    data["game_title"] = normalize_game_text(cfg.game_title, GAME_TITLE_MAX_LENGTH)
    data["game_notes"] = normalize_game_text(cfg.game_notes, GAME_NOTES_MAX_LENGTH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

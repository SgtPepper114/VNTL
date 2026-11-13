from __future__ import annotations

import copy
import math
import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFontComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    Config, OverlayConfig, ROLE_PROVIDER_DEFAULTS, ROLE_MODEL_OPTIONS,
    GAME_TITLE_MAX_LENGTH, GAME_NOTES_MAX_LENGTH, normalize_game_text,
)
from translator import _SYSTEM_PROMPT, _SUMMARIZE_SYSTEM, _DESCRIBE_SYSTEM


def _slider_to_threshold(pos: int) -> float:
    """Map slider position 0–100 to a threshold % on a log scale.
    pos=0  → 0.0%   (always trigger)
    pos=33 → 1.0%   (expression-level changes)
    pos=50 → 3.2%   (moderate changes / sprite swaps)
    pos=67 → 10.0%  (scene transitions)
    pos=100 → 100%
    """
    if pos <= 0:
        return 0.0
    return 10 ** (pos * 3 / 100 - 1)


def _threshold_to_slider(pct: float) -> int:
    """Inverse of _slider_to_threshold."""
    if pct <= 0.0:
        return 0
    pos = (math.log10(pct) + 1) * 100 / 3
    return max(1, min(100, round(pos)))

def _ensure_arrow_images() -> tuple[str, str]:
    """
    Generate up/down arrow PNGs using Pillow and return (up_path, down_path).
    Paths use forward slashes for Qt CSS compatibility on Windows.
    Returns ('', '') if Pillow is not installed.
    """
    config_dir = os.path.expanduser("~/.config/vntl")
    os.makedirs(config_dir, exist_ok=True)
    up_path   = os.path.join(config_dir, "arrow_up.png")
    down_path = os.path.join(config_dir, "arrow_down.png")
    try:
        from PIL import Image
        color = (170, 170, 204, 255)  # #aaaacc opaque
        w, h, cx = 9, 5, 4

        def _make(pointing_up: bool, path: str) -> None:
            if os.path.exists(path):
                return
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            px  = img.load()
            for row in range(h):
                half = row if pointing_up else (h - 1 - row)
                for x in range(cx - half, cx + half + 1):
                    if 0 <= x < w:
                        px[x, row] = color
            img.save(path)

        _make(True,  up_path)
        _make(False, down_path)
        return (
            up_path.replace("\\", "/"),
            down_path.replace("\\", "/"),
        )
    except ImportError:
        return ("", "")


_UP_ARROW_PATH, _DOWN_ARROW_PATH = _ensure_arrow_images()

_DARK_BG = "#141419"

_STYLE = f"""
    QDialog {{
        background-color: {_DARK_BG};
    }}
    QWidget {{
        background-color: transparent;
        color: #ffffff;
    }}
    QLabel {{
        color: #ffffff;
    }}
    QLineEdit {{
        background-color: #1e1e26;
        color: #ffffff;
        border: 1px solid #3a3a4a;
        border-radius: 4px;
        padding: 4px 6px;
    }}
    QComboBox {{
        background-color: #1e1e26;
        color: #ffffff;
        border: 1px solid #3a3a4a;
        border-radius: 4px;
        padding: 4px 6px;
        min-width: 120px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox::down-arrow {{
        image: url("{_DOWN_ARROW_PATH}");
        width: 9px;
        height: 5px;
    }}
    QComboBox QAbstractItemView {{
        background-color: #1e1e26;
        color: #ffffff;
        selection-background-color: #2e2e42;
        border: 1px solid #3a3a4a;
    }}
    QAbstractItemView {{
        background-color: #1e1e26;
        color: #ffffff;
        border: 1px solid #3a3a4a;
        outline: 0px;
    }}
    QAbstractScrollArea > QWidget {{
        background-color: #1e1e26;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: #3a3a4a;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: #7070c0;
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{
        background: #5050a0;
        border-radius: 2px;
    }}
    QCheckBox {{
        color: #ffffff;
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        background-color: #1e1e26;
        border: 1px solid #3a3a4a;
        border-radius: 3px;
        width: 14px;
        height: 14px;
    }}
    QCheckBox::indicator:checked {{
        background-color: #5050a0;
        border-color: #7070c0;
    }}
    QSpinBox {{
        background-color: #1e1e26;
        color: #ffffff;
        border: 1px solid #3a3a4a;
        border-radius: 4px;
        padding: 4px 24px 4px 6px;
        min-width: 60px;
        max-width: 72px;
    }}
    QSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid #3a3a4a;
        border-top-right-radius: 4px;
        background-color: #2a2a36;
    }}
    QSpinBox::up-button:hover {{
        background-color: #3a3a52;
    }}
    QSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 18px;
        border-left: 1px solid #3a3a4a;
        border-bottom-right-radius: 4px;
        background-color: #2a2a36;
    }}
    QSpinBox::down-button:hover {{
        background-color: #3a3a52;
    }}
    QSpinBox::up-arrow {{
        image: url("{_UP_ARROW_PATH}");
        width: 9px;
        height: 5px;
    }}
    QSpinBox::down-arrow {{
        image: url("{_DOWN_ARROW_PATH}");
        width: 9px;
        height: 5px;
    }}
    QTabWidget::pane {{
        border: 1px solid #3a3a4a;
        border-radius: 4px;
        background-color: transparent;
    }}
    QTabBar::tab {{
        background-color: #1e1e26;
        color: #aaaacc;
        padding: 6px 18px;
        border: 1px solid #3a3a4a;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: #2a2a3a;
        color: #ffffff;
    }}
    QTabBar::tab:hover:!selected {{
        background-color: #242432;
        color: #ccccee;
    }}
    QTextEdit {{
        background-color: #1e1e26;
        color: #ffffff;
        border: 1px solid #3a3a4a;
        border-radius: 4px;
        padding: 4px 6px;
    }}
"""

_BTN_SWATCH = (
    "QPushButton {{"
    "  background-color: rgba({r},{g},{b},{a});"
    "  border: 1px solid #3a3a4a;"
    "  border-radius: 4px;"
    "  min-width: 48px; max-width: 48px;"
    "  min-height: 22px; max-height: 22px;"
    "}}"
    "QPushButton:hover {{ border-color: #7070c0; }}"
)

_BTN_ACTION = (
    "QPushButton {"
    "  background-color: #2e2e3e;"
    "  color: #ffffff;"
    "  border: 1px solid #3a3a4a;"
    "  border-radius: 4px;"
    "  padding: 6px 20px;"
    "}"
    "QPushButton:hover { background-color: #3e3e52; }"
)

_BTN_SAVE = (
    "QPushButton {"
    "  background-color: #3a3a80;"
    "  color: #ffffff;"
    "  border: 1px solid #5050a0;"
    "  border-radius: 4px;"
    "  padding: 6px 20px;"
    "}"
    "QPushButton:hover { background-color: #5050a0; }"
)

# Display name → internal provider key
_PROVIDER_KEY_MAP: dict[str, str] = {
    "DeepSeek":  "deepseek",
    "Anthropic": "anthropic",
    "OpenAI":    "openai",
    "Google":    "google",
    "Ollama":    "ollama",
}
_PROVIDERS = list(_PROVIDER_KEY_MAP.keys())


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    font = QFont()
    font.setBold(True)
    font.setPointSize(10)
    lbl.setFont(font)
    lbl.setStyleSheet("color: rgba(160,160,220,255); margin-top: 6px;")
    return lbl


def _subsection_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: rgba(140,140,200,200); font-size: 8pt; margin-top: 4px;")
    return lbl


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("background-color: #2a2a3a; max-height: 1px;")
    return line


class SettingsDialog(QDialog):
    """
    Dark-themed settings dialog opened from the overlay's gear button.

    Appearance changes (colors, opacity) are applied live to the overlay.
    Non-appearance changes are applied by the save callback when Save is clicked.
    Cancel reverts all live appearance changes.
    """

    def __init__(self, cfg: Config, overlay, save_fn, screenshot_service=None, hooker=None) -> None:
        super().__init__(overlay)

        self._cfg = cfg
        self._overlay = overlay
        self._save_fn = save_fn
        self._screenshot_service = screenshot_service
        self._hooker = hooker

        # Snapshot current appearance so Cancel can revert exactly
        self._orig_overlay: OverlayConfig = copy.deepcopy(cfg.overlay)

        self.setWindowTitle("VNTL 日译中设置")
        self.setMinimumWidth(620)
        self.resize(680, 760)
        self.setStyleSheet(_STYLE)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint
        )

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(18, 16, 18, 16)

        tabs = QTabWidget()

        for title, builder in [
            ("游戏",     self._build_game),
            ("外观",     self._build_appearance),
            ("接口与模型", self._build_models_and_apis),
            ("手柄",     self._build_controller),
            ("上下文",   self._build_context),
            ("提示词",   self._build_prompts),
        ]:
            page = QWidget()
            page.setStyleSheet("background-color: transparent;")
            lay = QVBoxLayout(page)
            lay.setSpacing(8)
            lay.setContentsMargins(12, 12, 12, 12)
            builder(lay)
            lay.addStretch()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            tabs.addTab(scroll, title)

        root.addWidget(tabs)
        root.addWidget(_separator())
        self._build_buttons(root)
        self._fix_combo_popups()

    def _build_game(self, parent: QVBoxLayout) -> None:
        parent.addWidget(QLabel(f"当前游戏名称（最多 {GAME_TITLE_MAX_LENGTH} 字）："))
        self._game_title_edit = QLineEdit()
        self._game_title_edit.setMaxLength(GAME_TITLE_MAX_LENGTH)
        self._game_title_edit.setPlaceholderText("填写当前正在玩的作品名称，可留空")
        self._game_title_edit.setText(normalize_game_text(self._cfg.game_title, GAME_TITLE_MAX_LENGTH))
        parent.addWidget(self._game_title_edit)

        parent.addWidget(QLabel("补充说明（可选）："))
        self._game_notes_edit = QTextEdit()
        self._game_notes_edit.setAcceptRichText(False)
        self._game_notes_edit.setPlaceholderText(
            "可填写当前篇章、游戏版本，以及翻译偏好，例如：游戏角色名不翻译、术语固定译法、对白语气。"
        )
        self._game_notes_edit.setMinimumHeight(180)
        self._game_notes_edit.setPlainText(normalize_game_text(self._cfg.game_notes, GAME_NOTES_MAX_LENGTH))
        parent.addWidget(self._game_notes_edit)
        self._game_notes_count_label = QLabel()
        self._game_notes_count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        parent.addWidget(self._game_notes_count_label)
        self._game_notes_edit.textChanged.connect(self._on_game_notes_changed)
        self._on_game_notes_changed()

        instructions = QLabel(
            "点击“保存”后从下一次请求生效。补充说明中的翻译偏好优先于默认翻译规则和旧译文。"
            "已经显示的译文不会自动修改；点击 ↻ 可按新偏好重新翻译当前已识别的日文。\n\n"
            "游戏资料会用于识屏、翻译和剧情摘要，但不覆盖画面原文或算作已读剧情；"
            "模型仍以当前画面及已读内容为依据，不凭作品知识补写剧情。\n\n"
            "修改游戏信息不会清空已读记忆。切换作品或路线时，请清空剧情记忆，"
            "或载入对应的记忆文件。"
        )
        instructions.setWordWrap(True)
        parent.addWidget(instructions)

    def _on_game_notes_changed(self) -> None:
        text = self._game_notes_edit.toPlainText()
        if len(text) > GAME_NOTES_MAX_LENGTH:
            position = self._game_notes_edit.textCursor().position()
            self._game_notes_edit.blockSignals(True)
            try:
                text = text[:GAME_NOTES_MAX_LENGTH]
                self._game_notes_edit.setPlainText(text)
                cursor = self._game_notes_edit.textCursor()
                cursor.setPosition(min(position, self._game_notes_edit.document().characterCount() - 1))
                self._game_notes_edit.setTextCursor(cursor)
            finally:
                self._game_notes_edit.blockSignals(False)
        self._game_notes_count_label.setText(f"{len(text)} / {GAME_NOTES_MAX_LENGTH} 字")

    def _build_controller(self, parent: QVBoxLayout) -> None:
        self._controller_enabled_check = QCheckBox("启用 Switch2 Pro 背键翻译")
        self._controller_enabled_check.setChecked(self._cfg.controller_enabled)
        parent.addWidget(self._controller_enabled_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("翻译按键："))
        self._controller_button_combo = QComboBox()
        self._controller_button_combo.addItem("GR · 右背键", "GR")
        self._controller_button_combo.addItem("GL · 左背键", "GL")
        index = self._controller_button_combo.findData(self._cfg.controller_button)
        self._controller_button_combo.setCurrentIndex(max(0, index))
        self._controller_button_combo.setEnabled(self._controller_enabled_check.isChecked())
        self._controller_enabled_check.toggled.connect(self._controller_button_combo.setEnabled)
        row.addWidget(self._controller_button_combo)
        row.addStretch()
        parent.addLayout(row)

        instructions = QLabel(
            "请通过 USB 连接 Switch2 Pro 手柄，并使用手柄上已启用的背键。\n\n"
            "按下一次翻译当前游戏画面，长按不会连续触发。背键直接触发翻译，不发送键盘按键。\n\n"
            "这里只读取现有背键状态，不会更改手柄或游戏的按键映射。点击“保存”后立即生效。"
        )
        instructions.setWordWrap(True)
        parent.addWidget(instructions)
        parent.addWidget(_separator())

        parent.addWidget(QLabel("当前连接状态："))
        self._controller_status_label = QLabel()
        self._controller_status_label.setWordWrap(True)
        self._controller_status_label.setToolTip("显示当前运行状态；修改设置后请点击保存。")
        parent.addWidget(self._controller_status_label)
        self._refresh_controller_status()
        self._controller_status_timer = QTimer(self)
        self._controller_status_timer.setInterval(1000)
        self._controller_status_timer.timeout.connect(self._refresh_controller_status)
        self.finished.connect(self._controller_status_timer.stop)
        self._controller_status_timer.start()

    def _refresh_controller_status(self) -> None:
        status_fn = getattr(self._overlay, "controller_status_fn", None)
        try:
            status = status_fn() if callable(status_fn) else "手柄状态暂不可用。"
        except Exception:
            status = "暂时无法读取手柄状态，请稍后重试。"
        self._controller_status_label.setText(str(status or "手柄状态暂不可用。"))

    def _fix_combo_popups(self) -> None:
        """Force every QComboBox popup in the dialog to be opaque dark."""
        bg  = QColor("#1e1e26")
        fg  = QColor("#ffffff")
        sel = QColor("#2e2e42")
        for combo in self.findChildren(QComboBox):
            view = combo.view()
            view.setAutoFillBackground(True)
            p = QPalette(view.palette())
            p.setColor(QPalette.ColorRole.Base,            bg)
            p.setColor(QPalette.ColorRole.AlternateBase,   bg)
            p.setColor(QPalette.ColorRole.Window,          bg)
            p.setColor(QPalette.ColorRole.Text,            fg)
            p.setColor(QPalette.ColorRole.Highlight,       sel)
            p.setColor(QPalette.ColorRole.HighlightedText, fg)
            view.setPalette(p)

    def _build_appearance(self, parent: QVBoxLayout) -> None:
        self._jp_furigana_check = QCheckBox("日文汉字上方显示假名注音")
        self._jp_furigana_check.setChecked(self._cfg.overlay.jp_furigana_enabled)
        self._jp_furigana_check.toggled.connect(self._on_furigana_changed)
        parent.addWidget(self._jp_furigana_check)
        furigana_hint = QLabel(
            "在本地自动生成平假名注音，不增加模型请求。人名、地名和特殊读法可能不准确。"
        )
        furigana_hint.setWordWrap(True)
        furigana_hint.setStyleSheet("color: #aaaacc; font-size: 11px;")
        parent.addWidget(furigana_hint)
        parent.addWidget(_separator())

        # Opacity slider
        row = QHBoxLayout()
        lbl = QLabel("背景不透明度：")
        lbl.setFixedWidth(165)
        row.addWidget(lbl)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(int(self._cfg.overlay.bg_color[3] / 255 * 100))
        row.addWidget(self._opacity_slider)

        self._opacity_lbl = QLabel(f"{int(self._cfg.overlay.bg_color[3] / 255 * 100)}%")
        self._opacity_lbl.setFixedWidth(36)
        self._opacity_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._opacity_lbl)

        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        parent.addLayout(row)

        # Color rows
        self._bg_btn = self._add_color_row(
            parent, "背景颜色：", self._cfg.overlay.bg_color,
            self._set_bg_color,
        )
        self._jp_btn = self._add_color_row(
            parent, "日文颜色：", self._cfg.overlay.jp_color,
            self._set_jp_color,
        )
        self._en_btn = self._add_color_row(
            parent, "中文译文颜色：", self._cfg.overlay.en_color,
            self._set_en_color,
        )

        # Font rows
        for attr_family, attr_size, row_label in [
            ("jp_font_family", "jp_font_size", "日文字体："),
            ("en_font_family", "en_font_size", "中文译文字体："),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(row_label)
            lbl.setFixedWidth(165)
            row.addWidget(lbl)

            size_spin = QSpinBox()
            size_spin.setRange(6, 72)
            size_spin.setValue(getattr(self._cfg.overlay, attr_size))
            row.addWidget(size_spin)

            font_combo = QFontComboBox()
            font_combo.setEditable(False)
            font_combo.setCurrentFont(QFont(getattr(self._cfg.overlay, attr_family)))
            row.addWidget(font_combo)

            parent.addLayout(row)

            font_combo.currentFontChanged.connect(
                lambda f, a=attr_family: self._on_font_changed(a, f.family()))
            size_spin.valueChanged.connect(
                lambda v, a=attr_size: self._on_font_changed(a, v))

    def _add_color_row(
        self,
        parent: QVBoxLayout,
        label: str,
        color: list[int],
        setter,
    ) -> QPushButton:
        """Add a label + color-swatch button row. Returns the swatch button."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(165)
        row.addWidget(lbl)

        btn = QPushButton()
        self._refresh_swatch(btn, color)
        btn.clicked.connect(lambda: self._pick_color(btn, setter))
        row.addWidget(btn)
        row.addStretch()

        parent.addLayout(row)
        return btn

    def _build_context(self, parent: QVBoxLayout) -> None:
        # Clipboard toggle
        self._clipboard_check = QCheckBox("自动翻译剪贴板文字")
        self._clipboard_check.setChecked(self._cfg.clipboard_enabled)
        parent.addWidget(self._clipboard_check)

        cb_note = QLabel(
            "如果只想按需截图翻译，可关闭此项，避免复制文字时自动翻译。"
        )
        cb_note.setWordWrap(True)
        cb_note.setStyleSheet(
            "color: rgba(160,160,160,180); font-size: 8pt; font-style: italic;"
        )
        parent.addWidget(cb_note)

        # Screenshot toggle
        self._screenshot_check = QCheckBox(
            "辅助场景描述（截图模型须支持视觉）"
        )
        self._screenshot_check.setChecked(self._cfg.screenshot_enabled)
        parent.addWidget(self._screenshot_check)

        sc_note = QLabel(
            "为已提取的台词附加画面信息。请先在右键菜单中绑定游戏截图窗口；"
            "按需截图读字使用“截图翻译”按钮。"
        )
        sc_note.setWordWrap(True)
        sc_note.setStyleSheet(
            "color: rgba(160,160,160,180); font-size: 8pt; font-style: italic;"
        )
        parent.addWidget(sc_note)

        # Scene change threshold slider
        parent.addSpacing(4)
        thresh_row = QHBoxLayout()
        lbl_thresh = QLabel("场景变化阈值：")
        lbl_thresh.setFixedWidth(165)
        thresh_row.addWidget(lbl_thresh)

        self._scene_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._scene_thresh_slider.setRange(0, 100)
        self._scene_thresh_slider.setValue(_threshold_to_slider(self._cfg.scene_change_threshold))
        thresh_row.addWidget(self._scene_thresh_slider)

        _init_pct = _slider_to_threshold(self._scene_thresh_slider.value())
        self._scene_thresh_lbl = QLabel(f"{_init_pct:.1f}%")
        self._scene_thresh_lbl.setFixedWidth(48)
        self._scene_thresh_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        thresh_row.addWidget(self._scene_thresh_lbl)

        self._scene_thresh_slider.valueChanged.connect(
            lambda v: self._scene_thresh_lbl.setText(f"{_slider_to_threshold(v):.1f}%")
        )
        parent.addLayout(thresh_row)

        thresh_note = QLabel("数值越小越敏感。约 1% 可捕捉表情变化，3% 用于人物变化，10% 用于场景切换。")
        thresh_note.setWordWrap(True)
        thresh_note.setStyleSheet(
            "color: rgba(160,160,160,180); font-size: 8pt; font-style: italic;"
        )
        parent.addWidget(thresh_note)

        cooldown_row = QHBoxLayout()
        cooldown_lbl = QLabel("场景识别间隔：")
        cooldown_lbl.setFixedWidth(165)
        cooldown_row.addWidget(cooldown_lbl)
        self._scene_cooldown_spin = QDoubleSpinBox()
        self._scene_cooldown_spin.setRange(0.0, 10.0)
        self._scene_cooldown_spin.setSingleStep(0.5)
        self._scene_cooldown_spin.setDecimals(1)
        self._scene_cooldown_spin.setSuffix(" 秒")
        self._scene_cooldown_spin.setValue(self._cfg.scene_change_cooldown)
        self._scene_cooldown_spin.setMaximumWidth(90)
        cooldown_row.addWidget(self._scene_cooldown_spin)
        cooldown_row.addStretch()
        parent.addLayout(cooldown_row)

        parent.addSpacing(4)

        # Summarize-at-tokens
        tok_row = QHBoxLayout()
        lbl = QLabel("开始整理记忆的阈值：")
        lbl.setFixedWidth(165)
        tok_row.addWidget(lbl)
        self._summ_spin = QSpinBox()
        self._summ_spin.setRange(1_000, 500_000)
        self._summ_spin.setSingleStep(1_000)
        self._summ_spin.setValue(self._cfg.context_summarize_tokens)
        self._summ_spin.setMaximumWidth(110)
        tok_row.addWidget(self._summ_spin)
        tok_row.addWidget(QLabel("tokens"))
        tok_row.addStretch()
        parent.addLayout(tok_row)

        # Summary target size
        sum_tok_row = QHBoxLayout()
        lbl_st = QLabel("摘要目标长度：")
        lbl_st.setFixedWidth(165)
        sum_tok_row.addWidget(lbl_st)
        self._summary_tokens_spin = QSpinBox()
        self._summary_tokens_spin.setRange(500, 20_000)
        self._summary_tokens_spin.setSingleStep(500)
        self._summary_tokens_spin.setValue(self._cfg.summary_max_tokens)
        self._summary_tokens_spin.setMaximumWidth(110)
        sum_tok_row.addWidget(self._summary_tokens_spin)
        sum_tok_row.addWidget(QLabel("tokens"))
        sum_tok_row.addStretch()
        parent.addLayout(sum_tok_row)

        # Min recent lines
        lines_row = QHBoxLayout()
        lbl2 = QLabel("保留最近完整对话：")
        lbl2.setFixedWidth(165)
        lines_row.addWidget(lbl2)
        self._lines_spin = QSpinBox()
        self._lines_spin.setRange(5, 100)
        self._lines_spin.setValue(self._cfg.context_min_recent_lines)
        lines_row.addWidget(self._lines_spin)
        lines_row.addWidget(QLabel("条"))
        lines_row.addStretch()
        parent.addLayout(lines_row)
        summary_note = QLabel(
            "达到 token 阈值且有较早对话可整理时，本次翻译结束后立即在后台整理记忆，"
            "不等待下一条翻译，也不占用字幕位置。最近的指定条数保留完整内容。"
            "整理失败会保留原记忆，不会在空闲时连续重试。"
        )
        summary_note.setWordWrap(True)
        summary_note.setStyleSheet("color: #aaaacc; font-size: 11px;")
        parent.addWidget(summary_note)

        # Line batch window
        batch_row = QHBoxLayout()
        batch_lbl = QLabel("台词合并等待时间：")
        batch_lbl.setFixedWidth(165)
        batch_row.addWidget(batch_lbl)
        self._batch_win_spin = QSpinBox()
        self._batch_win_spin.setRange(0, 2000)
        self._batch_win_spin.setSingleStep(50)
        self._batch_win_spin.setSuffix(" 毫秒")
        self._batch_win_spin.setValue(self._cfg.line_batch_window_ms)
        self._batch_win_spin.setMaximumWidth(100)
        batch_row.addWidget(self._batch_win_spin)
        batch_row.addStretch()
        parent.addLayout(batch_row)

        # Text debounce
        debounce_row = QHBoxLayout()
        debounce_lbl = QLabel("逐字显示等待时间：")
        debounce_lbl.setFixedWidth(165)
        debounce_row.addWidget(debounce_lbl)
        self._debounce_spin = QSpinBox()
        self._debounce_spin.setRange(10, 2000)
        self._debounce_spin.setSingleStep(10)
        self._debounce_spin.setSuffix(" 毫秒")
        self._debounce_spin.setValue(self._cfg.debounce_ms)
        self._debounce_spin.setMaximumWidth(100)
        debounce_row.addWidget(self._debounce_spin)
        debounce_row.addStretch()
        parent.addLayout(debounce_row)


    def _build_models_and_apis(self, parent: QVBoxLayout) -> None:
        # --- API 密钥（只需填写使用的服务商） subsection ---
        parent.addWidget(_subsection_label("API 密钥（只需填写使用的服务商）"))

        parent.addLayout(self._input_row(
            "DeepSeek 密钥：", self._cfg.deepseek_api_key,
            password=True, dest="_deepseek_key_edit",
        ))
        parent.addLayout(self._input_row(
            "Anthropic 密钥：", self._cfg.anthropic_api_key,
            password=True, dest="_anthropic_key_edit",
        ))
        parent.addLayout(self._input_row(
            "OpenAI 密钥：", self._cfg.openai_api_key,
            password=True, dest="_openai_key_edit",
        ))
        parent.addLayout(self._input_row(
            "Google 密钥：", self._cfg.google_api_key,
            password=True, dest="_google_key_edit",
        ))
        parent.addLayout(self._input_row(
            "Ollama 地址：", self._cfg.ollama_base_url,
            password=False, dest="_ollama_url_edit",
        ))

        parent.addSpacing(6)

        # --- Roles subsection ---
        parent.addWidget(_subsection_label("用途（服务商 + 模型）"))

        self._role_combos:        dict[str, QComboBox]      = {}
        self._role_model_combos:  dict[str, QComboBox]      = {}
        self._role_model_edits:   dict[str, QLineEdit]      = {}
        self._role_model_stacks:  dict[str, QStackedWidget] = {}
        self._role_think_checks:  dict[str, QCheckBox]      = {}
        self._role_google_thinking_combos: dict[str, QComboBox] = {}
        self._role_max_tokens_spins: dict[str, QSpinBox]    = {}

        for role_key, label, provider_attr, model_attr, thinking_attr in [
            ("translator", "日译中：", "translator_provider", "translator_model", "translator_ollama_thinking"),
            ("descriptor", "截图 / OCR：", "descriptor_provider", "descriptor_model", "descriptor_ollama_thinking"),
            ("summarizer", "剧情摘要：", "summarizer_provider", "summarizer_model", "summarizer_ollama_thinking"),
        ]:
            provider_val = getattr(self._cfg, provider_attr)
            model_val    = getattr(self._cfg, model_attr)
            options      = ROLE_MODEL_OPTIONS.get(role_key, {}).get(provider_val, [])

            row = QHBoxLayout()

            lbl = QLabel(label)
            lbl.setFixedWidth(90)
            row.addWidget(lbl)

            # Provider combo
            prov_combo = QComboBox()
            prov_combo.addItems(_PROVIDERS)
            prov_combo.setStyleSheet("QComboBox { min-width: 88px; }")
            prov_combo.setFixedWidth(110)
            display = next(
                (k for k, v in _PROVIDER_KEY_MAP.items() if v == provider_val),
                "DeepSeek",
            )
            prov_combo.setCurrentText(display)
            row.addWidget(prov_combo)

            # Model combo (page 0 of stack)
            model_combo = QComboBox()
            model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            model_combo.setMinimumContentsLength(20)
            model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            model_combo.addItems(options)
            model_combo.addItem("其他模型…")

            # Other text field (page 1 of stack)
            model_edit = QLineEdit()
            model_edit.setPlaceholderText("输入模型名称…")

            # Choose initial stack page
            if model_val in options:
                model_combo.setCurrentText(model_val)
                initial_idx = 0
            else:
                model_combo.setCurrentText("其他模型…")
                model_edit.setText(model_val)
                initial_idx = 1

            stack = QStackedWidget()
            stack.addWidget(model_combo)
            stack.addWidget(model_edit)
            stack.setCurrentIndex(initial_idx)
            stack.setFixedHeight(prov_combo.sizeHint().height())
            stack.setMinimumWidth(180)
            row.addWidget(stack, 1)

            think_check = QCheckBox("Ollama 思考")
            think_check.setChecked(getattr(self._cfg, thinking_attr))
            think_check.setEnabled(provider_val == "ollama")
            think_check.setToolTip("仅对 Ollama 生效；开启后可能增加响应时间。DeepSeek 默认关闭思考以加快翻译。")

            parent.addLayout(row)

            # Max output tokens sub-row (indented under the provider/model row)
            tok_row = QHBoxLayout()
            tok_row.addSpacing(12)
            lbl_tok = QLabel("最大输出长度：")
            lbl_tok.setFixedWidth(100)
            tok_row.addWidget(lbl_tok)
            tok_spin = QSpinBox()
            tok_spin.setRange(128, 16384)
            tok_spin.setSingleStep(128)
            tok_spin.setValue(getattr(self._cfg, f"{role_key}_max_tokens"))
            tok_spin.setMaximumWidth(80)
            tok_spin.setToolTip("单次输出的 token 上限。增加上限不会自动生成更多内容。")
            tok_row.addWidget(tok_spin)
            google_label = QLabel("Google 思考：")
            tok_row.addWidget(google_label)
            google_combo = QComboBox()
            for display_label, value in (("低", "low"), ("中", "medium"), ("高", "high")):
                google_combo.addItem(display_label, value)
            default_level = "medium" if role_key == "translator" else "low"
            current_level = getattr(self._cfg, f"{role_key}_google_thinking_level", default_level)
            level_index = google_combo.findData(current_level)
            google_combo.setCurrentIndex(level_index if level_index >= 0 else google_combo.findData(default_level))
            google_combo.setEnabled(provider_val == "google")
            google_combo.setStyleSheet("QComboBox { min-width: 42px; }")
            google_combo.setFixedWidth(74)
            google_combo.setToolTip("仅对 Google Gemini 生效。低档更快，高档可增加推理时间与费用。")
            tok_row.addWidget(google_combo)
            tok_row.addWidget(think_check)
            tok_row.addStretch()
            parent.addLayout(tok_row)

            self._role_combos[role_key]          = prov_combo
            self._role_model_combos[role_key]    = model_combo
            self._role_model_edits[role_key]     = model_edit
            self._role_model_stacks[role_key]    = stack
            self._role_think_checks[role_key]    = think_check
            self._role_google_thinking_combos[role_key] = google_combo
            self._role_max_tokens_spins[role_key] = tok_spin

            # Switch to text field when "Other…" is selected
            def _on_model_combo_changed(
                text: str, stk: QStackedWidget = stack, edit: QLineEdit = model_edit,
            ) -> None:
                if text == "其他模型…":
                    stk.setCurrentIndex(1)
                    edit.clear()
                    edit.setFocus()

            model_combo.currentTextChanged.connect(_on_model_combo_changed)

            # Rebuild model options and auto-fill when provider changes
            def _on_provider_changed(
                text: str,
                stk:  QStackedWidget = stack,
                mc:   QComboBox      = model_combo,
                edit: QLineEdit      = model_edit,
                rk:   str            = role_key,
                chk:  QCheckBox      = think_check,
                gc:   QComboBox      = google_combo,
            ) -> None:
                new_internal = _PROVIDER_KEY_MAP.get(text, "deepseek")
                chk.setEnabled(new_internal == "ollama")
                gc.setEnabled(new_internal == "google")
                new_options  = ROLE_MODEL_OPTIONS.get(rk, {}).get(new_internal, [])
                new_default  = ROLE_PROVIDER_DEFAULTS.get(rk, {}).get(new_internal, "")
                current = edit.text().strip() if stk.currentIndex() == 1 else mc.currentText()

                mc.blockSignals(True)
                mc.clear()
                mc.addItems(new_options)
                mc.addItem("其他模型…")
                mc.blockSignals(False)

                mc.setCurrentText(current if current in new_options else new_default)
                stk.setCurrentIndex(0)

            prov_combo.currentTextChanged.connect(_on_provider_changed)


    def _input_row(
        self, label: str, value: str, *, password: bool = False, dest: str
    ) -> QHBoxLayout:
        """Build a label + QLineEdit row and store the edit as self.<dest>."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(165)
        row.addWidget(lbl)

        edit = QLineEdit(value)
        if password:
            edit.setEchoMode(QLineEdit.EchoMode.Password)
        row.addWidget(edit)

        setattr(self, dest, edit)
        return row

    def _build_prompts(self, parent: QVBoxLayout) -> None:
        """Build the Prompts tab with one QTextEdit per LLM role."""
        _roles = [
            ("日译中",  "translator_system_prompt", _SYSTEM_PROMPT,    "_translator_prompt_edit"),
            ("场景描述",  "descriptor_system_prompt", _DESCRIBE_SYSTEM,  "_descriptor_prompt_edit"),
            ("剧情摘要",  "summarizer_system_prompt", _SUMMARIZE_SYSTEM, "_summarizer_prompt_edit"),
        ]
        for display_name, cfg_attr, default_text, widget_attr in _roles:
            hdr = QHBoxLayout()
            lbl = QLabel(f"<b>{display_name}</b>")
            hdr.addWidget(lbl)
            hdr.addStretch()
            reset_btn = QPushButton("恢复默认")
            reset_btn.setFixedHeight(24)
            hdr.addWidget(reset_btn)
            parent.addLayout(hdr)

            edit = QTextEdit()
            edit.setPlainText(getattr(self._cfg, cfg_attr) or default_text)
            edit.setMinimumHeight(80)
            edit.setAcceptRichText(False)
            edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            parent.addWidget(edit, 1)   # stretch=1 → all three share vertical space equally
            setattr(self, widget_attr, edit)

            reset_btn.clicked.connect(
                lambda checked, e=edit, d=default_text: e.setPlainText(d)
            )

    def _build_buttons(self, parent: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.addStretch()

        cancel = QPushButton("取消")
        cancel.setStyleSheet(_BTN_ACTION)
        cancel.clicked.connect(self._on_cancel)
        row.addWidget(cancel)

        save = QPushButton("保存")
        save.setStyleSheet(_BTN_SAVE)
        save.clicked.connect(self._on_save)
        row.addWidget(save)

        parent.addLayout(row)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _refresh_swatch(btn: QPushButton, color: list[int]) -> None:
        r, g, b, a = color
        btn.setStyleSheet(_BTN_SWATCH.format(r=r, g=g, b=b, a=a))

    def _pick_color(self, btn: QPushButton, setter) -> None:
        """Open a QColorDialog and call setter with [r,g,b,a] if accepted."""
        if setter == self._set_bg_color:
            current = list(self._cfg.overlay.bg_color)
        elif setter == self._set_jp_color:
            current = list(self._cfg.overlay.jp_color)
        else:
            current = list(self._cfg.overlay.en_color)

        r, g, b, a = current
        color = QColorDialog.getColor(
            QColor(r, g, b, a),
            self,
            "选择颜色",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            new = [color.red(), color.green(), color.blue(), color.alpha()]
            setter(new)
            self._refresh_swatch(btn, new)

    # Font setter — mutate cfg and push live to overlay
    def _on_furigana_changed(self, enabled: bool) -> None:
        self._cfg.overlay.jp_furigana_enabled = enabled
        self._overlay.apply_appearance(self._cfg.overlay)

    def _on_font_changed(self, attr: str, value) -> None:
        setattr(self._cfg.overlay, attr, value)
        self._overlay.apply_appearance(self._cfg.overlay)

    # Color setters — mutate cfg and push live to overlay
    def _set_bg_color(self, c: list[int]) -> None:
        self._cfg.overlay.bg_color = c
        self._overlay.apply_appearance(self._cfg.overlay)

    def _set_jp_color(self, c: list[int]) -> None:
        self._cfg.overlay.jp_color = c
        self._overlay.apply_appearance(self._cfg.overlay)

    def _set_en_color(self, c: list[int]) -> None:
        self._cfg.overlay.en_color = c
        self._overlay.apply_appearance(self._cfg.overlay)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_opacity_changed(self, value: int) -> None:
        self._opacity_lbl.setText(f"{value}%")
        self._cfg.overlay.bg_color[3] = int(value / 100 * 255)
        self._refresh_swatch(self._bg_btn, self._cfg.overlay.bg_color)
        self._overlay.apply_appearance(self._cfg.overlay)

    def _on_save(self) -> None:
        self._cfg.game_title = normalize_game_text(self._game_title_edit.text(), GAME_TITLE_MAX_LENGTH)
        self._cfg.game_notes = normalize_game_text(self._game_notes_edit.toPlainText(), GAME_NOTES_MAX_LENGTH)

        # Controller edits stay local to the dialog until Save is clicked.
        self._cfg.controller_enabled = self._controller_enabled_check.isChecked()
        selected_button = self._controller_button_combo.currentData()
        self._cfg.controller_button = selected_button if selected_button in ("GR", "GL") else "GR"

        # API keys
        self._cfg.deepseek_api_key  = self._deepseek_key_edit.text().strip()
        self._cfg.anthropic_api_key = self._anthropic_key_edit.text().strip()
        self._cfg.openai_api_key    = self._openai_key_edit.text().strip()
        self._cfg.google_api_key    = self._google_key_edit.text().strip()
        self._cfg.ollama_base_url   = self._ollama_url_edit.text().strip()

        # Per-role provider + model + thinking + max tokens
        for role_key, provider_attr, model_attr, thinking_attr, tokens_attr in [
            ("translator", "translator_provider", "translator_model", "translator_ollama_thinking", "translator_max_tokens"),
            ("descriptor", "descriptor_provider", "descriptor_model", "descriptor_ollama_thinking", "descriptor_max_tokens"),
            ("summarizer", "summarizer_provider", "summarizer_model", "summarizer_ollama_thinking", "summarizer_max_tokens"),
        ]:
            combo_text = self._role_combos[role_key].currentText()
            internal   = _PROVIDER_KEY_MAP.get(combo_text, "deepseek")
            stack      = self._role_model_stacks[role_key]
            if stack.currentIndex() == 1:
                model_val = self._role_model_edits[role_key].text().strip()
            else:
                model_val = self._role_model_combos[role_key].currentText()
                if not model_val or model_val == "其他模型…":
                    model_val = ""
            if not model_val:
                model_val = ROLE_PROVIDER_DEFAULTS.get(role_key, {}).get(internal, "")
            setattr(self._cfg, provider_attr, internal)
            setattr(self._cfg, model_attr,    model_val)
            setattr(self._cfg, thinking_attr, self._role_think_checks[role_key].isChecked())
            setattr(self._cfg, f"{role_key}_google_thinking_level",
                    self._role_google_thinking_combos[role_key].currentData())
            setattr(self._cfg, tokens_attr,   self._role_max_tokens_spins[role_key].value())

        # Context settings
        self._cfg.clipboard_enabled        = self._clipboard_check.isChecked()
        self._cfg.screenshot_enabled       = self._screenshot_check.isChecked()
        self._cfg.context_summarize_tokens = self._summ_spin.value()
        self._cfg.summary_max_tokens       = self._summary_tokens_spin.value()
        self._cfg.context_min_recent_lines = self._lines_spin.value()
        self._cfg.line_batch_window_ms     = self._batch_win_spin.value()
        debounce = self._debounce_spin.value()
        self._cfg.debounce_ms = debounce
        if self._hooker is not None:
            self._hooker.set_debounce(debounce)

        # Scene change threshold (apply immediately to screenshot service)
        _pct = _slider_to_threshold(self._scene_thresh_slider.value())
        self._cfg.scene_change_threshold = _pct
        if self._screenshot_service is not None:
            self._screenshot_service.threshold = _pct

        cooldown = self._scene_cooldown_spin.value()
        self._cfg.scene_change_cooldown = cooldown
        if self._screenshot_service is not None:
            self._screenshot_service.cooldown = cooldown

        # System prompts (empty string = use built-in default)
        for attr, edit, default in (
            ("translator_system_prompt", self._translator_prompt_edit, _SYSTEM_PROMPT),
            ("descriptor_system_prompt", self._descriptor_prompt_edit, _DESCRIBE_SYSTEM),
            ("summarizer_system_prompt", self._summarizer_prompt_edit, _SUMMARIZE_SYSTEM),
        ):
            value = edit.toPlainText().strip()
            setattr(self._cfg, attr, "" if value == default.strip() else value)

        # Appearance is already live in cfg.overlay — write everything to disk
        self._save_fn(self._cfg)

        # Update snapshot so a subsequent Cancel doesn't revert past this save
        self._orig_overlay = copy.deepcopy(self._cfg.overlay)
        self.accept()

    def _on_cancel(self) -> None:
        self.reject()

    def reject(self) -> None:
        # Cancel, Escape and the window close button all discard live previews.
        self._cfg.overlay = copy.deepcopy(self._orig_overlay)
        self._overlay.apply_appearance(self._cfg.overlay)
        super().reject()

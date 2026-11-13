"""Provide Chinese and Japanese fallback fonts on Windows and in UI previews."""
import os
from pathlib import Path

from PyQt6.QtGui import QFont, QFontDatabase


def initialize_fonts(app, cfg):
    directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in ("msyh.ttc", "msyhbd.ttc", "msgothic.ttc", "seguisym.ttf"):
        path = directory / name
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    families = set(QFontDatabase.families())
    chinese = next((f for f in ("Microsoft YaHei UI", "Microsoft YaHei", "SimSun") if f in families), app.font().family())
    japanese = next((f for f in ("Yu Gothic", "MS Gothic", "Microsoft YaHei") if f in families), chinese)
    app.setFont(QFont(chinese, 10))
    if cfg.overlay.en_font_family not in families:
        cfg.overlay.en_font_family = chinese
    if cfg.overlay.jp_font_family not in families:
        cfg.overlay.jp_font_family = japanese

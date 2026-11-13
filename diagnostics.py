"""Offline native-runtime check for source and PyInstaller distributions."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def check_ocr(arguments: list[str]) -> int:
    """Usage: vntl --check-ocr IMAGE OUTPUT.json. Uses no game state or API."""
    if len(arguments) != 2:
        return 2
    image_path, output_path = map(Path, arguments)
    result = {"ok": False}
    try:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        from PIL import Image
        from local_ocr import MeikiReader
        app = QApplication.instance() or QApplication([])
        image = Image.open(image_path).convert("RGB")
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as executor:
            found = executor.submit(MeikiReader().recognize, image).result(timeout=60)
        result.update(ok=True, characters=len(found.text), confidence=found.confidence,
                      elapsed_ms=(time.perf_counter() - started) * 1000,
                      qt_platform=app.platformName(), provider="CPUExecutionProvider")
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1

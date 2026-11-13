"""Lazy CPU-only Japanese OCR with pinned, verified local Meiki models.

The caller owns the worker thread. Construction performs no imports of inference
libraries, downloads, or inference. Initialization failures propagate so a failed
observation cannot be mistaken for a genuinely empty game screen.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import math
import ssl
import statistics
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)
MEIKI_VERSION = "0.3.5"


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    boxes: tuple[tuple[int, int, int, int], ...] = ()
    uncertain: bool = False


@dataclass(frozen=True)
class _Model:
    repo: str
    revision: str
    filename: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


# Immutable upstream snapshots used in the local Japanese subtitle benchmark.
# Copying these three files directly into model_dir also prepares a fully offline
# installation; neither a Hugging Face account nor a token is needed.
MODELS = (
    _Model("rtr46/meiki.text.detect.v0", "a9cffa4f60cbf72ddb87edf19c6f98a01cd042e6",
           "meiki.text.detect.v0.1.960x544.onnx",
           "40b6a016667745cae7d3055929ae3b8b1e7716aac795f5904cd3c2c7c3b8404b", 14503825),
    _Model("rtr46/meiki.txt.recognition.v0", "a28cf5874dc2438ebb1c86336be26bcec51e3375",
           "meiki.text.rec.v0.960x32.onnx",
           "3e96bc772fbee9717e536a6353032bb944c3382dd2f6960ef4890decda43b000", 18593254),
    _Model("rtr46/meiki.txt.recognition.v0", "a28cf5874dc2438ebb1c86336be26bcec51e3375",
           "meiki.text.rec.v0.vertical.32x480.onnx",
           "2c2a83a23bc3b7e6c63962175f507ecc6c5e85cc174f17bdec37d9bbd0bf895a", 12872961),
)


def _verified(path: Path, model: _Model) -> bool:
    if not path.is_file() or path.stat().st_size != model.size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == model.sha256


def _download_model(model: _Model, destination: Path) -> None:
    """Fetch one public, immutable model; verify before atomic installation."""
    temporary: Path | None = None
    try:
        logger.info("Downloading Japanese OCR model %s", model.filename)
        with urllib.request.urlopen(model.url, timeout=30, context=ssl.create_default_context()) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as output:
                temporary = Path(output.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > model.size:
                        raise RuntimeError(f"Unexpected OCR model size: {model.filename}")
                    output.write(chunk)
        if temporary is None or not _verified(temporary, model):
            raise RuntimeError(f"OCR model integrity check failed: {model.filename}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_models(directory: Path) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in MODELS:
        path = directory / model.filename
        if not _verified(path, model):
            _download_model(model, path)
        paths.append(path)
    return tuple(paths)


def _create_engine(paths: tuple[Path, ...], threads: int):
    if importlib.metadata.version("meikiocr") != MEIKI_VERSION:
        raise RuntimeError(f"Japanese OCR requires meikiocr=={MEIKI_VERSION}")
    import onnxruntime as ort
    from meikiocr import MeikiOCR

    class LocalMeikiOCR(MeikiOCR):
        # Upstream 0.3.5 does not expose session options/model paths. Override
        # only construction; all detection/recognition logic remains upstream.
        def __init__(self):
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            sessions = tuple(ort.InferenceSession(str(path), sess_options=options,
                                                 providers=["CPUExecutionProvider"]) for path in paths)
            self.det_session, self.rec_session, self.vrec_session = sessions
            self.active_provider = self.det_session.get_providers()[0]
            self.max_batch_size = 8

        def _fix_swapped_pairs(self, text, chars):
            # Upstream has a small lexical replacement table. Keep OCR evidence
            # unchanged here instead of correcting words from a built-in lexicon.
            return text, chars

    return LocalMeikiOCR()


def _image_as_bgr(image: Image.Image):
    import numpy as np
    return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def _bbox(value, width: int, height: int) -> tuple[int, int, int, int] | None:
    try:
        if len(value) != 4 or not all(math.isfinite(float(v)) for v in value):
            return None
        x1, y1, x2, y2 = (int(round(float(v))) for v in value)
    except (TypeError, ValueError, OverflowError):
        return None
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _confidence(value) -> float:
    try:
        score = float(value)
        return max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _isolated_trailing_mark(chars: list[dict], boxes: list, vertical: bool) -> bool:
    """Remove only a tiny, low-confidence, separated trailing component.

    Never classify a glyph by its text (口/ロ/一 are legitimate Japanese). An
    ordinary full-sized character or normally spaced punctuation stays intact.
    Ambiguous cursor-like shapes are retained and reported through confidence.
    """
    if vertical or len(chars) < 5 or any(box is None for box in boxes):
        return False
    if _confidence(chars[-1].get("conf")) >= 0.35:
        return False
    typical_width = statistics.median(box[2] - box[0] for box in boxes[:-1])
    typical_height = statistics.median(box[3] - box[1] for box in boxes[:-1])
    x1, y1, x2, y2 = boxes[-1]
    return (x2 - x1 < typical_width * 0.65 and y2 - y1 < typical_height * 0.65
            and x1 - boxes[-2][2] > typical_width * 0.75)


def _advance_shape(image: Image.Image, box, confidence: float) -> bool:
    """Recognize only a down-triangle or a folded-page UI outline from pixels.

    Meiki gives these small symbols a full line-height character box, so box
    height alone cannot distinguish them. Inspect the bright connected ink
    instead. A real 口/ロ is a closed ring with an empty center; 一 is a thin bar.
    Neither has a solid downward taper or an open paper outline with an internal
    lower-right fold. No recognized character value participates in this test.
    """
    if box is None or confidence >= 0.85:
        return False
    import cv2
    import numpy as np

    pixels = np.asarray(image.crop(box).convert("RGB"))
    # Bright neutral ink excludes the common colored backgrounds behind text.
    mask = ((pixels.min(axis=2) > 170) & (np.ptp(pixels, axis=2) < 60)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return False
    component = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    x, y, width, height, area = (int(v) for v in stats[component])
    line_height = box[3] - box[1]
    if area < 12 or area < float(mask.sum()) * 0.8 or not line_height * 0.45 <= width <= line_height * 1.2:
        return False
    ink = labels[y:y + height, x:x + width] == component

    # Filled down-arrow: wide top, narrow bottom, mostly monotonically tapering
    # rows, located toward the bottom of the character cell.
    if (line_height * 0.2 <= height <= line_height * 0.65
            and 1.35 <= width / height <= 3.0
            and y >= line_height * 0.3
            and 0.4 <= area / (width * height) <= 0.7):
        row_width = ink.sum(axis=1)
        band = max(2, height // 5)
        if (float(row_width[:band].mean()) >= width * 0.65
                and float(row_width[-band:].mean()) <= width * 0.4
                and float((np.diff(row_width.astype(int)) <= 2).mean()) >= 0.8):
            return True

    # Folded-page outline. Require the visible fold inside the lower-right
    # quadrant, mostly empty elsewhere, and an open outline. Closed Japanese
    # box glyphs have a large external contour even when their ink is thin.
    if confidence >= 0.65 or not (line_height * 0.65 <= height <= line_height * 1.05 and 0.55 <= width / height <= 1.05):
        return False
    top = max(2, round(height * 0.12))
    side = max(2, round(width * 0.12))
    if float(ink[:top].any(axis=0).mean()) < 0.85 or float(ink[:, :side].any(axis=1).mean()) < 0.85:
        return False
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or max(cv2.contourArea(c) for c in contours) / (width * height) >= 0.5:
        return False
    interior = ink[round(height * 0.15):round(height * 0.9), round(width * 0.15):round(width * 0.9)].copy()
    if int(interior.sum()) < 8:
        return False
    fold = ink[round(height * 0.55):round(height * 0.9), round(width * 0.5):round(width * 0.9)]
    return bool(int(fold.sum()) >= int(interior.sum()) * 0.75 and fold.any(axis=0).sum() >= 3 and fold.any(axis=1).sum() >= 3)


def _to_result(lines, width: int, height: int, image: Image.Image | None = None) -> OCRResult:
    texts, bounds, confidences = [], [], []
    uncertain = False
    for line in lines:
        text = line.get("text", "")
        if not text:
            continue
        chars = list(line.get("chars") or [])
        boxes = [_bbox(char.get("bbox"), width, height) for char in chars]
        consistent = "".join(char.get("char", "") for char in chars) == text
        vertical = bool(line.get("is_vertical"))
        shape = (consistent and not vertical and len(chars) >= 3 and image is not None
                 and _advance_shape(image, boxes[-1], _confidence(chars[-1].get("conf"))))
        isolated = consistent and _isolated_trailing_mark(chars, boxes, vertical)
        if shape or isolated:
            chars, boxes = chars[:-1], boxes[:-1]
            text = "".join(char.get("char", "") for char in chars)
            # A verified UI shape is not part of the text confidence. The less
            # specific spacing/size heuristic remains an uncertain observation.
            uncertain = uncertain or (isolated and not shape)
        texts.append(text)
        confidences.extend(_confidence(char.get("conf")) for char in chars)
        valid = [box for box in boxes if box is not None]
        if valid:
            bounds.append((min(b[0] for b in valid), min(b[1] for b in valid),
                           max(b[2] for b in valid), max(b[3] for b in valid)))
        if not consistent or not chars or len(valid) != len(chars):
            uncertain = True
    confidence = statistics.mean(confidences) if confidences else 0.0
    if confidences and (confidence < 0.8 or min(confidences) < 0.5):
        uncertain = True
    return OCRResult("\n".join(texts), confidence, tuple(bounds), uncertain)


class MeikiReader:
    def __init__(self, model_dir: Path | None = None, threads: int = 4):
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.model_dir = Path(model_dir) if model_dir is not None else Path.home() / ".config/vntl/models/meiki"
        self.threads = threads
        self._engine = None

    def recognize(self, image: Image.Image) -> OCRResult:
        if image.width <= 0 or image.height <= 0:
            raise ValueError("OCR image must have a positive size")
        if self._engine is None:
            paths = _prepare_models(self.model_dir)
            self._engine = _create_engine(paths, self.threads)
        # Once constructed the upstream inference path only uses local sessions.
        lines = self._engine.run_ocr(_image_as_bgr(image))
        return _to_result(lines, image.width, image.height, image)

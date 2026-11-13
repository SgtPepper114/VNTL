import hashlib
import io
import tempfile
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw, ImageFont

import local_ocr
from local_ocr import MeikiReader, OCRResult


def line(text, confidence=0.95, start=0, y=0):
    return {"text": text, "is_vertical": False, "chars": [
        {"char": char, "conf": confidence, "bbox": [start + i * 20, y, start + (i + 1) * 20, y + 30]}
        for i, char in enumerate(text)
    ]}


class LocalOCRTests(unittest.TestCase):
    def test_reader_is_lazy_and_reuses_engine_without_preparing_models_again(self):
        engine = Mock()
        engine.run_ocr.return_value = [line("ミカ：口ロ一")]
        paths = tuple(Path(str(i)) for i in range(3))
        with patch.object(local_ocr, "_prepare_models", return_value=paths) as prepare, \
                patch.object(local_ocr, "_create_engine", return_value=engine) as create, \
                patch.object(local_ocr, "_image_as_bgr", return_value="pixels"):
            reader = MeikiReader(Path("local-models"), threads=4)
            prepare.assert_not_called()
            create.assert_not_called()
            first = reader.recognize(Image.new("RGB", (300, 100)))
            second = reader.recognize(Image.new("RGB", (300, 100)))
        prepare.assert_called_once_with(Path("local-models"))
        create.assert_called_once_with(paths, 4)
        self.assertEqual(first, second)
        self.assertEqual(first.text, "ミカ：口ロ一")
        self.assertFalse(first.uncertain)
        self.assertEqual(engine.run_ocr.call_count, 2)

    def test_default_cache_is_per_user_and_invalid_threads_rejected(self):
        self.assertEqual(MeikiReader().model_dir, Path.home() / ".config/vntl/models/meiki")
        for threads in (0, -2, 1.5, True):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                MeikiReader(threads=threads)

    def test_failure_is_not_misreported_as_empty_screen_and_can_retry(self):
        reader = MeikiReader()
        with patch.object(local_ocr, "_prepare_models", side_effect=OSError("missing model")):
            with self.assertRaises(OSError):
                reader.recognize(Image.new("RGB", (10, 10)))
        self.assertIsNone(reader._engine)

    def test_raw_text_shape_whitespace_and_lines_are_preserved(self):
        result = local_ocr._to_result([line("ミカ：１２段"), line("<口> ロ 一　", y=40)], 400, 100)
        self.assertEqual(result.text, "ミカ：１２段\n<口> ロ 一　")
        self.assertEqual(result.boxes, ((0, 0, 120, 30), (0, 40, 160, 70)))
        self.assertAlmostEqual(result.confidence, 0.95)
        with self.assertRaises(FrozenInstanceError):
            result.text = "変更"

    def test_empty_detection_is_normal_empty_observation(self):
        self.assertEqual(local_ocr._to_result([], 100, 100), OCRResult("", 0.0))

    def test_low_confidence_character_is_retained_and_flagged(self):
        raw = line("口ロ一")
        raw["chars"][1]["conf"] = 0.2
        result = local_ocr._to_result([raw], 100, 100)
        self.assertEqual(result.text, "口ロ一")
        self.assertTrue(result.uncertain)
        self.assertAlmostEqual(result.confidence, (0.95 + 0.2 + 0.95) / 3)

    def test_only_geometrically_isolated_tiny_low_confidence_tail_is_filtered(self):
        raw = line("何かいる口")
        raw["chars"][-1].update(conf=0.2, bbox=[120, 15, 130, 25])
        result = local_ocr._to_result([raw], 200, 100)
        self.assertEqual(result.text, "何かいる")
        self.assertEqual(result.boxes, ((0, 0, 80, 30),))
        self.assertTrue(result.uncertain)

    def test_normal_spacing_or_high_confidence_tail_is_not_removed(self):
        for confidence, bbox in ((0.2, [80, 15, 90, 25]), (0.9, [120, 15, 130, 25]), (0.2, [120, 0, 140, 30])):
            raw = line("何かいる口")
            raw["chars"][-1].update(conf=confidence, bbox=bbox)
            with self.subTest(confidence=confidence, bbox=bbox):
                self.assertEqual(local_ocr._to_result([raw], 200, 100).text, "何かいる口")

    def test_vertical_text_never_uses_horizontal_cursor_filter(self):
        raw = line("何かいる口")
        raw["is_vertical"] = True
        raw["chars"][-1].update(conf=0.2, bbox=[120, 15, 130, 25])
        self.assertEqual(local_ocr._to_result([raw], 200, 100).text, "何かいる口")

    def test_down_triangle_is_removed_by_pixels_even_with_full_height_box(self):
        raw = line("何かいる一")
        for index, char in enumerate(raw["chars"]):
            char["bbox"] = [index * 40, 0, (index + 1) * 40, 44]
        raw["chars"][-1]["conf"] = 0.6
        image = Image.new("RGB", (200, 44))
        ImageDraw.Draw(image).polygon([(164, 25), (196, 25), (180, 41)], fill="white")
        result = local_ocr._to_result([raw], 200, 44, image)
        self.assertEqual(result.text, "何かいる")
        self.assertFalse(result.uncertain)

    def test_folded_page_shape_is_removed_without_testing_recognized_glyph(self):
        raw = line("何かいるし")
        for index, char in enumerate(raw["chars"]):
            char["bbox"] = [index * 40, 0, (index + 1) * 40, 44]
        raw["chars"][-1]["conf"] = 0.3
        image = Image.new("RGB", (200, 44))
        draw = ImageDraw.Draw(image)
        draw.line([(165, 5), (193, 5), (193, 33)], fill="white", width=2)
        draw.line([(165, 5), (165, 40), (185, 40)], fill="white", width=2)
        draw.line([(193, 26), (184, 26), (184, 36)], fill="white", width=2)
        result = local_ocr._to_result([raw], 200, 44, image)
        self.assertEqual(result.text, "何かいる")

    def test_real_font_kanji_katakana_and_long_vowel_are_preserved_even_at_low_confidence(self):
        font_path = next((path for path in (
            Path("C:/Windows/Fonts/YuGothR.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")) if path.is_file()), None)
        if font_path is None:
            self.skipTest("A Japanese font is required for the actual glyph-shape regression")
        font = ImageFont.truetype(str(font_path), 38)
        for glyph in ("口", "ロ", "一", "ー", "し", "円", "回"):
            with self.subTest(glyph=glyph):
                raw = line("文字文字" + glyph)
                for index, char in enumerate(raw["chars"]):
                    char["bbox"] = [index * 44, 0, (index + 1) * 44, 48]
                raw["chars"][-1]["conf"] = 0.2
                image = Image.new("RGB", (220, 48))
                draw = ImageDraw.Draw(image)
                for index, char in enumerate(raw["text"]):
                    draw.text((index * 44, 0), char, font=font, fill="white", anchor="lt")
                result = local_ocr._to_result([raw], 220, 48, image)
                self.assertEqual(result.text, raw["text"])

    def test_closed_rectangular_kanji_outline_cannot_match_folded_page(self):
        image = Image.new("RGB", (40, 44))
        ImageDraw.Draw(image).rectangle((5, 5, 33, 40), outline="white", width=2)
        self.assertFalse(local_ocr._advance_shape(image, (0, 0, 40, 44), 0.2))

    def test_inconsistent_character_metadata_does_not_rewrite_native_text(self):
        raw = line("何かいる口")
        raw["text"] = "原文はそのまま"
        raw["chars"][-1].update(conf=0.2, bbox=[120, 15, 130, 25])
        result = local_ocr._to_result([raw], 200, 100)
        self.assertEqual(result.text, "原文はそのまま")
        self.assertTrue(result.uncertain)

    def test_malformed_confidence_and_boxes_are_marked_uncertain(self):
        raw = line("文字")
        raw["chars"][0].update(conf=float("nan"), bbox=[0, 0, float("inf"), 30])
        raw["chars"][1]["bbox"] = [-1, -2, 300, 400]
        result = local_ocr._to_result([raw], 100, 100)
        self.assertEqual(result.text, "文字")
        self.assertEqual(result.boxes, ((0, 0, 100, 100),))
        self.assertTrue(result.uncertain)

    def test_verified_local_models_do_not_touch_network(self):
        payload = b"fixed model bytes"
        model = local_ocr._Model("owner/repo", "revision", "model.onnx", hashlib.sha256(payload).hexdigest(), len(payload))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / model.filename
            target.write_bytes(payload)
            with patch.object(local_ocr, "MODELS", (model,)), patch.object(local_ocr, "_download_model") as download:
                self.assertEqual(local_ocr._prepare_models(Path(directory)), (target,))
            download.assert_not_called()

    def test_download_uses_pinned_public_revision_and_checks_integrity(self):
        payload = b"fixed model bytes"
        model = local_ocr._Model("owner/repo", "revision", "model.onnx", hashlib.sha256(payload).hexdigest(), len(payload))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / model.filename
            with patch.object(local_ocr.urllib.request, "urlopen", return_value=io.BytesIO(payload)) as request:
                local_ocr._download_model(model, target)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(request.call_args.args[0], "https://huggingface.co/owner/repo/resolve/revision/model.onnx")
            self.assertTrue(request.call_args.kwargs["context"].check_hostname)

    def test_failed_download_keeps_previous_file_and_removes_partial(self):
        model = local_ocr._Model("owner/repo", "revision", "model.onnx", "0" * 64, 3)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / model.filename
            target.write_bytes(b"previous")
            with patch.object(local_ocr.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(RuntimeError):
                    local_ocr._download_model(model, target)
            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_engine_constructor_uses_cpu_local_paths_without_global_monkeypatch(self):
        class Base:
            def __init__(self):
                raise AssertionError("Must not call upstream downloader")
            def run_ocr(self, image):
                return image

        options = Mock()
        factory = Mock(return_value=options)
        sessions = [Mock(), Mock(), Mock()]
        sessions[0].get_providers.return_value = ["CPUExecutionProvider"]
        session_factory = Mock(side_effect=sessions)
        fake_ort = types.SimpleNamespace(SessionOptions=factory, InferenceSession=session_factory,
                                         GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="optimized"))
        paths = tuple(Path(f"model-{i}.onnx") for i in range(3))
        with patch.object(local_ocr.importlib.metadata, "version", return_value="0.3.5"), \
                patch.dict("sys.modules", {"onnxruntime": fake_ort, "meikiocr": types.SimpleNamespace(MeikiOCR=Base)}):
            engine = local_ocr._create_engine(paths, 4)
        self.assertIs(fake_ort.SessionOptions, factory)
        self.assertEqual((options.intra_op_num_threads, options.inter_op_num_threads), (4, 1))
        self.assertEqual(session_factory.call_count, 3)
        for call, path in zip(session_factory.call_args_list, paths):
            self.assertEqual(call.args, (str(path),))
            self.assertEqual(call.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(engine.run_ocr("unchanged pixels"), "unchanged pixels")
        self.assertEqual(engine._fix_swapped_pairs("談冗", []), ("談冗", []))

    def test_unexpected_meiki_version_fails_before_constructing_engine(self):
        with patch.object(local_ocr.importlib.metadata, "version", return_value="99.0"):
            with self.assertRaisesRegex(RuntimeError, "0.3.5"):
                local_ocr._create_engine(tuple(), 4)


if __name__ == "__main__":
    unittest.main()

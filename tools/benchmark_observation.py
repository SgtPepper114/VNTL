"""Measure the production observation path on complete, frozen game frames.

This command calls the API roles saved in VNTL settings. It writes experimental
logs/results only under --output, and never loads or saves the player's story.
Example (PowerShell):
  uv run python tools/benchmark_observation.py --crop-top 60 --vision-baseline \
    --output validation/benchmark frame1.png frame2.png

The measured delay starts when a complete frame becomes available. Typewriter
trigger accuracy must separately be evaluated on a timestamped video sequence.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

import translator as translator_module
from config import load_config
from context_manager import ContextManager
from local_ocr import MeikiReader
from observation_service import ObservationService
from screenshot_service import ScreenshotService
from translator import Translator


async def trial(path, cfg, reader):
    frame = Image.open(path).convert("RGB")
    context = ContextManager()
    translator = Translator(cfg, context)
    output = {"frame": path.name}
    shown = []

    async def capture():
        return frame.copy()

    service = ObservationService(
        cfg, context, translator, capture=capture, target_key=lambda: 1,
        on_display=lambda jp, zh, replace: shown.append((jp, zh)),
        on_memory=lambda: None, on_status=lambda message: None, reader=reader,
    )
    start = time.perf_counter()
    service.start()
    try:
        async with asyncio.timeout(60):
            while service._cache is None:
                await asyncio.sleep(.005)
        output["frame_to_cached_translation_ms"] = (time.perf_counter() - start) * 1000
        pressed = time.perf_counter()
        service.request_display()
        output["cached_button_callback_ms"] = (time.perf_counter() - pressed) * 1000
        output["japanese"], output["chinese"] = shown[-1]
        # Let the independent scene request finish for call counts/context review.
        if service._scene_task:
            await service._scene_task
        output["metrics"] = dict(service.metrics)
        output["observations"] = context.to_dict()
        return output
    finally:
        await service.aclose()


async def baseline(path, cfg):
    context = ContextManager()
    translator = Translator(cfg, context)
    before = time.perf_counter()
    jp, scene = await translator.read_screen(ScreenshotService.encode_frame(Image.open(path).convert("RGB")))
    context.current_scene_description = scene
    zh = await translator.translate(jp)
    return {"frame": path.name, "vision_then_translation_ms": (time.perf_counter() - before) * 1000,
            "japanese": jp, "chinese": zh}


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    translator_module._TRANSLATE_LOG = args.output / "translate.log"
    translator_module._DESCRIBE_LOG = args.output / "describe.log"
    translator_module._SUMMARIZE_LOG = args.output / "summarize.log"
    cfg = copy.deepcopy(load_config())
    cfg.observation_enabled = True
    cfg.translation_mode = "manual"
    cfg.ocr_stable_ms, cfg.ocr_poll_ms = 600, 200
    cfg.ocr_region_top_percent = args.crop_top
    cfg.scene_observation_enabled = True
    reader = MeikiReader()
    warm = Image.open(args.frames[0]).convert("RGB")
    before = time.perf_counter()
    await asyncio.to_thread(reader.recognize, warm.crop((0, round(warm.height * args.crop_top / 100), warm.width, warm.height)))
    result = {"startup_and_warmup_ms": (time.perf_counter() - before) * 1000,
              "settings": {"translator": cfg.translator_model, "descriptor": cfg.descriptor_model,
                           "stable_ms": cfg.ocr_stable_ms, "poll_ms": cfg.ocr_poll_ms,
                           "crop_top_percent": args.crop_top},
              "notes": ["Warmed CPU OCR; frozen complete frame; actual production service and API calls.",
                        "Button timing includes callback dispatch, excludes Qt paint/controller polling.",
                        "Scene descriptions run in parallel; text translation uses available context.",
                        "Optional legacy baseline starts with an already complete screenshot and has no stability wait."],
              "local": [], "baseline": []}
    for path in args.frames:
        local = await trial(path, cfg, reader)
        result["local"].append(local)
        if args.vision_baseline:
            result["baseline"].append(await baseline(path, cfg))
        (args.output / "api-pipeline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"frame": path.name, "frame_to_cache_ms": round(local["frame_to_cached_translation_ms"], 1),
                          "callback_ms": round(local["cached_button_callback_ms"], 3)}, ensure_ascii=False), flush=True)
    result["median_local_ms"] = statistics.median(item["frame_to_cached_translation_ms"] for item in result["local"])
    result["median_callback_ms"] = statistics.median(item["cached_button_callback_ms"] for item in result["local"])
    if result["baseline"]:
        result["median_baseline_ms"] = statistics.median(item["vision_then_translation_ms"] for item in result["baseline"])
    (args.output / "api-pipeline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k.startswith("median_")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--crop-top", type=int, choices=range(0, 91), default=0)
    parser.add_argument("--vision-baseline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))

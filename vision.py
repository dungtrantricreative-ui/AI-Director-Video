"""
vision.py — phân tích thị giác cho từng scene.

THAY THẾ: trước đây dùng Qwen3-VL-Plus qua DashScope API (gọi mạng, tốn phí theo token).
Bây giờ dùng Qwen3-VL-4B-Instruct tải từ Hugging Face, chạy local qua `transformers`,
load 1 lần, cache vào thư mục theo config, float16, GPU nếu có.

Output JSON giữ nguyên schema `vision_analysis.json` mô tả trong
ref-asr-vision-pipeline.md để không phá vỡ các stage sau (semantic graph,
script writer, storyboard) vốn đã được thiết kế để tiêu thụ schema đó.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from platform_utils import resolve_torch_device
from progress_utils import print_progress_bar

VISION_SYSTEM_PROMPT = (
    "You are a visual analyst for a video commentary pipeline. "
    "Look at the provided frames from one video scene and describe concrete, "
    "visible facts first, then a short interpretation. "
    "Respond ONLY with a single JSON object with these exact keys: "
    "visual_summary (string), characters (array of strings), location (string), "
    "actions (array of strings), emotion (string), shot_type (string), "
    "visual_intensity (number 0-1), tags (array of strings). "
    "No markdown, no extra text, only the JSON object."
)


class VisionAnalyzer:
    """Bọc model + processor Qwen3-VL-4B-Instruct, load một lần và tái sử dụng."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("processing.vision_model_name", "Qwen/Qwen3-VL-4B-Instruct")
        self.cache_dir = str(cfg.resolve_path("paths.model_cache_dir"))
        self.device = self._resolve_device(cfg.get("processing.vision_device", "auto"))
        self.dtype = self._resolve_dtype(cfg.get("processing.vision_dtype", "float16"))
        self.max_new_tokens = cfg.get("processing.vision_max_new_tokens", 512)
        self.model = None
        self.processor = None

    @staticmethod
    def _resolve_device(preferred: str) -> str:
        return resolve_torch_device(preferred)

    @staticmethod
    def _resolve_dtype(preferred: str):
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
            preferred, torch.float16
        )

    def load(self) -> None:
        """Load model + processor vào GPU/CPU. Gọi 1 lần trước khi phân tích cả loạt scene."""
        print(f"[vision] Loading {self.model_name} on {self.device} ({self.dtype})... "
              f"(lần đầu sẽ tải model; log tải % / tốc độ của huggingface_hub sẽ hiện ngay bên dưới)")
        # Không bọc Heartbeat: huggingface_hub đã tự in thanh tiến độ tải (%, MB/s,
        # ETA) qua tqdm cho từng file trọng số. Bọc thêm sẽ khiến 2 log cùng ghi \r
        # đè lên nhau, chỉ còn thấy "vẫn đang chạy..." thay vì log tải thật.
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device in ("cpu", "mps"):
            self.model.to(self.device)
        self.model.eval()

    def unload(self) -> None:
        """Giải phóng model khỏi VRAM/RAM sau khi phân tích xong toàn bộ scene."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def analyze_scene(self, scene_id: str, frame_paths: list[str]) -> dict[str, Any]:
        """
        Phân tích một scene dựa trên các khung hình đại diện (frame_paths).
        Trả về dict theo đúng schema vision_analysis (xem VISION_SYSTEM_PROMPT).
        """
        if self.model is None:
            raise RuntimeError("VisionAnalyzer chưa được load(). Gọi .load() trước.")

        images = [Image.open(p).convert("RGB") for p in frame_paths if Path(p).exists()]
        if not images:
            return self._empty_result(scene_id, reason="no_frames")

        content = [{"type": "image", "image": img} for img in images]
        content.append({
            "type": "text",
            "text": "Analyze this scene and return the JSON object described in the system prompt.",
        })
        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_prompt], images=images, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        parsed = self._parse_json_response(output_text)
        parsed["scene_id"] = scene_id
        return parsed

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        """Cố gắng parse JSON từ output model; nếu lỗi, trả về kết quả rỗng an toàn."""
        text = text.strip()
        # Model đôi khi bọc JSON trong ```json ... ``` dù đã được yêu cầu không làm vậy.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: tìm { ... } đầu tiên trong chuỗi
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        return {
            "visual_summary": data.get("visual_summary", ""),
            "characters": data.get("characters", []),
            "location": data.get("location", ""),
            "actions": data.get("actions", []),
            "emotion": data.get("emotion", ""),
            "shot_type": data.get("shot_type", ""),
            "visual_intensity": float(data.get("visual_intensity", 0.0) or 0.0),
            "tags": data.get("tags", []),
        }

    @staticmethod
    def _empty_result(scene_id: str, reason: str) -> dict[str, Any]:
        return {
            "scene_id": scene_id,
            "visual_summary": "",
            "characters": [],
            "location": "",
            "actions": [],
            "emotion": "",
            "shot_type": "",
            "visual_intensity": 0.0,
            "tags": [],
            "review_flag": reason,
        }


def run_vision_analysis(cfg, preprocess_result: dict[str, Any], checkpoint_mgr=None) -> list[dict[str, Any]]:
    """Entry point cho stage 'vision'. Ghi vision_analysis.json vào pipeline/."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    scenes = preprocess_result["scenes"]
    keyframes = preprocess_result["keyframes"]

    analyzer = VisionAnalyzer(cfg)
    analyzer.load()

    results = []
    total_scenes = len(scenes)
    try:
        for scene_idx, scene in enumerate(scenes, start=1):
            scene_id = scene["scene_id"]
            frame_paths = keyframes.get(scene_id, [])
            analysis = analyzer.analyze_scene(scene_id, frame_paths)
            analysis["start"] = scene["start"]
            analysis["end"] = scene["end"]
            results.append(analysis)
            print_progress_bar(
                scene_idx, total_scenes,
                prefix="[vision] analyzing",
                suffix=f"{scene_id} ({len(frame_paths)} frames)",
            )
    finally:
        # Dọn VRAM ngay cả khi có lỗi giữa chừng.
        analyzer.unload()

    with open(pipeline_dir / "vision_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[vision] Xong: {len(results)} scene đã phân tích.")

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("vision", results)

    return results

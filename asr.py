"""
asr.py — sinh Dialogue Timeline (asr_timeline.json) bằng faster-whisper.

Tự động chọn device/compute_type theo config + fallback về CPU nếu CUDA
hoặc cublas không khả dụng (đúng theo hành vi mô tả trong references gốc,
nhưng giờ đọc toàn bộ tham số từ config.toml thay vì hard-code path Windows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel

from platform_utils import resolve_torch_device
from progress_utils import print_progress_bar


def _resolve_device(preferred: str) -> str:
    # faster-whisper (CTranslate2) chỉ hỗ trợ "cuda" hoặc "cpu", không có "mps".
    device = resolve_torch_device(preferred)
    return "cpu" if device == "mps" else device


def _resolve_compute_type(preferred: str, device: str) -> str:
    if preferred == "auto":
        return "float16" if device == "cuda" else "int8"
    return preferred


def load_whisper_model(cfg) -> WhisperModel:
    """
    Load faster-whisper model. Model được tự động tải về `paths.model_cache_dir`
    (faster-whisper dùng huggingface_hub cache dưới nền, ta chỉ cần trỏ
    HF cache dir bằng biến môi trường tại thời điểm import — xem run.py).
    """
    model_size = cfg.get("processing.asr_model_size", "small")
    device = _resolve_device(cfg.get("processing.asr_device", "auto"))
    compute_type = _resolve_compute_type(cfg.get("processing.asr_compute_type", "auto"), device)
    cache_dir = str(cfg.resolve_path("paths.model_cache_dir"))

    print(f"[asr] Loading faster-whisper '{model_size}' on {device} ({compute_type})... "
          f"(lần đầu sẽ tải model; log tải % / tốc độ của huggingface_hub sẽ hiện ngay bên dưới)")
    try:
        # Không bọc Heartbeat ở đây: huggingface_hub đã tự in thanh tiến độ tải
        # (%, MB/s, ETA) qua tqdm. Bọc thêm sẽ khiến 2 log cùng ghi \r đè lên
        # nhau và người dùng chỉ thấy dòng "vẫn đang chạy..." thay vì log thật.
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
    except RuntimeError as e:
        if device == "cuda" and "cublas" in str(e).lower():
            print("[asr] cublas không khả dụng, fallback sang CPU (int8).")
            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root=cache_dir,
            )
        else:
            raise
    return model


def transcribe(model: WhisperModel, audio_path: Path, language: str | None = None) -> list[dict[str, Any]]:
    """Chạy ASR + VAD trên file audio, trả về danh sách segment chuẩn hoá."""
    lang = language or None
    segments, info = model.transcribe(str(audio_path), language=lang, vad_filter=True)
    total_dur = getattr(info, "duration", 0.0) or 0.0

    results = []
    for seg in segments:
        results.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "confidence": round(seg.avg_logprob, 4),
        })
        if total_dur > 0:
            # faster-whisper trả segment tuần tự theo thời gian -> seg.end / total_dur ~ % đã xử lý.
            done_pct = min(seg.end / total_dur, 1.0)
            print_progress_bar(
                int(done_pct * 1000), 1000,
                prefix="[asr] transcribing",
                suffix=f"{seg.end:.0f}s/{total_dur:.0f}s ({len(results)} đoạn)",
            )
    print_progress_bar(1000, 1000, prefix="[asr] transcribing", suffix=f"xong ({len(results)} đoạn)")
    return results


def run_asr(cfg, preprocess_result: dict[str, Any], checkpoint_mgr=None) -> list[dict[str, Any]]:
    """Entry point cho stage 'asr'. Ghi asr_timeline.json vào pipeline/."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(preprocess_result["audio_path"])
    language = cfg.get("processing.asr_language", "") or None

    model = load_whisper_model(cfg)
    print("[asr] Transcribing...")
    timeline = transcribe(model, audio_path, language=language)

    # Giải phóng VRAM sau khi dùng xong model ASR.
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    with open(pipeline_dir / "asr_timeline.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    print(f"[asr] Xong: {len(timeline)} đoạn thoại.")

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("asr", timeline)

    return timeline

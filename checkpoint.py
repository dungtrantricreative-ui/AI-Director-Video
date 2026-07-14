"""
checkpoint.py — cơ chế checkpoint để pipeline có thể resume khi Colab bị ngắt kết nối.

Mỗi stage (asr, scenes, vision, script, tts, render) sau khi chạy xong sẽ ghi:
  - dữ liệu kết quả (JSON hoặc đường dẫn file) vào ./checkpoints/<stage>.json
  - một marker "done" để lần chạy sau biết là có thể bỏ qua stage đó.

Cách dùng:
    ckpt = CheckpointManager(checkpoint_dir)
    if ckpt.is_done("asr"):
        asr_result = ckpt.load("asr")
    else:
        asr_result = run_asr(...)
        ckpt.save("asr", asr_result)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def is_done(self, stage: str) -> bool:
        """Kiểm tra xem stage đã hoàn thành và có thể bỏ qua hay chưa."""
        p = self._path(stage)
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("_done", False))
        except (json.JSONDecodeError, OSError):
            return False

    def save(self, stage: str, payload: Any) -> None:
        """Lưu kết quả của một stage và đánh dấu là đã hoàn thành."""
        p = self._path(stage)
        wrapper = {
            "_done": True,
            "_stage": stage,
            "_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": payload,
        }
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
        print(f"[checkpoint] Đã lưu stage '{stage}' → {p}")

    def load(self, stage: str) -> Any:
        """Đọc lại kết quả đã lưu của một stage."""
        p = self._path(stage)
        with open(p, "r", encoding="utf-8") as f:
            wrapper = json.load(f)
        return wrapper["data"]

    def clear(self, stage: str | None = None) -> None:
        """Xoá checkpoint của 1 stage cụ thể, hoặc toàn bộ nếu stage=None."""
        if stage is None:
            for p in self.dir.glob("*.json"):
                p.unlink()
            print("[checkpoint] Đã xoá toàn bộ checkpoint.")
        else:
            p = self._path(stage)
            if p.exists():
                p.unlink()
                print(f"[checkpoint] Đã xoá checkpoint stage '{stage}'.")

    def status(self) -> dict[str, bool]:
        """Trả về trạng thái done/chưa done của các stage đã biết."""
        known_stages = ["preprocess", "asr", "scenes", "vision", "script", "tts", "render"]
        return {s: self.is_done(s) for s in known_stages}

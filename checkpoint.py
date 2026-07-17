"""
checkpoint.py — Enhanced checkpoint system with near real-time frequency.

Every significant operation now creates a checkpoint, allowing resume from
almost any point in the pipeline. Checkpoints are automatically synced to
cloud storage (Tigris, or any other S3-compatible provider) when configured.

Changes from original:
  - Micro-checkpoints within stages (every scene, every batch)
  - Cloud sync after each checkpoint
  - Project-aware checkpointing
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Enhanced checkpoint manager with micro-checkpoints and cloud sync."""

    def __init__(self, checkpoint_dir: str | Path, project_id: str = "", cloud_storage=None,
                 auto_sync_cloud: bool = True, auto_save_interval: int = 0):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id
        self.cloud = cloud_storage
        # auto_sync_cloud: đọc từ config processing.auto_sync_cloud. Nếu False,
        # KHÔNG tự động sync checkpoint lên cloud sau mỗi lần lưu (người
        # dùng có thể vẫn chủ động sync tay qua menu "5. Đồng bộ lên cloud").
        self.auto_sync_cloud = auto_sync_cloud
        # auto_save_interval: đọc từ project.auto_save_interval trong config.toml
        # (giây). 0 (mặc định) = giữ nguyên hành vi cũ: throttle sync
        # micro-checkpoint lên cloud theo SỐ LẦN gọi (mỗi 5 lần, xem
        # save_micro). Nếu > 0, chuyển throttle sang theo THỜI GIAN: sync mỗi
        # khi đã trôi qua ít nhất auto_save_interval giây kể từ lần sync gần
        # nhất, bất kể số lần save_micro() đã gọi.
        self.auto_save_interval = max(0, auto_save_interval)
        self._save_count = 0
        # Đếm riêng số lần save_micro() được gọi, dùng để throttle sync lên
        # cloud (KHÔNG dùng chung _save_count — biến đó chỉ tăng trong
        # save(), thường gọi 1 lần/stage nên gần như luôn = 0 -> throttle
        # bằng _save_count sẽ vô tình sync MỌI micro-checkpoint, ngược ý định
        # "mỗi 5 lần" ghi trong docstring).
        self._micro_save_count = 0
        self._last_save_time = time.time()

        # Register signal handlers for graceful shutdown
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Save emergency checkpoint on SIGINT/SIGTERM."""
        print(f"\n[checkpoint] Received signal {signum}, saving emergency checkpoint...")
        self.save_emergency()
        # Restore original signal handler and re-raise
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        sys.exit(1)

    def _path(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def _micro_path(self, stage: str, item_id: str) -> Path:
        return self.dir / f"{stage}_{item_id}.json"

    def is_done(self, stage: str) -> bool:
        """Check if a stage is completed."""
        p = self._path(stage)
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("_done", False))
        except (json.JSONDecodeError, OSError):
            return False

    def is_micro_done(self, stage: str, item_id: str) -> bool:
        """Check if a micro-checkpoint is done."""
        p = self._micro_path(stage, item_id)
        return p.exists()

    def list_micro_done(self, stage: str) -> set[str]:
        """Trả về tập hợp item_id đã có micro-checkpoint cho 1 stage.

        Dùng để resume: gọi hàm này TRƯỚC khi chạy lại 1 stage, để biết
        item nào (scene, clip...) đã xong và có thể bỏ qua, thay vì luôn
        chạy lại từ đầu dù micro-checkpoint đã được ghi.
        """
        prefix = f"{stage}_"
        done = set()
        for p in self.dir.glob(f"{prefix}*.json"):
            if p.name.endswith(".json.tmp"):
                continue
            item_id = p.stem[len(prefix):]
            done.add(item_id)
        return done

    def force_sync_micro(self, stage: str, item_id: str) -> None:
        """Ép sync 1 micro-checkpoint lên cloud ngay, bỏ qua throttle.

        Dùng ở item cuối cùng của vòng lặp để đảm bảo item cuối luôn được
        đẩy lên cloud dù chưa rơi đúng vào bội số của chu kỳ throttle.
        """
        p = self._micro_path(stage, item_id)
        if p.exists():
            self._sync_to_cloud(p, f"checkpoints/{stage}_{item_id}.json")

    def save(self, stage: str, payload: Any) -> None:
        """Save stage checkpoint and sync to cloud."""
        p = self._path(stage)
        wrapper = {
            "_done": True,
            "_stage": stage,
            "_project_id": self.project_id,
            "_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "_save_count": self._save_count,
            "data": payload,
        }
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

        self._save_count += 1
        self._last_save_time = time.time()
        print(f"[checkpoint] Saved stage '{stage}' -> {p.name}")

        # Sync to cloud
        self._sync_to_cloud(p, f"checkpoints/{stage}.json")

    def save_micro(self, stage: str, item_id: str, payload: Any) -> None:
        """Save micro-checkpoint within a stage (per-scene, per-batch)."""
        p = self._micro_path(stage, item_id)
        wrapper = {
            "_done": True,
            "_stage": stage,
            "_item_id": item_id,
            "_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": payload,
        }
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

        # Throttle sync micro-checkpoint lên cloud để tránh spam API:
        # - Nếu auto_save_interval > 0 (project.auto_save_interval trong
        #   config.toml): sync theo THỜI GIAN, mỗi khi đã trôi qua ít nhất
        #   auto_save_interval giây kể từ lần sync micro gần nhất.
        # - Ngược lại (mặc định 0): giữ hành vi cũ, sync mỗi 5 lần gọi
        #   save_micro() (dùng bộ đếm riêng _micro_save_count vì _save_count
        #   chỉ tăng trong save(), không phản ánh số lần save_micro() thực sự
        #   được gọi).
        self._micro_save_count += 1
        if self.auto_save_interval > 0:
            should_sync = (time.time() - self._last_save_time) >= self.auto_save_interval
        else:
            should_sync = self._micro_save_count % 5 == 0
        if should_sync:
            self._sync_to_cloud(p, f"checkpoints/{stage}_{item_id}.json")
            self._last_save_time = time.time()

    def load(self, stage: str) -> Any:
        """Load stage checkpoint data."""
        p = self._path(stage)
        with open(p, "r", encoding="utf-8") as f:
            wrapper = json.load(f)
        return wrapper["data"]

    def load_micro(self, stage: str, item_id: str) -> Any:
        """Load micro-checkpoint data."""
        p = self._micro_path(stage, item_id)
        with open(p, "r", encoding="utf-8") as f:
            wrapper = json.load(f)
        return wrapper["data"]

    def clear(self, stage: str | None = None) -> None:
        """Clear checkpoint(s)."""
        if stage is None:
            for p in self.dir.glob("*.json"):
                p.unlink()
            print("[checkpoint] Cleared all checkpoints.")
        else:
            # Clear main + micro + partial
            for p in self.dir.glob(f"{stage}*.json"):
                p.unlink()
            print(f"[checkpoint] Cleared checkpoints for '{stage}'.")

    def clear_all_for_project(self) -> None:
        """Clear all checkpoints for this project."""
        self.clear()

    def status(self) -> dict[str, bool]:
        """Return status of all known stages."""
        known_stages = ["preprocess", "asr", "vision", "semantic_graph", "script", "tts", "render"]
        return {s: self.is_done(s) for s in known_stages}

    def save_emergency(self) -> None:
        """Save emergency marker so we know where we stopped."""
        p = self.dir / "_emergency_stop.json"
        wrapper = {
            "_emergency": True,
            "_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_completed_stages": list(self.status().items()),
        }
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
        print("[checkpoint] Emergency checkpoint saved.")

    def _sync_to_cloud(self, local_path: Path, remote_relative: str) -> None:
        """Sync a single file to cloud storage."""
        if self.cloud is None or not self.auto_sync_cloud:
            return
        try:
            remote_key = f"projects/{self.project_id}/{remote_relative}"
            self.cloud._upload_file(local_path, remote_key)
        except Exception:
            # Don't crash pipeline on cloud sync failure
            pass

    def sync_all_to_cloud(self, force: bool = False) -> dict[str, int]:
        """Sync all checkpoint files to cloud storage.

        force=True bỏ qua cờ auto_sync_cloud (dùng khi người dùng chủ động
        bấm "đồng bộ" từ menu, chứ không phải auto-sync ngầm)."""
        if self.cloud is None:
            return {"uploaded": 0, "errors": 0}
        if not self.auto_sync_cloud and not force:
            return {"uploaded": 0, "errors": 0}

        uploaded = 0
        errors = 0
        for p in self.dir.glob("*.json"):
            remote_key = f"projects/{self.project_id}/checkpoints/{p.name}"
            ok = self.cloud._upload_file(p, remote_key)
            if ok:
                uploaded += 1
            else:
                errors += 1

        print(f"[checkpoint] Cloud sync: {uploaded} uploaded, {errors} errors")
        return {"uploaded": uploaded, "errors": errors}

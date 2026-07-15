#!/usr/bin/env python3
"""
run.py — entry point DUY NHẤT của pipeline. Đọc config.toml và chạy toàn bộ
các bước: preprocess -> asr -> vision -> semantic graph -> script (narration +
storyboard) -> tts -> render, với checkpoint sau mỗi bước để có thể resume nếu
tiến trình bị ngắt giữa chừng (Colab, máy cá nhân, server...).

Chạy (Linux/macOS):
    python3 run.py
Chạy (Windows):
    python run.py

Không cần tham số dòng lệnh. Mọi cấu hình đọc từ config.toml (cùng thư mục).
Cấu trúc project là phẳng (flat) — mọi module .py nằm cùng cấp, không còn
package con `modules/`, nên không cần chỉnh sys.path để import lẫn nhau.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config  # noqa: E402
from checkpoint import CheckpointManager  # noqa: E402
from platform_utils import ensure_ffmpeg  # noqa: E402
from progress_utils import StepTracker, print_progress_bar  # noqa: E402


def ensure_python_packages(cfg=None) -> None:
    """Kiểm tra nhanh các package quan trọng, tự pip install nếu thiếu."""
    checks = {
        "faster_whisper": "faster-whisper",
        "scenedetect": "scenedetect",
        "cv2": "opencv-python",
        "transformers": "transformers",
        "torch": "torch",
        "openai": "openai",
        "edge_tts": "edge-tts",
        "srt": "srt",
    }
    missing = []
    print(f"[deps] Kiểm tra {len(checks)} package quan trọng...")
    for i, (module_name, pip_name) in enumerate(checks.items(), start=1):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
        print_progress_bar(i, len(checks), prefix="[deps] kiểm tra", suffix=module_name)

    if missing:
        print(f"[deps] Thiếu package: {missing}. Đang cài đặt (có thể mất vài phút tuỳ mạng)... "
              f"(log tải/cài của pip hiện bên dưới)")
        # Bỏ cờ "-q" để pip tự in log tải/cài (%, tốc độ) của nó; không bọc
        # Heartbeat vì sẽ đè lên thanh tiến độ \r gốc của pip.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing],
            check=True,
        )
        print("[deps] Đã cài xong package còn thiếu.")
    else:
        print("[deps] Mọi package quan trọng đã sẵn sàng.")


def ask_task_config(cfg) -> dict:
    """
    Thu thập thông tin sáng tạo (title, plot summary, hook) trước khi phân tích.
    Không bắt buộc tương tác — nếu chạy non-interactive (không có TTY), dùng giá trị
    mặc định từ config.toml và bỏ qua bước chọn hook (dùng hook đầu tiên do LLM sinh).
    """
    task_config = {
        "narration_pov": cfg.get("processing.narration_pov", "third_person"),
        "content_type": cfg.get("processing.content_type", "movie"),
        "genre": cfg.get("processing.genre", "drama"),
        "target_duration_sec": cfg.get("processing.target_duration_sec", 180),
    }

    interactive = sys.stdin.isatty()
    if not interactive:
        print("[task] Chạy non-interactive: dùng cấu hình mặc định trong config.toml, "
              "bỏ qua bước hỏi tên phim / hook mở đầu.")
        task_config["title"] = ""
        task_config["plot_summary"] = ""
        return task_config

    print("=" * 70)
    print("CẤU HÌNH NỘI DUNG / CONTENT CONFIG (Enter để dùng mặc định)")
    print("=" * 70)
    task_config["title"] = input("Tên phim / tiêu đề: ").strip()
    task_config["plot_summary"] = input("Tóm tắt cốt truyện (bỏ trống nếu không có): ").strip()
    return task_config


def choose_hook(cfg, task_config: dict) -> str | None:
    """Sinh 10 hook và cho người dùng chọn nếu đang chạy tương tác; nếu không, tự chọn hook đầu tiên."""
    from script_writer import generate_hooks

    try:
        hooks = generate_hooks(cfg, task_config, task_config.get("plot_summary", ""))
    except Exception as e:
        print(f"[hook] Không sinh được hook ({e}), bỏ qua bước chọn hook.")
        return None

    if not hooks:
        return None

    interactive = sys.stdin.isatty()
    print("\n=== 开篇钩子 / Opening hooks ===")
    for i, h in enumerate(hooks, start=1):
        print(f"{i}. [{h.get('style', '')}] {h.get('text', '')}")

    if not interactive:
        chosen = hooks[0]["text"]
        print(f"\n[hook] Non-interactive: tự động chọn hook #1: {chosen}")
        return chosen

    choice = input("\nChọn số thứ tự hook (Enter để dùng #1): ").strip()
    if not choice:
        return hooks[0]["text"]
    try:
        idx = int(choice) - 1
        return hooks[idx]["text"]
    except (ValueError, IndexError):
        print("Lựa chọn không hợp lệ, dùng hook #1.")
        return hooks[0]["text"]


def main() -> None:
    print("=" * 70)
    print("AI DIRECTOR VIDEO COMMENTARY — Colab Pipeline")
    print("=" * 70)

    ensure_ffmpeg()

    cfg = load_config("config.toml")
    ensure_python_packages(cfg)

    # Nạp hf_token từ config.toml vào biến môi trường TRƯỚC khi bất kỳ module nào
    # (asr.py, vision.py) import huggingface_hub/transformers, để việc tải model
    # dùng đúng token đã cấu hình thay vì luôn gửi request "unauthenticated".
    hf_token = cfg.get("api.hf_token", "")
    if hf_token and not hf_token.startswith("PASTE_"):
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("[main] Đã nạp hf_token từ config.toml vào biến môi trường.")

    checkpoint_dir = cfg.resolve_path("paths.checkpoint_dir")

    ckpt = CheckpointManager(checkpoint_dir)

    print("\n[checkpoint] Trạng thái hiện tại:")
    for stage, done in ckpt.status().items():
        print(f"  - {stage}: {'✓ done' if done else '  pending'}")
    print()

    import preprocess, asr, vision, semantic_graph, script_writer, tts, render

    stages = ["preprocess", "asr", "vision", "semantic_graph", "script", "tts", "render"]
    tracker = StepTracker(stages)

    def run_stage(stage: str, compute_fn, has_checkpoint: bool = True):
        """
        Chạy 1 stage: bỏ qua nếu đã có checkpoint local, ngược lại chạy
        compute_fn() rồi lưu.
        """
        tracker.start(stage)
        if has_checkpoint and ckpt.is_done(stage):
            result = ckpt.load(stage)
            print(f"[main] Bỏ qua {stage} (đã có checkpoint).")
            tracker.finish(stage, skipped=True)
        else:
            result = compute_fn()
            tracker.finish(stage)
        return result

    preprocess_result = run_stage("preprocess", lambda: preprocess.run_preprocess(cfg, ckpt))
    asr_timeline = run_stage("asr", lambda: asr.run_asr(cfg, preprocess_result, ckpt))
    vision_analysis = run_stage("vision", lambda: vision.run_vision_analysis(cfg, preprocess_result, ckpt))

    # Semantic graph không có checkpoint riêng (chạy nhanh + luôn cần dữ liệu mới nhất).
    semantic_blocks = run_stage(
        "semantic_graph",
        lambda: semantic_graph.run_semantic_graph(cfg, preprocess_result, asr_timeline, vision_analysis, ckpt),
        has_checkpoint=False,
    )

    def _run_script():
        task_config = ask_task_config(cfg)
        hook = choose_hook(cfg, task_config)
        return script_writer.run_script_writer(
            cfg, task_config, semantic_blocks, asr_timeline, vision_analysis,
            hook=hook, director_brief=task_config.get("plot_summary", ""), checkpoint_mgr=ckpt,
        )

    storyboard = run_stage("script", _run_script)
    tts_result = run_stage("tts", lambda: tts.run_tts(cfg, storyboard, ckpt))
    render_result = run_stage("render", lambda: render.run_render(cfg, storyboard, tts_result, ckpt))

    print("\n" + "=" * 70)
    print("HOÀN TẤT / DONE")
    print("=" * 70)
    print(f"final_preview.mp4 : {render_result['final_preview_path']}")
    print(f"narration_subtitle.srt : {render_result['srt_path']}")
    print(f"Validation: {'PASS' if render_result['validation_report']['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()

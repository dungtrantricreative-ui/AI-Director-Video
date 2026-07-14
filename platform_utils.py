"""
platform_utils.py — các hàm tiện ích đa nền tảng (Linux / macOS / Windows),
dùng chung bởi run.py và setup.py.

Không còn giả định "luôn chạy trên Debian/Colab với apt-get" — mỗi hàm ở đây
tự phát hiện hệ điều hành (platform.system()) và trình quản lý gói phù hợp,
rồi mới quyết định cách cài đặt hoặc chỉ dẫn người dùng cài thủ công.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys




def current_os() -> str:
    """Trả về 'linux' | 'macos' | 'windows' | 'other'."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return "other"


def _has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def print_system_info() -> None:
    print("=" * 70)
    print("THÔNG TIN HỆ THỐNG / SYSTEM INFO")
    print("=" * 70)
    print(f"OS: {platform.platform()} ({current_os()})")
    print(f"Python: {sys.version.splitlines()[0]}")

    try:
        import psutil  # type: ignore
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"RAM: {ram_gb:.1f} GB")
    except ImportError:
        pass

    # GPU: nvidia-smi (Linux/Windows có CUDA), fallback kiểm tra Apple Silicon (MPS).
    if _has_cmd("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"GPU: {result.stdout.strip()}")
            else:
                print("GPU: không phát hiện (nvidia-smi có mặt nhưng không trả kết quả).")
        except Exception:
            print("GPU: không phát hiện.")
    elif current_os() == "macos" and platform.machine() == "arm64":
        print("GPU: Apple Silicon (dùng qua backend MPS của PyTorch nếu có).")
    else:
        print("GPU: không phát hiện (không có nvidia-smi / không phải Apple Silicon).")
    print()


def ensure_ffmpeg(auto_install: bool = True) -> None:
    """
    Kiểm tra FFmpeg có trên PATH; nếu không và auto_install=True, thử cài bằng
    trình quản lý gói phù hợp với hệ điều hành hiện tại:
      - Linux : apt-get (Debian/Ubuntu/Colab) -> dnf -> pacman
      - macOS : Homebrew (brew)
      - Windows: winget -> choco
    Nếu không cài tự động được, ném lỗi kèm hướng dẫn cài thủ công theo từng OS.
    """
    if _has_cmd("ffmpeg"):
        return

    if not auto_install:
        raise RuntimeError(_manual_ffmpeg_instructions())

    os_name = current_os()
    print(f"[deps] FFmpeg chưa có trên PATH, đang thử tự cài cho {os_name}...")

    # Không bọc Heartbeat: apt-get/dnf/pacman/brew/winget/choco đều tự in log
    # cài đặt của riêng chúng ra stdout/stderr theo thời gian thực. Bọc thêm
    # Heartbeat sẽ chỉ chen dòng "vẫn đang chạy..." vào giữa log gốc.
    try:
        if os_name == "linux":
            if _has_cmd("apt-get"):
                subprocess.run(["apt-get", "update", "-qq"], check=True)
                # Không dùng "-qq" ở đây để log cài đặt gói (tải/giải nén) hiện đầy đủ.
                subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
            elif _has_cmd("dnf"):
                subprocess.run(["dnf", "install", "-y", "ffmpeg"], check=True)
            elif _has_cmd("pacman"):
                subprocess.run(["pacman", "-Sy", "--noconfirm", "ffmpeg"], check=True)
            else:
                raise RuntimeError(_manual_ffmpeg_instructions())
        elif os_name == "macos":
            if _has_cmd("brew"):
                subprocess.run(["brew", "install", "ffmpeg"], check=True)
            else:
                raise RuntimeError(_manual_ffmpeg_instructions())
        elif os_name == "windows":
            if _has_cmd("winget"):
                subprocess.run(
                    ["winget", "install", "-e", "--id", "Gyan.FFmpeg", "--accept-source-agreements",
                     "--accept-package-agreements"],
                    check=True,
                )
            elif _has_cmd("choco"):
                subprocess.run(["choco", "install", "ffmpeg", "-y"], check=True)
            else:
                raise RuntimeError(_manual_ffmpeg_instructions())
        else:
            raise RuntimeError(_manual_ffmpeg_instructions())
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(_manual_ffmpeg_instructions())

    if not _has_cmd("ffmpeg"):
        # Trên Windows, PATH có thể cần mở lại terminal sau khi winget/choco cài xong.
        raise RuntimeError(
            _manual_ffmpeg_instructions()
            + "\n(Nếu vừa cài bằng winget/choco, hãy đóng và mở lại terminal rồi thử lại.)"
        )


def _manual_ffmpeg_instructions() -> str:
    return (
        "Không tìm thấy FFmpeg và không thể tự cài. Cài thủ công theo hệ điều hành:\n"
        "  - Linux (Debian/Ubuntu): sudo apt-get install -y ffmpeg\n"
        "  - Linux (Fedora):        sudo dnf install -y ffmpeg\n"
        "  - Linux (Arch):          sudo pacman -S ffmpeg\n"
        "  - macOS (Homebrew):      brew install ffmpeg\n"
        "  - Windows (winget):      winget install Gyan.FFmpeg\n"
        "  - Windows (choco):       choco install ffmpeg\n"
        "  - Hoặc tải bản build từ https://ffmpeg.org/download.html và thêm vào PATH."
    )


def resolve_torch_device(preferred: str = "auto") -> str:
    """
    Chọn device cho PyTorch: tôn trọng giá trị người dùng chỉ định (cuda/cpu/mps),
    hoặc tự phát hiện khi preferred == "auto":
      CUDA (NVIDIA, Linux/Windows) > MPS (Apple Silicon, macOS) > CPU.
    """
    if preferred != "auto":
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

"""
config.py — bộ đọc cấu hình trung tâm.

Toàn bộ project đọc cấu hình từ `config.toml` thông qua module này.
Không còn bất kỳ chỗ nào dùng os.getenv() hoặc python-dotenv.

Dùng `tomllib` (built-in từ Python 3.11+), fallback về thư viện `tomli`
cho Python < 3.11 (bắt buộc trong requirements.txt).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:
    import tomli as tomllib  # type: ignore[import-not-found]


class Config:
    """
    Wrapper mỏng quanh dict đã parse từ config.toml.

    Cho phép truy cập kiểu `cfg.get("api.cerebras_api_key")` (dot-path)
    hoặc `cfg["api"]["cerebras_api_key"]` (dict thường).
    """

    def __init__(self, data: dict[str, Any], config_path: Path):
        self._data = data
        self.config_path = config_path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Đọc giá trị theo đường dẫn dạng 'section.key', trả về default nếu thiếu."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Trả về toàn bộ một section (vd: cfg.section('processing'))."""
        return self._data.get(name, {})

    def set(self, dotted_key: str, value: Any) -> None:
        """Ghi đè 1 giá trị trong bộ nhớ (KHÔNG ghi xuống config.toml trên đĩa).
        Dùng khi cần cập nhật cấu hình lúc chạy (vd: người dùng nhập lại đường
        dẫn video vì đường dẫn trong config.toml không tồn tại)."""
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def resolve_path(self, dotted_key: str, default: str | None = None) -> Path:
        """
        Đọc một giá trị đường dẫn từ config và chuẩn hoá thành Path tuyệt đối,
        tương đối theo thư mục chứa config.toml (không phải theo cwd hiện tại).
        """
        raw = self.get(dotted_key, default)
        if raw is None:
            raise KeyError(f"Missing required path config: {dotted_key}")
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (self.config_path.parent / p).resolve()
        return p

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | os.PathLike = "config.toml") -> Config:
    """
    Load và parse config.toml. Ném lỗi rõ ràng nếu file không tồn tại
    hoặc thiếu section bắt buộc.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {config_path}\n"
            f"Hãy copy config.toml.example thành config.toml rồi điền key (xem README mục Cài đặt)."
        )

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    required_sections = ["api", "tts", "processing", "paths"]
    missing = [s for s in required_sections if s not in data]
    if missing:
        raise ValueError(
            f"config.toml thiếu (các) section bắt buộc: {missing}. "
            f"Cần có đủ [api], [tts], [processing], [paths]."
        )

    return Config(data, config_path)


# Instance dùng chung, lazy-load lần đầu gọi get_config().
_default_config: Config | None = None


def get_config(path: str | os.PathLike = "config.toml") -> Config:
    """Trả về Config đã cache (singleton nhẹ) để tránh đọc file lặp lại."""
    global _default_config
    if _default_config is None:
        _default_config = load_config(path)
    return _default_config


def reset_config_cache() -> None:
    """Dùng trong test/Colab khi cần load lại config sau khi config.toml bị sửa thủ công."""
    global _default_config
    _default_config = None

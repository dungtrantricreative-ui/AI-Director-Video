# AI Director — Video Commentary Pipeline

Pipeline Python độc lập, tự động tạo video bình luận/thuyết minh (review phim,
tóm tắt phim, "recap") từ một file video gốc: tách audio → nhận diện scene →
ASR → phân tích hình ảnh → viết kịch bản (hook + lời bình) → TTS → dựng video
cuối cùng kèm phụ đề.

Chạy được trên **Google Colab, Linux, macOS và Windows** — không phụ thuộc
vào bất kỳ nền tảng cụ thể nào.

> Dự án này là bản viết lại độc lập bằng Python, bám theo đúng schema/thuật
> toán mô tả trong repo skill gốc `ai-director-video-commentary` (vốn được
> thiết kế cho Claude Code agent, không phải code chạy được). Xem chi tiết
> khác biệt ở mục [Nguồn gốc & khác biệt](#nguồn-gốc--khác-biệt) bên dưới.

## Mục lục

- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cấu trúc project](#cấu-trúc-project)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Chạy pipeline](#chạy-pipeline)
- [Kết quả](#kết-quả)
- [Checkpoint / resume](#checkpoint--resume)
- [Giới hạn / điều cần biết](#giới-hạn--điều-cần-biết)
- [Nguồn gốc & khác biệt](#nguồn-gốc--khác-biệt)

## Yêu cầu hệ thống

- Python **3.10+** (khuyến nghị 3.11+ để có `tomllib` built-in)
- FFmpeg (script cài đặt tự động dò và cài nếu thiếu)
- GPU NVIDIA (CUDA) hoặc Apple Silicon (MPS) để chạy nhanh; vẫn chạy được
  trên CPU nhưng chậm hơn nhiều ở bước phân tích hình ảnh
- Cerebras API key (bắt buộc, dùng để sinh kịch bản) — lấy tại
  [cloud.cerebras.ai](https://cloud.cerebras.ai)

## Cấu trúc project

Toàn bộ project nằm phẳng trong một thư mục duy nhất — **không có thư mục
con nào cả** (không `modules/`, không `scripts/`, không `references/`), để
clone/deploy lên GitHub nhanh và đơn giản nhất có thể:

```
ai-director/
  config.py                     ← loader TOML dùng chung cho mọi module
  checkpoint.py                  ← quản lý checkpoint/resume
  platform_utils.py              ← tiện ích đa nền tảng (ffmpeg, GPU/device detection)
  preprocess.py                  ← ffprobe, tách audio, scene detection, keyframes
  asr.py                          ← faster-whisper
  vision.py                       ← Qwen3-VL-4B-Instruct (transformers)
  semantic_graph.py               ← hợp nhất ASR + vision thành semantic blocks
  script_writer.py                ← LLM (Cerebras) viết hook + narration + storyboard
  tts.py                           ← edge-tts
  render.py                        ← ffmpeg render + validate SRT/video
  storyboard_to_srt.py             ← CLI: xuất SRT từ storyboard.json
  validate_storyboard.py           ← CLI: validate storyboard.json theo schema
  setup.py                         ← chạy 1 lần đầu tiên để cài đặt + tạo config.toml
  run.py                            ← entry point duy nhất, chạy toàn bộ pipeline
  config.toml.example              ← mẫu cấu hình, copy thành config.toml rồi điền key
  requirements.txt
  ref-*.md, ref-*.json, ref-*.srt   ← tài liệu schema gốc (Video Semantic Graph, Storyboard...)
```

Các thư mục dữ liệu (`input/`, `output/`, `checkpoints/`, `model_cache/`)
**không có sẵn trong repo** — code tự tạo bằng `mkdir(parents=True,
exist_ok=True)` ngay khi cần (checkpoint đầu tiên, video đầu ra, model tải
về...), nên không cần commit thư mục rỗng lên GitHub. Bạn chỉ cần tự tạo
`input/` (hoặc trỏ `paths.input_video` sang nơi khác) khi đặt video nguồn vào.

## Cài đặt

### Cách 1 — script tự động (khuyến nghị, mọi nền tảng)

```bash
# Linux / macOS
python3 setup.py

# Windows
python setup.py
```

`setup.py` sẽ:
1. In thông tin hệ thống (OS, Python, RAM, GPU/Apple Silicon).
2. Tự cài FFmpeg bằng trình quản lý gói phù hợp với OS hiện tại
   (`apt-get`/`dnf`/`pacman` trên Linux, `brew` trên macOS, `winget`/`choco`
   trên Windows) — nếu không có sẵn, script in hướng dẫn cài thủ công.
3. Cài các package Python trong `requirements.txt`.
4. Hỏi bạn nhập: Cerebras API key, Hugging Face token (tuỳ chọn), giọng Edge
   TTS, đường dẫn video đầu vào, thư mục output, tên model Qwen3-VL.
5. Ghi cấu hình ra `config.toml`.
6. Tải trước model faster-whisper (small) + Qwen3-VL-4B-Instruct vào
   `./model_cache`.

### Cách 2 — cài thủ công

```bash
# 1. Cài FFmpeg
#    Linux (Debian/Ubuntu): sudo apt-get install -y ffmpeg
#    macOS (Homebrew):      brew install ffmpeg
#    Windows (winget):      winget install Gyan.FFmpeg

# 2. Cài package Python
pip install -r requirements.txt

# 3. Tạo config.toml từ mẫu và điền Cerebras API key
cp config.toml.example config.toml    # Windows (PowerShell): copy config.toml.example config.toml

# 4. (Tuỳ chọn) Tải trước model — nếu bỏ qua, model sẽ tự tải khi chạy run.py lần đầu
python3 -c "
from pathlib import Path
cache_dir = Path('./model_cache').resolve()
cache_dir.mkdir(parents=True, exist_ok=True)

from faster_whisper import WhisperModel
WhisperModel('small', device='cpu', compute_type='int8', download_root=str(cache_dir))

from transformers import AutoProcessor
AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', cache_dir=str(cache_dir), trust_remote_code=True)
"
```

> Nếu Hugging Face yêu cầu đăng nhập (model gated) hoặc bạn muốn tránh giới
> hạn tốc độ tải, đặt token trước khi chạy lệnh trên:
> ```bash
> export HF_TOKEN=hf_xxxxxxxx          # Windows (PowerShell): $env:HF_TOKEN="hf_xxxxxxxx"
> ```
>
> Lệnh trên chỉ tải **processor** của model vision (nhẹ); **trọng số**
> Qwen3-VL-4B-Instruct (~8-10GB) sẽ tự tải về `model_cache/` ngay lần đầu
> pipeline chạy tới bước `vision.py` — không cần tải thủ công thêm, chỉ cần
> đảm bảo đủ dung lượng ổ đĩa và mạng ổn định ở lần chạy đầu tiên.
>
> Nếu đổi `processing.vision_model_name` trong `config.toml` sang model khác
> (không phải `Qwen/Qwen3-VL-4B-Instruct`), sửa lại tên model trong lệnh
> `AutoProcessor.from_pretrained(...)` ở trên cho khớp.

### Google Colab

Upload toàn bộ thư mục project (hoặc giải nén file zip) rồi chạy trong cell:

```python
%cd ai-director
!python setup.py
```

## Cấu hình

Mọi tham số nằm trong `config.toml` (copy từ `config.toml.example`), gồm 4 section:

- `[api]` — Cerebras API key/endpoint/model, HF token.
- `[tts]` — giọng, tốc độ, âm lượng edge-tts.
- `[processing]` — model ASR/vision, ngưỡng scene detection, thể loại/độ dài
  narration, công thức tính thời lượng phụ đề.
- `[paths]` — đường dẫn video đầu vào, thư mục output/checkpoint/model cache.

Không dùng `.env` / `os.getenv()` ở bất kỳ đâu trong code — mọi cấu hình đọc
qua `config.py`.

## Chạy pipeline

1. Đặt video nguồn vào đường dẫn đã khai trong `config.toml` (mặc định
   `./input/source.mp4` — tự tạo thư mục `input/` nếu chưa có, hoặc đổi
   `paths.input_video` sang đường dẫn khác).
2. Chạy:

```bash
python3 run.py     # Linux/macOS
python run.py       # Windows
```

Pipeline chạy tuần tự: **preprocess → ASR → vision → semantic graph → script
(narration + storyboard) → TTS → render**, checkpoint sau mỗi bước vào
`./checkpoints/`. Nếu tiến trình bị ngắt giữa chừng, chạy lại đúng lệnh trên —
pipeline tự resume từ bước gần nhất, không chạy lại từ đầu.

## Kết quả

```
output/
  pipeline/            ← sản phẩm trung gian để debug (asr_timeline.json,
                          vision_analysis.json, semantic_blocks.json,
                          storyboard.json, voiceover.mp3, keyframes/...)
  deliverables/
    final_preview.mp4   ← video hoàn chỉnh
    narration_subtitle.srt
```

## Checkpoint / resume

```python
from checkpoint import CheckpointManager
ckpt = CheckpointManager("./checkpoints")
ckpt.clear("script")   # xoá checkpoint để chạy lại riêng bước viết kịch bản
ckpt.clear()           # xoá toàn bộ, chạy lại từ đầu
```

## Giới hạn / điều cần biết

- **Cerebras model**: tên model trong `[api] cerebras_model` có thể thay đổi
  theo thời gian trên Cerebras Inference Cloud — sửa trong `config.toml` nếu
  cần.
- **Qwen3-VL-4B-Instruct** cần tải ~8-10GB trọng số lần đầu chạy stage vision;
  chạy được ở `float16` trên GPU 15GB VRAM trở lên (T4 free tier của Colab
  chạy được nhưng có thể chậm).
- Trên **Apple Silicon (M1/M2/M3...)**, `vision_device`/`asr_device = "auto"`
  sẽ ưu tiên CUDA → MPS → CPU; riêng `faster-whisper` (CTranslate2) chưa hỗ
  trợ MPS nên tự động chạy CPU.
- Các bước tương tác (chọn hook mở đầu, xác nhận storyboard) không bắt buộc
  dừng lại chờ người dùng nếu chạy non-interactive (ví dụ chạy nền, CI) — nếu
  muốn luôn dừng lại xác nhận, chạy `run.py` trong terminal/cell tương tác
  bình thường.
- Không tự sinh BGM/cover — người dùng tự hoàn thiện trong CapCut/Premiere
  nếu cần.

## Nguồn gốc & khác biệt

Repo skill gốc `ai-director-video-commentary` không phải là chương trình
Python — đó là một **skill cho Claude Code** (`skill.md` + `references/*.md`),
tức tài liệu hướng dẫn để Claude Code (agent) tự thực hiện từng bước bằng tay
(gọi ffmpeg, đọc frame, viết kịch bản bằng model built-in...). Repo gốc không
có `main.py`, không có module gọi DashScope, không có module TTS.

Bản này viết lại toàn bộ thành code Python độc lập, bám sát schema/thuật toán
mô tả trong `references/*.md` (Video Semantic Graph, Storyboard, công thức
tính thời lượng phụ đề, thuật toán render...), với 4 điểm khác biệt:

| Thành phần | Repo gốc (thiết kế cho Claude Code) | Bản này |
|---|---|---|
| Cấu hình | không có (giả định do agent tự hỏi) | `config.toml` (`tomllib`/`tomli`) |
| Vision analysis | Qwen3-VL-Plus qua DashScope API | Qwen3-VL-4B-Instruct local qua `transformers` |
| Viết kịch bản | Claude built-in (không cần API) | LLM qua Cerebras API (OpenAI-compatible) |
| TTS | không có (để trống, dùng CapCut) | `edge-tts`, xuất `voiceover.mp3` |

Các file `ref-*.md`/`ref-*.json`/`ref-*.srt` được giữ lại làm tài liệu mô tả
schema mà code Python đã hiện thực hoá — không còn là "spec cho agent" nữa.

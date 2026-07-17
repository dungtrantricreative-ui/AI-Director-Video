# AI Director Video Commentary — Enhanced Version

An AI-powered video commentary pipeline that automatically generates narrated commentary videos from any input video. Features project management, cloud sync (Tigris, S3-compatible), and near real-time checkpointing.

## What This Does

Input: Any video file (movie clip, drama scene, etc.)
Output: A new video with AI-generated voiceover commentary, properly synced with the original footage, plus SRT subtitles.

**Pipeline stages:**
1. **Preprocess** — Probe video, extract audio, detect scenes, extract keyframes
2. **ASR** — Speech-to-text transcription using faster-whisper
3. **Vision** — AI analyzes each scene's visual content
4. **Semantic Graph** — Combines audio + visual analysis into structured blocks
5. **Script** — AI writes viral-style narration commentary
6. **TTS** — Text-to-speech voiceover generation
7. **Render** — Final video assembly with synced commentary

---

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.toml` with your API keys:

```toml
[api]
cerebras_api_key = "your-cerebras-key"
mistral_api_key = "your-mistral-key"

[cloud]
access_key = "your-tigris-access-key"
secret_key = "your-tigris-secret-key"
bucket_name = "ai-director-video"
endpoint_url = "https://t3.storage.dev"
region_name = "auto"
addressing_style = "virtual"
enabled = true
```

### 3. Run

```bash
python run.py
```

You'll see the project management menu:

```
======================================================================
  AI DIRECTOR VIDEO — PROJECT MANAGER
======================================================================
  1. Create new project
  2. Continue existing project
  3. List all projects
  4. Delete a project
  5. Sync project to cloud (Tigris)
  6. Download project from cloud
  7. Run pipeline on a project
  0. Exit
======================================================================
```

---

## Project Management

### Creating a Project

1. Select **1. Create new project**
2. Enter a project ID (e.g., `my-movie-v1`)
3. Enter video file path
4. Enter project title

Each project gets its own directory:
```
projects/
  my-movie-v1/
    _project_meta.json      # Project metadata
    checkpoints/            # Pipeline checkpoints
    output/
      pipeline/            # Intermediate files
      deliverables/        # Final video + subtitles
```

### Continuing a Project

1. Select **2. Continue existing project**
2. Choose from the list
3. Pipeline resumes from last completed stage

### Cloud Sync

Projects are automatically synced to cloud storage. You can also:
- **5. Sync project to cloud** — Manual upload
- **6. Download project from cloud** — Download from another machine

---

## Features

### Near Real-Time Checkpoints

The pipeline saves checkpoints at multiple levels:
- **Stage checkpoints** — After each major stage (preprocess, ASR, vision, etc.)
- **Micro-checkpoints** — Within stages (per-scene in vision, per-clip in TTS/render)
- **Emergency checkpoints** — On SIGINT/SIGTERM (Ctrl+C)

Configure frequency in `config.toml`:
```toml
[processing]
micro_checkpoint_interval = 1  # Save every item (most frequent)
```

### Cloud Storage (Tigris)

All project data is synced to Tigris, or any other S3-compatible storage provider:
- Checkpoints sync automatically after each save
- Full project upload/download for cross-machine workflow
- Credentials stored in `config.toml` (not environment variables)

### Smart Resume

When you continue a project:
1. Scans all checkpoints to determine current stage
2. Skips completed stages automatically
3. Resumes from exact point of interruption
4. Works across different machines (with cloud sync)

---

## Configuration Reference

### `[api]` — API Keys

| Key | Description |
|-----|-------------|
| `cerebras_api_key` | Cerebras API key for script writing |
| `cerebras_model` | Model name (default: `zai-glm-4.7`) |
| `hf_token` | Hugging Face token (optional, for local vision) |
| `mistral_api_key` | Mistral API key for vision analysis |

### `[tts]` — Voice Settings

| Key | Description | Default |
|-----|-------------|---------|
| `voice` | Edge-TTS voice name | `vi-VN-HoangMinhNeural` |
| `rate` | Speech rate | `+0%` |
| `volume` | Volume adjustment | `+0%` |
| `pitch` | Pitch adjustment | `+0Hz` |

**Available voices:**
- Vietnamese: `vi-VN-HoangMinhNeural`, `vi-VN-NamMinhNeural`
- English: `en-US-JennyNeural`, `en-US-GuyNeural`
- Chinese: `zh-CN-XiaoxiaoNeural`, `zh-CN-YunxiNeural`

### `[processing]` — Pipeline Settings

| Key | Description | Default |
|-----|-------------|---------|
| `asr_model_size` | Whisper model size | `small` |
| `vision_backend` | Vision analysis backend | `mistral` |
| `micro_checkpoint_interval` | Checkpoint frequency | `1` |
| `narration_pov` | Narration point of view | `third_person` |
| `content_type` | Content type | `movie` |
| `target_duration_sec` | Target output duration | `180` |

### `[cloud]` — Cloud Storage

| Key | Description | Default |
|-----|-------------|---------|
| `access_key` | Cloud storage access key | — |
| `secret_key` | Cloud storage secret key | — |
| `bucket_name` | Storage bucket name | `ai-director-video` |
| `endpoint_url` | S3 endpoint URL | `https://t3.storage.dev` (Tigris) |
| `region_name` | Region (`auto` for Tigris) | `auto` |
| `addressing_style` | `path` or `virtual` (Tigris requires `virtual`) | `virtual` |
| `enabled` | Enable cloud sync | `true` |

### `[paths]` — File Paths

| Key | Description | Default |
|-----|-------------|---------|
| `input_video` | Input video path | `./input/source.mp4` |
| `output_dir` | Output directory | `./output` |
| `checkpoint_dir` | Checkpoint directory | `./checkpoints` |
| `projects_dir` | Projects directory | `./projects` |

---

## Command Line Usage

### Interactive Mode (Default)

```bash
python run.py
```

Shows project menu for managing and running projects.

### Non-Interactive Mode

```bash
# Run directly without menu (uses default project)
python run.py --no-menu
```

Or set in `config.toml`:
```toml
[project]
show_project_menu_on_start = false
```

---

## File Structure

```
AI-Director-Video/
├── run.py                    # Main entry point
├── config.toml               # Configuration (your API keys)
├── config.toml.example       # Example configuration
├── requirements.txt          # Python dependencies
│
├── checkpoint.py             # Enhanced checkpoint system
├── cloud_storage.py          # Cloud storage (Tigris / S3-compatible)
├── project_manager.py        # Project management
│
├── preprocess.py             # Video preprocessing
├── asr.py                    # Speech-to-text
├── vision.py                 # Visual analysis
├── semantic_graph.py         # Data combination
├── script_writer.py          # Narration generation
├── tts.py                    # Text-to-speech
├── render.py                 # Video rendering
│
├── config.py                 # Config loader
├── platform_utils.py         # Platform utilities
├── progress_utils.py         # Progress tracking
│
├── projects/                 # Project directories
│   └── <project-id>/
│       ├── _project_meta.json
│       ├── checkpoints/
│       └── output/
│
├── input/                    # Input videos
├── output/                   # Default output
├── checkpoints/              # Default checkpoints
└── model_cache/              # Downloaded AI models
```

---

## Troubleshooting

### "No such file or directory" for video
- Check `paths.input_video` in `config.toml`
- Or enter video path when prompted

### Cloud sync fails
- Check `access_key` and `secret_key` in `[cloud]`
- Ensure bucket name is correct and, for Tigris, that `region_name = "auto"` and `addressing_style = "virtual"`
- Pipeline continues locally even if cloud sync fails

### Checkpoint not found
- Delete the corrupted checkpoint: `rm checkpoints/<stage>.json`
- Or create fresh project and re-run

### Out of memory (CUDA)
- Reduce `vision_batch_size` in config
- Use `vision_backend = "mistral"` instead of `"local"`
- Use smaller `asr_model_size`

---

## API Keys Setup

### Cerebras (for script writing)
1. Sign up at https://cerebras.ai
2. Get API key from dashboard
3. Add to `config.toml`: `cerebras_api_key = "your-key"`

### Mistral (for vision analysis)
1. Sign up at https://mistral.ai
2. Get API key from dashboard
3. Add to `config.toml`: `mistral_api_key = "your-key"`

### Tigris (for cloud storage)
1. Sign up at https://www.tigrisdata.com and create an access key + bucket
2. Fill in `[cloud]` in `config.toml` with the access key, secret key, and bucket name
3. Set `cloud.enabled = true` to use cloud sync (endpoint/region/addressing_style already default to Tigris)

---

## Tips

1. **Start simple** — Use short videos (1-3 minutes) first
2. **Check checkpoints** — View `checkpoints/` folder to see progress
3. **Use cloud sync** — Sync important projects to continue on another machine
4. **Customize voice** — Try different TTS voices in config
5. **Adjust duration** — Set `target_duration_sec` for desired output length

---

## License

This project is provided as-is for educational and personal use.

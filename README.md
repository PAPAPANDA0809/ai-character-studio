# Reelmint

Local pipeline: post idea → scene plan (local LLM) → Flux keyframes → Wan 2.2 first-last-frame clips (5 s each) → stitched MP4 + caption + subtitles.

## Launch

Double-click `start_studio.bat`. It starts ComfyUI (if it isn't running) in its own window, starts the web app and opens http://127.0.0.1:7860.

1. Type a post idea (and optionally a fixed character description), pick keyframe count and format, press **Plan scenes**.
2. Edit prompts, on-screen lines, motions and caption if you want, then **1. Generate keyframes**.
3. Regenerate any keyframe you don't like (new seed), then **2. Animate & stitch**.
4. Results land in `app\projects\<project>\`: `final.mp4`, `final_subtitled.mp4`, `caption.txt`, `subtitles.srt`, `keyframes\`, `clips\`.

N keyframes give N-1 clips of 5 s (81 frames at 16 fps).

## Layout

| Path | What |
|---|---|
| `ComfyUI_windows_portable\` | ComfyUI 0.38 portable + ComfyUI-GGUF node |
| `models\` | all model files (wired in via `ComfyUI\extra_model_paths.yaml`) |
| `app\characters\` | master character pictures you upload |
| `app\` | the web app (stdlib Python, no extra packages) |
| `app\config.json` | model names, steps, sizes, LLM provider |
| `start_comfyui.bat` | ComfyUI only (UI at http://127.0.0.1:8188) |

## Models

Trimmed on 3 Oct 2026 to the engines in use (about 138 GB of other models removed). Bigger models are meant to run on rented GPUs instead.

| Engine | Files | Character picture | Licence |
|---|---|---|---|
| Qwen-Image 2.1 (default) | int8 7.3 GB + Qwen3-VL-8B int8 9.4 GB + VAE | yes | Qwen research licence (commercial use needs an agreement) |
| Z-Image Turbo | Q8_0 GGUF 7.2 GB + Qwen3-4B fp8 5.6 GB + Flux VAE (`ae`) | no | Apache 2.0 |
| Wan 2.2 14B (video) | I2V A14B high + low noise Q4_K_S GGUF, lightx2v 4-step LoRAs, UMT5-XXL Q5_K_M GGUF, Wan 2.1 VAE; uses `WanFirstLastFrameToVideo` | start/end pictures | Apache 2.0 |
| Wan 2.2 5B (text-to-video) | TI2V-5B Q5_K_M GGUF + Wan 2.2 VAE | text only | Apache 2.0 |

- Trained characters: LoRAs in `models\loras`, listed under `loras` in `app\config.json` (Jacelyn, for Qwen-Image 2.1).
- Text: Ollama `qwen3:4b` (unloaded after each call so the GPU is free).
- Upscaler: 4x-UltraSharp, for engines that don't draw HD directly.

Measured on this PC (RTX 3070 Ti 8 GB, 32 GB RAM): Qwen-Image 2.1 ~30 s per picture, ~4 min with a character picture; Wan 2.2 14B 18-24 min per 480p clip. The removed engines' workflow code is still in `app\comfy.py`; downloading their files again (`setup\extra*.txt`) and re-listing them in the engine tables brings them back. LTX-2.3 took ~2.5 h per 5 s clip here, which is why it was dropped.

## Switching the text model

In `app\config.json` → `llm.provider`:
- `"ollama"`: any Ollama model, set `ollama_model` (e.g. `qwen3:8b`, `llama3.2:3b`).
- `"openai"`: any OpenAI-compatible endpoint. Set `openai_base_url`, `openai_model`, `openai_api_key` (LM Studio: `http://127.0.0.1:1234/v1`; OpenAI: `https://api.openai.com/v1`; OpenRouter: `https://openrouter.ai/api/v1`).
- `"template"`: no LLM, simple built-in plan from the idea text.

No restart needed; config is re-read on every request.

When Claude isn't available on the PC, `llm.local_rewrite` decides what happens:
- `false` (default): Plan scenes uses your prompt exactly as written (a single picture gets it word for word; several keyframes get it plus a shot type each, so they differ), and Improve says Claude is needed.
- `true`: the local model above rewrites the idea into scenes and Improve uses it too.

## Speed tips

- ComfyUI runs with `--disable-pinned-memory --cache-ram 6` so the 14B video model doesn't push Windows into heavy paging on 32 GB RAM.
- Close GPU-heavy apps (browsers with many tabs, games) before animating: the 8 GB card is tight.

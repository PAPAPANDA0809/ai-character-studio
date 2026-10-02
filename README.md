# AI Character Studio

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

- Image: FLUX.1-dev Q4_K_S GGUF, T5-XXL Q5_K_M GGUF, CLIP-L, Flux VAE. 20 steps.
- Image with a master character picture: FLUX.1 Kontext dev Q4_K_S GGUF (same encoders/VAE). Pick or upload the picture in the app; every keyframe then keeps that character.
- Video: Wan 2.2 I2V A14B high-noise + low-noise Q4_K_S GGUF with the lightx2v 4-step LoRAs, UMT5-XXL Q5_K_M GGUF, Wan 2.1 VAE. Uses ComfyUI's `WanFirstLastFrameToVideo`.
- Text: Ollama `qwen3:4b` (unloaded after each call so the GPU is free).

## Switching the text model

In `app\config.json` → `llm.provider`:
- `"ollama"`: any Ollama model, set `ollama_model` (e.g. `qwen3:8b`, `llama3.2:3b`).
- `"openai"`: any OpenAI-compatible endpoint. Set `openai_base_url`, `openai_model`, `openai_api_key` (LM Studio: `http://127.0.0.1:1234/v1`; OpenAI: `https://api.openai.com/v1`; OpenRouter: `https://openrouter.ai/api/v1`).
- `"template"`: no LLM, simple built-in plan from the idea text.

No restart needed; config is re-read on every request.

## Speed tips

- ComfyUI runs with `--disable-pinned-memory --cache-ram 6` so the 14B video model doesn't push Windows into heavy paging on 32 GB RAM.
- Close GPU-heavy apps (browsers with many tabs, games) before animating: the 8 GB card is tight.

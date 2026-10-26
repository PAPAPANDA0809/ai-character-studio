# Reelmint

Local pipeline: post idea → scene plan (local LLM) → Flux keyframes → Wan 2.2 first-last-frame clips (5 s each) → stitched MP4 + caption + subtitles.

## Launch

Double-click `start_studio.bat`. It starts ComfyUI (if it isn't running) in its own window, starts the web app through Windows Task Scheduler and opens http://127.0.0.1:7860.

The **Reelmint Background Server** task runs independently of ChatGPT, starts at Windows sign-in, and restarts the web server if it exits. The task does not rent GPUs or submit generation jobs. The PC must remain on and awake for phone access. The web server does not launch ComfyUI automatically at sign-in; use `start_studio.bat` when local generation is needed.

To install or refresh background startup, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File setup\enable_background.ps1` from the repository root. For troubleshooting, see `app/reelmint-supervisor.log` and `app/reelmint-server.log`. To turn it off, disable and end **Reelmint Background Server** in Windows Task Scheduler.

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

### Krea 2: GPU and hosted API

Open **Advanced settings → Image engine → Krea setup** on Create or in a project's Settings.

- **Krea 2 Turbo · GPU** uses native ComfyUI nodes, FP8 weights, Euler/simple sampling, 8 steps and CFG 1. New Fast GPU rentals download its three model files in the background after the video and full-quality Qwen files. It becomes selectable once those files are ready. Existing rentals created before this integration do not gain the new download automatically; reconnect when no jobs are running. A 24 GB+ GPU is recommended. The local 8 GB card needs substantial CPU/RAM offloading and is not the recommended target.
- For local installation, run `python setup/download_models.py setup/krea2.txt` **from the repository root**. This downloads about 19 GB from the official Comfy-Org repository. Model paths are listed in `setup/krea2.txt`. Recent ComfyUI with `CLIPLoader` type `krea2` is required; no custom nodes are needed.
- **Krea 2 Turbo / Medium / Large · API** run without ComfyUI or a rented GPU. An administrator can save a shared server API key in Krea setup, or set `KREA_API_KEY`. Keys are stored in the ignored `app/secrets.json` as `krea_api_key`; they are never returned to the browser. Saving a key enables the options but does not validate its balance or spend credit.
- Listed base API prices are $0.015 / $0.03 / $0.06 per image (check [Krea pricing](https://www.krea.ai/app/api/pricing) for changes). API credit is separate from Krea subscriptions. The current API schema accepts native `1K`; higher Reelmint output sizes are resized, not native 2K/4K generations. Requests use the saved prompt with Krea prompt expansion disabled, track asynchronous jobs, and preserve the normal picture versions and downloads. An API failure never silently retries with a different paid provider.
- These initial Krea options generate from text. Character-reference generation and image editing continue through Qwen. Local Qwen LoRAs are not interchangeable with Krea weights.
- Stopping a hosted Krea request stops Reelmint waiting; a job already submitted may still complete and be billed by Krea. Check your Krea jobs before retrying a timed-out request.

References: [Krea API specification](https://api.krea.ai/openapi.json), [official ComfyUI workflow](https://docs.comfy.org/tutorials/image/krea/krea-2), and [Krea model licence](https://www.krea.ai/krea-2-licensing).

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

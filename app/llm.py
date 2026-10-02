"""Pluggable text model used for scene planning and captions.

Providers (set "llm.provider" in config.json):
  ollama   - local Ollama server (default, qwen3:4b)
  openai   - any OpenAI-compatible endpoint (LM Studio, llama.cpp server, OpenAI, OpenRouter...)
  template - no model at all; builds a simple plan from the idea text
"""
import glob
import json
import os
import re
import shutil
import subprocess
import urllib.request

PLAN_SYSTEM = """You are a creative director for short social media videos featuring a recurring character.
Given a post idea, you plan a sequence of KEYFRAMES. A video model will animate a 5-second clip
between each pair of consecutive keyframes, so consecutive keyframes must be visually continuous
(same character, same outfit, similar location, only a moderate change in pose, camera or action).

Return ONLY a JSON object with this shape:
{
  "title": "short title",
  "character": "one detailed sentence describing the character's fixed look: age, face, hair, outfit, colours",
  "setting": "one sentence describing the main location and lighting",
  "keyframes": [
    {"image_prompt": "full visual description of this still frame (subject, pose, framing, background, lighting)",
     "subtitle": "short on-screen line for this moment (max 10 words)"}
  ],
  "motions": ["for each transition keyframe i -> i+1: describe the motion and camera movement in one sentence"],
  "caption": "engaging post caption, 1-3 sentences, may include emoji",
  "hashtags": ["#tag1", "#tag2"]
}
The number of motions must be exactly one less than the number of keyframes.
Do not mention the character's name inside image_prompt; describe the look instead."""


def _post(url, payload, headers=None, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(cfg, system, user, want_json=True):
    p = cfg["provider"]
    if p == "ollama":
        body = {
            "model": cfg["ollama_model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": False,
            "options": {"temperature": cfg.get("temperature", 0.7), "num_ctx": 8192},
            # unload right away so the GPU is free for ComfyUI
            "keep_alive": 0,
        }
        if want_json:
            body["format"] = "json"
        return _post(cfg["ollama_url"].rstrip("/") + "/api/chat", body)["message"]["content"]
    if p == "openai":
        body = {
            "model": cfg["openai_model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": cfg.get("temperature", 0.7),
        }
        if want_json:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": "Bearer " + cfg["openai_api_key"]} if cfg.get("openai_api_key") else {}
        out = _post(cfg["openai_base_url"].rstrip("/") + "/chat/completions", body, headers)
        return out["choices"][0]["message"]["content"]
    raise ValueError("unknown llm provider: " + p)


ENHANCE_SYSTEM = """You turn a rough social media post idea into a stronger creative brief for an AI image/video generator.
Keep the user's intent, characters and language. Make it specific and visual: who the character is, what happens
(a clear beginning, middle and payoff), the setting, mood, lighting, camera feel and the hook for the post.
Write 3-6 sentences of plain prose. No headings, no lists, no preamble - return only the improved brief."""


def _secret(name):
    """Keys live in app/secrets.json (never committed), e.g. {"claude_api_key": "sk-ant-..."}."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json"), encoding="utf-8") as f:
            return json.load(f).get(name, "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def _claude_cli():
    """Path to the Claude Code CLI: a standalone install on PATH, else the copy bundled with the Claude desktop app."""
    found = shutil.which("claude") or next(iter(glob.glob(os.path.expanduser(r"~\.local\bin\claude.exe"))), None)
    if found:
        return found
    # the desktop app is a packaged (MSIX) app, so outside it the copy lives under LocalCache
    bundled = sorted(glob.glob(os.path.expandvars(r"%APPDATA%\Claude\claude-code\*\*\claude.exe")) +
                     glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\claude-code\*\*\claude.exe")),
                     key=os.path.getmtime)
    return bundled[-1] if bundled else None


def enhance(cfg, idea, mode="video", style="", character=""):
    """Rewrite a rough idea into a richer brief: Claude Code on this PC (owner's login), else a Claude API key,
    else the local LLM.
    Returns (text, engine_label)."""
    what = {"images": "a photo set / carousel", "t2v": "a short text-to-video", "video": "a short video built from keyframes"}
    user = f"Idea: {idea}\nFormat: {what.get(mode, 'a short video')}\n"
    if style:
        user += f"Visual style: {style}\n"
    if character:
        user += f"Character: {character}\n"
    cli = _claude_cli()
    if cli:  # Claude Code on this PC, signed in with the owner's Claude account
        r = subprocess.run([cli, "-p", "--output-format", "text", "--max-turns", "1"],
                           input=ENHANCE_SYSTEM + "\n\n" + user, capture_output=True, text=True, encoding="utf-8",
                           timeout=180, cwd=os.path.dirname(os.path.abspath(__file__)))
        out = (r.stdout or "").strip()
        if r.returncode == 0 and out:
            return out, "Claude (your Claude login)"
        if "log" not in (out + (r.stderr or "")).lower():  # anything other than "not logged in" is a real error
            raise RuntimeError("Claude Code failed: " + (r.stderr or out)[-300:])
    key = os.environ.get("ANTHROPIC_API_KEY") or _secret("claude_api_key")
    if key:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        model = cfg.get("claude_model", "claude-opus-5-5")
        try:
            resp = client.beta.messages.create(
                model=model, max_tokens=2000, system=ENHANCE_SYSTEM,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except anthropic.AuthenticationError:
            raise RuntimeError("Claude API key was rejected; check claude_api_key in app/secrets.json")
        except anthropic.RateLimitError:
            raise RuntimeError("Claude is rate-limited right now; try again in a minute")
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Claude API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            raise RuntimeError("couldn't reach the Claude API (no internet?)")
        if resp.stop_reason == "refusal":
            raise RuntimeError("Claude declined to rewrite this idea")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text, f"Claude ({resp.model})"
    if cfg["provider"] == "template":
        return idea, "no AI configured"
    text = chat(cfg, ENHANCE_SYSTEM, user, want_json=False)
    text = text.split("</think>")[-1].strip()  # small local models sometimes leak their reasoning
    return text, f"local {cfg.get('ollama_model') if cfg['provider'] == 'ollama' else cfg.get('openai_model')}"


def _extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model did not return JSON:\n" + text[:500])
    return json.loads(m.group(0))


def template_plan(idea, n, character=""):
    character = character or "a friendly young content creator with short dark hair, wearing a casual denim jacket"
    beats = ["opening wide shot", "medium shot, starting the action", "close-up, reacting",
             "medium shot, continuing the action", "wide shot, finishing", "close-up, smiling at the camera"]
    kfs = [{"image_prompt": f"{idea}, {beats[i % len(beats)]}", "subtitle": ""} for i in range(n)]
    return {
        "title": idea[:60], "character": character, "setting": "",
        "keyframes": kfs,
        "motions": ["smooth natural movement, slow camera push in"] * (n - 1),
        "caption": idea, "hashtags": [],
    }


REFERENCE_RULE = ("IMPORTANT: the character's appearance comes from a reference picture. In image_prompt and motions, "
                  "call them only 'the character' and NEVER describe their face, fur, hair, body, clothing, hats, "
                  "glasses or accessories; describe only pose, action, expression, framing, location and lighting. "
                  "Set \"character\" to an empty string.")


IMAGES_RULE = ("This is a PHOTO SET (carousel post), not a video: the keyframes are independent images of the same "
               "character with varied poses, framing and moments. Return an empty \"motions\" list.")


T2V_RULE = ("This is a TEXT-TO-VIDEO post: each keyframe is a separate 5-second video scene. In image_prompt describe "
            "the scene AND the action/camera movement that happens during those 5 seconds, fully self-contained "
            "(repeat the character's look in every scene). Return an empty \"motions\" list.")


def plan(cfg, idea, n_keyframes, style="", character="", has_reference=False, mode="video"):
    if cfg["provider"] == "template":
        p = template_plan(idea, n_keyframes, "" if has_reference else character)
        if has_reference:
            for k in p["keyframes"]:
                k["image_prompt"] = "the character: " + k["image_prompt"]
        return p
    user = f"Post idea: {idea}\nNumber of keyframes: {n_keyframes}\n"
    if mode == "images":
        user += IMAGES_RULE + "\n"
    elif mode == "t2v":
        user += T2V_RULE + "\n"
    if style:
        user += f"Visual style: {style}\n"
    if has_reference:
        user += REFERENCE_RULE + "\n"
        if character:
            user += f"(Who the character is, for context only: {character})\n"
    elif character:
        user += f"The character must look like this: {character}\n"
    last_err = None
    for _ in range(3):
        try:
            data = _extract_json(chat(cfg, PLAN_SYSTEM, user))
            if has_reference:
                data["character"] = ""
            return normalize_plan(data, n_keyframes, idea, "" if has_reference else character)
        except Exception as e:  # retry on malformed output
            last_err = e
    raise RuntimeError(f"planning failed: {last_err}")


def normalize_plan(data, n, idea, character=""):
    kfs = data.get("keyframes") or []
    kfs = [k if isinstance(k, dict) else {"image_prompt": str(k)} for k in kfs]
    kfs = [{"image_prompt": str(k.get("image_prompt") or idea), "subtitle": str(k.get("subtitle") or "")} for k in kfs]
    while len(kfs) < n:
        kfs.append(dict(kfs[-1]) if kfs else {"image_prompt": idea, "subtitle": ""})
    kfs = kfs[:n]
    motions = [str(m) for m in (data.get("motions") or [])]
    while len(motions) < max(0, n - 1):
        motions.append("smooth natural movement, gentle camera motion")
    tags = data.get("hashtags") or []
    if isinstance(tags, str):
        tags = tags.split()
    return {
        "title": data.get("title") or idea[:60],
        "character": character or data.get("character", ""),
        "setting": data.get("setting", ""),
        "keyframes": kfs,
        "motions": motions[: max(0, n - 1)],
        "caption": data.get("caption", ""),
        "hashtags": [t if t.startswith("#") else "#" + t for t in map(str, tags)],
    }

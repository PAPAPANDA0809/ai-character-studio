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
import urllib.parse
import urllib.request

PLAN_SYSTEM = """You are a creative director for short social media posts built around one main subject: a person, an
animal, a product, food, an object or a place - whatever the post idea is about. Never add people the idea doesn't ask for.
Given a post idea, you plan a sequence of KEYFRAMES. A video model will animate a 5-second clip
between each pair of consecutive keyframes, so consecutive keyframes must be visually continuous
(same subject, same look, similar location, only a moderate change in pose, camera or action).

Return ONLY a JSON object with this shape:
{
  "title": "short title",
  "character": "one detailed sentence describing the main subject's fixed look (for a person: age, face, hair, outfit, colours; for anything else: shape, colours, materials); empty if there is no recurring subject",
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


# Servers on this PC (Ollama, LM Studio) are reached directly: a Windows system proxy (VPN app) can break them.
# Internet endpoints keep using the system proxy.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def open_url(req, timeout):
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    local = urllib.parse.urlparse(url).hostname in ("127.0.0.1", "localhost", "::1")
    return (DIRECT.open if local else urllib.request.urlopen)(req, timeout=timeout)


def _post(url, payload, headers=None, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with open_url(req, timeout) as r:
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
Keep the user's intent, subject and language. The subject may be a person, an animal, a product, food, an object or a
place; never add people the idea doesn't ask for. Write it the way an image model reads best: simple, concrete
words - who or what, what they do, what they wear, where, the light and the framing (close-up / half body / full
body). Only describe what the framing shows: for half body or closer, leave out shoes, socks and anything else below
the waist (the generator squeezes them into the frame, e.g. a sneaker floating at the hip). Keep poses simple and
natural. Leave out metaphors, feelings about the post, "the hook", camera brands and
tiny details (single strands of hair, condensation, exact finger positions) - each extra detail is a chance for
the generator to get anatomy wrong. Follow the Format line exactly: a single photo is ONE moment in ONE frame
(never a sequence, several shots, panels or a beginning/middle/end, or the generator draws a collage); a photo set
gives each photo its own moment; a video has a short beginning, middle and end.
Length: 40-70 words for photos, at most 100 words for videos. One plain paragraph. No headings, no lists, no
preamble - return only the improved prompt."""


def _secret(name):
    """Keys live in app/secrets.json (never committed), e.g. {"claude_api_key": "sk-ant-..."}."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json"), encoding="utf-8") as f:
            return json.load(f).get(name, "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


# Text in, text out: no tools (any signed-in user's idea reaches this prompt), no saved session per call.
CLI_ARGS = ["-p", "--output-format", "text", "--max-turns", "1", "--tools", "", "--no-session-persistence"]


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


def enhance(cfg, idea, mode="video", style="", character="", has_reference=False, count=None):
    """Rewrite a rough idea into a richer brief: Claude Code on this PC (owner's login), else a Claude API key,
    else the local LLM. count = number of pictures/scenes the project will have.
    Returns (text, engine_label)."""
    n = int(count) if str(count or "").isdigit() else None
    if mode == "images":
        fmt = ("ONE single photo: one moment, one frame" if n == 1 else
               f"a photo set of {n} separate photos" if n else "a photo set / carousel")
    elif mode == "t2v":
        fmt = "ONE 5-second video scene" if n == 1 else f"a short text-to-video of {n} scenes" if n else "a short text-to-video"
    else:
        fmt = "a short video built from keyframes"
    user = f"Idea: {idea}\nFormat: {fmt}\n"
    if style:
        user += f"Visual style: {style}\n"
    if character:
        user += f"Character: {character}\n"
    if has_reference:
        user += ("The main subject's look comes from a reference picture. It may be a person, an animal, a product, "
                 "food or an object - go by the idea and do NOT add a person (hands, model, someone holding it) unless "
                 "the idea asks for one. Don't describe the subject's face, hair, body, shape or design. If it is a "
                 "person and the idea mentions clothing or an outfit (even vaguely, like 'a summer outfit'), describe "
                 "it concretely: each garment, colour, fabric, shoes and accessories - otherwise the generator copies "
                 "the outfit from the reference picture. Also describe the scene, placement or pose, action, mood "
                 "and camera.\n")
        user += MULTI_REF_RULE.format(n=int(has_reference)) if int(has_reference) > 1 else ""
    try:
        text, label = _claude(cfg, ENHANCE_SYSTEM, user)
    except RuntimeError as e:
        if not cfg.get("uncensored_fallback"):
            raise RuntimeError(str(e).replace("declined this prompt", "declined to rewrite this idea"))
        text, label = None, ""
    if text and not refused(text):
        return text, label
    if cfg.get("uncensored_fallback"):  # Claude declined or isn't there: the uncensored local Qwen writes it
        label = f"local {cfg.get('ollama_model')}" + (" (Claude declined)" if text else "")
        return local_write(cfg, ENHANCE_SYSTEM, user), label
    if cfg["provider"] == "template":
        return idea, "no AI configured"
    if not cfg.get("local_rewrite", False):
        raise RuntimeError("Improve needs Claude, which isn't available on this PC right now. "
                           "Plan scenes will use your prompt exactly as written.")
    text = chat(cfg, ENHANCE_SYSTEM, user, want_json=False)
    text = text.split("</think>")[-1].strip()  # small local models sometimes leak their reasoning
    return text, f"local {cfg.get('ollama_model') if cfg['provider'] == 'ollama' else cfg.get('openai_model')}"


# MiniMax's own prompt format (docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md and _ref_en.md in MiniMaxAI/MiniMax-H3,
# condensed): H3 makes the sound too, and follows shots, camera moves and dialogue written this way far better
# than a one-line motion. Claude writes it from the project's plan; without Claude the plain prompt is used.
H3_SYSTEM = """You write the prompt for MiniMax H3, a video model that generates the picture AND the sound together,
following MiniMax's official prompt format exactly. You get the scene the user planned (subject, action, setting,
style) and must turn it into one H3 prompt. Keep the user's subject, action, setting, wording and intent; add only
what makes it filmable: concrete shots, camera moves, lighting, sound. Never add people the scene doesn't have.
Write in English (dialogue stays in its own language inside <d>). Return ONLY the prompt text, no preamble.

FORMAT (text-to-video):
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

With a first-frame picture (image-to-video), the first line is exactly:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
then a blank line, then the three fields. [Shot 1] continues from what <Picture 1> shows (same person, clothes,
composition, scene), then the action develops.

With first AND last pictures (first-and-last-frame), the first line is exactly:
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the S.SS-second mark of the target video.
(S.SS = the clip length with two decimals), then a blank line, then the three fields. Use ONE shot: describe the
motion path from the state in Picture 1 to the pose, spacing and composition of Picture 2 at the end.

With character/reference pictures (reference-to-video) use the six-field format instead:
subject_definitions:
<Subject 1> is the person in <Picture 1>: ...   (one line per picture; <Subject 2> for <Picture 2> ...)

summary:
[reference generation] one short paragraph: the target video and how the subjects are used.

retention_analysis:
<Subject 1> (appears in [Shot 1], ...): fully_preserved - face, hair, body and clothes are retained.

detailed_description:
One or two sentences of style, then [Shot 1] ... (name <Subject N> where each appears; 150-300 words)

overall_soundscape: ...

non_diegetic_music: ...

RULES
- integrated_multimodal_description / detailed_description is a timeline. Start [Shot 1] with the style (Live-action,
  cinematic / 2D-animated / 3D CG ...) and the opening composition, then the subject, its look and position, the
  action, the camera and the sounds that happen in the shot.
- [Shot 1] has no timestamp. Later shots: "[Shot 2] At 00:03.500, the camera cuts to ..." with strictly increasing
  times inside the clip length. Never write time ranges like [0s-3s]. Cut only when subject, space, state, viewpoint
  or time changes; a reframe is camera motion inside the shot. A 5-second clip is usually 1-2 shots.
- Camera moves as natural sentences with type (+ amplitude + speed only when they matter): zoom in/out, push in /
  pull out, pan left/right, truck left/right, tilt up/down, pedestal up/down, arc shot, tracking shot, static shot,
  POV, shake slightly/strongly; "with small/large amplitude", "at slow/fast speed". E.g. "The camera pushes in with
  small amplitude at slow speed toward her face."
- Dialogue or singing: give the speaker a stable ID the first time with voice details, then the words inside <d>
  with a language tag: The young woman with a soft, low voice (S1) says: <d>[English] I get off at the next station.</d>
  Only use dialogue the scene asks for. Voiceover: "says in an off-screen voiceover" and state the lips stay closed.
- On-screen text goes in English double quotes, verbatim.
- overall_soundscape: 1-4 sentences of ambience, action sounds and non-verbal human sounds (wind, footsteps, fabric,
  breathing, laughter). No dialogue or music here. N/A only for a silent clip.
- non_diegetic_music: 1-3 sentences on instruments, tempo, rhythm and dynamics of the score only (no mood words);
  N/A when the scene wants none. Music the characters hear (radio, singing) belongs in the shots instead.
- Describe what should appear; never list what to avoid (there is no negative prompt).
- Length: 80-200 words for the main field of a text/image clip; nothing beyond the fields."""


def h3_request(task, scene, seconds, character="", style="", refs=0, motion=""):
    """The user message for H3_SYSTEM: task = t2va | i2va | fl2va | ref2va."""
    kind = {"t2va": "Text-to-video (no pictures).",
            "i2va": "Image-to-video: <Picture 1> is the first frame.",
            "fl2va": "First-and-last-frame: Picture 1 is the first frame, Picture 2 the last frame.",
            "ref2va": f"Reference-to-video with {refs} character picture(s) <Picture 1>..<Picture {refs}>: the main "
                      "subject's face, hair and body come from <Picture 1>; don't describe the face, keep its clothes "
                      "unless the scene says otherwise."}[task]
    user = f"Task: {kind}\nClip length: {seconds:.2f} seconds\nScene: {scene}\n"
    if motion:
        user += f"Motion and camera for this clip: {motion}\n"
    if character:
        user += f"Character (fixed look): {character}\n"
    if style:
        user += f"Visual style: {style}\n"
    return user


def h3_prompt(cfg, task, scene, seconds, character="", style="", refs=0, motion=""):
    """An H3-format prompt from Claude, or None when Claude isn't available or declines (the caller then sends the
    plain prompt)."""
    user = h3_request(task, scene, seconds, character, style, refs, motion)
    try:
        text, _ = _claude(cfg, H3_SYSTEM, user, timeout=240)
    except RuntimeError:
        text = None
    if not _h3_shaped(text) and cfg.get("uncensored_fallback"):  # declined or missing: the local Qwen tries
        try:
            text = local_write(cfg, H3_SYSTEM, user)
        except (RuntimeError, ValueError, OSError):
            text = None
    return text.strip() if _h3_shaped(text) else None


def _h3_shaped(text):
    return bool(text) and ("integrated_multimodal_description:" in text or "detailed_description:" in text)


REFUSAL = re.compile(r"^\W*(?:I'?m sorry|I am sorry|sorry|I can(?:'|no)t|I cannot|I won'?t|I'?m (?:not able|unable)|"
                     r"I am (?:not able|unable)|I'?m not comfortable|I don'?t (?:feel comfortable|think I can))", re.I)


def refused(text):
    """Claude sometimes answers a declined request in plain words instead of a refusal stop."""
    return bool(REFUSAL.search((text or "")[:200]))


def local_write(cfg, system, user):
    """The local Ollama model (an uncensored Qwen3) writes the text when Claude declines. Reelmint's minors check runs
    on what goes in and on what comes out, whatever image engine the project uses."""
    import krea
    krea.validate_prompt(user)
    try:
        text = chat(cfg, system, user, want_json=False)
    except OSError as e:
        raise RuntimeError(f"the local model ({cfg.get('ollama_model')}) isn't reachable; is Ollama running? ({e})")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).split("</think>")[-1].strip()
    if not text:
        raise RuntimeError("the local model returned nothing")
    krea.validate_prompt(text)
    return text


def _claude(cfg, system, user, timeout=180):
    """(text, label) from Claude Code on this PC (owner's login), else the Claude API key; (None, "") when neither
    is available. Raises RuntimeError for a real failure (rate limit, refusal, rejected key...)."""
    cli = _claude_cli()
    if cli:
        r = subprocess.run([cli, *CLI_ARGS], input=system + "\n\n" + user, capture_output=True, text=True,
                           encoding="utf-8", timeout=timeout, cwd=os.path.dirname(os.path.abspath(__file__)))
        out = (r.stdout or "").strip()
        if r.returncode == 0 and out:
            return out, "Claude (your Claude login)"
        if "log" not in (out + (r.stderr or "")).lower():  # anything other than "not logged in" is a real error
            raise RuntimeError("Claude Code failed: " + (r.stderr or out)[-300:])
    key = os.environ.get("ANTHROPIC_API_KEY") or _secret("claude_api_key")
    if not key:
        return None, ""
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    try:
        resp = client.beta.messages.create(
            model=cfg.get("claude_model", "claude-opus-5-5"), max_tokens=2000, system=system,
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
        raise RuntimeError("Claude declined this prompt")
    return "".join(b.text for b in resp.content if b.type == "text").strip(), f"Claude ({resp.model})"


def _extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model did not return JSON:\n" + text[:500])
    return json.loads(m.group(0))


def template_plan(idea, n, character="", default_character=True):
    if default_character:
        character = character or "a friendly young content creator with short dark hair, wearing a casual denim jacket"
    beats = ["opening wide shot", "medium shot, starting the action", "close-up, reacting",
             "medium shot, continuing the action", "wide shot, finishing", "close-up, smiling at the camera"]
    kfs = [{"image_prompt": idea if n == 1 else f"{idea}, {beats[i % len(beats)]}", "subtitle": ""} for i in range(n)]
    return {
        "title": idea[:60], "character": character, "setting": "",
        "keyframes": kfs,
        "motions": ["smooth natural movement, slow camera push in"] * (n - 1),
        "caption": idea, "hashtags": [],
    }


REFERENCE_RULE = ("IMPORTANT: the main subject's appearance comes from a reference picture. It may be a person, an animal, "
                  "a product, food or an object - go by the post idea, and NEVER add a person (hands, model, someone "
                  "holding it) the idea doesn't ask for. In image_prompt and motions call it only 'the subject' and do "
                  "NOT describe its face, fur, hair, body, shape or design. If it is a person and the post idea mentions "
                  "an outfit, clothing or accessories (even vaguely), describe that outfit concretely - each garment, "
                  "colour, fabric, shoes, accessories - and repeat it in every image_prompt (otherwise don't mention "
                  "clothing); also describe pose or placement, action, framing, location and lighting. "
                  "Set \"character\" to an empty string.")

# has_reference is the number of character pictures; with more than one the generator sees them as Picture 1, 2, 3
MULTI_REF_RULE = ("There are {n} reference pictures, in order: picture 1 is the main subject; the others are extra "
                  "references (another view of the subject, an outfit, a product, a place). Where the idea uses one "
                  "of them (e.g. 'the dress from picture 2'), keep that exact wording 'picture 2' / 'picture 3' and say "
                  "what to take from it, without describing what it looks like.\n")


IMAGES_RULE = ("This is a PHOTO SET (carousel post), not a video: the keyframes are independent images of the same "
               "subject with varied poses, angles, framing and moments. Return an empty \"motions\" list.")


T2V_RULE = ("This is a TEXT-TO-VIDEO post: each keyframe is a separate 5-second video scene. In image_prompt describe "
            "the scene AND the action/camera movement that happens during those 5 seconds, fully self-contained "
            "(repeat the subject's look in every scene). Return an empty \"motions\" list.")


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
        user += MULTI_REF_RULE.format(n=int(has_reference)) if int(has_reference) > 1 else ""
        if character:
            user += f"(Who the character is, for context only: {character})\n"
    elif character:
        user += f"The character must look like this: {character}\n"
    user += ("Follow the post idea closely: every keyframe must come from it (same subject, setting, actions, "
             "props, mood and hook). Split it into the moments it describes, reusing its own wording for each "
             "moment; do not invent a different story, new props or new characters.\n")
    cli = _claude_cli() if cfg.get("claude_planner", True) else None
    if cli:  # Claude on this PC plans the scenes too, so the plan matches the improved prompt
        try:
            r = subprocess.run([cli, *CLI_ARGS],
                               input=PLAN_SYSTEM + "\n\n" + user, capture_output=True, text=True, encoding="utf-8",
                               timeout=240, cwd=os.path.dirname(os.path.abspath(__file__)))
            if r.returncode == 0 and r.stdout.strip():
                data = _extract_json(r.stdout)
                if has_reference:
                    data["character"] = ""
                plan_ = normalize_plan(data, n_keyframes, idea, "" if has_reference else character)
                if n_keyframes == 1 and mode != "video":
                    # a single picture/scene IS the user's prompt, word for word (also with a character picture);
                    # Claude only writes title/caption/hashtags. Its character/setting text would be appended to the
                    # prompt, so only what the user typed in the Character box is kept.
                    plan_["keyframes"][0]["image_prompt"] = idea
                    plan_["character"] = "" if has_reference else character
                    plan_["setting"] = ""
                plan_["planner"] = "claude"
                return plan_
        except Exception:
            pass  # fall back below
    if not cfg.get("local_rewrite", False):  # no Claude: keep the user's own words, the local model rewrites nothing
        plan_ = template_plan(idea, n_keyframes, "" if has_reference else character, default_character=False)
        plan_["planner"] = "your own words (Claude not available)"
        return plan_
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

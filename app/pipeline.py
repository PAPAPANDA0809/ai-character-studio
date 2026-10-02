"""Project storage and the generation pipeline (plan -> keyframes -> clips -> stitched video + captions)."""
import glob
import json
import os
import queue
import random
import re
import shutil
import subprocess
import threading
import time
import traceback

import comfy
import llm

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS = os.path.join(APP_DIR, "projects")
CHARACTERS = os.path.join(APP_DIR, "characters")
MODELS = os.path.join(os.path.dirname(APP_DIR), "models")
os.makedirs(PROJECTS, exist_ok=True)
os.makedirs(CHARACTERS, exist_ok=True)

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def list_characters():
    return sorted(f for f in os.listdir(CHARACTERS) if f.lower().endswith(IMG_EXT))


def save_character(name, data):
    base, ext = os.path.splitext(name)
    base = re.sub(r"[^\w\- ]+", "", base).strip() or "character"
    ext = ext.lower() if ext.lower() in IMG_EXT else ".png"
    fname = base + ext
    with open(os.path.join(CHARACTERS, fname), "wb") as f:
        f.write(data)
    return fname


def load_config():
    with open(os.path.join(APP_DIR, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def find_ffmpeg(name):
    found = shutil.which(name)
    if found:
        return found
    pattern = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\*\bin\ffmpeg.exe")
    hits = glob.glob(pattern)
    if hits:
        return hits[0]
    raise FileNotFoundError("ffmpeg not found; install it or set 'ffmpeg' in config.json")


# ---------------------------------------------------------------- storage

def pdir(pid, *parts):
    return os.path.join(PROJECTS, pid, *parts)


def load(pid):
    with open(pdir(pid, "project.json"), encoding="utf-8") as f:
        return json.load(f)


_save_lock = threading.Lock()


def save(p):
    with _save_lock:
        tmp = pdir(p["id"], "project.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2, ensure_ascii=False)
        os.replace(tmp, pdir(p["id"], "project.json"))


def list_projects():
    out = []
    for d in sorted(os.listdir(PROJECTS), reverse=True):
        try:
            p = load(d)
            out.append({"id": p["id"], "title": p["plan"]["title"], "status": p["status"], "created": p["created"]})
        except Exception:
            pass
    return out


def gallery(kind):
    """Every generated image (kind='images') or finished/clip video (kind='videos'), newest first."""
    items = []
    for d in os.listdir(PROJECTS):
        try:
            p = load(d)
        except Exception:
            continue
        title = p["plan"]["title"]
        if kind == "images":
            for f in sorted(glob.glob(pdir(d, "keyframes", "kf_*.png"))):
                items.append({"project": d, "title": title, "path": "keyframes/" + os.path.basename(f),
                              "mtime": os.path.getmtime(f), "label": f"{title} #{int(f[-6:-4]) + 1}"})
        else:
            finals = [f for f in ("final_subtitled.mp4", "final.mp4") if os.path.exists(pdir(d, f))]
            if finals:
                f = pdir(d, finals[0])
                items.append({"project": d, "title": title, "path": finals[0], "mtime": os.path.getmtime(f),
                              "label": title, "final": True})
            else:  # unfinished video projects still show their clips
                for f in sorted(glob.glob(pdir(d, "clips", "clip_*.mp4"))):
                    items.append({"project": d, "title": title, "path": "clips/" + os.path.basename(f),
                                  "mtime": os.path.getmtime(f), "label": f"{title} clip {int(f[-6:-4]) + 1}"})
    return sorted(items, key=lambda x: -x["mtime"])


def log(p, msg):
    p.setdefault("log", []).append(time.strftime("%H:%M:%S ") + msg)
    p["log"] = p["log"][-200:]
    p["message"] = msg
    save(p)


# ---------------------------------------------------------------- job queue (one GPU => one worker)

jobs = queue.Queue()
current = {"pid": None, "task": None}


def enqueue(pid, task, **kw):
    p = load(pid)
    p["status"] = "queued:" + task
    save(p)
    jobs.put((pid, task, kw))


def worker():
    while True:
        pid, task, kw = jobs.get()
        current.update(pid=pid, task=task)
        p = load(pid)
        try:
            p["status"] = "running:" + task
            save(p)
            {"keyframes": do_keyframes, "animate": do_animate, "stitch": restitch, "t2v": do_t2v}[task](p, **kw)
            p = load(pid)
            p["status"] = "done:" + task
            log(p, f"{task} finished")
        except Exception as e:
            p = load(pid)
            p["status"] = "error:" + task
            log(p, f"ERROR in {task}: {e}")
            traceback.print_exc()
        finally:
            current.update(pid=None, task=None)


def _reset_stale():
    """Jobs don't survive a restart; mark anything left queued/running as interrupted."""
    for d in os.listdir(PROJECTS):
        try:
            p = load(d)
            if p["status"].startswith(("queued", "running")):
                p["status"] = "interrupted:" + p["status"].split(":", 1)[1]
                save(p)
        except Exception:
            pass


_reset_stale()
threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- steps

def create_project(idea, n_keyframes, style, character, aspect, seed=None, reference="",
                   image_engine=None, video_engine=None, mode="video"):
    cfg = load_config()
    mode = mode if mode in ("images", "t2v") else "video"
    n_keyframes = max(2 if mode == "video" else 1, min(int(n_keyframes), 12 if mode == "images" else 8))
    reference = reference if reference in list_characters() else ""
    plan = llm.plan(cfg["llm"], idea, n_keyframes, style, character, has_reference=bool(reference), mode=mode)
    pid = time.strftime("%Y%m%d-%H%M%S-") + re.sub(r"[^a-z0-9]+", "-", plan["title"].lower()).strip("-")[:30]
    os.makedirs(pdir(pid, "keyframes"))
    os.makedirs(pdir(pid, "clips"))
    p = {
        "id": pid, "created": time.time(), "idea": idea, "style": style, "aspect": aspect,
        "seed": int(seed) if seed not in (None, "") else random.randint(1, 2**31 - 1),
        "plan": plan, "keyframe_seeds": {}, "status": "planned", "log": [],
        "reference": reference,
        "image_engine": image_engine if image_engine in comfy.IMAGE_ENGINES else cfg.get("default_image_engine", "flux-dev"),
        "video_engine": video_engine if video_engine in comfy.VIDEO_ENGINES else cfg.get("default_video_engine", "wan22-14b"),
        "clip_fps": {}, "mode": mode,
        "t2v_engine": cfg.get("default_t2v_engine", "wan22-5b"),
    }
    save(p)
    log(p, f"planned {n_keyframes} {dict(images='images', t2v='scenes').get(mode, 'keyframes')} with {cfg['llm']['provider']}")
    return p


def engines():
    def ready(f):  # present and not still being written by a download
        path = os.path.join(MODELS, f)
        return os.path.exists(path) and time.time() - os.path.getmtime(path) > 90

    def entry(key, e):
        return {"id": key, "label": e["label"], "installed": all(ready(f) for f in e["files"])}
    cfg = load_config()
    return {
        "image": [entry(k, e) for k, e in comfy.IMAGE_ENGINES.items()],
        "video": [entry(k, e) for k, e in comfy.VIDEO_ENGINES.items()],
        "t2v": [entry(k, e) for k, e in comfy.T2V_ENGINES.items()],
        "default_t2v": cfg.get("default_t2v_engine", "wan22-5b"),
        "reference": entry("flux-kontext", comfy.REFERENCE_ENGINE),
        "default_image": cfg.get("default_image_engine", "flux-dev"),
        "default_video": cfg.get("default_video_engine", "wan22-14b"),
    }


def set_engines(pid, image_engine=None, video_engine=None, mode=None, t2v_engine=None):
    p = load(pid)
    if t2v_engine in comfy.T2V_ENGINES:
        p["t2v_engine"] = t2v_engine
    if mode in ("images", "video", "t2v"):
        if mode == "video" and len(p["plan"]["keyframes"]) < 2:
            raise ValueError("a video needs at least 2 keyframes; add one first")
        p["mode"] = mode
    if image_engine in comfy.IMAGE_ENGINES:
        p["image_engine"] = image_engine
    if video_engine in comfy.VIDEO_ENGINES:
        p["video_engine"] = video_engine
    save(p)
    return p


def set_reference(pid, reference):
    p = load(pid)
    p["reference"] = reference if reference in list_characters() else ""
    save(p)
    return p


def update_plan(pid, plan):
    p = load(pid)
    n = len(plan["keyframes"])
    plan["motions"] = (plan.get("motions") or [])[: max(0, n - 1)]
    while len(plan["motions"]) < n - 1:
        plan["motions"].append("smooth natural movement, gentle camera motion")
    p["plan"] = plan
    save(p)
    return p


def keyframe_prompt(p, i):
    pl = p["plan"]
    parts = [p.get("style", ""), pl["keyframes"][i]["image_prompt"], pl.get("character", ""), pl.get("setting", "")]
    return ". ".join(s.strip().rstrip(".") for s in parts if s and s.strip())


def reference_prompt(p, i):
    """Kontext prompt: describe the new scene and tell it to keep the reference character unchanged."""
    pl = p["plan"]
    scene = ". ".join(s.strip().rstrip(".") for s in [pl["keyframes"][i]["image_prompt"], pl.get("setting", "")]
                      if s and s.strip())
    style = p.get("style", "").strip()
    return (f"Place the exact same character from the reference image into a new scene: {scene}. "
            f"Keep the character's face, body proportions, colors and outfit identical to the reference. "
            + (f"Style: {style}." if style else ""))


def size(cfg, p):
    return cfg["sizes"].get(p.get("aspect", "portrait"), cfg["sizes"]["portrait"])


def do_keyframes(p, indices=None, reseed=False):
    cfg = load_config()
    c = comfy.Comfy(cfg["comfy_url"])
    if not c.alive():
        raise RuntimeError("ComfyUI is not running at " + cfg["comfy_url"])
    w, h = size(cfg, p)
    n = len(p["plan"]["keyframes"])
    indices = list(range(n)) if indices is None else [int(i) for i in indices]
    ref = p.get("reference")
    if ref:
        ref_name = c.upload_image(os.path.join(CHARACTERS, ref), "aics_ref_" + re.sub(r"[^\w.\-]", "_", ref))
    for i in indices:
        key = str(i)
        if reseed or key not in p["keyframe_seeds"]:
            # one shared seed for the first pass helps keep the character consistent
            p["keyframe_seeds"][key] = random.randint(1, 2**31 - 1) if reseed else p["seed"]
        log(p, f"keyframe {i + 1}/{n}: generating with {'flux-kontext' if ref else p.get('image_engine', 'flux-dev')}")
        t = time.time()
        prefix = f"aics/{p['id']}/kf{i:02d}"
        if ref:
            wf = comfy.kontext_workflow(cfg, reference_prompt(p, i), ref_name, w, h, p["keyframe_seeds"][key], prefix)
        else:
            wf = comfy.image_workflow(cfg, p.get("image_engine", "flux-dev"), keyframe_prompt(p, i), w, h,
                                      p["keyframe_seeds"][key], prefix)
        out = c.run(wf)
        img = next(o["images"][0] for o in out.values() if o.get("images"))
        with open(pdir(p["id"], "keyframes", f"kf_{i:02d}.png"), "wb") as f:
            f.write(c.fetch(img))
        p["keyframes_version"] = time.time()
        log(p, f"keyframe {i + 1}/{n}: done in {time.time() - t:.0f}s")
    write_caption(p)


def do_animate(p, only=None):
    cfg = load_config()
    ff = find_ffmpeg(cfg["ffmpeg"])
    c = comfy.Comfy(cfg["comfy_url"])
    if not c.alive():
        raise RuntimeError("ComfyUI is not running at " + cfg["comfy_url"])
    w, h = size(cfg, p)
    engine = p.get("video_engine", "wan22-14b")
    fps = {"ltxv-2b": cfg["ltxv"]["fps"], "hunyuan15": cfg["hunyuan15"]["fps"]}.get(engine, cfg["video"]["fps"])
    n = len(p["plan"]["keyframes"])
    for i in range(n):
        if not os.path.exists(pdir(p["id"], "keyframes", f"kf_{i:02d}.png")):
            raise RuntimeError(f"keyframe {i + 1} is missing; generate keyframes first")
    c.free()  # drop Flux from VRAM before loading Wan
    clips = range(n - 1) if only is None else [int(only)]
    for i in clips:
        log(p, f"clip {i + 1}/{n - 1}: uploading keyframes")
        a = c.upload_image(pdir(p["id"], "keyframes", f"kf_{i:02d}.png"), f"aics_{p['id']}_kf{i:02d}.png")
        b = c.upload_image(pdir(p["id"], "keyframes", f"kf_{i + 1:02d}.png"), f"aics_{p['id']}_kf{i + 1:02d}.png")
        prompt = ". ".join(x for x in [p["plan"]["motions"][i], p["plan"].get("character", ""), p.get("style", "")] if x)
        t = time.time()
        log(p, f"clip {i + 1}/{n - 1}: animating with {engine} (this is the slow part)")
        out = c.run(comfy.video_workflow(cfg, engine, prompt, a, b, w, h, p["seed"] + i, f"aics/{p['id']}/clip{i:02d}"))
        frames = next(o["images"] for o in out.values() if o.get("images"))
        fdir = pdir(p["id"], "clips", f"frames_{i:02d}")
        shutil.rmtree(fdir, ignore_errors=True)
        os.makedirs(fdir)
        for k, fr in enumerate(frames):
            with open(os.path.join(fdir, f"{k:04d}.png"), "wb") as f:
                f.write(c.fetch(fr))
        run_ffmpeg(ff, ["-framerate", str(fps), "-i", os.path.join(fdir, "%04d.png"), *H264,
                        pdir(p["id"], "clips", f"clip_{i:02d}.mp4")])
        p.setdefault("clip_fps", {})[str(i)] = fps
        p["clips_version"] = time.time()
        log(p, f"clip {i + 1}/{n - 1}: done in {time.time() - t:.0f}s ({len(frames)} frames)")
    if all(glob.glob(pdir(p["id"], "clips", f"frames_{i:02d}", "*.png")) for i in range(n - 1)):
        stitch(p, cfg, ff)
    else:
        log(p, "some clips are not animated yet; skipping stitch")


def do_t2v(p, only=None):
    """Video-only mode: each scene is generated straight from text, then the clips are joined."""
    cfg = load_config()
    ff = find_ffmpeg(cfg["ffmpeg"])
    c = comfy.Comfy(cfg["comfy_url"])
    if not c.alive():
        raise RuntimeError("ComfyUI is not running at " + cfg["comfy_url"])
    w, h = size(cfg, p)
    engine = p.get("t2v_engine", "wan22-5b")
    fps = cfg["ltxv"]["fps"] if engine == "ltxv-2b" else cfg["wan5b"]["fps"]
    n = len(p["plan"]["keyframes"])
    c.free()
    for i in (range(n) if only is None else [int(only)]):
        prompt = ". ".join(x for x in [p.get("style", ""), p["plan"]["keyframes"][i]["image_prompt"],
                                       p["plan"].get("character", ""), p["plan"].get("setting", "")] if x)
        t = time.time()
        log(p, f"scene {i + 1}/{n}: generating video with {engine}")
        out = c.run(comfy.t2v_workflow(cfg, engine, prompt, w, h, p["seed"] + i, f"aics/{p['id']}/scene{i:02d}"))
        frames = next(o["images"] for o in out.values() if o.get("images"))
        save_clip_frames(p, c, ff, i, frames, fps)
        log(p, f"scene {i + 1}/{n}: done in {time.time() - t:.0f}s ({len(frames)} frames)")
    if all(glob.glob(pdir(p["id"], "clips", f"frames_{i:02d}", "*.png")) for i in range(n)):
        stitch(p, cfg, ff)


def save_clip_frames(p, c, ff, i, frames, fps):
    fdir = pdir(p["id"], "clips", f"frames_{i:02d}")
    shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir)
    for k, fr in enumerate(frames):
        with open(os.path.join(fdir, f"{k:04d}.png"), "wb") as f:
            f.write(c.fetch(fr))
    run_ffmpeg(ff, ["-framerate", str(fps), "-i", os.path.join(fdir, "%04d.png"), *H264,
                    pdir(p["id"], "clips", f"clip_{i:02d}.mp4")])
    p.setdefault("clip_fps", {})[str(i)] = fps
    p["clips_version"] = time.time()


def restitch(p):
    cfg = load_config()
    stitch(p, cfg, find_ffmpeg(cfg["ffmpeg"]))


H264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart"]


def run_ffmpeg(ff, args, cwd=None):
    r = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", *args], capture_output=True, text=True, cwd=cwd)
    if r.returncode:
        raise RuntimeError("ffmpeg failed: " + r.stderr[-1500:])


def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_caption(p):
    pl = p["plan"]
    with open(pdir(p["id"], "caption.txt"), "w", encoding="utf-8") as f:
        caption = pl.get("caption", "").strip()
        tags = [t for t in pl.get("hashtags", []) if t.lower() not in caption.lower()]
        f.write(caption + ("\n\n" + " ".join(tags) if tags else "") + "\n")


def stitch(p, cfg, ff):
    n = len(p["plan"]["keyframes"])
    t2v = p.get("mode") == "t2v"
    nclips = n if t2v else n - 1
    clip_fps = [p.get("clip_fps", {}).get(str(i), cfg["video"]["fps"]) for i in range(nclips)]
    fps = max(clip_fps)  # clips from different engines are retimed to the fastest frame rate
    log(p, "stitching clips")
    seq = pdir(p["id"], "clips", "all_frames")
    shutil.rmtree(seq, ignore_errors=True)
    os.makedirs(seq)
    k = 0
    bounds = []  # (start_sec, end_sec) per clip on the final timeline
    for i in range(nclips):
        files = sorted(glob.glob(pdir(p["id"], "clips", f"frames_{i:02d}", "*.png")))
        if not files:
            raise RuntimeError(f"clip {i + 1} has no frames; animate it first")
        if clip_fps[i] != fps:  # repeat frames so the clip keeps its real duration
            files = [files[min(len(files) - 1, int(j * clip_fps[i] / fps))]
                     for j in range(round(len(files) * fps / clip_fps[i]))]
        if i > 0 and not t2v:
            files = files[1:]  # first frame duplicates the previous clip's last frame (same keyframe)
        start = k
        for fsrc in files:
            shutil.copyfile(fsrc, os.path.join(seq, f"{k:05d}.png"))
            k += 1
        bounds.append((start / fps, k / fps))

    # subtitles: keyframe i's line over clip i; the last clip is split between the last two lines
    kfs = p["plan"]["keyframes"]
    cues = []
    for i, (s, e) in enumerate(bounds):
        if not t2v and i == len(bounds) - 1 and kfs[-1].get("subtitle"):
            mid = (s + e) / 2
            cues += [(s, mid, kfs[i].get("subtitle", "")), (mid, e, kfs[-1]["subtitle"])]
        else:
            cues.append((s, e, kfs[i].get("subtitle", "")))
    cues = [c for c in cues if c[2].strip()]
    with open(pdir(p["id"], "subtitles.srt"), "w", encoding="utf-8") as f:
        for j, (s, e, text) in enumerate(cues, 1):
            f.write(f"{j}\n{srt_time(s)} --> {srt_time(e)}\n{text.strip()}\n\n")

    write_caption(p)

    run_ffmpeg(ff, ["-framerate", str(fps), "-i", os.path.join(seq, "%05d.png"), *H264, "final.mp4"],
               cwd=pdir(p["id"]))
    if cues:
        style = "Fontname=Arial,Fontsize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,MarginV=40"
        run_ffmpeg(ff, ["-i", "final.mp4", "-vf", f"subtitles=subtitles.srt:force_style='{style}'", *H264,
                        "final_subtitled.mp4"], cwd=pdir(p["id"]))
    shutil.rmtree(seq, ignore_errors=True)
    p["final_version"] = time.time()
    log(p, f"final video written ({k / fps:.1f}s)")

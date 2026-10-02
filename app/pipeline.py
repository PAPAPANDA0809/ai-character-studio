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
os.makedirs(PROJECTS, exist_ok=True)


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
            {"keyframes": do_keyframes, "animate": do_animate, "stitch": restitch}[task](p, **kw)
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


threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- steps

def create_project(idea, n_keyframes, style, character, aspect, seed=None):
    cfg = load_config()
    n_keyframes = max(2, min(int(n_keyframes), 8))
    plan = llm.plan(cfg["llm"], idea, n_keyframes, style, character)
    pid = time.strftime("%Y%m%d-%H%M%S-") + re.sub(r"[^a-z0-9]+", "-", plan["title"].lower()).strip("-")[:30]
    os.makedirs(pdir(pid, "keyframes"))
    os.makedirs(pdir(pid, "clips"))
    p = {
        "id": pid, "created": time.time(), "idea": idea, "style": style, "aspect": aspect,
        "seed": int(seed) if seed not in (None, "") else random.randint(1, 2**31 - 1),
        "plan": plan, "keyframe_seeds": {}, "status": "planned", "log": [],
    }
    save(p)
    log(p, f"planned {n_keyframes} keyframes with {cfg['llm']['provider']}")
    return p


def update_plan(pid, plan):
    p = load(pid)
    n = len(plan["keyframes"])
    plan["motions"] = (plan.get("motions") or [])[: n - 1]
    while len(plan["motions"]) < n - 1:
        plan["motions"].append("smooth natural movement, gentle camera motion")
    p["plan"] = plan
    save(p)
    return p


def keyframe_prompt(p, i):
    pl = p["plan"]
    parts = [p.get("style", ""), pl["keyframes"][i]["image_prompt"], pl.get("character", ""), pl.get("setting", "")]
    return ". ".join(s.strip().rstrip(".") for s in parts if s and s.strip())


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
    for i in indices:
        key = str(i)
        if reseed or key not in p["keyframe_seeds"]:
            # one shared seed for the first pass helps keep the character consistent
            p["keyframe_seeds"][key] = random.randint(1, 2**31 - 1) if reseed else p["seed"]
        log(p, f"keyframe {i + 1}/{n}: generating")
        t = time.time()
        out = c.run(comfy.flux_workflow(cfg, keyframe_prompt(p, i), w, h, p["keyframe_seeds"][key],
                                        f"aics/{p['id']}/kf{i:02d}"))
        img = next(o["images"][0] for o in out.values() if o.get("images"))
        with open(pdir(p["id"], "keyframes", f"kf_{i:02d}.png"), "wb") as f:
            f.write(c.fetch(img))
        p["keyframes_version"] = time.time()
        log(p, f"keyframe {i + 1}/{n}: done in {time.time() - t:.0f}s")


def do_animate(p, only=None):
    cfg = load_config()
    ff = find_ffmpeg(cfg["ffmpeg"])
    c = comfy.Comfy(cfg["comfy_url"])
    if not c.alive():
        raise RuntimeError("ComfyUI is not running at " + cfg["comfy_url"])
    w, h = size(cfg, p)
    fps = cfg["video"]["fps"]
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
        log(p, f"clip {i + 1}/{n - 1}: animating (this is the slow part)")
        out = c.run(comfy.wan_flf_workflow(cfg, prompt, a, b, w, h, p["seed"] + i, f"aics/{p['id']}/clip{i:02d}"),
                    on_progress=None)
        frames = next(o["images"] for o in out.values() if o.get("images"))
        fdir = pdir(p["id"], "clips", f"frames_{i:02d}")
        shutil.rmtree(fdir, ignore_errors=True)
        os.makedirs(fdir)
        for k, fr in enumerate(frames):
            with open(os.path.join(fdir, f"{k:04d}.png"), "wb") as f:
                f.write(c.fetch(fr))
        run_ffmpeg(ff, ["-framerate", str(fps), "-i", os.path.join(fdir, "%04d.png"), *H264,
                        pdir(p["id"], "clips", f"clip_{i:02d}.mp4")])
        p["clips_version"] = time.time()
        log(p, f"clip {i + 1}/{n - 1}: done in {time.time() - t:.0f}s ({len(frames)} frames)")
    if all(glob.glob(pdir(p["id"], "clips", f"frames_{i:02d}", "*.png")) for i in range(n - 1)):
        stitch(p, cfg, ff)
    else:
        log(p, "some clips are not animated yet; skipping stitch")


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


def stitch(p, cfg, ff):
    fps = cfg["video"]["fps"]
    n = len(p["plan"]["keyframes"])
    log(p, "stitching clips")
    seq = pdir(p["id"], "clips", "all_frames")
    shutil.rmtree(seq, ignore_errors=True)
    os.makedirs(seq)
    k = 0
    bounds = []  # (start_sec, end_sec) per clip on the final timeline
    for i in range(n - 1):
        files = sorted(glob.glob(pdir(p["id"], "clips", f"frames_{i:02d}", "*.png")))
        if not files:
            raise RuntimeError(f"clip {i + 1} has no frames; animate it first")
        if i > 0:
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
        if i == len(bounds) - 1 and kfs[-1].get("subtitle"):
            mid = (s + e) / 2
            cues += [(s, mid, kfs[i].get("subtitle", "")), (mid, e, kfs[-1]["subtitle"])]
        else:
            cues.append((s, e, kfs[i].get("subtitle", "")))
    cues = [c for c in cues if c[2].strip()]
    with open(pdir(p["id"], "subtitles.srt"), "w", encoding="utf-8") as f:
        for j, (s, e, text) in enumerate(cues, 1):
            f.write(f"{j}\n{srt_time(s)} --> {srt_time(e)}\n{text.strip()}\n\n")

    pl = p["plan"]
    with open(pdir(p["id"], "caption.txt"), "w", encoding="utf-8") as f:
        f.write(pl.get("caption", "").strip() + "\n\n" + " ".join(pl.get("hashtags", [])) + "\n")

    run_ffmpeg(ff, ["-framerate", str(fps), "-i", os.path.join(seq, "%05d.png"), *H264, "final.mp4"],
               cwd=pdir(p["id"]))
    if cues:
        style = "Fontname=Arial,Fontsize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,MarginV=40"
        run_ffmpeg(ff, ["-i", "final.mp4", "-vf", f"subtitles=subtitles.srt:force_style='{style}'", *H264,
                        "final_subtitled.mp4"], cwd=pdir(p["id"]))
    shutil.rmtree(seq, ignore_errors=True)
    p["final_version"] = time.time()
    log(p, f"final video written ({k / fps:.1f}s)")

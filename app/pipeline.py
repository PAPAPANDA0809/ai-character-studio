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

import cloud
import cloudgpu
import comfy
import llm
import push

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS = os.path.join(APP_DIR, "projects")
CHARACTERS = os.path.join(APP_DIR, "characters")
MODELS = os.path.join(os.path.dirname(APP_DIR), "models")
os.makedirs(PROJECTS, exist_ok=True)
os.makedirs(CHARACTERS, exist_ok=True)

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
QUALITIES = ("standard", "sharp", "high", "2k", "ultra")  # picture resolution: 360p, 720p, 1080p, 2K (1440p), 4K
NATIVE_HD = ("qwen-image-21", "z-image", "ideogram4")  # draw 720p/1080p directly: the 4x upscaler made Z-Image look painted
PERSON_WORDS = re.compile(r"(?i)\b(she|he|her|his|him|woman|women|man|men|girl|boy|lady|guy|person|people|model|"
                          r"wife|husband|mom|mum|dad|kid|child|baby|face|smil\w*)\b")
CLOTHING_WORDS = re.compile(r"(?i)\b(wear\w*|outfit|cloth\w*|dress\w*|shirt|t-shirt|tee|top|blouse|jacket|coat|raincoat|"
                            r"hoodie|sweater|cardigan|skirt|shorts|jeans|pants|trousers|bikini|swimsuit|shoes|sneakers|"
                            r"boots|sandals|heels|hat|cap|scarf|uniform|suit|gown|kimono|kebaya|hijab)\b")


def char_dir(user):
    d = os.path.join(CHARACTERS, user)
    os.makedirs(d, exist_ok=True)
    return d


def list_characters(user):
    return sorted(f for f in os.listdir(char_dir(user)) if f.lower().endswith(IMG_EXT))


def save_character(user, name, data):
    base, ext = os.path.splitext(name)
    base = re.sub(r"[^\w\- ]+", "", base).strip() or "character"
    ext = ext.lower() if ext.lower() in IMG_EXT else ".png"
    fname = base + ext
    with open(os.path.join(char_dir(user), fname), "wb") as f:
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

# Project ids look like "<user>--<timestamp-title>" and live in projects/<user>/<timestamp-title>.
PID_RE = re.compile(r"[a-z0-9_]{2,20}--[\w\-]+")


def user_of(pid):
    """Owner of a project id; None for anything malformed (e.g. "bob--../alice/x"), so it never matches a user."""
    return pid.split("--", 1)[0] if isinstance(pid, str) and PID_RE.fullmatch(pid) else None


def pdir(pid, *parts):
    user, _, rest = pid.partition("--")
    return os.path.join(PROJECTS, user, rest, *parts)


def project_ids(user=None):
    users = [user] if user else [u for u in os.listdir(PROJECTS) if os.path.isdir(os.path.join(PROJECTS, u))]
    out = []
    for u in users:
        d = os.path.join(PROJECTS, u)
        if os.path.isdir(d):
            out += [f"{u}--{x}" for x in os.listdir(d) if os.path.exists(os.path.join(d, x, "project.json"))]
    return out


def migrate_legacy(user):
    """Move projects/characters made before sign-in existed into this user's folders."""
    for d in os.listdir(PROJECTS):
        src = os.path.join(PROJECTS, d)
        if os.path.exists(os.path.join(src, "project.json")):
            os.makedirs(os.path.join(PROJECTS, user), exist_ok=True)
            os.replace(src, os.path.join(PROJECTS, user, d))
            p = load(f"{user}--{d}")
            p["id"] = f"{user}--{d}"
            save(p)
    for f in os.listdir(CHARACTERS):
        if f.lower().endswith(IMG_EXT):
            os.replace(os.path.join(CHARACTERS, f), os.path.join(char_dir(user), f))


def load(pid):
    with open(pdir(pid, "project.json"), encoding="utf-8") as f:
        return json.load(f)


_save_lock = threading.Lock()


def replace_file(tmp, dest):
    """os.replace, retried: on Windows it fails with "Access is denied" while another thread is reading dest
    (the page polls project.json every 3 s), which used to kill the running job."""
    for _ in range(40):
        try:
            return os.replace(tmp, dest)
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, dest)


def save(p):
    with _save_lock:
        tmp = pdir(p["id"], "project.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2, ensure_ascii=False)
        replace_file(tmp, pdir(p["id"], "project.json"))


def with_present(p):
    """A project for the page, plus which pictures exist on disk (one can be deleted from the Images tab)."""
    n = len(p["plan"]["keyframes"])
    return {**p, "present": [os.path.exists(pdir(p["id"], "keyframes", f"kf_{i:02d}.png")) for i in range(n)],
            "clips_present": [os.path.exists(pdir(p["id"], "clips", f"clip_{i:02d}.mp4")) for i in range(n)]}


def list_projects(user):
    out = []
    for d in sorted(project_ids(user), reverse=True):
        try:
            p = load(d)
            kfs = p["plan"]["keyframes"]
            out.append({"id": p["id"], "title": p["plan"]["title"], "status": p["status"], "created": p["created"],
                        "pinned": bool(p.get("pinned")), "mode": p.get("mode", "video"),
                        # for the phone feed: what was asked for, and which pictures exist
                        "prompt": p.get("idea") or (kfs[0]["image_prompt"] if kfs else ""),
                        "message": p.get("message", ""), "keyframes_version": p.get("keyframes_version", 0),
                        "present": [os.path.exists(pdir(p["id"], "keyframes", f"kf_{i:02d}.png")) for i in range(len(kfs))]})
        except Exception:
            pass
    return out


def gallery(kind, user):
    """Every generated image (kind='images') or finished/clip video (kind='videos'), newest first."""
    items = []
    for d in project_ids(user):
        try:
            p = load(d)
        except Exception:
            continue
        title = p["plan"]["title"]
        if kind == "images":
            for f in sorted(glob.glob(pdir(d, "keyframes", "kf_*.png"))):
                items.append({"project": d, "title": title, "path": "keyframes/" + os.path.basename(f),
                              "mtime": os.path.getmtime(f), "label": f"{title} #{int(f[-6:-4]) + 1}"})
            for f in sorted(glob.glob(pdir(d, "keyframes", "history", "kf_*.png"))):
                n_ = int(os.path.basename(f)[3:5]) + 1
                items.append({"project": d, "title": title, "path": "keyframes/history/" + os.path.basename(f),
                              "mtime": os.path.getmtime(f), "label": f"{title} #{n_} (older version)"})
        else:
            finals = [f for f in ("final_subtitled.mp4", "final.mp4") if os.path.exists(pdir(d, f))]
            if finals:
                f = pdir(d, finals[0])
                items.append({"project": d, "title": title, "path": finals[0], "mtime": os.path.getmtime(f),
                              "label": title, "final": True})
            elif p.get("mode") == "images":
                for f in sorted(glob.glob(pdir(d, "clips", "img_*.mp4"))):
                    items.append({"project": d, "title": title, "path": "clips/" + os.path.basename(f),
                                  "mtime": os.path.getmtime(f), "label": f"{title} #{int(f[-6:-4]) + 1}"})
            else:  # unfinished video projects still show their clips
                for f in sorted(glob.glob(pdir(d, "clips", "clip_*.mp4"))):
                    items.append({"project": d, "title": title, "path": "clips/" + os.path.basename(f),
                                  "mtime": os.path.getmtime(f), "label": f"{title} clip {int(f[-6:-4]) + 1}"})
    return sorted(items, key=lambda x: -x["mtime"])


TRASH = os.path.join(APP_DIR, "trash")


def _to_trash(user, src, rel):
    """Deleted things are moved (not erased) to app/trash/<user>/<date>/... so a mistake can be undone by hand."""
    dst = os.path.join(TRASH, user, time.strftime("%Y%m%d"), rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        dst += f".{int(time.time() * 1000)}"
    shutil.move(src, dst)


def delete_character(user, name):
    """Move one of the user's character pictures to the trash. Only names from their own list are accepted."""
    if name not in list_characters(user):
        return False
    _to_trash(user, os.path.join(char_dir(user), name), "characters/" + name)
    return True


def delete_items(user, projects=(), files=()):
    """Delete whole projects and/or single files ({project, path}) owned by user. Returns how many were removed."""
    n = 0
    for pid in projects:
        if user_of(pid) == user and os.path.isdir(pdir(pid)) and current["pid"] != pid:
            _to_trash(user, pdir(pid), "projects/" + pid)
            n += 1
    for it in files:
        pid, rel = it.get("project", ""), it.get("path", "")
        if user_of(pid) != user or ".." in rel or not os.path.isdir(pdir(pid)):
            continue
        src = os.path.normpath(pdir(pid, *rel.split("/")))
        if not src.startswith(os.path.normpath(pdir(pid))) or not os.path.isfile(src):
            continue
        _to_trash(user, src, f"files/{pid}/{rel}")
        p = load(pid)
        if rel.startswith("keyframes/history/"):
            name = rel.rsplit("/", 1)[-1]
            for vers in p.get("versions", {}).values():
                if name in vers:
                    vers.remove(name)
            p.get("version_info", {}).pop(name, None)
        elif m := re.fullmatch(r"keyframes/kf_(\d+)\.png", rel):
            k = str(int(m.group(1)))
            vers = p.get("versions", {}).get(k, [])
            p.get("pic_info", {}).pop(k, None)
            if vers:  # the newest older version (and its prompt record) takes the deleted picture's place
                name = vers.pop()
                os.replace(pdir(pid, "keyframes", "history", name), src)
                if name in p.get("version_info", {}):
                    p.setdefault("pic_info", {})[k] = p["version_info"].pop(name)
        elif rel.startswith("clips/img_"):
            p.get("image_clips", {}).pop(str(int(rel[-6:-4])), None)
        p["keyframes_version"] = p["clips_version"] = time.time()
        save(p)
        n += 1
    return n


def log(p, msg):
    p.setdefault("log", []).append(time.strftime("%H:%M:%S ") + msg)
    p["log"] = p["log"][-200:]
    p["message"] = msg
    save(p)


# ---------------------------------------------------------------- job queue (one GPU => one worker)

jobs = queue.Queue()
current = {"pid": None, "task": None}
last_engine = [None]  # picture engine ComfyUI last ran for us (video jobs free memory themselves)
cancelled = set()


class Cancelled(Exception):
    pass


def check_cancel(p):
    if p["id"] in cancelled:
        raise Cancelled()


def cancel(pid):
    """Stop a project's running job (interrupting ComfyUI) and drop its queued jobs."""
    with jobs.mutex:
        kept = [j for j in jobs.queue if j[0] != pid]
        dropped = len(jobs.queue) - len(kept)
        jobs.queue.clear()
        jobs.queue.extend(kept)
    running = current["pid"] == pid
    if running:  # the job's Comfy client sees this within ~2 s and drops/interrupts only its own prompt
        cancelled.add(pid)
    if dropped and not running:
        p = load(pid)
        p["status"] = "stopped:" + p["status"].split(":", 1)[-1]
        log(p, "stopped by user")
    return running or bool(dropped)


def enqueue(pid, task, **kw):
    p = load(pid)
    p["status"] = "queued:" + task
    save(p)
    jobs.put((pid, task, kw))


def worker():
    while True:
        pid, task, kw = jobs.get()
        current.update(pid=pid, task=task)
        try:
            p = load(pid)
            p["status"] = "running:" + task
            save(p)
            {"keyframes": do_keyframes, "animate": do_animate, "stitch": restitch, "t2v": do_t2v,
             "animate_image": do_animate_image}[task](p, **kw)
            p = load(pid)
            p["status"] = "done:" + task
            log(p, f"{task} finished")
            _announce(p, task, "done")
        except Exception as e:
            if not isinstance(e, (Cancelled, comfy.Stopped)):
                traceback.print_exc()
            try:
                p = load(pid)
            except Exception:  # project deleted while queued: nothing to report into, keep the worker alive
                continue
            if isinstance(e, (Cancelled, comfy.Stopped)) or pid in cancelled:
                p["status"] = "stopped:" + task
                log(p, "stopped by user")
            else:
                p["status"] = "error:" + task
                log(p, f"ERROR in {task}: {e}")
                _announce(p, task, "error", str(e))
        finally:
            cancelled.discard(pid)
            current.update(pid=None, task=None)


def _announce(p, task, status, detail=""):
    """A phone notification for the project's owner, if they turned notifications on."""
    try:
        push.job_done(user_of(p["id"]), p["id"], p["plan"].get("title") or "Reelmint", task, status, detail)
    except Exception:
        traceback.print_exc()


def _reset_stale():
    """Jobs don't survive a restart; mark anything left queued/running as interrupted."""
    for d in project_ids():
        try:
            p = load(d)
            if p["status"].startswith(("queued", "running")):
                p["status"] = "interrupted:" + p["status"].split(":", 1)[1]
                save(p)
        except Exception:
            pass


def _clear_comfy_orphans():
    try:
        n = comfy.Comfy(load_config()["comfy_url"]).clear_orphans()
        if n:
            print(f"stopped {n} ComfyUI job(s) left over from an earlier app run")
    except Exception:
        pass  # ComfyUI not running yet: nothing to clear


def start():
    """Called by the server once it owns its port: a second copy that fails to start must not touch jobs."""
    _reset_stale()
    threading.Thread(target=_clear_comfy_orphans, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=cloudgpu.resume, daemon=True).start()  # a rented GPU from before the restart


# ---------------------------------------------------------------- steps

def lora_entry(cfg, file, engine):
    """The configured LoRA with this file, if it was trained for this image engine (else None)."""
    return next((l for l in cfg.get("loras", []) if l["file"] == file and l["engine"] == engine), None)


def with_trigger(text, lora):
    """Start the prompt the way the LoRA's training captions started ("jacelynrm, a woman, ..."): the trigger
    word only holds the face when it leads the prompt (mid-sentence after the style text, the face drifted).
    The name as people write it ("@jacelyn", "Jacelyn") also becomes the trigger word."""
    if not lora or not lora.get("trigger"):
        return text
    trig = lora["trigger"]
    name = lora.get("name") or lora.get("label")  # versions share the character's name ("Jacelyn v2" -> "Jacelyn")
    if name:
        text = re.sub(rf"@?\b{re.escape(name)}\b", trig, text, flags=re.I)
    head = f"{trig}, {lora['class']}" if lora.get("class") else trig
    return text if text.lower().startswith(head.lower()) else f"{head}, {text}"


MAX_REFS = 3  # Qwen-Image 2.1 takes up to 16, but each one adds minutes on an 8 GB card


def clean_refs(user, refs):
    """Character pictures that exist for this user, in order, without repeats (one name or a list)."""
    have = list_characters(user)
    out = []
    for r in ([refs] if isinstance(refs, str) else refs or []):
        if r in have and r not in out:
            out.append(r)
    return out[:MAX_REFS]


def refs_of(p):
    """A project's character pictures in order (picture 1 is the main subject); older projects have one."""
    return p.get("references") or ([p["reference"]] if p.get("reference") else [])


def create_project(user, idea, n_keyframes, style, character, aspect, seed=None, reference="",
                   image_engine=None, video_engine=None, mode="video", quality=None, video_res=None, lora=None):
    cfg = load_config()
    mode = mode if mode in ("images", "t2v") else "video"
    n_keyframes = max(2 if mode == "video" else 1, min(int(n_keyframes), 12 if mode == "images" else 8))
    refs = clean_refs(user, reference)
    reference = refs[0] if refs else ""
    plan = llm.plan(cfg["llm"], idea, n_keyframes, style, character, has_reference=len(refs), mode=mode)
    base = user + "--" + time.strftime("%Y%m%d-%H%M%S-") + re.sub(r"[^a-z0-9]+", "-", plan["title"].lower()).strip("-")[:30]
    pid, k = base, 2
    while os.path.exists(pdir(pid)):  # same title in the same second
        pid, k = f"{base}-{k}", k + 1
    os.makedirs(pdir(pid, "keyframes"))
    os.makedirs(pdir(pid, "clips"))
    p = {
        "id": pid, "created": time.time(), "idea": idea, "style": style, "aspect": aspect,
        "seed": int(seed) if seed not in (None, "") else random.randint(1, 2**31 - 1),
        "plan": plan, "keyframe_seeds": {}, "status": "planned", "log": [],
        "reference": reference, "references": refs,
        "image_engine": (image_engine if image_engine in comfy.REFERENCE_ENGINES else comfy.REFERENCE_ENGINES[0]) if reference else
                        image_engine if image_engine in comfy.IMAGE_ENGINES else cfg.get("default_image_engine", "qwen-image-21"),
        "video_engine": video_engine if video_engine in comfy.VIDEO_ENGINES else cfg.get("default_video_engine", "wan22-14b"),
        "clip_fps": {}, "mode": mode,
        "quality": quality if quality in QUALITIES else ("sharp" if mode == "images" else "standard"),
        "video_res": video_res if video_res in ("480p", "720p") else "480p",
        "t2v_engine": cfg.get("default_t2v_engine", "wan22-5b"),
    }
    p["lora"] = lora if lora and lora_entry(cfg, lora, p["image_engine"]) else ""
    save(p)
    log(p, f"planned {n_keyframes} {dict(images='images', t2v='scenes').get(mode, 'keyframes')} with {plan.get('planner') or cfg['llm']['provider']}")
    return p


def model_path(cfg, f):
    """Where a model file (e.g. "unet/x.gguf") lives: the main models folder or one of extra_model_dirs; None if absent."""
    for root in [MODELS, *cfg.get("extra_model_dirs", [])]:
        path = os.path.join(root, *f.split("/"))
        if os.path.exists(path):
            return path
    return None


def engines():
    cfg = load_config()

    def ready(f):  # present and not still being written by a download
        path = model_path(cfg, f)
        return bool(path) and time.time() - os.path.getmtime(path) > 90

    def entry(key, e):
        if e.get("rented"):  # files only on the rented card
            on = cloudgpu.state["status"] == "ready" and cloudgpu.state["extras"].get(e["rented"])
            return {"id": key, "label": e["label"], "installed": bool(on), "note": "with Fast GPU"}
        label = e["label"]
        if key == "qwen-image-21" and cloudgpu.state["status"] == "ready":  # say where it runs and how
            label = "Qwen-Image 2.1 · " + ("full quality on Fast GPU" if cloudgpu.state["extras"].get("qwen_hq")
                                           else "on Fast GPU (full quality in a few minutes)")
        if (key in comfy.VIDEO_ENGINES or key in comfy.T2V_ENGINES) and cloudgpu.state["status"] == "ready" \
                and cloudgpu.state.get("video"):  # every video job goes to the rented card's engine (on_rented)
            label = f"{cloudgpu.video_label(cloudgpu.state['video'], cloudgpu.state['h3_unet'])} with sound · on Fast GPU" + (
                "" if cloudgpu.state["extras"].get("video") else " (downloading, a few minutes)")
        return {"id": key, "label": label, "installed": all(ready(f) for f in e["files"])}
    return {
        "loras": [{**l, "installed": ready("loras/" + l["file"])} for l in cfg.get("loras", [])],
        "reference_engines": list(comfy.REFERENCE_ENGINES),
        "image": [entry(k, e) for k, e in comfy.IMAGE_ENGINES.items()],
        "video": [entry(k, e) for k, e in comfy.VIDEO_ENGINES.items()],
        "t2v": [entry(k, e) for k, e in comfy.T2V_ENGINES.items()],
        "default_t2v": cfg.get("default_t2v_engine", "wan22-5b"),
        "gpt_image": cloud.available(),
        "default_image": cfg.get("default_image_engine", "qwen-image-21"),
        "default_video": cfg.get("default_video_engine", "wan22-14b"),
    }


def set_engines(pid, image_engine=None, video_engine=None, mode=None, t2v_engine=None, quality=None, video_res=None,
                lora=None, style=None):
    p = load(pid)
    if style is not None:  # the visual style added to every prompt ("" = none)
        p["style"] = " ".join(str(style).split())[:300]
    cfg = load_config()
    if lora is not None:  # "" = no LoRA; a LoRA is only kept for the engine it was trained for (checked below)
        p["lora"] = lora
    if image_engine in comfy.IMAGE_ENGINES and image_engine not in comfy.REFERENCE_ENGINES:
        p["reference"], p["references"] = "", []
    if quality in QUALITIES:
        p["quality"] = quality
    if video_res in ("480p", "720p"):
        p["video_res"] = video_res
    if t2v_engine in comfy.T2V_ENGINES:
        p["t2v_engine"] = t2v_engine
    if mode in ("images", "video", "t2v"):
        if mode == "video" and len(p["plan"]["keyframes"]) < 2:
            raise ValueError("a video needs at least 2 keyframes; add one first")
        p["mode"] = mode
        if mode == "video":  # image/t2v plans have no motions; animate needs one per transition
            fit_motions(p["plan"])
    if image_engine in comfy.IMAGE_ENGINES:
        p["image_engine"] = image_engine
    if video_engine in comfy.VIDEO_ENGINES:
        p["video_engine"] = video_engine
    if p.get("lora") and not lora_entry(cfg, p["lora"], p.get("image_engine")):
        p["lora"] = ""  # switched to an engine the LoRA wasn't trained for
    save(p)
    return p


def set_reference(pid, reference):
    p = load(pid)
    p["references"] = clean_refs(user_of(pid), reference)
    p["reference"] = p["references"][0] if p["references"] else ""
    if p["reference"] and p.get("image_engine") not in comfy.REFERENCE_ENGINES:
        p["image_engine"] = comfy.REFERENCE_ENGINES[0]
    save(p)
    return p


def fit_motions(plan):
    """Exactly one motion per transition (keyframes - 1), padding with a neutral one."""
    n = len(plan["keyframes"])
    plan["motions"] = (plan.get("motions") or [])[: max(0, n - 1)]
    while len(plan["motions"]) < n - 1:
        plan["motions"].append("smooth natural movement, gentle camera motion")


def update_plan(pid, plan):
    p = load(pid)
    fit_motions(plan)
    p["plan"] = plan
    save(p)
    return p


def pin(pid, pinned):
    p = load(pid)
    p["pinned"] = bool(pinned)
    save(p)
    return {"id": pid, "pinned": p["pinned"]}


def rename(pid, title):
    p = load(pid)
    title = " ".join(str(title or "").split())[:80]
    if not title:
        raise ValueError("the name can't be empty")
    p["plan"]["title"] = title
    save(p)
    return p


def keyframe_prompt(p, i):
    pl = p["plan"]
    parts = [p.get("style", ""), pl["keyframes"][i]["image_prompt"], pl.get("character", ""), pl.get("setting", "")]
    return ". ".join(s.strip().rstrip(".") for s in parts if s and s.strip())


def reference_prompt(p, i):
    """Character-picture prompt: keep the subject from the reference (a person, animal, product or object) and let
    the user's words decide clothing, pose and scene; an outfit named in the prompt replaces the reference outfit."""
    pl = p["plan"]
    scene = ". ".join(s.strip().rstrip(".") for s in [pl["keyframes"][i]["image_prompt"], pl.get("setting", "")]
                      if s and s.strip())
    style = p.get("style", "").strip()
    # Image models read "face", "hair", "people" (even in "do not add people") as a request to draw a person, so
    # person and clothing words are only added when the user's own description is about a person / clothes.
    n = len(refs_of(p))  # Qwen's encoder labels them "Picture 1", "Picture 2", ... in this order
    src = "picture 1" if n > 1 else "the reference image"
    keep = f"Keep the exact same subject from {src}, with its identity and details unchanged"
    if PERSON_WORDS.search(scene):
        keep += ": same face, eyes, hair, skin tone and body"
    if CLOTHING_WORDS.search(scene):
        keep += f". Use the clothing described here instead of the outfit in {src}"
    if n > 1:
        keep += f". Pictures 2{' and 3' if n > 2 else ''} are extra references: use them only as the description says"
    return f"{keep}. {scene}. " + (f"Style: {style}." if style else "")


def video_size(cfg, p):
    return cfg["video_sizes"].get(p.get("video_res", "480p"), cfg["video_sizes"]["480p"])[p.get("aspect", "portrait")]


def size(cfg, p):
    return cfg["sizes"].get(p.get("aspect", "portrait"), cfg["sizes"]["portrait"])


def pic_info(p, i, engine, edit=None):
    """What a picture was made from, kept with it so an older version shows its own prompt."""
    return {"prompt": p["plan"]["keyframes"][i]["image_prompt"], "edit": edit or "", "engine": engine, "time": time.time()}


def save_keyframe(p, i, data, info=None):
    """Write picture i; an existing picture is kept in keyframes/history so "New version" never loses it.
    info (pic_info) is stored under pic_info[i] and travels to version_info[file] when the picture becomes a version."""
    cur = pdir(p["id"], "keyframes", f"kf_{i:02d}.png")
    if os.path.exists(cur):
        hist = pdir(p["id"], "keyframes", "history")
        os.makedirs(hist, exist_ok=True)
        name = f"kf_{i:02d}_{int(time.time() * 1000)}.png"
        os.replace(cur, os.path.join(hist, name))
        p.setdefault("versions", {}).setdefault(str(i), []).append(name)
        old = p.get("pic_info", {}).pop(str(i), None)
        if old:
            p.setdefault("version_info", {})[name] = old
    with open(cur, "wb") as f:
        f.write(data)
    if info:
        p.setdefault("pic_info", {})[str(i)] = info


def use_version(pid, index, name):
    """Make an older version the current picture (the current one becomes a version)."""
    p = load(pid)
    i = int(index)
    vers = p.get("versions", {}).get(str(i), [])
    if name not in vers:
        raise ValueError("unknown version")
    hist = pdir(pid, "keyframes", "history")
    cur = pdir(pid, "keyframes", f"kf_{i:02d}.png")
    swap = f"kf_{i:02d}_{int(time.time() * 1000)}.png"
    infos, vinfos = p.setdefault("pic_info", {}), p.setdefault("version_info", {})
    if os.path.exists(cur):
        os.replace(cur, os.path.join(hist, swap))
        vers.append(swap)
        if str(i) in infos:
            vinfos[swap] = infos.pop(str(i))
    os.replace(os.path.join(hist, name), cur)
    vers.remove(name)
    infos.pop(str(i), None)
    if name in vinfos:  # its prompt comes back with it
        infos[str(i)] = vinfos.pop(name)
        p["plan"]["keyframes"][i]["image_prompt"] = infos[str(i)]["prompt"]
    p["keyframes_version"] = time.time()
    save(p)
    return p


def do_keyframes(p, indices=None, reseed=False, engine=None, edit=None):
    """Qwen-Image 2.1 pictures and edits run on the rented GPU while a fast-GPU session is ready; everything else here."""
    remote = cloudgpu.url()
    if remote and (engine or p.get("image_engine")) == "ideogram4" and not edit and not refs_of(p):
        log(p, f"using the rented {cloudgpu.info()['label']}")
        with cloudgpu.using():
            return _keyframes(p, indices, reseed, engine, edit, remote)
    if remote and engine != "gpt-image" and (edit or p.get("image_engine") == "qwen-image-21"):
        lora = lora_entry(load_config(), p.get("lora", ""), "qwen-image-21")
        if not lora or cloudgpu.has_lora(lora["file"]):
            log(p, f"using the rented {cloudgpu.info()['label']}")
            with cloudgpu.using():
                return _keyframes(p, indices, reseed, engine, edit, remote)
        log(p, f'the trained character "{lora["label"]}" is still uploading to the rented GPU, so this runs on this PC')
    return _keyframes(p, indices, reseed, engine, edit)


def _keyframes(p, indices=None, reseed=False, engine=None, edit=None, remote=None):
    """edit = an instruction ("make her hair red"): Qwen-Image 2.1 changes the finished picture as told; the
    result becomes a new version like any regenerated picture. remote = ComfyUI on the rented GPU."""
    cfg = load_config()
    if remote and cloudgpu.state["extras"].get("qwen_hq"):  # the rented card has the full-quality (bf16) Qwen files
        cfg["qwen_image_21"] = {**cfg["qwen_image_21"], **cloudgpu.HQ_QWEN}
    c = comfy.Comfy(remote or cfg["comfy_url"], lambda: p["id"] in cancelled, remote=bool(remote))
    if not c.alive():
        raise RuntimeError("ComfyUI is not running at " + (remote or cfg["comfy_url"]))
    q = p.get("quality", "standard")
    hq = q in ("high", "sharp", "2k", "ultra")  # upscale pass; "high"/"2k"/"ultra" also render bigger with more steps
    table = {"high": "hires_sizes", "2k": "hires_sizes", "ultra": "hires_sizes", "sharp": "sharp_sizes"}.get(q)
    w, h = (cfg[table].get(p.get("aspect", "portrait")) if table else None) or size(cfg, p)
    steps = cfg["image"]["hq_steps"] if q in ("high", "2k", "ultra") else None
    n = len(p["plan"]["keyframes"])
    indices = list(range(n)) if indices is None else [int(i) for i in indices]
    refs = refs_of(p)
    gone = [r for r in refs if r not in list_characters(user_of(p["id"]))]
    if gone:
        raise RuntimeError(f'the character picture "{gone[0]}" was deleted; pick another one (or None) under Character picture')
    ref = bool(refs)
    ref_names = [c.upload_image(os.path.join(char_dir(user_of(p["id"])), r),
                                "aics_ref_" + re.sub(r"[^\w.\-]", "_", user_of(p["id"]) + "_" + r)) for r in refs]
    for i in indices:
        check_cancel(p)
        key = str(i)
        if reseed or key not in p["keyframe_seeds"]:
            # one shared seed for the first pass helps keep the character consistent
            p["keyframe_seeds"][key] = random.randint(1, 2**31 - 1) if reseed else p["seed"]
        use = engine or p.get("image_engine")
        if use != "gpt-image" and use not in comfy.IMAGE_ENGINES:  # older project made with a removed engine
            use = cfg.get("default_image_engine", "qwen-image-21")
        if ref and use != "gpt-image" and use not in comfy.REFERENCE_ENGINES:
            use = comfy.REFERENCE_ENGINES[0]
        if edit:  # only Qwen-Image 2.1 edits; the picture itself is the reference
            use = "qwen-image-21"
        if use == "ideogram4" and not (remote and cloudgpu.state["extras"].get("ideogram")):
            raise RuntimeError("Ideogram 4.0 runs on a rented card: connect a Fast GPU and wait until Ideogram is "
                               "ready in its panel, or pick Qwen-Image 2.1")
        if use == "qwen-image-21" and remote and cloudgpu.state["extras"].get("qwen_hq"):
            use_label = "qwen-image-21 (full quality)"
        else:
            use_label = use
        log(p, f"keyframe {i + 1}/{n}: {'editing' if edit else 'generating'} with {use_label}")
        t = time.time()
        out_size = cfg["output_sizes"][q][p.get("aspect", "portrait")] if p.get("mode") == "images" else (w, h)
        if use == "gpt-image":  # cloud, pay-per-use: only when picked for this picture
            data = cloud.gpt_image(cfg, keyframe_prompt(p, i), p.get("aspect", "portrait"), out_size)
            save_keyframe(p, i, data, pic_info(p, i, "gpt-image"))
            p.setdefault("keyframe_engines", {})[key] = "gpt-image"
            p["keyframes_version"] = time.time()
            log(p, f"keyframe {i + 1}/{n}: done with GPT-image in {time.time() - t:.0f}s")
            continue
        prefix = f"aics/{p['id']}/kf{i:02d}"
        rw, rh, upscale = w, h, hq
        if use in NATIVE_HD and q in ("sharp", "high"):
            # these engines draw 720p/1080p directly: sharper real detail and no 4x pass (which took 10 s to 7 min
            # depending on RAM pressure; tested 3 Oct: 416 s vs 79 s for one 1080p Qwen-Image 2.1 picture)
            upscale = False
            if p.get("mode") == "images":
                ow, oh = cfg["output_sizes"][q][p.get("aspect", "portrait")]
                rw, rh = (ow + 15) // 16 * 16, (oh + 15) // 16 * 16
        lora = lora_entry(cfg, p.get("lora", ""), use)  # a trained character, only for the engine it was made for
        if lora and not model_path(cfg, "loras/" + lora["file"]):
            raise RuntimeError(f'the trained character "{lora["label"]}" ({lora["file"]}) is not in models\\loras yet')
        if edit:
            src = pdir(p["id"], "keyframes", f"kf_{i:02d}.png")
            if not os.path.exists(src):
                raise RuntimeError(f"picture {i + 1} isn't generated yet; generate it before editing")
            name = c.upload_image(src, f"aics_{p['id']}_edit{i:02d}.png")
            # the instruction as typed: the trained character's trigger lead-in would read as a new picture request
            wf = comfy.reference_workflow(cfg, use, edit, [name], rw, rh, p["keyframe_seeds"][key], prefix, steps, edit=True)
        elif ref:
            wf = comfy.reference_workflow(cfg, use, with_trigger(reference_prompt(p, i), lora), ref_names, rw, rh,
                                          p["keyframe_seeds"][key], prefix, steps)
        else:
            wf = comfy.image_workflow(cfg, use, with_trigger(keyframe_prompt(p, i), lora), rw, rh,
                                      p["keyframe_seeds"][key], prefix, steps)
        if lora:
            wf = comfy.add_lora(wf, lora["file"], lora.get("strength", 1.0))
        if upscale:
            wf = comfy.add_upscale(cfg, wf, full=(q == "ultra"))
        if p.get("mode") == "images":  # pictures are delivered at an exact size: 360p / 720p / 1080p / 2K / 4K
            ow, oh = cfg["output_sizes"][q][p.get("aspect", "portrait")]
            wf = comfy.add_final_size(wf, ow, oh)
        if not remote and last_engine[0] != use:  # unload the previous engine's models first: 32 GB RAM can't cache several big ones
            c.free()
            last_engine[0] = use
        try:
            out = c.run(wf)
            img = next(o["images"][0] for o in out.values() if o.get("images"))
            data = c.fetch(img)
        except Exception as e:
            if p["id"] in cancelled or not cloud.available() or edit:
                raise
            # last resort: the local engine failed, so make this one picture with GPT-image
            log(p, f"keyframe {i + 1}/{n}: local engine failed ({str(e)[:80]}), using GPT-image instead")
            data = cloud.gpt_image(cfg, keyframe_prompt(p, i), p.get("aspect", "portrait"), out_size)
            use = "gpt-image"
        save_keyframe(p, i, data, pic_info(p, i, use, edit))
        p.setdefault("keyframe_engines", {})[key] = use
        p["keyframes_version"] = time.time()
        log(p, f"keyframe {i + 1}/{n}: done in {time.time() - t:.0f}s")
    write_caption(p)


def on_rented(p, run):
    """Video jobs run on the rented GPU while a fast-GPU session is ready, with that card's video engine (MiniMax H3 or
    LTX-2.5, both with sound) instead of this PC's; run(remote) gets that ComfyUI's address, or None for this PC."""
    remote = cloudgpu.url() if cloudgpu.state.get("video") else None
    if not remote:
        return run(None)
    t = time.time()
    while not cloudgpu.state["extras"].get("video"):  # pictures come first; the video engine a few minutes later
        if time.time() - t > 20 * 60 or not cloudgpu.url():
            raise RuntimeError(f"the rented card's video engine ({cloudgpu.info()['video_label']}) didn't finish "
                               "downloading; try again in a moment, or stop and reconnect the Fast GPU")
        if int(time.time() - t) % 60 < 5:
            log(p, f"waiting for {cloudgpu.info()['video_label']} to finish downloading on the rented card")
        check_cancel(p)
        cloudgpu._check_extras()
        time.sleep(5)
    log(p, f"using the rented {cloudgpu.info()['label']} ({cloudgpu.info()['video_label']})")
    with cloudgpu.using():
        return run(remote)


def clip_engine(p, cfg, remote, key="video_engine", local_default="wan22-14b", engines=None):
    """The engine a video job uses: the rented card's (cfg gets its file names) or the project's own on this PC."""
    if remote:
        engine = cloudgpu.state["video"]
        if engine == "ltx25":
            cfg["ltx25"] = {**cfg["ltx25"], **comfy.LTX25_POD}
        return engine
    engines = comfy.VIDEO_ENGINES if engines is None else engines
    return p.get(key) if p.get(key) in engines else local_default


def clip_workflow(cfg, engine, prompt, start, end, w, h, seed, prefix):
    """A clip from first/last frames (start only = animate one picture; neither = text-to-video)."""
    if engine == "h3":
        return comfy.h3_workflow(prompt, start, end, w, h, seed, prefix, text_encoder=cloudgpu.state.get("h3_te"),
                                 unet=cloudgpu.state.get("h3_unet"), short=cloudgpu.state.get("h3_short") or 768)
    if not start:
        return comfy.t2v_workflow(cfg, engine, prompt, w, h, seed, prefix)
    return comfy.video_workflow(cfg, engine, prompt, start, end, w, h, seed, prefix)


def do_animate(p, only=None):
    return on_rented(p, lambda remote: _animate(p, only, remote))


def _animate(p, only=None, remote=None):
    cfg = load_config()
    ff = find_ffmpeg(cfg["ffmpeg"])
    c = comfy.Comfy(remote or cfg["comfy_url"], lambda: p["id"] in cancelled, remote=bool(remote))
    if not c.alive():
        raise RuntimeError("ComfyUI is not running at " + (remote or cfg["comfy_url"]))
    w, h = video_size(cfg, p)
    engine = clip_engine(p, cfg, remote)
    fps = engine_fps(cfg, engine)
    n = len(p["plan"]["keyframes"])
    for i in range(n):
        if not os.path.exists(pdir(p["id"], "keyframes", f"kf_{i:02d}.png")):
            raise RuntimeError(f"keyframe {i + 1} is missing; generate keyframes first")
    c.free()  # drop Flux from VRAM before loading Wan
    last_engine[0] = None
    clips = range(n - 1) if only is None else [int(only)]
    for i in clips:
        check_cancel(p)
        log(p, f"clip {i + 1}/{n - 1}: uploading keyframes")
        a = c.upload_image(pdir(p["id"], "keyframes", f"kf_{i:02d}.png"), f"aics_{p['id']}_kf{i:02d}.png")
        b = c.upload_image(pdir(p["id"], "keyframes", f"kf_{i + 1:02d}.png"), f"aics_{p['id']}_kf{i + 1:02d}.png")
        prompt = ". ".join(x for x in [p["plan"]["motions"][i], p["plan"].get("character", ""), p.get("style", "")] if x)
        t = time.time()
        log(p, f"clip {i + 1}/{n - 1}: animating with {engine} (this is the slow part)")
        out = c.run(clip_workflow(cfg, engine, prompt, a, b, w, h, p["seed"] + i, f"aics/{p['id']}/clip{i:02d}"))
        frames = next(o["images"] for o in out.values() if o.get("images"))
        save_clip_frames(p, c, ff, i, frames, fps, out)
        log(p, f"clip {i + 1}/{n - 1}: done in {time.time() - t:.0f}s ({len(frames)} frames)")
    if all(glob.glob(pdir(p["id"], "clips", f"frames_{i:02d}", "*.png")) for i in range(n - 1)):
        stitch(p, cfg, ff)
    else:
        log(p, "some clips are not animated yet; skipping stitch")


def do_t2v(p, only=None):
    return on_rented(p, lambda remote: _t2v(p, only, remote))


def _t2v(p, only=None, remote=None):
    """Video-only mode: each scene is generated straight from text, then the clips are joined."""
    cfg = load_config()
    ff = find_ffmpeg(cfg["ffmpeg"])
    c = comfy.Comfy(remote or cfg["comfy_url"], lambda: p["id"] in cancelled, remote=bool(remote))
    if not c.alive():
        raise RuntimeError("ComfyUI is not running at " + (remote or cfg["comfy_url"]))
    w, h = video_size(cfg, p)
    engine = clip_engine(p, cfg, remote, "t2v_engine", "wan22-5b", comfy.T2V_ENGINES)
    fps = engine_fps(cfg, engine)
    n = len(p["plan"]["keyframes"])
    c.free()
    last_engine[0] = None
    for i in (range(n) if only is None else [int(only)]):
        check_cancel(p)
        prompt = ". ".join(x for x in [p.get("style", ""), p["plan"]["keyframes"][i]["image_prompt"],
                                       p["plan"].get("character", ""), p["plan"].get("setting", "")] if x)
        t = time.time()
        log(p, f"scene {i + 1}/{n}: generating video with {engine}")
        out = c.run(clip_workflow(cfg, engine, prompt, None, None, w, h, p["seed"] + i, f"aics/{p['id']}/scene{i:02d}"))
        frames = next(o["images"] for o in out.values() if o.get("images"))
        save_clip_frames(p, c, ff, i, frames, fps, out)
        log(p, f"scene {i + 1}/{n}: done in {time.time() - t:.0f}s ({len(frames)} frames)")
    if all(glob.glob(pdir(p["id"], "clips", f"frames_{i:02d}", "*.png")) for i in range(n)):
        stitch(p, cfg, ff)


def engine_fps(cfg, engine):
    return {"ltxv-2b": cfg["ltxv"]["fps"], "hunyuan15": cfg["hunyuan15"]["fps"], "wan22-5b": cfg["wan5b"]["fps"],
            "h3": comfy.H3["fps"], **{e: cfg[e]["fps"] for e in comfy.LTX2_ENGINES}}.get(engine, cfg["video"]["fps"])


def save_clip_audio(c, out, dest):
    """Engines with sound (LTX-2.x) return a SaveAudio file: keep it at dest. Returns dest, or None (old file removed)."""
    aud = next((o["audio"][0] for o in (out or {}).values() if o.get("audio")), None)
    if os.path.exists(dest):
        os.remove(dest)
    if not aud:
        return None
    with open(dest, "wb") as f:
        f.write(c.fetch(aud))
    return dest


def save_clip_frames(p, c, ff, i, frames, fps, out=None):
    fdir = pdir(p["id"], "clips", f"frames_{i:02d}")
    shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir)
    for k, fr in enumerate(frames):
        with open(os.path.join(fdir, f"{k:04d}.png"), "wb") as f:
            f.write(c.fetch(fr))
    audio = save_clip_audio(c, out, pdir(p["id"], "clips", f"audio_{i:02d}.flac"))
    run_ffmpeg(ff, ["-framerate", str(fps), "-i", os.path.join(fdir, "%04d.png"),
                    *(["-i", audio, "-c:a", "aac", "-shortest"] if audio else []), *H264,
                    pdir(p["id"], "clips", f"clip_{i:02d}.mp4")])
    p.setdefault("clip_fps", {})[str(i)] = fps
    p["clips_version"] = time.time()


def do_animate_image(p, index, motion=""):
    return on_rented(p, lambda remote: _animate_image(p, index, motion, remote))


def _animate_image(p, index, motion="", remote=None):
    """Images-only projects: turn one finished picture into a 5-second clip."""
    cfg = load_config()
    ff = find_ffmpeg(cfg["ffmpeg"])
    c = comfy.Comfy(remote or cfg["comfy_url"], lambda: p["id"] in cancelled, remote=bool(remote))
    if not c.alive():
        raise RuntimeError("ComfyUI is not running at " + (remote or cfg["comfy_url"]))
    i = int(index)
    src = pdir(p["id"], "keyframes", f"kf_{i:02d}.png")
    if not os.path.exists(src):
        raise RuntimeError(f"picture {i + 1} isn't generated yet")
    engine = clip_engine(p, cfg, remote)
    fps = engine_fps(cfg, engine)
    w, h = video_size(cfg, p)
    c.free()
    last_engine[0] = None
    a = c.upload_image(src, f"aics_{p['id']}_kf{i:02d}.png")
    k = p["plan"]["keyframes"][i]
    prompt = ". ".join(x for x in [motion or "natural subtle movement, gentle breathing and hair motion, slow cinematic camera push-in",
                                   k["image_prompt"]] if x)
    log(p, f"picture {i + 1}: animating with {engine}")
    t = time.time()
    out = c.run(clip_workflow(cfg, engine, prompt, a, None, w, h, p["seed"] + 100 + i, f"aics/{p['id']}/img{i:02d}"))
    frames = next(o["images"] for o in out.values() if o.get("images"))
    fdir = pdir(p["id"], "clips", f"imgframes_{i:02d}")
    shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir)
    for n_, fr in enumerate(frames):
        with open(os.path.join(fdir, f"{n_:04d}.png"), "wb") as f:
            f.write(c.fetch(fr))
    vw, vh = video_size(cfg, p)
    audio = save_clip_audio(c, out, pdir(p["id"], "clips", f"imgaudio_{i:02d}.flac"))
    run_ffmpeg(ff, ["-framerate", str(fps), "-i", os.path.join(fdir, "%04d.png"),
                    *(["-i", audio, "-c:a", "aac", "-shortest"] if audio else []),
                    "-vf", f"scale={vw}:{vh}:flags=lanczos,setsar=1", *H264, pdir(p["id"], "clips", f"img_{i:02d}.mp4")])
    shutil.rmtree(fdir, ignore_errors=True)
    p.setdefault("image_clips", {})[str(i)] = time.time()
    p["clips_version"] = time.time()
    log(p, f"picture {i + 1}: video done in {time.time() - t:.0f}s")


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


def stitch_audio(p, ff, bounds, clip_fps, t2v):
    """One soundtrack for the final video from the clips' own sound (LTX-2.x); silence where a clip has none.
    Each piece is cut to its clip's slot on the timeline, so sound stays in sync. None when no clip has sound."""
    files = [pdir(p["id"], "clips", f"audio_{i:02d}.flac") for i in range(len(bounds))]
    if not any(os.path.exists(f) for f in files):
        return None
    inputs, parts = [], []
    for i, ((s, e), f) in enumerate(zip(bounds, files)):
        skip = 0 if t2v or i == 0 else 1 / clip_fps[i]  # the first frame was dropped (same keyframe as the previous clip)
        if os.path.exists(f):
            inputs += ["-i", f]
            src = f"[{len(inputs) // 2 - 1}:a]atrim=start={skip:.4f},asetpts=PTS-STARTPTS,"
        else:
            src = "anullsrc=r=48000:cl=stereo,"
        parts.append(f"{src}aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{e - s:.4f},"
                     f"asetpts=PTS-STARTPTS[a{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[a{i}]" for i in range(len(bounds))) + f"concat=n={len(bounds)}:v=0:a=1[out]"
    dest = pdir(p["id"], "clips", "soundtrack.wav")
    run_ffmpeg(ff, [*inputs, "-filter_complex", graph, "-map", "[out]", dest])
    return dest


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

    audio = stitch_audio(p, ff, bounds, clip_fps, t2v)
    vw, vh = video_size(cfg, p)  # engines round sizes differently (LTX: multiples of 32); deliver the exact size
    run_ffmpeg(ff, ["-framerate", str(fps), "-i", os.path.join(seq, "%05d.png"),
                    *(["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"] if audio else []),
                    "-vf", f"scale={vw}:{vh}:flags=lanczos,setsar=1", *H264, "final.mp4"],
               cwd=pdir(p["id"]))
    if cues:
        style = "Fontname=Arial,Fontsize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,MarginV=40"
        run_ffmpeg(ff, ["-i", "final.mp4", "-vf", f"subtitles=subtitles.srt:force_style='{style}'", "-c:a", "copy", *H264,
                        "final_subtitled.mp4"], cwd=pdir(p["id"]))
    shutil.rmtree(seq, ignore_errors=True)
    p["final_version"] = time.time()
    log(p, f"final video written ({k / fps:.1f}s)")

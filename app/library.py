"""Pose and outfit pictures to pick instead of describing them in words.

Two free galleries by the same author (terms: free to use, also commercially; the images themselves may not be
redistributed, and not used for real people or minors):
  POSE-DB-01    ai-pose-gallery.pages.dev     288 poses x 5 angles, faceless grey mannequin line drawings
  COSTUME-DB-01 ai-costume-gallery.pages.dev  1,407 outfits on an invisible body, no person
Importing keeps each gallery's index and small thumbnails here (app/library, never committed); the full picture is
fetched the first time a project uses it. People can also add their own pose and outfit pictures (per user).
The chosen pictures go to the image engine as extra reference pictures after the character pictures."""
import concurrent.futures
import json
import os
import re
import threading
import urllib.parse
import urllib.request
import facegallery

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")
KINDS = ("pose", "outfit")
SOURCES = {"pose": "https://ai-pose-gallery.pages.dev/", "outfit": "https://ai-costume-gallery.pages.dev/"}
UA = {"User-Agent": "Reelmint (personal library import)"}
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")

# the galleries are in Japanese
POSE_LINES = {"全身1人・セクシー系": "Full body · sexy", "全身1人・標準系": "Full body · standard",
              "上半身1人": "Upper body", "2人絡み": "Two people"}
POSE_KINDS = {"かわいい系": "cute", "しゃがみ": "squatting", "のけぞり・反り": "arching back", "ひねり": "twisting",
              "セクシー・露骨寄り系": "sexy, explicit", "側位": "side by side, lying", "前かがみ": "bending forward",
              "密着": "close together", "寝そべり": "lying down", "対面座位": "sitting face to face", "座り": "sitting",
              "座位": "sitting together", "座位密着": "sitting close together", "後背位": "from behind",
              "横座り・女の子座り": "side-sitting", "正常位": "lying face to face", "特殊": "special", "立ち": "standing",
              "立位": "standing together", "立位密着": "standing close together", "背面座位": "sitting, facing away",
              "胸寄せ・谷間強調系": "cleavage", "膝立ち": "kneeling", "膝立ち・四つん這い": "kneeling, all fours",
              "顔寄せ系": "face close-up", "騎乗位": "straddling"}
ANGLES = {"正面": "front", "上から": "from above", "下から": "from below", "横": "side", "斜め": "three-quarter"}
OUTFIT_CATS = {"春": "Spring", "夏": "Summer", "秋": "Autumn", "冬": "Winter", "ランジェリー": "Lingerie",
               "パーティ": "Party", "水着": "Swimwear", "セクシー": "Sexy"}

importing = {k: None for k in KINDS}  # None, or {"done": n, "total": n, "error": ""} while/after an import runs
_lock = threading.Lock()
_index = {}


def _dir(*parts):
    return os.path.join(ROOT, *parts)


def _get(url, timeout=60):
    import krea  # its certifi bundle: Windows' own chain for these sites has an expired certificate
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout, context=krea.tls_context()) as r:
        return r.read()


def gallery_items(kind, html):
    """The gallery's own data (the page embeds it as `const D=[...]`) turned into library items."""
    m = re.search(r"const D=(\[.*?\]);\s*\n", html, re.S)
    if not m:
        raise RuntimeError(f"the {kind} gallery's page changed; its picture list wasn't found")
    out = []
    for x in json.loads(m.group(1)):
        if kind == "pose":
            line = x.get("系統", "")
            out.append({"id": "pose:" + x["id"], "group": x["id"].rsplit("_", 1)[0],
                        "cat": POSE_LINES.get(line, line), "sub": POSE_KINDS.get(x.get("分類", ""), x.get("分類", "")),
                        "angle": ANGLES.get(x.get("アングル", ""), x.get("アングル", "")), "name": x.get("ポーズ名", ""),
                        "thumb_url": SOURCES[kind] + x["サムネ"], "full_url": SOURCES[kind] + x["原寸"]})
        else:
            cat = x.get("カテゴリ", "")
            out.append({"id": "outfit:" + x["id"], "group": x["id"], "cat": OUTFIT_CATS.get(cat, cat), "sub": "",
                        "angle": "", "name": x.get("キーワード", ""),
                        "thumb_url": SOURCES[kind] + x["サムネ"], "full_url": SOURCES[kind] + x["原寸"]})
    return out


def _safe(item_id):
    return re.sub(r"[^\w.\-]", "_", item_id.split(":", 1)[1])


def _ext(url):
    return os.path.splitext(url.split("?")[0])[1].lower() or ".png"


def thumb_path(item):
    return _dir(item["id"].split(":")[0], "thumbs", _safe(item["id"]) + _ext(item["thumb_url"]))


def full_path(item):
    return _dir(item["id"].split(":")[0], "full", _safe(item["id"]) + _ext(item["full_url"]))


def index(kind):
    """The imported gallery's items (cached)."""
    with _lock:
        if kind not in _index:
            try:
                with open(_dir(kind, "index.json"), encoding="utf-8") as f:
                    _index[kind] = json.load(f)
            except (FileNotFoundError, ValueError):
                _index[kind] = []
        return _index[kind]


def run_import(kind, workers=8):
    """Fetch the gallery's list and every thumbnail (about 12 MB for poses, 11 MB for outfits). Already fetched
    thumbnails are skipped, so a second import only adds what's new."""
    if kind not in KINDS:
        raise ValueError("unknown library: " + str(kind))
    state = importing[kind] = {"done": 0, "total": 0, "error": ""}
    try:
        items = gallery_items(kind, _get(SOURCES[kind]).decode("utf-8"))
        state["total"] = len(items)
        os.makedirs(_dir(kind, "thumbs"), exist_ok=True)
        os.makedirs(_dir(kind, "full"), exist_ok=True)

        def one(item):
            path = thumb_path(item)
            if not os.path.exists(path):
                for attempt in range(3):
                    try:
                        data = _get(item["thumb_url"])
                        with open(path + ".part", "wb") as f:
                            f.write(data)
                        os.replace(path + ".part", path)
                        break
                    except OSError:
                        if attempt == 2:
                            return
            state["done"] += 1

        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, items))
        with open(_dir(kind, "index.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        with _lock:
            _index.pop(kind, None)
    except Exception as e:  # shown in the picker
        state["error"] = str(e)[:300]
    return state


def start_import(kind):
    if importing.get(kind) and not importing[kind].get("error") and importing[kind]["done"] < max(1, importing[kind]["total"]):
        return importing[kind]  # already running
    threading.Thread(target=run_import, args=(kind,), daemon=True).start()
    return {"done": 0, "total": 0, "error": ""}


def user_dir(user, kind):
    return _dir("users", re.sub(r"[^\w.\-]", "_", user), kind)


def own_items(user, kind):
    d = user_dir(user, kind)
    names = sorted((n for n in os.listdir(d) if n.lower().endswith(IMG_EXT)), key=lambda n: -os.path.getmtime(os.path.join(d, n))) \
        if os.path.isdir(d) else []
    return [{"id": f"mine-{kind}:{n}", "group": n, "cat": "My pictures", "sub": "", "angle": "",
             "name": os.path.splitext(n)[0]} for n in names]


FAV_KINDS = ("face", *KINDS)  # faces are the user's character pictures


def _fav_path(user):
    return os.path.join(_dir("users", re.sub(r"[^\w.\-]", "_", user)), "favorites.json")


def favorites(user):
    """{kind: [ids]}: the pictures this user starred, newest first."""
    try:
        with open(_fav_path(user), encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError):
        data = {}
    return {k: [i for i in data.get(k, []) if isinstance(i, str)] for k in FAV_KINDS}


def set_favorite(user, kind, item_id, on):
    if kind not in FAV_KINDS or not isinstance(item_id, str) or not item_id or len(item_id) > 300:
        raise ValueError("unknown picture")
    favs = favorites(user)
    favs[kind] = ([item_id] if on else []) + [i for i in favs[kind] if i != item_id]
    os.makedirs(os.path.dirname(_fav_path(user)), exist_ok=True)
    with open(_fav_path(user), "w", encoding="utf-8") as f:
        json.dump(favs, f)
    return favs


def listing(user):
    """What the picker shows: each kind's items (own pictures first) with their thumbnail address."""
    out = {}
    for kind in KINDS:
        items = own_items(user, kind) + [{k: v for k, v in it.items() if not k.endswith("_url")} for it in index(kind)]
        for it in items:
            it["thumb"] = "/library/thumb/" + urllib.parse.quote(it["id"], safe="")
        out[kind] = {"items": items, "imported": bool(index(kind)), "importing": importing[kind],
                     "source": SOURCES[kind]}
    out["favorites"] = favorites(user)
    out["face"] = facegallery.listing()
    return out


def kind_of(item_id):
    head = str(item_id or "").split(":", 1)[0]
    return head.replace("mine-", "") if head.replace("mine-", "") in KINDS else None


def find(item_id):
    kind = kind_of(item_id)
    return next((it for it in index(kind) if it["id"] == item_id), None) if kind else None


def valid(user, item_id, kind):
    """item_id if it names a picture of this kind the user can use, else ""."""
    if not item_id or kind_of(item_id) != kind:
        return ""
    if item_id.startswith("mine-"):
        return item_id if os.path.isfile(os.path.join(user_dir(user, kind), item_id.split(":", 1)[1])) else ""
    return item_id if find(item_id) else ""


def thumb_file(user, item_id):
    if str(item_id).startswith("mine-"):
        name = item_id.split(":", 1)[1]
        if name != os.path.basename(name):
            return None
        path = os.path.join(user_dir(user, kind_of(item_id)), name)
        return path if os.path.isfile(path) else None
    it = find(item_id)
    return thumb_path(it) if it and os.path.exists(thumb_path(it)) else None


def local_file(user, item_id):
    """The full-size picture on this PC, fetched from the gallery the first time it's used."""
    if str(item_id).startswith("mine-"):
        path = thumb_file(user, item_id)
        if not path:
            raise RuntimeError("that pose or outfit picture was deleted; pick another one")
        return path
    it = find(item_id)
    if not it:
        raise RuntimeError(f"{item_id} isn't in the picture library; import the gallery again")
    path = full_path(it)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = _get(it["full_url"], timeout=120)
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    return path


def save_own(user, kind, name, data):
    if kind not in KINDS:
        raise ValueError("unknown library: " + str(kind))
    if not data or len(data) > 25 * 1024 * 1024:
        raise ValueError("Pick a picture under 25 MB")
    base, ext = os.path.splitext(os.path.basename(name or "picture.png"))
    ext = ext.lower() if ext.lower() in IMG_EXT else ".png"
    base = re.sub(r"[^\w\-. ]", "_", base).strip() or "picture"
    d = user_dir(user, kind)
    os.makedirs(d, exist_ok=True)
    out, n = base + ext, 1
    while os.path.exists(os.path.join(d, out)):
        n += 1
        out = f"{base} {n}{ext}"
    with open(os.path.join(d, out), "wb") as f:
        f.write(data)
    return f"mine-{kind}:{out}"


def delete_own(user, item_id):
    path = thumb_file(user, item_id) if str(item_id).startswith("mine-") else None
    if not path:
        raise ValueError("Only your own pictures can be removed")
    os.remove(path)

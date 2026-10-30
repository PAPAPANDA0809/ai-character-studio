"""Shared local face references; selected originals become private character pictures."""
import filecmp
import json
from pathlib import Path
import re
import shutil
import threading

ROOT = Path(__file__).resolve().parent / "library" / "face"
SOURCE = "https://ai-portrait-gallery.pages.dev/"
HAIR = {"ショート": "Short", "ミディアム": "Medium", "セミロング": "Semi-long", "ロング": "Long"}
_cache = None
_lock = threading.RLock()


def index():
    global _cache
    with _lock:
        if _cache is None:
            try:
                _cache = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                _cache = []
        return _cache


def listing():
    return {"items": [{**{k: v for k, v in item.items() if k != "original"},
                       "thumb": "/library/thumb/" + item["id"]} for item in index()],
            "imported": bool(index()), "importing": None, "source": SOURCE}


def find(item_id):
    return next((item for item in index() if item["id"] == item_id), None)


def thumb_file(item_id):
    item = find(item_id)
    if not item:
        return None
    path = ROOT / "thumbs" / (item["group"] + ".webp")
    return str(path) if path.is_file() else None


def select(item_id, character_directory):
    item = find(item_id)
    if not item:
        raise ValueError("Choose a face from the gallery")
    source = Path(item["original"])
    if not source.is_file():
        raise ValueError("This gallery picture was moved. Reimport the local face gallery.")
    destination = Path(character_directory)
    destination.mkdir(parents=True, exist_ok=True)
    with _lock:
        number = 1
        while True:
            name = f"gallery-{item['group']}" + (f"-{number}" if number > 1 else "") + ".png"
            target = destination / name
            if target.exists():
                if filecmp.cmp(source, target, shallow=False):
                    return name
                number += 1
                continue
            try:
                stream = target.open("xb")
            except FileExistsError:
                continue
            try:
                with source.open("rb") as original, stream:
                    shutil.copyfileobj(original, stream)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            return name


def import_local(directory):
    """Index an already-downloaded library and make small local thumbnails; no network calls."""
    from PIL import Image
    global _cache
    source = Path(directory).resolve()
    records = json.loads((source / "gallery-manifest.json").read_text(encoding="utf-8"))
    receipts = json.loads((source / "download-receipts.json").read_text(encoding="utf-8"))
    (ROOT / "thumbs").mkdir(parents=True, exist_ok=True)
    items = []
    seen = set()
    for record in records:
        identifier = record["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in seen:
            raise ValueError("Invalid or duplicate face gallery identifier")
        seen.add(identifier)
        original = (source / receipts[identifier]["file"]).resolve()
        if not original.is_relative_to(source) or not original.is_file():
            raise ValueError("Gallery original is missing or outside the gallery folder")
        if original.stat().st_size != record["bytes"]:
            raise ValueError("Incomplete gallery original: " + identifier)
        category = {"face": "Face portraits", "styling": "Styling"}[record["section"]]
        hair = HAIR.get(record["hair"], record["hair"])
        thumb = ROOT / "thumbs" / (identifier + ".webp")
        if not thumb.is_file():
            with Image.open(original) as image:
                image = image.convert("RGB")
                image.thumbnail((240, 320))
                temporary = thumb.with_suffix(".part")
                image.save(temporary, "WEBP", quality=78, method=2)
            temporary.replace(thumb)
        items.append({"id": "face:" + identifier, "group": identifier, "cat": category,
                      "sub": hair, "angle": "", "name": hair + " hair · " + identifier,
                      "keywords": record.get("style", ""), "original": str(original)})
    temporary = ROOT / "index.json.part"
    temporary.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    temporary.replace(ROOT / "index.json")
    with _lock:
        _cache = items
    return {"total": len(items), "portraits": sum(item["cat"] == "Face portraits" for item in items),
            "styling": sum(item["cat"] == "Styling" for item in items), "directory": str(source)}

"""Pay-per-use cloud image engine (OpenAI GPT-image). Key lives in app/secrets.json as "openai_api_key".
GPT Image 2.5 Sunburst (Sep 2026, best quality, same price as GPT Image 2) by default; an account without it falls back to
gpt-image-2. Character pictures go to the edits endpoint as input images, which keeps the person."""
import base64
import os
import io
import json
import urllib.error
import urllib.request

import llm

MODEL = "gpt-image-2.5-sunburst"
FALLBACK = "gpt-image-2"

SIZES = {"portrait": "1024x1536", "landscape": "1536x1024", "square": "1024x1024"}
# GPT Image 2/2.5 draw any size with both edges multiples of 16, the long edge up to 3840 and 655,360-8,294,400 pixels,
# so Reelmint's 720p/1080p/2K/4K come out at their own size (1080 -> 1088, the nearest multiple of 16) with no resizing.
# 360p is under the minimum. About $0.04 (720p) to $0.10 (4K) a picture with 2.5 at high quality (GPT Image 2: ~4x).
MIN_PIXELS, MAX_PIXELS, MAX_EDGE = 655_360, 8_294_400, 3840


def native_size(out_size):
    """(w, h) GPT can draw for this output size: multiples of 16, inside its limits (None = let it pick)."""
    if not out_size:
        return None
    w, h = (int(v) for v in out_size)
    if w * h < MIN_PIXELS:  # 360p: scale up to the smallest size GPT draws
        k = (MIN_PIXELS / (w * h)) ** 0.5
        w, h = w * k, h * k
    w, h = int(-(-w // 16) * 16), int(-(-h // 16) * 16)
    if max(w, h) > MAX_EDGE or w * h > MAX_PIXELS:
        return None
    return w, h


def available():
    return bool(llm._secret("openai_api_key"))


def _post(payload, key, refs=None):
    """generations (JSON), or edits (multipart, image[] = the character pictures) when there are refs."""
    if not refs:
        req = urllib.request.Request("https://api.openai.com/v1/images/generations", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    else:
        b = "----reelmint" + base64.b16encode(os.urandom(8)).decode()
        parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                 for k, v in payload.items()]
        for path in refs:  # at most 2048 px, JPEG: a 19 MB character PNG uploaded slowly for no gain
            from PIL import Image
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((2048, 2048), Image.LANCZOS)
                out = io.BytesIO()
                im.save(out, "JPEG", quality=92)
            name = os.path.splitext(os.path.basename(path))[0] + ".jpg"
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="image[]"; filename="{name}"\r\n'
                         f'Content-Type: image/jpeg\r\n\r\n'.encode() + out.getvalue() + b"\r\n")
        body = b"".join(parts) + f"--{b}--\r\n".encode()
        req = urllib.request.Request("https://api.openai.com/v1/images/edits", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}",
                                              "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def gpt_image(cfg, prompt, aspect, out_size=None, refs=None):
    """Generate one picture; returns PNG bytes, resized/cropped to out_size (w, h) when given. refs = paths of
    character pictures to keep (picture 1 = the main subject)."""
    key = llm._secret("openai_api_key")
    if not key:
        raise RuntimeError("GPT-image needs an OpenAI API key in app/secrets.json (\"openai_api_key\")")
    c = cfg.get("gpt_image", {})
    size = native_size(out_size)
    payload = {"model": c.get("model", MODEL), "prompt": prompt[:4000], "n": 1,
               "size": f"{size[0]}x{size[1]}" if size else SIZES.get(aspect, "1024x1536"), "quality": c.get("quality", "high")}
    try:
        try:
            res = _post(payload, key, refs)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code in (400, 403, 404) and "model" in body and payload["model"] != FALLBACK:
                payload["model"] = FALLBACK  # this account doesn't have GPT Image 2.5 yet
                res = _post(payload, key, refs)
            else:
                raise urllib.error.HTTPError(e.url, e.code, e.msg, e.hdrs, io.BytesIO(body.encode()))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 401:
            raise RuntimeError("OpenAI didn't accept the API key in app/secrets.json (\"openai_api_key\")")
        if e.code == 400 and "size" in body:  # model wants a different size list: let it choose
            payload["size"] = "auto"
            res = _post(payload, key, refs)
        elif e.code == 400 and ("safety" in body or "moderation" in body):
            raise RuntimeError("GPT-image refused this prompt (OpenAI content rules)")
        else:
            raise RuntimeError(f"GPT-image error {e.code}: {body[:300]}")
    item = res["data"][0]
    if item.get("b64_json"):
        data = base64.b64decode(item["b64_json"])
    else:
        with urllib.request.urlopen(item["url"], timeout=120) as r:
            data = r.read()
    if out_size:
        from PIL import Image, ImageOps
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if size is None or img.size != tuple(size):  # GPT picked another size: fit it to the one asked for
            img = ImageOps.fit(img, tuple(out_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "PNG")  # GPT's own size kept as it is (1088x1920 for 1080p: no resizing)
        data = buf.getvalue()
    return data

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
    payload = {"model": c.get("model", MODEL), "prompt": prompt[:4000], "n": 1,
               "size": SIZES.get(aspect, "1024x1536"), "quality": c.get("quality", "high")}
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
        img = ImageOps.fit(Image.open(io.BytesIO(data)).convert("RGB"), tuple(out_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        data = buf.getvalue()
    return data

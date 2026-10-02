"""Pay-per-use cloud image engine (OpenAI GPT-image). Key lives in app/secrets.json as "openai_api_key"."""
import base64
import io
import json
import urllib.error
import urllib.request

import llm

SIZES = {"portrait": "1024x1536", "landscape": "1536x1024", "square": "1024x1024"}


def available():
    return bool(llm._secret("openai_api_key"))


def _post(payload, key):
    req = urllib.request.Request("https://api.openai.com/v1/images/generations", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def gpt_image(cfg, prompt, aspect, out_size=None):
    """Generate one picture; returns PNG bytes, resized/cropped to out_size (w, h) when given."""
    key = llm._secret("openai_api_key")
    if not key:
        raise RuntimeError("GPT-image needs an OpenAI API key in app/secrets.json (\"openai_api_key\")")
    c = cfg.get("gpt_image", {})
    payload = {"model": c.get("model", "gpt-image-2"), "prompt": prompt[:4000], "n": 1,
               "size": SIZES.get(aspect, "1024x1536"), "quality": c.get("quality", "high")}
    try:
        res = _post(payload, key)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 400 and "size" in body:  # model wants a different size list: let it choose
            payload["size"] = "auto"
            res = _post(payload, key)
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

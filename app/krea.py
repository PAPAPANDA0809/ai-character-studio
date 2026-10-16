"""Krea 2 hosted image generation. Secrets stay on the server; no automatic paid retries."""
import io
import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.krea.ai"
SECRETS = os.path.join(os.path.dirname(__file__), "secrets.json")
KEY_LOCK = threading.Lock()
API_ENGINES = {
    "krea-api-turbo": ("medium-turbo", "Krea 2 Turbo · API", "$0.015"),
    "krea-api-medium": ("medium", "Krea 2 Medium · API", "$0.03"),
    "krea-api-large": ("large", "Krea 2 Large · API", "$0.06"),
}
GPU_FILES = ["diffusion_models/krea2_turbo_fp8_scaled.safetensors",
             "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "vae/qwen_image_vae.safetensors"]
MINOR = re.compile(r"\b(?:underage|child(?:ren)?|minor|preteen|teen(?:ager)?|schoolgirl|schoolboy|"
                   r"adolescent|little (?:girl|boy)|young (?:girl|boy)|(?:[0-9]|1[0-7])\s*[- ]?\s*"
                   r"(?:years?\s*[- ]?\s*old|y/?o))\b", re.I)
SEXUAL = re.compile(r"\b(?:naked|nude|nudity|sexual|sex|porn\w*|erotic\w*|undress\w*|"
                    r"genitals?|topless|bottomless|strip\w*)\b|remove.{0,40}clothes", re.I)


def validate_prompt(prompt):
    """Reject child sexual content before either hosted or GPU work is submitted."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter an image prompt")
    if MINOR.search(prompt) and SEXUAL.search(prompt):
        raise ValueError("Krea cannot generate sexual content involving minors. Use a non-sexual, age-appropriate prompt.")


def api_key():
    environment = os.environ.get("KREA_API_KEY", "").strip()
    if environment:
        return environment
    try:
        with open(SECRETS, encoding="utf-8") as stream:
            value = json.load(stream).get("krea_api_key", "")
            return value.strip() if isinstance(value, str) else ""
    except (FileNotFoundError, ValueError):
        return ""


def available():
    return bool(api_key())


def save_key(key):
    if not isinstance(key, str) or not 8 <= len(key.strip()) <= 4096 or re.search(r"\s", key.strip()):
        raise ValueError("Enter a valid Krea API key")
    if os.environ.get("KREA_API_KEY"):
        raise ValueError("Krea uses KREA_API_KEY on this server. Update that environment variable instead.")
    with KEY_LOCK:
        try:
            with open(SECRETS, encoding="utf-8") as stream:
                secrets = json.load(stream)
        except FileNotFoundError:
            secrets = {}
        secrets["krea_api_key"] = key.strip()
        tmp = SECRETS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as stream:
            json.dump(secrets, stream, indent=2)
        os.replace(tmp, SECRETS)


def entries():
    ready = available()
    return [{"id": key, "label": label, "installed": ready,
             "note": "paid per image" if ready else "add API key in Krea setup",
             "description": f"{price} per image · generated at Krea's native 1K size; larger outputs are resized. Text to image."}
            for key, (_, label, price) in API_ENGINES.items()]


def tls_context():
    """Prefer the runtime's CA bundle over stale Windows certificate chains. Never disable verification."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def request(path, key, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45, context=tls_context()) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        messages = {401: "Krea rejected the API key. Update it in Krea setup.",
                    402: "Your Krea API balance is empty. API credit is separate from the Krea app subscription.",
                    429: "Krea is busy or your request limit was reached. Try again later."}
        # Do not echo provider errors: they can contain prompts, credentials or signed URLs.
        raise RuntimeError(messages.get(error.code, f"Krea API request failed ({error.code}). Check your Krea account.")) from None
    except (OSError, ValueError):
        raise RuntimeError("Krea could not be reached. Check your connection and Krea jobs before retrying; a submitted job may still finish.") from None


def generate(engine, prompt, aspect, seed, out_size, should_stop=lambda: False):
    from PIL import Image, ImageOps
    validate_prompt(prompt)
    key = api_key()
    if not key:
        raise RuntimeError("Add your Krea API key in Krea setup first.")
    if engine not in API_ENGINES:
        raise ValueError("Unknown Krea API model")
    if should_stop():
        raise RuntimeError("Stopped before sending the Krea request")
    payload = {"prompt": prompt, "aspect_ratio": {"portrait": "9:16", "landscape": "16:9", "square": "1:1"}[aspect],
               "resolution": "1K", "seed": seed, "creativity": "raw"}
    job = request("/generate/image/krea/krea-2/" + API_ENGINES[engine][0], key, payload)
    job_id = job.get("job_id", "")
    if not re.fullmatch(r"[a-zA-Z0-9-]{1,128}", job_id):
        raise RuntimeError("Krea returned no job ID. Check your Krea jobs before retrying.")
    deadline = time.monotonic() + 600
    while job.get("status") != "completed":
        if should_stop():
            raise RuntimeError("Stopped waiting for Krea. The submitted job may still complete and be charged in your Krea account.")
        if job.get("status") in ("failed", "cancelled"):
            raise RuntimeError("Krea generation " + job["status"] + ". Check the job in your Krea account.")
        if time.monotonic() > deadline:
            raise RuntimeError("Krea is taking longer than expected. Check your Krea jobs before retrying.")
        time.sleep(2)
        job = request("/jobs/" + urllib.parse.quote(job_id, safe=""), key)
    if should_stop():
        raise RuntimeError("Stopped waiting for Krea; the image is available in your Krea account.")
    urls = (job.get("result") or {}).get("urls") or []
    if not urls or not isinstance(urls, list) or not isinstance(urls[0], str):
        raise RuntimeError("Krea completed without an image URL")
    url = urllib.parse.urlsplit(urls[0])
    if url.scheme != "https" or url.username or url.password or url.hostname in ("localhost", "127.0.0.1", "::1"):
        raise RuntimeError("Krea returned an invalid image URL")
    # The API token is deliberately omitted from the separate asset download.
    with urllib.request.urlopen(urls[0], timeout=90, context=tls_context()) as response:
        data = response.read(50 * 1024 * 1024 + 1)
    if len(data) > 50 * 1024 * 1024:
        raise RuntimeError("Krea image was too large")
    with Image.open(io.BytesIO(data)) as original:
        image = original.convert("RGB")
        if out_size:
            image = ImageOps.fit(image, tuple(out_size), Image.Resampling.LANCZOS)
        result = io.BytesIO()
        image.save(result, "PNG")
    return result.getvalue()

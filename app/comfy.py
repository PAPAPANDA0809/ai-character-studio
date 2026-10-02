"""Minimal ComfyUI API client plus the two workflows (Flux keyframe, Wan 2.2 first-last-frame)."""
import json
import time
import urllib.parse
import urllib.request
import uuid

CLIENT_ID = uuid.uuid4().hex


class Comfy:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def _get(self, path, timeout=30):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return r.read()

    def _post(self, path, payload, timeout=60):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return json.loads(body) if body else {}

    def alive(self):
        try:
            self._get("/system_stats", timeout=3)
            return True
        except Exception:
            return False

    def upload_image(self, path, name):
        boundary = uuid.uuid4().hex
        with open(path, "rb") as f:
            data = f.read()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                f"Content-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(self.url + "/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["name"]

    def free(self):
        try:
            self._post("/free", {"unload_models": True, "free_memory": True})
        except Exception:
            pass

    def run(self, workflow, timeout=3600, on_progress=None):
        """Queue a workflow, wait for it, return the history entry's outputs."""
        res = self._post("/prompt", {"prompt": workflow, "client_id": CLIENT_ID})
        if res.get("node_errors"):
            raise RuntimeError("ComfyUI rejected workflow: " + json.dumps(res["node_errors"])[:1500])
        pid = res["prompt_id"]
        start = time.time()
        while time.time() - start < timeout:
            hist = json.loads(self._get("/history/" + pid))
            if pid in hist:
                entry = hist[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise RuntimeError("ComfyUI error: " + json.dumps(msgs)[:1500])
                return entry["outputs"]
            if on_progress:
                on_progress(time.time() - start)
            time.sleep(2)
        raise TimeoutError("ComfyUI job timed out")

    def fetch(self, file_info):
        q = urllib.parse.urlencode({"filename": file_info["filename"], "subfolder": file_info.get("subfolder", ""),
                                    "type": file_info.get("type", "output")})
        return self._get("/view?" + q, timeout=120)


def flux_workflow(cfg, prompt, width, height, seed, prefix):
    c = cfg["image"]
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": c["unet"]}},
        "2": {"class_type": "DualCLIPLoaderGGUF",
              "inputs": {"clip_name1": c["t5"], "clip_name2": c["clip_l"], "type": "flux"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["3", 0], "guidance": c["guidance"]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                         "seed": seed, "steps": c["steps"], "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": c["vae"]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


WAN_NEGATIVE = ("blurry, low quality, distorted face, deformed hands, extra limbs, flicker, jitter, static frame, "
                "watermark, text, subtitles, overexposed, worst quality, jpeg artifacts")


def wan_flf_workflow(cfg, prompt, start_name, end_name, width, height, seed, prefix):
    v = cfg["video"]
    steps = v["steps"]
    half = steps // 2
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": v["unet_high"]}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": v["unet_low"]}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": v["lora_high"], "strength_model": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["2", 0], "lora_name": v["lora_low"], "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": v["shift"]}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": v["shift"]}},
        "7": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": v["text_encoder"], "type": "wan"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["7", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": WAN_NEGATIVE, "clip": ["7", 0]}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": v["vae"]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": start_name}},
        "12": {"class_type": "LoadImage", "inputs": {"image": end_name}},
        "13": {"class_type": "WanFirstLastFrameToVideo",
               "inputs": {"positive": ["8", 0], "negative": ["9", 0], "vae": ["10", 0],
                          "width": width, "height": height, "length": v["frames"], "batch_size": 1,
                          "start_image": ["11", 0], "end_image": ["12", 0]}},
        "14": {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["5", 0], "positive": ["13", 0], "negative": ["13", 1], "latent_image": ["13", 2],
                          "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0,
                          "end_at_step": half, "return_with_leftover_noise": "enable"}},
        "15": {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["6", 0], "positive": ["13", 0], "negative": ["13", 1], "latent_image": ["14", 0],
                          "add_noise": "disable", "noise_seed": seed, "steps": steps, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "start_at_step": half,
                          "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["10", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": prefix}},
    }

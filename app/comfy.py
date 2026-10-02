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


# Engines shown in the app's dropdowns. "files" are checked against the models folder to mark what's installed.
IMAGE_ENGINES = {
    "flux-dev": {"label": "FLUX.1 dev (best quality, ~50s)", "files": ["unet/flux1-dev-Q4_K_S.gguf"]},
    "flux-schnell": {"label": "FLUX.1 schnell (fast drafts, ~15s)", "files": ["unet/flux1-schnell-Q4_K_S.gguf"]},
    "sdxl-juggernaut": {"label": "Juggernaut XL (SDXL photoreal, ~20s)",
                        "files": ["checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"]},
    "qwen-image": {"label": "Qwen-Image (best prompt following + text in image, slow)",
                   "files": ["unet/qwen-image-Q3_K_M.gguf", "loras/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
                             "text_encoders/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "vae/qwen_image_vae.safetensors"]},
}
REFERENCE_ENGINE = {"label": "FLUX.1 Kontext (uses the master character picture)",
                    "files": ["unet/flux1-kontext-dev-Q4_K_S.gguf"]}
VIDEO_ENGINES = {
    "wan22-14b": {"label": "Wan 2.2 14B first+last frame (best quality, slow)",
                  "files": ["unet/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf", "unet/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"]},
    "ltxv-2b": {"label": "LTX-Video 2B first+last frame (fast, lower quality)",
                "files": ["checkpoints/ltxv-2b-0.9.8-distilled.safetensors"]},
    "hunyuan15": {"label": "HunyuanVideo 1.5 480p (start frame only; won't land on the next keyframe)",
                  "files": ["diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors",
                            "text_encoders/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                            "text_encoders/byt5_small_glyphxl_fp16.safetensors",
                            "vae/hunyuanvideo15_vae_fp16.safetensors",
                            "clip_vision/sigclip_vision_patch14_384.safetensors"]},
}


def image_workflow(cfg, engine, prompt, width, height, seed, prefix):
    c = cfg["image"]
    if engine == "flux-schnell":
        return flux_workflow(cfg, prompt, width, height, seed, prefix, unet=c["schnell_unet"], steps=c["schnell_steps"])
    if engine == "sdxl-juggernaut":
        return sdxl_workflow(cfg, prompt, width, height, seed, prefix)
    if engine == "qwen-image":
        return qwen_image_workflow(cfg, prompt, width, height, seed, prefix)
    return flux_workflow(cfg, prompt, width, height, seed, prefix)


def video_workflow(cfg, engine, prompt, start_name, end_name, width, height, seed, prefix):
    if engine == "ltxv-2b":
        return ltxv_flf_workflow(cfg, prompt, start_name, end_name, width, height, seed, prefix)
    if engine == "hunyuan15":
        return hunyuan15_i2v_workflow(cfg, prompt, start_name, width, height, seed, prefix)
    return wan_flf_workflow(cfg, prompt, start_name, end_name, width, height, seed, prefix)


def qwen_image_workflow(cfg, prompt, width, height, seed, prefix):
    q = cfg["qwen_image"]
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": q["unet"]}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": q["lora"], "strength_model": 1.0}},
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.1, "sampling": "flow"}},
        "4": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": q["text_encoder"], "type": "qwen_image"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": " ", "clip": ["4", 0]}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["3", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
                         "seed": seed, "steps": q["steps"], "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["9", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def hunyuan15_i2v_workflow(cfg, prompt, start_name, width, height, seed, prefix):
    """HunyuanVideo 1.5 image-to-video. It has no end-frame input, so only the start keyframe is used."""
    hv = cfg["hunyuan15"]
    w, h = width // 16 * 16, height // 16 * 16
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": hv["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": hv["shift"]}},
        "3": {"class_type": "DualCLIPLoaderGGUF",
              "inputs": {"clip_name1": hv["text_encoder"], "clip_name2": hv["byt5"], "type": "hunyuan_video_15"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": WAN_NEGATIVE, "clip": ["3", 0]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": hv["vae"]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": start_name}},
        "8": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": hv["clip_vision"]}},
        "9": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["8", 0], "image": ["7", 0], "crop": "center"}},
        "10": {"class_type": "HunyuanVideo15ImageToVideo",
               "inputs": {"positive": ["4", 0], "negative": ["5", 0], "vae": ["6", 0], "width": w, "height": h,
                          "length": hv["frames"], "batch_size": 1, "start_image": ["7", 0],
                          "clip_vision_output": ["9", 0]}},
        "11": {"class_type": "KSampler",
               "inputs": {"model": ["2", 0], "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2],
                          "seed": seed, "steps": hv["steps"], "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "denoise": 1.0}},
        "12": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["11", 0], "vae": ["6", 0], "tile_size": 256, "overlap": 64,
                          "temporal_size": 32, "temporal_overlap": 8}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


SDXL_NEGATIVE = "blurry, lowres, deformed, bad anatomy, extra fingers, watermark, text, logo, cartoonish, oversaturated"


def sdxl_workflow(cfg, prompt, width, height, seed, prefix):
    c = cfg["image"]
    # SDXL wants ~1MP; render at that size with the same aspect, the video step rescales it
    scale = (1024 * 1024 / (width * height)) ** 0.5
    w, h = int(width * scale) // 64 * 64, int(height * scale) // 64 * 64
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": c["sdxl_checkpoint"]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": SDXL_NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
                         "seed": seed, "steps": c["sdxl_steps"], "cfg": c["sdxl_cfg"], "sampler_name": "dpmpp_2m",
                         "scheduler": "karras", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "ImageScale", "inputs": {"image": ["6", 0], "upscale_method": "lanczos",
                                                     "width": width, "height": height, "crop": "center"}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


def flux_workflow(cfg, prompt, width, height, seed, prefix, unet=None, steps=None):
    c = cfg["image"]
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet or c["unet"]}},
        "2": {"class_type": "DualCLIPLoaderGGUF",
              "inputs": {"clip_name1": c["t5"], "clip_name2": c["clip_l"], "type": "flux"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["3", 0], "guidance": c["guidance"]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                         "seed": seed, "steps": steps or c["steps"], "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": c["vae"]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def kontext_workflow(cfg, prompt, ref_name, width, height, seed, prefix):
    """FLUX.1 Kontext: generate a new scene that keeps the character from the reference image."""
    c = cfg["image"]
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": c["kontext_unet"]}},
        "2": {"class_type": "DualCLIPLoaderGGUF",
              "inputs": {"clip_name1": c["t5"], "clip_name2": c["clip_l"], "type": "flux"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": c["vae"]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        # a ~0.4MP reference keeps attention cost close to a plain Flux render on 8GB
        "12": {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"image": ["11", 0], "upscale_method": "lanczos",
                          "megapixels": c.get("kontext_ref_megapixels", 0.4), "resolution_steps": 16}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["12", 0], "vae": ["8", 0]}},
        "14": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["3", 0], "latent": ["13", 0]}},
        "4": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["14", 0], "guidance": c["kontext_guidance"]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                         "seed": seed, "steps": c["kontext_steps"], "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def ltxv_flf_workflow(cfg, prompt, start_name, end_name, width, height, seed, prefix):
    """LTX-Video 2B distilled with the two keyframes as guides at the first and last frame."""
    v = cfg["ltxv"]
    w, h = width // 32 * 32, height // 32 * 32
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": v["checkpoint"]}},
        "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": cfg["image"]["t5"], "type": "ltxv"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": WAN_NEGATIVE, "clip": ["2", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": start_name}},
        "6": {"class_type": "LoadImage", "inputs": {"image": end_name}},
        "7": {"class_type": "LTXVPreprocess", "inputs": {"image": ["5", 0], "img_compression": 35}},
        "8": {"class_type": "LTXVPreprocess", "inputs": {"image": ["6", 0], "img_compression": 35}},
        "9": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": w, "height": h, "length": v["frames"], "batch_size": 1}},
        "10": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["3", 0], "negative": ["4", 0], "vae": ["1", 2], "latent": ["9", 0],
                          "image": ["7", 0], "frame_idx": 0, "strength": 1.0}},
        "11": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["10", 0], "negative": ["10", 1], "vae": ["1", 2], "latent": ["10", 2],
                          "image": ["8", 0], "frame_idx": -1, "strength": 1.0}},
        "12": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["11", 0], "negative": ["11", 1], "frame_rate": float(v["fps"])}},
        "13": {"class_type": "LTXVScheduler",
               "inputs": {"steps": v["steps"], "max_shift": 2.05, "base_shift": 0.95, "stretch": True,
                          "terminal": 0.1, "latent": ["11", 2]}},
        "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "15": {"class_type": "SamplerCustom",
               "inputs": {"model": ["1", 0], "add_noise": True, "noise_seed": seed, "cfg": 1.0,
                          "positive": ["12", 0], "negative": ["12", 1], "sampler": ["14", 0],
                          "sigmas": ["13", 0], "latent_image": ["11", 2]}},
        "16": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["12", 0], "negative": ["12", 1], "latent": ["15", 0]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 2], "vae": ["1", 2]}},
        "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": prefix}},
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

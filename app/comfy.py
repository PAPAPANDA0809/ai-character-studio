"""Minimal ComfyUI API client plus the two workflows (Flux keyframe, Wan 2.2 first-last-frame)."""
import json
import time
import urllib.parse
import urllib.request
import uuid

CLIENT_ID = uuid.uuid4().hex


class Stopped(Exception):
    """The job's Stop button was pressed while ComfyUI had our prompt queued or running."""


class Comfy:
    def __init__(self, url, should_stop=None):
        self.url = url.rstrip("/")
        self.should_stop = should_stop  # polled while waiting on a prompt

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
            if self.should_stop and self.should_stop():
                self.stop(pid)
                raise Stopped()
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

    def stop(self, prompt_id):
        """Drop this prompt if it is still waiting; interrupt it (and only it) if it is running.
        ComfyUI notices an interrupt at the next sampling step, which can take a while on big video models."""
        for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
            try:
                self._post(path, body)
            except Exception:
                pass

    def clear_orphans(self):
        """Stop app prompts ("aics/..." outputs) left behind by an earlier app process: their jobs are gone,
        but ComfyUI would keep the GPU busy with them and newer jobs would wait behind. Returns how many."""
        q = json.loads(self._get("/queue", timeout=5))
        n = 0
        for item in q.get("queue_running", []) + q.get("queue_pending", []):
            prompt_id, wf, extra = item[1], item[2], item[3] if len(item) > 3 else {}
            ours = any(str(nd.get("inputs", {}).get("filename_prefix", "")).startswith("aics/") for nd in wf.values())
            if ours and extra.get("client_id") != CLIENT_ID:
                self.stop(prompt_id)
                n += 1
        return n

    def fetch(self, file_info):
        q = urllib.parse.urlencode({"filename": file_info["filename"], "subfolder": file_info.get("subfolder", ""),
                                    "type": file_info.get("type", "output")})
        return self._get("/view?" + q, timeout=120)


# Engines shown in the app's dropdowns. "files" are checked against the models folder to mark what's installed.
IMAGE_ENGINES = {
    "z-image": {"label": "Z-Image Turbo (best photoreal, ~30s)",
                "files": ["unet/z_image_turbo-Q8_0.gguf", "text_encoders/qwen_3_4b_fp8_mixed.safetensors"]},
    "flux-kontext": {"label": "FLUX Kontext (uses your character picture, ~1.5 min)",
                     "files": ["unet/flux1-kontext-dev-Q4_K_S.gguf"]},
    "flux-dev": {"label": "FLUX.1 dev (~50s)", "files": ["unet/flux1-dev-Q4_K_S.gguf"]},
    "flux-schnell": {"label": "FLUX.1 schnell (fast drafts, ~20s)", "files": ["unet/flux1-schnell-Q4_K_S.gguf"]},
    "sdxl-juggernaut": {"label": "Juggernaut XL (SDXL photoreal, ~30s)",
                        "files": ["checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"]},
    "qwen-image": {"label": "Qwen-Image (best prompt following + text in image, ~1.5 min)",
                   "files": ["unet/qwen-image-Q3_K_M.gguf", "loras/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
                             "text_encoders/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "vae/qwen_image_vae.safetensors"]},
    "flux2-klein-4b": {"label": "FLUX.2 klein 4B (new 2026, fastest ~15 s, character picture OK)",
                       "files": ["unet/flux-2-klein-4b-fp8.safetensors", "text_encoders/qwen_3_4b_fp8_mixed.safetensors",
                                 "vae/flux2-vae.safetensors"]},
    "flux2-klein-9b": {"label": "FLUX.2 klein 9B (character picture OK, 30 s to 5+ min on 32 GB RAM, non-commercial licence)",
                       "files": ["unet/flux-2-klein-9b-Q4_K_M.gguf", "text_encoders/qwen_3_8b_fp8mixed.safetensors",
                                 "vae/flux2-vae.safetensors"]},
    "qwen-image-21": {"label": "Qwen-Image 2.1 (newest, Sep 2026, slow, character picture OK, research licence)",
                      "files": ["diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
                                "text_encoders/qwen3vl_8b_int8_convrot.safetensors", "vae/qwen_image_2.1_vae_bf16.safetensors"]},
}
# Engines that can use the character picture as a reference
REFERENCE_ENGINES = ("flux-kontext", "flux2-klein-4b", "flux2-klein-9b", "qwen-image-21")
REFERENCE_ENGINE = {"label": "FLUX.1 Kontext (uses the master character picture)",
                    "files": ["unet/flux1-kontext-dev-Q4_K_S.gguf"]}

LTX23_FILES = ["unet/ltx-2.3-22b-distilled-Q4_K_S.gguf", "checkpoints/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
               "vae/ltx-2.3-22b-distilled_video_vae.safetensors", "checkpoints/ltx-2.3-22b-distilled_audio_vae.safetensors",
               "text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"]
LTX25_FILES = ["unet/LTX-2.5-Distilled-Q4_K_S.gguf", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
               "vae/ltx-2.5-video-vae-bf16.safetensors", "vae/ltx-2.5-audio-vae-bf16.safetensors"]
LTX2_ENGINES = ("ltx23", "ltx25")
VIDEO_ENGINES = {
    "wan22-14b": {"label": "Wan 2.2 14B first+last frame (best quality, ~20 min/clip)",
                  "files": ["unet/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf", "unet/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"]},
    "ltxv-2b": {"label": "LTX-Video 2B first+last frame (fast, ~1 min/clip)",
                "files": ["checkpoints/ltxv-2b-0.9.8-distilled.safetensors"]},
    "hunyuan15": {"label": "HunyuanVideo 1.5 480p (~8 min/clip, start frame only)",
                  "files": ["diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors",
                            "text_encoders/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                            "text_encoders/byt5_small_glyphxl_fp16.safetensors",
                            "vae/hunyuanvideo15_vae_fp16.safetensors",
                            "clip_vision/sigclip_vision_patch14_384.safetensors"]},
    "ltx23": {"label": "LTX-2.3 22B first+last frame, with sound (new 2026, slow on 8 GB)", "files": LTX23_FILES},
    "ltx25": {"label": "LTX-2.5 22B first+last frame, with sound (newest, slow on 8 GB)", "files": LTX25_FILES},
}


T2V_ENGINES = {
    "wan22-5b": {"label": "Wan 2.2 5B text-to-video (experimental, ~6 min, weak at 9:16)",
                 "files": ["unet/Wan2.2-TI2V-5B-Q5_K_M.gguf", "vae/wan2.2_vae.safetensors",
                           "text_encoders/umt5-xxl-encoder-Q5_K_M.gguf"]},
    "ltxv-2b": {"label": "LTX-Video 2B text-to-video (fast, ~40s)",
                "files": ["checkpoints/ltxv-2b-0.9.8-distilled.safetensors"]},
    "ltx23": {"label": "LTX-2.3 22B text-to-video, with sound (new 2026, slow on 8 GB)", "files": LTX23_FILES},
    "ltx25": {"label": "LTX-2.5 22B text-to-video, with sound (newest, slow on 8 GB)", "files": LTX25_FILES},
}


def t2v_workflow(cfg, engine, prompt, width, height, seed, prefix):
    w, h = width // 32 * 32, height // 32 * 32
    if engine in LTX2_ENGINES:
        return ltx2_workflow(cfg, engine, prompt, None, None, width, height, seed, prefix)
    if engine == "ltxv-2b":
        v = cfg["ltxv"]
        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": v["checkpoint"]}},
            "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": cfg["image"]["t5"], "type": "ltxv"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": WAN_NEGATIVE, "clip": ["2", 0]}},
            "9": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w, "height": h, "length": v["frames"], "batch_size": 1}},
            "12": {"class_type": "LTXVConditioning", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": float(v["fps"])}},
            "13": {"class_type": "LTXVScheduler", "inputs": {"steps": v["steps"], "max_shift": 2.05, "base_shift": 0.95,
                                                            "stretch": True, "terminal": 0.1, "latent": ["9", 0]}},
            "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
            "15": {"class_type": "SamplerCustom",
                   "inputs": {"model": ["1", 0], "add_noise": True, "noise_seed": seed, "cfg": 1.0,
                              "positive": ["12", 0], "negative": ["12", 1], "sampler": ["14", 0],
                              "sigmas": ["13", 0], "latent_image": ["9", 0]}},
            "17": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["1", 2]}},
            "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": prefix}},
        }
    t = cfg["wan5b"]
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": t["unet"]}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": t["shift"]}},
        "3": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": cfg["video"]["text_encoder"], "type": "wan"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": WAN_NEGATIVE, "clip": ["3", 0]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": t["vae"]}},
        "7": {"class_type": "Wan22ImageToVideoLatent",
              "inputs": {"vae": ["6", 0], "width": w, "height": h, "length": t["frames"], "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0],
                         "seed": seed, "steps": t["steps"], "cfg": t["cfg"], "sampler_name": "uni_pc",
                         "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecodeTiled",
              "inputs": {"samples": ["8", 0], "vae": ["6", 0], "tile_size": 256, "overlap": 64,
                         "temporal_size": 32, "temporal_overlap": 8}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def add_upscale(cfg, wf, full=False):
    """Insert a 4x-UltraSharp pass (then halve it => 2x sharper output) right before the SaveImage node."""
    save = next(k for k, n in wf.items() if n["class_type"] == "SaveImage")
    src = wf[save]["inputs"]["images"]
    wf["90"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": cfg["image"]["upscaler"]}}
    wf["91"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["90", 0], "image": src}}
    wf["92"] = {"class_type": "ImageScaleBy", "inputs": {"image": ["91", 0], "upscale_method": "lanczos",
                                                         "scale_by": 1.0 if full else 0.5}}
    wf[save]["inputs"]["images"] = ["92", 0]
    return wf


def add_final_size(wf, width, height):
    """Resize (center-crop if needed) right before saving, to an exact delivery size like 1080x1920."""
    save = next(k for k, n in wf.items() if n["class_type"] == "SaveImage")
    wf["95"] = {"class_type": "ImageScale", "inputs": {"image": wf[save]["inputs"]["images"], "upscale_method": "lanczos",
                                                       "width": width, "height": height, "crop": "center"}}
    wf[save]["inputs"]["images"] = ["95", 0]
    return wf


def image_workflow(cfg, engine, prompt, width, height, seed, prefix, steps=None):
    c = cfg["image"]
    if engine == "flux-schnell":
        return flux_workflow(cfg, prompt, width, height, seed, prefix, unet=c["schnell_unet"], steps=c["schnell_steps"])
    if engine == "sdxl-juggernaut":
        return sdxl_workflow(cfg, prompt, width, height, seed, prefix)
    if engine == "qwen-image":
        return qwen_image_workflow(cfg, prompt, width, height, seed, prefix)
    if engine == "flux-kontext":
        return flux_workflow(cfg, prompt, width, height, seed, prefix, steps=steps)
    if engine == "z-image":
        return zimage_workflow(cfg, prompt, width, height, seed, prefix)
    if engine in ("flux2-klein-4b", "flux2-klein-9b"):
        return flux2_klein_workflow(cfg, engine, prompt, width, height, seed, prefix)
    if engine == "qwen-image-21":
        return qwen_image21_workflow(cfg, prompt, width, height, seed, prefix, steps=steps)
    return flux_workflow(cfg, prompt, width, height, seed, prefix, steps=steps)


def reference_workflow(cfg, engine, prompt, ref_name, width, height, seed, prefix, steps=None):
    """A new picture that keeps the character from the uploaded reference picture."""
    if engine in ("flux2-klein-4b", "flux2-klein-9b"):
        return flux2_klein_workflow(cfg, engine, prompt, width, height, seed, prefix, ref_name=ref_name)
    if engine == "qwen-image-21":
        return qwen_image21_workflow(cfg, prompt, width, height, seed, prefix, ref_name=ref_name, steps=steps)
    return kontext_workflow(cfg, prompt, ref_name, width, height, seed, prefix, steps)


def video_workflow(cfg, engine, prompt, start_name, end_name, width, height, seed, prefix):
    if engine in LTX2_ENGINES:
        return ltx2_workflow(cfg, engine, prompt, start_name, end_name, width, height, seed, prefix)
    if engine == "ltxv-2b":
        return ltxv_flf_workflow(cfg, prompt, start_name, end_name, width, height, seed, prefix)
    if engine == "hunyuan15":
        return hunyuan15_i2v_workflow(cfg, prompt, start_name, width, height, seed, prefix)
    return wan_flf_workflow(cfg, prompt, start_name, end_name, width, height, seed, prefix)


def add_zimage_polish(cfg, wf, prompt, seed, denoise):
    """Second pass: re-render the picture with Z-Image at low denoise. Keeps the character and composition
    from the first engine (e.g. FLUX Kontext) but gives Z-Image's photographic skin, hair and light."""
    z = cfg["zimage"]
    save = next(k for k, n in wf.items() if n["class_type"] == "SaveImage")
    src = wf[save]["inputs"]["images"]
    vae = next(["%s" % k, 0] for k, n in wf.items() if n["class_type"] == "VAELoader")
    wf.update({
        "60": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": z["unet"]}},
        "61": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["60", 0], "shift": z["shift"], "sampling": "flow"}},
        "62": {"class_type": "CLIPLoader", "inputs": {"clip_name": z["text_encoder"], "type": "lumina2", "device": "default"}},
        "63": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["62", 0]}},
        "64": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["63", 0]}},
        "65": {"class_type": "VAEEncode", "inputs": {"pixels": src, "vae": vae}},
        "66": {"class_type": "KSampler",
               "inputs": {"model": ["61", 0], "positive": ["63", 0], "negative": ["64", 0], "latent_image": ["65", 0],
                          "seed": seed, "steps": z["steps"], "cfg": 1.0, "sampler_name": "res_multistep",
                          "scheduler": "simple", "denoise": denoise}},
        "67": {"class_type": "VAEDecode", "inputs": {"samples": ["66", 0], "vae": vae}},
    })
    wf[save]["inputs"]["images"] = ["67", 0]
    return wf


def zimage_workflow(cfg, prompt, width, height, seed, prefix):
    z = cfg["zimage"]
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": z["unet"]}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": z["shift"], "sampling": "flow"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": z["text_encoder"], "type": "lumina2", "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                         "seed": seed, "steps": z["steps"], "cfg": 1.0, "sampler_name": "res_multistep",
                         "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": z["vae"]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


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


def kontext_workflow(cfg, prompt, ref_name, width, height, seed, prefix, steps=None):
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
                         "seed": seed, "steps": steps or c["kontext_steps"], "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def ltxv_flf_workflow(cfg, prompt, start_name, end_name, width, height, seed, prefix):
    """LTX-Video 2B distilled with the two keyframes as guides at the first and last frame."""
    wf = _ltxv_flf(cfg, prompt, start_name, end_name or start_name, width, height, seed, prefix)
    if not end_name:  # single picture: guide only the first frame
        for k in ("6", "8", "11"):
            wf.pop(k)
        wf["12"]["inputs"].update(positive=["10", 0], negative=["10", 1])
        wf["13"]["inputs"]["latent"] = ["10", 2]
        wf["15"]["inputs"]["latent_image"] = ["10", 2]
    return wf


def _ltxv_flf(cfg, prompt, start_name, end_name, width, height, seed, prefix):
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
    wf = _wan_flf(cfg, prompt, start_name, end_name or start_name, width, height, seed, prefix)
    if not end_name:  # single picture: plain image-to-video
        wf.pop("12")
        wf["13"]["inputs"].pop("end_image")
    return wf


def _wan_flf(cfg, prompt, start_name, end_name, width, height, seed, prefix):
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


def _unet(name):
    """GGUF files go through ComfyUI-GGUF, everything else through the built-in loader."""
    if name.endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}}


def flux2_klein_workflow(cfg, engine, prompt, width, height, seed, prefix, ref_name=None):
    """FLUX.2 klein (distilled, 4 steps). With ref_name the picture is added as a reference latent (character kept)."""
    k = cfg["flux2_klein"]
    v = k["4b" if engine == "flux2-klein-4b" else "9b"]
    w, h = width // 16 * 16, height // 16 * 16
    wf = {
        "1": _unet(v["unet"]),
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": v["text_encoder"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": k["vae"]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "Flux2Scheduler", "inputs": {"steps": k["steps"], "width": w, "height": h}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "cfg": 1.0}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["9", 0], "guider": ["10", 0], "sampler": ["8", 0], "sigmas": ["7", 0],
                          "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }
    if ref_name:
        wf.update({
            "20": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
            "21": {"class_type": "ImageScaleToTotalPixels",
                   "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "megapixels": k.get("ref_megapixels", 1.0),
                              "resolution_steps": 16}},
            "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["5", 0]}},
            "23": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["3", 0], "latent": ["22", 0]}},
            "24": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["4", 0], "latent": ["22", 0]}},
        })
        wf["10"]["inputs"].update(positive=["23", 0], negative=["24", 0])
    return wf


def qwen_image21_workflow(cfg, prompt, width, height, seed, prefix, ref_name=None, steps=None):
    """Qwen-Image 2.1 (ComfyUI's int8 files). With ref_name the picture goes into its encoder as image 1."""
    q = cfg["qwen_image_21"]
    wf = {
        "1": _unet(q["unet"]),
        "2": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": q["text_encoder"], "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}},
        "5": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["3", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width // 16 * 16, "height": height // 16 * 16, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["6", 0],
                         "seed": seed, "steps": steps or q["steps"], "cfg": q["cfg"], "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }
    if ref_name:
        wf["20"] = {"class_type": "LoadImage", "inputs": {"image": ref_name}}
        wf["5"]["inputs"].update({"images.image_1": ["20", 0], "vae": ["4", 0]})
    return wf


LTX_NEGATIVE = ("blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
                "flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, "
                "extra limbs, disfigured hands, watermark, text, subtitles")


def ltx2_workflow(cfg, engine, prompt, start_name, end_name, width, height, seed, prefix):
    """LTX-2.3 / LTX-2.5 (distilled, 8 sigmas) in one pass. Video and sound are generated together:
    frames come back through SaveImage, the soundtrack through SaveAudio.
    start+end = first/last frame guides; start only = image-to-video; neither = text-to-video."""
    c = cfg[engine]
    w, h = width // 32 * 32, height // 32 * 32
    n, fps = c["frames"], c["fps"]
    if engine == "ltx23":  # Gemma 3 plus this model's embeddings connectors; the audio VAE has its own loader
        enc = {"class_type": "LTXAVTextEncoderLoader",
               "inputs": {"text_encoder": c["text_encoder"], "ckpt_name": c["connectors"], "device": "default"}}
        avae = {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": c["audio_vae"]}}
    else:  # LTX-2.5: Gemma 4 with the projection built in
        enc = {"class_type": "CLIPLoader", "inputs": {"clip_name": c["text_encoder"], "type": "ltxv", "device": "default"}}
        avae = {"class_type": "VAELoader", "inputs": {"vae_name": c["audio_vae"]}}
    wf = {
        "1": _unet(c["unet"]),
        "2": enc,
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": c["video_vae"]}},
        "4": avae,
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": LTX_NEGATIVE, "clip": ["2", 0]}},
        "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": float(fps)}},
        "8": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w, "height": h, "length": n, "batch_size": 1}},
    }
    pos, neg, lat = ["7", 0], ["7", 1], ["8", 0]

    def picture(key, name):
        wf[key] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf[key + "s"] = {"class_type": "ImageScale", "inputs": {"image": [key, 0], "upscale_method": "lanczos",
                                                               "width": w, "height": h, "crop": "center"}}
        wf[key + "p"] = {"class_type": "LTXVPreprocess",
                         "inputs": {"image": [key + "s", 0], "img_compression": c["img_compression"]}}
        return [key + "p", 0]

    guided = bool(start_name and end_name)
    if guided:  # first + last frame
        for key, name, idx in (("10", start_name, 0), ("12", end_name, -1)):
            g = key + "g"
            wf[g] = {"class_type": "LTXVAddGuide",
                     "inputs": {"positive": pos, "negative": neg, "vae": ["3", 0], "latent": lat,
                                "image": picture(key, name), "frame_idx": idx, "strength": c["guide_strength"]}}
            pos, neg, lat = [g, 0], [g, 1], [g, 2]
    elif start_name:  # one picture: it becomes the first frame
        wf["14"] = {"class_type": "LTXVImgToVideoInplace",
                    "inputs": {"vae": ["3", 0], "image": picture("10", start_name), "latent": lat, "strength": 1.0,
                               "bypass": False}}
        lat = ["14", 0]
    wf.update({
        "20": {"class_type": "LTXVEmptyLatentAudio",
               "inputs": {"frames_number": n, "frame_rate": fps, "batch_size": 1, "audio_vae": ["4", 0]}},
        "21": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": lat, "audio_latent": ["20", 0]}},
        "22": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "23": {"class_type": "SamplerEulerAncestral", "inputs": {"eta": 0.0, "s_noise": 1.0}},
        "24": {"class_type": "ManualSigmas", "inputs": {"sigmas": c["sigmas"]}},
        "25": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": pos, "negative": neg, "cfg": 1.0}},
        "26": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["22", 0], "guider": ["25", 0], "sampler": ["23", 0], "sigmas": ["24", 0],
                          "latent_image": ["21", 0]}},
        "27": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["26", 1]}},
        "29": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["27", 0], "vae": ["3", 0], "tile_size": 512,
                                                          "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}},
        "30": {"class_type": "SaveImage", "inputs": {"images": ["29", 0], "filename_prefix": prefix}},
        "31": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["27", 1], "audio_vae": ["4", 0]}},
        "32": {"class_type": "SaveAudio", "inputs": {"audio": ["31", 0], "filename_prefix": prefix + "_audio"}},
    })
    if guided:  # drop the guide frames before decoding
        wf["28"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": pos, "negative": neg, "latent": ["27", 0]}}
        wf["29"]["inputs"]["samples"] = ["28", 2]
    return wf

"""Minimal ComfyUI API client plus the two workflows (Flux keyframe, Wan 2.2 first-last-frame)."""
import json
import os
import time
import urllib.parse
import urllib.request
import uuid

CLIENT_ID = uuid.uuid4().hex
# ComfyUI runs on this PC: never go through the Windows system proxy (a VPN app's proxy resets uploads to it)
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Stopped(Exception):
    """The job's Stop button was pressed while ComfyUI had our prompt queued or running."""


class Comfy:
    def __init__(self, url, should_stop=None, remote=False):
        self.url = url.rstrip("/")
        self.should_stop = should_stop  # polled while waiting on a prompt
        self.remote = remote  # a rented GPU behind a slow upload: big pictures go up as JPEG

    def _get(self, path, timeout=30):
        with DIRECT.open(self.url + path, timeout=timeout) as r:
            return r.read()

    def _post(self, path, payload, timeout=60):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with DIRECT.open(req, timeout=timeout) as r:
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
        ctype = "image/png"
        if self.remote and len(data) > 600_000:  # a 1080p PNG is ~3 MB: ~0.4 MB as a high-quality JPEG
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.open(io.BytesIO(data)).convert("RGB").save(buf, "JPEG", quality=95)
            data, ctype, name = buf.getvalue(), "image/jpeg", os.path.splitext(name)[0] + ".jpg"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(self.url + "/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with DIRECT.open(req, timeout=300 if self.remote else 60) as r:
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
        misses = 0
        while time.time() - start < timeout:
            if self.should_stop and self.should_stop():
                self.stop(pid)
                raise Stopped()
            try:
                hist = json.loads(self._get("/history/" + pid))
                misses = 0
            except (OSError, ValueError):  # ComfyUI can drop a poll while it loads big models; the job keeps running
                misses += 1
                if misses > 15:
                    raise
                time.sleep(2)
                continue
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


# Engines shown in the app's dropdowns, best picture quality first (the user's ranking, 3 Oct 2026).
# Trimmed 3 Oct 2026 to the engines in use; the other models were removed from disk (bigger ones will run on
# rented GPUs). Their workflow functions below still work if the files are downloaded again and re-listed.
# "files" are checked against the models folders to mark what's installed.
KREA2 = {"unet": "krea2_turbo_fp8_scaled.safetensors", "text_encoder": "qwen3vl_4b_fp8_scaled.safetensors",
         "vae": "qwen_image_vae.safetensors"}
# Krea 2 with a character picture: conradlocke/krea2-identity-edit (rank-128 copy of V2, rented GPU only). Licence: free
# for personal use; selling outputs, client work or monetised posts need the author's commercial licence.
KREA2_REF_LORA = "krea2_identity_edit_v2_r128.safetensors"
KREA2_EDIT_LORA = "krea2_general_edit_v2_r128.safetensors"  # Krea 2 edits; same repo, licence and nodes
# MOSS-Image SFT on this PC (native in ComfyUI from v0.42; Comfy-Org/MOSS-Image files, Apache-2.0). Settings from
# ComfyUI's own "MOSS-Image: Text to Image" template: 25 steps, CFG 2.5, er_sde / simple, uncond skip at layer 22
# for the first half; its negative without the "nsfw" word.
MOSS = {"unet": "moss_image_sft_int8_convrot.safetensors", "text_encoder": "qwen3.5_9b_w4a8_gptq.safetensors",
        "vae": "flux2-vae_small.safetensors", "steps": 25, "cfg": 2.5,
        "negative": "worst quality, bad anatomy, watermark, jpeg artifact"}
# FLUX.3 [klein] distilled on this PC (native in ComfyUI from v0.41; black-forest-labs/flux-3-image-klein-comfy,
# gated: the user signs in and accepts the licence). FLUX Kommunity licence: free incl. commercial use under $5M
# revenue. Settings from ComfyUI's "FLUX 3 Image Klein distilled" templates: 16 steps, CFG 1, euler / simple;
# character pictures and edits through ReferenceMultiLatent.
FLUX3 = {"unet": "flux3-image-klein-int8-convrot.safetensors", "text_encoder": "qwen3vl_4b_int8_convrot.safetensors",
         "vae": "flux2-vae.safetensors", "steps": 16}
IMAGE_ENGINES = {
    "krea2-turbo": {"label": "Krea 2 Turbo · GPU", "files": ["diffusion_models/" + KREA2["unet"],
                    "text_encoders/" + KREA2["text_encoder"], "vae/" + KREA2["vae"]]},
    "qwen-image-21": {"label": "Qwen-Image 2.1 (best quality, ~30 s, ~4 min with character picture, research licence)",
                      "files": ["diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
                                "text_encoders/qwen3vl_8b_int8_convrot.safetensors", "vae/qwen_image_2.1_vae_bf16.safetensors"]},
    "flux3-klein": {"label": "FLUX.3 [klein] (this PC, fast, character pictures + edits)",
                    "files": ["diffusion_models/" + FLUX3["unet"], "text_encoders/" + FLUX3["text_encoder"],
                              "vae/" + FLUX3["vae"]]},
    "moss-image": {"label": "MOSS-Image (this PC, slow on 8 GB)",
                   "files": ["diffusion_models/" + MOSS["unet"], "text_encoders/" + MOSS["text_encoder"],
                             "vae/" + MOSS["vae"]]},
    "z-image": {"label": "Z-Image Turbo (excellent photoreal, ~45 s)",
                "files": ["unet/z_image_turbo-Q8_0.gguf", "text_encoders/qwen_3_4b_fp8_mixed.safetensors"]},
    # only on a rented card ("rented" names the download that brings it, see cloudgpu)
    "ideogram4": {"label": "Ideogram 4.0 (best for text in pictures, non-commercial licence)", "files": [],
                  "rented": "ideogram"},
    "flux2-dev": {"label": "FLUX.2 [dev] (rich light and detail, ~3 min, 48 GB+ card, non-commercial licence)", "files": [],
                  "rented": "flux2"},
}
# Engines that can use the character picture as a reference
REFERENCE_ENGINES = ("qwen-image-21", "krea2-turbo", "flux3-klein")  # Krea 2: one picture, rented GPU only

LTX23_FILES = ["unet/ltx-2.3-22b-distilled-Q4_K_S.gguf", "checkpoints/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
               "vae/ltx-2.3-22b-distilled_video_vae.safetensors", "checkpoints/ltx-2.3-22b-distilled_audio_vae.safetensors",
               "text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"]
LTX25_FILES = ["unet/LTX-2.5-Distilled-Q4_K_S.gguf", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
               "vae/ltx-2.5-video-vae-bf16.safetensors", "vae/ltx-2.5-audio-vae-bf16.safetensors"]
LTX2_ENGINES = ("ltx23", "ltx25")
VIDEO_ENGINES = {
    "wan22-14b": {"label": "Wan 2.2 14B first+last frame (best quality, ~20 min/clip)",
                  "files": ["unet/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf", "unet/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"]},
}


T2V_ENGINES = {
    "wan22-5b": {"label": "Wan 2.2 5B text-to-video (experimental, ~6 min, weak at 9:16)",
                 "files": ["unet/Wan2.2-TI2V-5B-Q5_K_M.gguf", "vae/wan2.2_vae.safetensors",
                           "text_encoders/umt5-xxl-encoder-Q5_K_M.gguf"]},
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
    if engine == "krea2-turbo":
        return krea2_workflow(cfg, prompt, width, height, seed, prefix)
    if engine == "moss-image":
        return moss_workflow(cfg, prompt, width, height, seed, prefix, steps=steps)
    if engine == "flux3-klein":
        return flux3_workflow(cfg, prompt, width, height, seed, prefix)
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
    if engine == "flux2-dev":
        return flux2_workflow(prompt, width, height, seed, prefix)
    if engine == "ideogram4":
        return ideogram4_workflow(prompt, width, height, seed, prefix)
    if engine in ("flux2-klein-4b", "flux2-klein-9b"):
        return flux2_klein_workflow(cfg, engine, prompt, width, height, seed, prefix)
    if engine == "qwen-image-21":
        return qwen_image21_workflow(cfg, prompt, width, height, seed, prefix, steps=steps)
    return flux_workflow(cfg, prompt, width, height, seed, prefix, steps=steps)


def reference_workflow(cfg, engine, prompt, ref_name, width, height, seed, prefix, steps=None, edit=False):
    """A new picture that keeps the character from the uploaded reference picture(s). ref_name is one uploaded name
    or a list; only Qwen-Image 2.1 takes several, the older engines use the first."""
    if engine == "qwen-image-21":
        return qwen_image21_workflow(cfg, prompt, width, height, seed, prefix, ref_name=ref_name, steps=steps, edit=edit)
    if engine == "flux3-klein":
        return flux3_workflow(cfg, prompt, width, height, seed, prefix,
                              ref_names=[ref_name] if isinstance(ref_name, str) else list(ref_name), edit=edit)
    ref_name = ref_name if isinstance(ref_name, str) else ref_name[0]
    if engine == "krea2-turbo":
        return krea2_workflow(cfg, prompt, width, height, seed, prefix, ref_name=ref_name, edit=edit)
    if engine in ("flux2-klein-4b", "flux2-klein-9b"):
        return flux2_klein_workflow(cfg, engine, prompt, width, height, seed, prefix, ref_name=ref_name)
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


def krea2_workflow(cfg, prompt, width, height, seed, prefix, ref_name=None, edit=False):
    """Official Krea 2 Turbo recipe: Euler/simple, 8 steps, CFG 1, zero negative conditioning.
    With ref_name: the Identity Edit V2 LoRA and the comfyui-krea2edit nodes (rented GPU only) put the person from
    the character picture into the new picture, with the model card's settings for one reference (ref_boost 2,
    grounding 768 px, reference placed beside the canvas so the pose and framing are free). Trained for <= 2 MP.
    edit=True: ref_name is the finished picture and prompt the instruction; General Edit V2 with the card's "most
    edits" settings (ref_boost 4, reference aligned so everything not named stays put). Removals need Krea 2 Raw at
    CFG 3 (not installed): Turbo tends to redraw the subject instead."""
    model = {**KREA2, **cfg.get("krea2", {})}
    if ref_name and width * height > 2_000_000:
        s = (2_000_000 / (width * height)) ** 0.5
        width, height = int(width * s) // 16 * 16, int(height * s) // 16 * 16
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": model["text_encoder"], "type": "krea2", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": (width + 7) // 8 * 8,
                                                          "height": (height + 7) // 8 * 8, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
              "latent_image": ["5", 0], "seed": seed, "steps": 8, "cfg": 1.0,
              "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": model["vae"]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["7", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }
    if ref_name:
        wf.update({
            "20": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
            # a phone photo would be VAE-encoded at full size otherwise; an edited picture keeps the output's detail
            "21": {"class_type": "ImageScaleToTotalPixels",
                   "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "resolution_steps": 16,
                              "megapixels": round(width * height / 1e6, 2) if edit else 1.0}},
            "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["7", 0]}},
            "23": {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": KREA2_EDIT_LORA if edit else KREA2_REF_LORA,
                              "strength_model": 1.0}},
            "24": {"class_type": "Krea2EditModelPatch",
                   "inputs": {"model": ["23", 0], "source_latent": ["22", 0], "vae": ["7", 0], "source_image": ["21", 0],
                              "target_latent": ["5", 0], "fit_mode": "fit", "ref_boost": 4.0 if edit else 2.0,
                              "ref_position": "aligned (trained)" if edit else "beside (right of target)"}},
        })
        # the text encoder sees the picture while reading the prompt (both nodes are required, per the node pack)
        wf["3"] = {"class_type": "Krea2EditGroundedEncode",
                   "inputs": {"clip": ["2", 0], "prompt": prompt, "image": ["21", 0], "grounding_px": 768}}
        wf["6"]["inputs"]["model"] = ["24", 0]
    return wf


def flux3_workflow(cfg, prompt, width, height, seed, prefix, ref_names=None, edit=False):
    """FLUX.3 [klein] distilled, ComfyUI's template recipe (see FLUX3); the step count is fixed (step-distilled).
    ref_names: character pictures (a new picture at our size) or, with edit=True, the picture being edited: then
    the output takes that picture's shape at our pixel count, so the edit keeps its framing."""
    f = {**FLUX3, **cfg.get("flux3", {})}
    w, h = (width + 15) // 16 * 16, (height + 15) // 16 * 16
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": f["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": f["text_encoder"], "type": "flux3", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "5": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
              "latent_image": ["5", 0], "seed": seed, "steps": f["steps"], "cfg": 1.0,
              "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }
    if ref_names:
        refs = {"positive": ["3", 0], "negative": ["4", 0], "vae": ["8", 0], "resolution": "resize",
                # an edited picture keeps the output's detail; character pictures go in at ~1 MP like the template
                "resolution.megapixels": round(w * h / (1024 * 1024), 2) if edit else 1.0}
        for i, name in enumerate(ref_names):
            wf[str(20 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
            refs[f"images.image{i}"] = [str(20 + i), 0]
        wf["11"] = {"class_type": "ReferenceMultiLatent", "inputs": refs}
        wf["7"]["inputs"].update(positive=["11", 0], negative=["11", 1])
        if edit:
            wf["7"]["inputs"]["latent_image"] = ["11", 2]
            del wf["5"]
    return wf


def moss_workflow(cfg, prompt, width, height, seed, prefix, steps=None):
    """MOSS-Image SFT, ComfyUI's template recipe (see MOSS). A LoRA goes between the loader and the uncond skip."""
    m = {**MOSS, **cfg.get("moss_image", {})}
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "moss_image", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": m["negative"], "clip": ["2", 0]}},
        "5": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": (width + 15) // 16 * 16,
                                                               "height": (height + 15) // 16 * 16, "batch_size": 1}},
        "6": {"class_type": "SkipLayerGuidanceDiTSimple",
              "inputs": {"model": ["1", 0], "double_layers": "", "single_layers": "22", "start_percent": 0.0,
                         "end_percent": 0.5}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "positive": ["3", 0], "negative": ["4", 0],
              "latent_image": ["5", 0], "seed": seed, "steps": steps or m["steps"], "cfg": m["cfg"],
              "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


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


def apply_lora_settings(wf, lora):
    """A LoRA's (or swapped model's) own sampling settings from config: "cfg", "negative", "sampler", "scheduler".
    Some fine-tunes want true CFG with a negative prompt where the base recipe runs CFG 1 (Noct Q: CFG 3), or
    another sampler (the TheseAlpacas NSFW LoRA: er_sde + beta, "euler gives poor results")."""
    for n in wf.values():
        if n["class_type"] == "KSampler":
            if "cfg" in lora:
                n["inputs"]["cfg"] = float(lora["cfg"])
            if "sampler" in lora:
                n["inputs"]["sampler_name"] = lora["sampler"]
            if "scheduler" in lora:
                n["inputs"]["scheduler"] = lora["scheduler"]
        if "negative" in lora and n["class_type"] == "TextEncodeQwenImage21":
            n["inputs"]["negative_prompt"] = lora["negative"]
    return wf


def set_unet(wf, name):
    """Swap the workflow's diffusion model for another file of the same family (a merged fine-tune such as Noct Q
    for Qwen-Image 2.1), keeping every other node."""
    loader = next(k for k, n in wf.items() if n["class_type"] in ("UNETLoader", "UnetLoaderGGUF"))
    wf[loader] = _unet(name)
    return wf


def add_lora(wf, lora_name, strength=1.0):
    """Put a LoRA (e.g. a trained character) between the workflow's model loader and everything that uses it."""
    loader = next(k for k, n in wf.items() if n["class_type"] in ("UNETLoader", "UnetLoaderGGUF", "CheckpointLoaderSimple"))
    for n in wf.values():
        for key, v in n["inputs"].items():
            if v == [loader, 0]:
                n["inputs"][key] = ["45", 0]
    wf["45"] = {"class_type": "LoraLoaderModelOnly",
                "inputs": {"model": [loader, 0], "lora_name": lora_name, "strength_model": float(strength)}}
    return wf


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


def qwen_image21_workflow(cfg, prompt, width, height, seed, prefix, ref_name=None, steps=None, edit=False):
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
    names = [ref_name] if isinstance(ref_name, str) else list(ref_name or [])
    for k, name in enumerate(names):  # several references: the encoder sees them as Picture 1, 2, ...
        wf[str(20 + k)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["5"]["inputs"][f"images.image_{k + 1}"] = [str(20 + k), 0]
    if names:
        wf["5"]["inputs"]["vae"] = ["4", 0]
    if edit:  # draw on the edited picture's own grid (any other size shifts the edit); ~1.6 MP keeps detail on 8 GB
        wf["5"]["inputs"]["resolution"] = 1280
        wf["7"]["inputs"]["latent_image"] = ["5", 2]
        del wf["6"]
    return wf


LTX_NEGATIVE = ("blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
                "flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, "
                "extra limbs, disfigured hands, watermark, text, subtitles")

# Engines that only run on a rented card (cloud_setup.sh downloads them; the files never come to this PC).
# MiniMax H3: #1 open image-to-video with sound (Oct 2026); 80 GB+ cards, and its licence excludes the US, EU, UK and
# Korea, so cloudgpu rents it only in other data centres. First+last frame model, base 20 steps (the turbo LoRA took
# 3 min instead of 7 for 5 s at 768x1344 on an RTX PRO 6000, at lower quality). The fp16 video VAE, not the int8 one:
# its decode kernel failed on an Iceland 5090 ("CUDA driver version is insufficient for CUDA runtime version", 6 Oct).
H3 = {"unet": "minimax_h3_fl2va_int8_convrot.safetensors", "video_vae": "minimax_h3_video_vae_fp16.safetensors",
      "audio_vae": "minimax_h3_audio_vae_fp32.safetensors", "text_encoder": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
      "text_encoder_blackwell": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "steps": 20, "frames": 124, "fps": 24,
      "ref_unet": "fasth3_omniref_8step_pruned_int8_convrot.safetensors"}
# FastH3 Omni (the reference-to-video model on the rented card): ComfyUI's video_fasth3_ominiref_r2v template.
# 8 steps, euler, the checkpoint's own video/audio sigma shifts; dense attention (the template's VSA sparse
# attention only speeds up full-size 15 s clips and needs kernels this app hasn't tested on rented cards).
OMNI = {"steps": 8, "shift_video": 12.0, "shift_audio": 3.0}
# Ideogram 4.0: #2 open text-to-image (Oct 2026), best at text inside pictures; two models (with and without the
# prompt) through DualModelGuider as in ComfyUI's template, "Quality" preset. Weights: non-commercial licence.
IDEOGRAM4 = {"unet": "ideogram4_int8_convrot.safetensors", "unet_uncond": "ideogram4_unconditional_int8_convrot.safetensors",
             "text_encoder": "qwen3vl_8b_fp8_scaled.safetensors", "vae": "flux2-vae.safetensors",
             "steps": 48, "mu": 0.0, "std": 1.5, "cfg": 7.0, "late_cfg": 3.0}
# LTX-2.5 on a rented card: the int8 transformer instead of the PC's 4-bit GGUF (same workflow)
LTX25_POD = {"unet": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
             "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
             "video_vae": "ltx-2.5-video-vae-bf16.safetensors", "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors"}


def h3_workflow(prompt, start_name, end_name, width, height, seed, prefix, text_encoder=None, unet=None, short=768,
                refs=None, ref_unet=None, frames=None, formatted=False):
    """MiniMax H3 with sound: start+end = first/last frame, start = image-to-video, neither = text-to-video. Frames come
    back through SaveImage and the soundtrack through SaveAudio, like LTX-2, so the pipeline stores both the same way.
    refs = character pictures (uploaded names): the reference-to-video model (FastH3 Omni) keeps those people, named
    <Picture 1>... in the prompt. formatted = the prompt is already in H3's own format (llm.h3_prompt), so no lead-in."""
    c = H3
    # H3's native canvas: a 768 px short edge, at most 768x1344, multiples of 32 (MiniMax's prompting tips); the
    # project's 480p/720p choice only gives the shape. short=480 on 32 GB cards (the slimmed model, 864x480).
    s_, long_ = min(width, height), max(width, height)
    long_ = min(1344 * short // 768 // 32 * 32, round(long_ * short / s_ / 32) * 32)
    width, height = (short, long_) if width <= height else (long_, short)
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet or c["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder or c["text_encoder"], "type": "minimax",
                                                     "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": c["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": c["audio_vae"]}},
        "6": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": width // 32 * 32,
                         "height": height // 32 * 32, "length": frames or c["frames"]}},
        "8": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": c["steps"],
                                                         "denoise": 1.0}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["6", 0]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["9", 0],
                                                                 "sigmas": ["8", 0], "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
        "16": {"class_type": "SaveAudio", "inputs": {"audio": ["14", 0], "filename_prefix": prefix + "_audio"}},
    }
    if refs:
        tags = " and ".join(f"<Picture {k + 1}>" for k in range(len(refs)))
        lead = ("Use <Picture 1> as the reference for the main character: keep the same face, hair and clothes"
                + (f"; {tags} show the people and things to use" if len(refs) > 1 else "") + ". ")
        wf["1"]["inputs"]["unet_name"] = ref_unet or c["ref_unet"]
        wf["6"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                   "inputs": {**wf["6"]["inputs"], "prompt": prompt if formatted else lead + prompt,
                              "audio_vae": ["4", 0], "ref_image_size": "match"}}
        # FastH3 Omni: 8 euler steps on the checkpoint's sigma shifts (the base model's 20 res_multistep steps elsewhere)
        wf["17"] = {"class_type": "MiniMaxH3SigmaShift",
                    "inputs": {"model": ["1", 0], "shift_video": OMNI["shift_video"], "shift_audio": OMNI["shift_audio"]}}
        wf["8"]["inputs"].update(model=["17", 0], steps=OMNI["steps"])
        wf["9"]["inputs"]["sampler_name"] = "euler"
        wf["11"]["inputs"]["model"] = ["17", 0]
        for k, name in enumerate(refs[:9]):
            wf[f"r{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            wf["6"]["inputs"][f"ref_images.ref_image_{k}"] = [f"r{k}", 0]
        return wf
    for key, name, slot in (("5", start_name, "first_frame"), ("7", end_name, "last_frame")):
        if name:
            wf[key] = {"class_type": "LoadImage", "inputs": {"image": name}}
            wf["6"]["inputs"][slot] = [key, 0]
    return wf


# FLUX.2 [dev] (Black Forest Labs, 32B, fp8 35.5 GB + 18 GB Mistral text encoder): only on rented 48 GB+ cards. 50 steps:
# ComfyUI's template uses 20, which looked flat next to Qwen-Image 2.1; 50 with a long prompt gave rich light and skin
# (7 Oct, L40S: 179 s at 1088x1920, Qwen 46 s). Weights: non-commercial licence.
FLUX2 = {"unet": "flux2_dev_fp8mixed.safetensors", "text_encoder": "mistral_3_small_flux2_fp8.safetensors",
         "vae": "flux2-vae.safetensors", "steps": 50, "guidance": 4.0}


def flux2_workflow(prompt, width, height, seed, prefix):
    """FLUX.2 [dev] text-to-image (ComfyUI's image_flux2_fp8 template without its reference pictures)."""
    c = FLUX2
    w, h = (width + 15) // 16 * 16, (height + 15) // 16 * 16
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": c["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": c["text_encoder"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": c["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": c["guidance"]}},
        "6": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": c["steps"], "width": w, "height": h}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["6", 0], "sampler": ["7", 0],
                                                                 "sigmas": ["8", 0], "latent_image": ["10", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def ideogram4_workflow(prompt, width, height, seed, prefix):
    """Ideogram 4.0 text-to-image (ComfyUI's image_ideogram4_t2i template). The model reads a JSON prompt; a plain one
    goes in as its high-level description."""
    c = IDEOGRAM4
    w, h = max((width + 15) // 16 * 16, 256), max((height + 15) // 16 * 16, 256)
    text = prompt if prompt.lstrip().startswith("{") else json.dumps({"high_level_description": prompt}, ensure_ascii=False)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": c["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": c["unet_uncond"], "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": c["text_encoder"], "type": "ideogram4", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": c["vae"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["3", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "CFGOverride", "inputs": {"model": ["1", 0], "cfg": c["late_cfg"], "start_percent": 0.7,
                                                      "end_percent": 1.0}},
        "8": {"class_type": "DualModelGuider", "inputs": {"model": ["7", 0], "model_negative": ["2", 0], "positive": ["5", 0],
                                                          "negative": ["6", 0], "cfg": c["cfg"]}},
        "9": {"class_type": "Ideogram4Scheduler", "inputs": {"steps": c["steps"], "width": w, "height": h, "mu": c["mu"],
                                                             "std": c["std"]}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["8", 0], "sampler": ["10", 0],
                                                                 "sigmas": ["9", 0], "latent_image": ["12", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix}},
    }


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

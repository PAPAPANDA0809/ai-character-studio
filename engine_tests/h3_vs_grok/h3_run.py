"""MiniMax H3 image-to-video test on a rented pod (ComfyUI reached through an SSH tunnel on 127.0.0.1:8189).
Built from ComfyUI's official templates video_minimax_h3_i2v.json and video_pika_h3_flex_fl2va.json.
Usage: python h3_run.py base|turbo|flex [seed]   (H3_UNET=minimax_h3_fl2va_pruned_int8_convrot.safetensors on a 32 GB card)
flex = Pika H3 Flex, 6 steps; its LoRA goes in ComfyUI/models/loras:
  wget https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/pika_h3_flex_fl2va_v1_bf16.safetensors"""
import json, os, sys, time, urllib.request, uuid

URL = "http://127.0.0.1:8189"
UNET = os.environ.get("H3_UNET", "minimax_h3_fl2va_int8_convrot.safetensors")
LORAS = {"turbo": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
         "flex": "pika_h3_flex_fl2va_v1_bf16.safetensors"}
# Flex's own schedule (knots 1000/750/500/250/100/40/0 at video shift 12) with fresh noise each step (lcm):
# picking 6 steps in a generic scheduler doesn't reproduce it
FLEX_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
HERE = os.path.dirname(os.path.abspath(__file__))
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # the VPN's system proxy breaks localhost


def call(path, data=None, headers=None):
    req = urllib.request.Request(URL + path, data=data, headers=headers or {})
    with DIRECT.open(req, timeout=120) as r:
        return r.read()


def upload(path):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{os.path.basename(path)}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{b}--\r\n".encode()
    return json.loads(call("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"}))["name"]


def workflow(image, prompt, mode, seed, width=768, height=1344, length=124):
    model = ["1", 0]
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                                                      "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                                                                "width": width, "height": height, "length": length,
                                                                "first_frame": ["5", 0]}},
    }
    if mode in LORAS:
        wf["7"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": LORAS[mode], "strength_model": 1.0}}
        model = ["7", 0]
    if mode == "flex":
        wf["8"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": FLEX_SIGMAS}}
    else:
        wf["8"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple",
                                                              "steps": 8 if mode == "turbo" else 20, "denoise": 1.0}}
    wf.update({
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm" if mode == "flex" else "res_multistep"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["6", 0]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["9", 0],
                                                                 "sigmas": ["8", 0], "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"h3/{mode}",
                                                     "format": "mp4", "format.codec": "h264"}},
    })
    return wf


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode not in ("base", "turbo", "flex"):
        sys.exit("usage: python h3_run.py base|turbo|flex [seed]")
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261003
    prompt = open(os.path.join(HERE, "prompt.txt"), encoding="utf-8").read().strip()
    image = "start_h3.png"  # copied into the pod's ComfyUI/input with scp (uploads through the tunnel crawl)
    pid = json.loads(call("/prompt", json.dumps({"prompt": workflow(image, prompt, mode, seed)}).encode(),
                          {"Content-Type": "application/json"}))["prompt_id"]
    t = time.time()
    print(f"{mode}: queued {pid}", flush=True)
    while True:
        time.sleep(10)
        h = json.loads(call(f"/history/{pid}")).get(pid)
        if not h:
            continue
        if h["status"].get("status_str") == "error":
            print(json.dumps(h["status"], indent=1)[:3000])
            sys.exit(1)
        for out in h["outputs"].values():
            for f in out.get("images", []) + out.get("videos", []) + out.get("gifs", []):
                # fetched afterwards with scp: downloads through the tunnel crawl too
                print(f"{mode}: done in {time.time() - t:.0f}s -> output/{f.get('subfolder', '')}/{f['filename']}")
                return


if __name__ == "__main__":
    import urllib.parse
    main()

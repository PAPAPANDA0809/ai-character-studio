"""Runs on the rented pod (cloud_setup.sh unpacks it from LC_B64; cloud_agent.py starts it once per job):
LongCat-Video-Avatar 1.6 (Meituan, MIT licence) makes a picture speak a voice track, with lip sync and body motion.

    python3 /root/longcat_run.py <job dir>

The job dir holds job.json {prompt, seconds, res, seed, int8}, image.png and audio.wav (16 kHz mono, made by the app),
and gets status ("loading the models", "part 2 of 5", DONE or ERROR: ...) and out.mp4 (25 fps, no sound: the app
adds the voice back). It follows Meituan's run_demo_avatar_single_audio_to_video.py at the pinned commit for 1.6
(8-step distilled sampling, the DMD + OPSD LoRA mix with --mix_steps 4 --mix_alpha 1.0), with these differences:
one GPU and one process (no torchrun), no vocal separation (the app sends a clean voice track; the separator's
packages are left out), and the frames go straight into an H.264 file through imageio-ffmpeg's own ffmpeg. The
model only knows FlashAttention or xformers; the pod has neither, so flash_attn below is a stand-in on PyTorch's
scaled_dot_product_attention (which picks its own fast kernel), put on the path before the model code loads."""
import json, math, os, subprocess, sys, time, traceback, types, wave

JOB = sys.argv[1]
CODE = "/root/longcat/code"
WEIGHTS = "/root/longcat/weights"
NUM_FRAMES, COND_FRAMES, FPS = 93, 13, 25  # LongCat's clip: 93 frames, each continuation re-uses the last 13
NEGATIVE = ("Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
            "messy background, three legs, many people in the background, walking backwards")


def status(text):
    with open(os.path.join(JOB, "status.tmp"), "w") as f:
        f.write(text)
    os.replace(os.path.join(JOB, "status.tmp"), os.path.join(JOB, "status"))


def segments_for(seconds):
    """Parts needed for this many seconds: the first makes 93 frames, each further one 80 more."""
    return 1 + max(0, math.ceil((seconds * FPS - NUM_FRAMES) / (NUM_FRAMES - COND_FRAMES)))


def install_flash_attn_shim():
    import torch
    import torch.nn.functional as F

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
        # [B, S, H, D] in and out, as flash_attn has it
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=causal, scale=softmax_scale)
        return o.transpose(1, 2)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                               dropout_p=0.0, softmax_scale=None, causal=False, **_):
        # packed [total, H, D] sequences, one attention per (query, key) pair of slices
        cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        out = torch.empty_like(q)
        for i in range(len(cq) - 1):
            qs, ks = q[cq[i]:cq[i + 1]], k[ck[i]:ck[i + 1]]
            o = F.scaled_dot_product_attention(qs.transpose(0, 1)[None], ks.transpose(0, 1)[None],
                                               v[ck[i]:ck[i + 1]].transpose(0, 1)[None], is_causal=causal,
                                               scale=softmax_scale)
            out[cq[i]:cq[i + 1]] = o[0].transpose(0, 1)
        return out

    import importlib.machinery
    shim = types.ModuleType("flash_attn")
    # transformers asks importlib about flash_attn: with a spec but no installed package it counts as absent, so the
    # UMT5 / Whisper models keep their own attention (without a spec, find_spec raises and the import fails)
    shim.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    shim.flash_attn_func = flash_attn_func
    shim.flash_attn_varlen_func = flash_attn_varlen_func
    shim.__version__ = "sdpa-shim"
    sys.modules["flash_attn"] = shim


def read_wav(path):
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 16000:
            raise RuntimeError("the voice track must be 16 kHz mono 16-bit WAV")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


class Writer:
    """RGB frames into an H.264 MP4 through imageio-ffmpeg's bundled ffmpeg."""
    def __init__(self, path, w, h):
        import imageio_ffmpeg
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", path],
                                  stdin=subprocess.PIPE)
        self.n = 0

    def add(self, frames, limit):
        for fr in frames:
            if self.n >= limit:
                return
            self.p.stdin.write(fr.tobytes())
            self.n += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg couldn't write the video")


def main():
    with open(os.path.join(JOB, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    seconds = max(1.0, float(job["seconds"]))
    res = job.get("res") if job.get("res") in ("480p", "720p") else "480p"
    n_seg = segments_for(seconds)

    status("loading the models (about 2 minutes)")
    sys.path.insert(0, CODE)
    install_flash_attn_shim()
    import numpy as np
    import torch
    import torch.distributed as dist
    from PIL import Image
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
    from longcat_video.modules.quantization import load_quantized_dit
    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1)
    context_parallel_util.init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)
    split = context_parallel_util.get_optimal_split(1)

    base = os.path.join(WEIGHTS, "LongCat-Video")
    ckpt = os.path.join(WEIGHTS, "LongCat-Video-Avatar-1.6")
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt, subfolder="scheduler")
    if job.get("int8"):
        dit = load_quantized_dit(ckpt, subfolder="base_model_int8", cp_split_hw=split)
    else:
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(ckpt, subfolder="base_model", cp_split_hw=split,
                                                                  torch_dtype=torch.bfloat16)
    dit = dit.to(0)
    dit.load_lora(os.path.join(ckpt, "loraA", "lora.safetensors"), "dmd", multiplier=1.0, lora_network_dim=128,
                  lora_network_alpha=64)
    dit.load_lora(os.path.join(ckpt, "loraB", "lora.safetensors"), "opsd", multiplier=1.0, lora_network_dim=128,
                  lora_network_alpha=64)
    dit.enable_lora_mix("dmd", "opsd", 1.0, 4)  # Meituan's 1.6 default: 4 DMD steps, then 4 OPSD steps
    whisper = os.path.join(ckpt, "whisper-large-v3")
    pipe = LongCatVideoAvatarPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                                      dit=dit, audio_encoder=get_audio_encoder(whisper, "avatar-v1.5").to(0),
                                      audio_feature_extractor=get_audio_feature_extractor(whisper, "avatar-v1.5"),
                                      model_type="avatar-v1.5")
    pipe.to(0)

    status("reading the voice")
    speech = read_wav(os.path.join(JOB, "audio.wav"))
    want = (NUM_FRAMES + (n_seg - 1) * (NUM_FRAMES - COND_FRAMES)) / FPS + 1.0  # what the parts cover, + 1 s
    if len(speech) < want * 16000:
        speech = np.append(speech, np.zeros(math.ceil(want * 16000) - len(speech), dtype=np.float32))
    emb = pipe.get_audio_embedding(speech, fps=FPS, device=0, sample_rate=16000, model_type="avatar-v1.5",
                                   use_loudness_norm=False)
    if torch.isnan(emb).any():
        raise RuntimeError("the voice track couldn't be read (broken audio embedding)")
    offsets = torch.arange(5) - 2

    def audio_for(start):
        idx = torch.arange(start, start + NUM_FRAMES).unsqueeze(1) + offsets.unsqueeze(0)
        return emb[torch.clamp(idx, 0, emb.shape[0] - 1)][None].to(0)

    gen = torch.Generator(device=0)
    gen.manual_seed(int(job.get("seed") or 42))
    prompt = job.get("prompt") or "A person is talking naturally to the camera."
    common = dict(prompt=prompt, negative_prompt=NEGATIVE, num_inference_steps=8, text_guidance_scale=1.0,
                  audio_guidance_scale=1.0, output_type="both", generator=gen, use_distill=True)
    limit = int(round(seconds * FPS))
    t0 = time.time()
    status(f"part 1 of {n_seg}")
    image = Image.open(os.path.join(JOB, "image.png")).convert("RGB")
    video, latent = pipe.generate_ai2v(image=image, resolution=res, num_frames=NUM_FRAMES,
                                       audio_emb=audio_for(0), **common)
    arr = (video[0] * 255).astype(np.uint8)
    h, w = arr.shape[1:3]
    out = Writer(os.path.join(JOB, "out.part.mp4"), w, h)
    out.add(arr, limit)
    ref_latent = latent[:, :, :1].clone()
    frames = [Image.fromarray(x) for x in arr]
    start = 0
    for s in range(1, n_seg):
        per = (time.time() - t0) / s
        status(f"part {s + 1} of {n_seg} · about {max(1, round(per * (n_seg - s) / 60))} min left")
        start += NUM_FRAMES - COND_FRAMES
        video, latent = pipe.generate_avc(video=frames, video_latent=latent, height=h, width=w, num_frames=NUM_FRAMES,
                                          num_cond_frames=COND_FRAMES, use_kv_cache=False, offload_kv_cache=False,
                                          enhance_hf=False, audio_emb=audio_for(start), ref_latent=ref_latent,
                                          ref_img_index=10, mask_frame_range=0, **common)
        arr = (video[0] * 255).astype(np.uint8)
        out.add(arr[COND_FRAMES:], limit)
        frames = [Image.fromarray(x) for x in arr]
        torch.cuda.empty_cache()
    out.close()
    os.replace(os.path.join(JOB, "out.part.mp4"), os.path.join(JOB, "out.mp4"))
    status("DONE")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        traceback.print_exc()
        status("ERROR: " + (str(e).strip().splitlines() or [type(e).__name__])[-1][:300])
        sys.exit(1)

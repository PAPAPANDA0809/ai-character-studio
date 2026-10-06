#!/bin/bash
# Reelmint fast-GPU pod (RunPod, runpod/pytorch image with torch 2.9.1 for CUDA 12.8): ComfyUI v0.38.0 (the version on
# the PC). The app sends this script to the pod's agent (cloud_agent.py), which runs it, and puts two lines after the
# first one: VIDEO=h3|ltx25 (the card's video engine: MiniMax H3 on 80 GB+ cards where its licence allows, else
# LTX-2.5) and H3_TE (H3's text encoder for this card).
# Progress goes to /root/status, ending in READY or ERROR: ...; trained characters arrive in /root/loras through the agent.
# Before READY: Qwen-Image 2.1 (int8) and the video engine. After READY, in the background: full-quality Qwen-Image 2.1
# (bf16) and Ideogram 4.0; /root/done_qwen_hq and /root/done_ideogram tell the app each one is in.
# Lessons from the first pods: pip resolving the full requirements tried to replace torch (gigabytes), and
# pypi.org downloads stalled in some data centres, so packages are pinned and installed without resolving,
# with short timeouts, retries and a mirror as fallback; Hugging Face downloads are fast everywhere.
VIDEO=${VIDEO:-ltx25}
st() { echo "$*" > /root/status; }
cd /root

# community hosts can hand out a card PyTorch can't use ("CUDA unknown error"): find out before downloading anything
st "testing the GPU"
gpu=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>&1 | tail -1)
case "$gpu" in
  True) ;;
  False) st "ERROR: FAULTY_GPU"; exit 1 ;;
  *) st "ERROR: PyTorch didn't load on the pod: $gpu"; exit 1 ;;
esac

st "installing ComfyUI"
[ -d ComfyUI ] || git clone -q --depth 1 --branch v0.38.0 https://github.com/comfyanonymous/ComfyUI || { st "ERROR: could not download ComfyUI"; exit 1; }
cd ComfyUI
mkdir -p /root/loras
printf 'reelmint:\n  base_path: /root\n  loras: loras\n' > extra_model_paths.yaml

H=https://huggingface.co
Q=$H/Comfy-Org/Qwen-Image-2.1/resolve/main
M=$H/Comfy-Org/MiniMax-H3/resolve/main
L=$H/comfyicu/LTX-2.5/resolve/main  # open mirror of Lightricks/LTX-2.5 (the original asks for a Hugging Face login)
I=$H/Comfy-Org/Ideogram-4/resolve/main
# cloud_dl.py (sent by the app in DL_B64): 8 connections a file, stalled pieces fetched again. A single curl
# connection crawled on some hosts and hung forever at 99 % on one (5 Oct).
echo "$DL_B64" | base64 -d > /root/dl.py
get() { python3 /root/dl.py "$1" "$2" 2>> /root/dl.log || echo "$2" >> /root/dl_fail; }
(
  get $Q/diffusion_models/qwen_image_2.1_int8_convrot.safetensors models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors &
  get $Q/text_encoders/qwen3vl_8b_int8_convrot.safetensors models/text_encoders/qwen3vl_8b_int8_convrot.safetensors &
  get $Q/vae/qwen_image_2.1_vae_bf16.safetensors models/vae/qwen_image_2.1_vae_bf16.safetensors &
  get $H/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors models/upscale_models/4x-UltraSharp.safetensors &
  if [ "$VIDEO" = h3 ]; then
    H3_UNET=${H3_UNET:-minimax_h3_fl2va_int8_convrot.safetensors}  # or the slimmed (pruned) one on 32-48 GB cards
    get $M/diffusion_models/$H3_UNET models/diffusion_models/$H3_UNET &
    get $M/text_encoders/$H3_TE models/text_encoders/$H3_TE &
    get $M/vae/minimax_h3_video_vae_int8_convrot.safetensors models/vae/minimax_h3_video_vae_int8_convrot.safetensors &
    get $M/vae/minimax_h3_audio_vae_fp32.safetensors models/vae/minimax_h3_audio_vae_fp32.safetensors &
  else
    get $L/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors models/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors &
    get $L/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors models/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors &
    get $L/vae/ltx-2.5-video-vae-bf16.safetensors models/vae/ltx-2.5-video-vae-bf16.safetensors &
    get $L/vae/ltx-2.5-audio-vae-bf16.safetensors models/vae/ltx-2.5-audio-vae-bf16.safetensors &
  fi
  wait
  touch /root/dl_done
) &
case "$VIDEO/$H3_UNET/$H3_TE" in
  h3/*pruned*/*nvfp4*) TOTAL=58000; NAME="MiniMax H3 lite" ;;
  h3/*pruned*) TOTAL=69000; NAME="MiniMax H3 lite" ;;
  h3/*/*nvfp4*) TOTAL=71000; NAME="MiniMax H3" ;;
  h3/*) TOTAL=82000; NAME="MiniMax H3" ;;
  *) TOTAL=56000; NAME="LTX-2.5" ;;
esac

st "installing Python packages"
PKGS="torchsde==0.2.6 trampoline==0.1.2 numpy==2.5.3 einops==0.8.2 transformers==5.17.0 tokenizers==0.23.2
 sentencepiece==0.2.2 safetensors==0.8.0 huggingface-hub==1.33.0 hf-xet==1.6.0 regex==2026.9.29 packaging==26.3 aiohttp==3.14.3
 aiohappyeyeballs==2.7.1 aiosignal==1.4.0 attrs==26.1.0 frozenlist==1.8.0 multidict==6.9.1 propcache==0.5.4 yarl==1.25.1 idna==3.20
 pyyaml==6.0.3 pillow==12.3.0 scipy==1.18.1 tqdm==4.70.1 psutil==7.2.2 alembic==1.20.0 mako==1.4.3 markupsafe==3.0.3
 sqlalchemy==2.1.1 typing-extensions==4.16.0 filelock==4.0.6 av==19.0.0 comfy-kitchen==0.2.36 comfy-aimdo==0.5.5 requests==2.34.2
 charset-normalizer==3.5.1 urllib3==2.8.0 certifi==2026.7.22 simpleeval==1.0.8 blake3==1.0.10 kornia==0.8.3 kornia-rs==0.2.0
 spandrel==0.4.2 pydantic==2.13.5 pydantic-core==2.46.5 annotated-types==0.8.0 typing-inspection==0.4.4 pydantic-settings==2.15.0
 python-dotenv==1.2.3 fsspec==2026.9.0 httpx==0.28.1 httpcore==1.0.9 anyio==4.15.1 sniffio==1.3.1 h11==0.16.0 click==8.5.0
 comfyui-frontend-package==1.53.6 protobuf==7.36.2"
ok=""
for idx in https://pypi.org/simple https://mirrors.aliyun.com/pypi/simple https://pypi.org/simple; do
  if pip install -q --break-system-packages --no-deps --timeout 15 --retries 8 --resume-retries 20 --index-url $idx $PKGS > /root/pip.log 2>&1; then
    ok=1; break
  fi
done
[ -n "$ok" ] || { st "ERROR: Python packages failed to install (see /root/pip.log)"; exit 1; }
# torchvision must match the image's torch build: PyPI's 0.24.1 is the CUDA 12.8 one, CUDA 13 needs pytorch.org's
CU=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null)
[ "$CU" = "13" ] && TVIDX=https://download.pytorch.org/whl/cu130 || TVIDX=https://pypi.org/simple
python3 -c "import torchvision" 2>/dev/null || pip install -q --break-system-packages --no-deps --timeout 15 --retries 8 \
  --resume-retries 20 --index-url $TVIDX torchvision==0.24.1 >> /root/pip.log 2>&1 || {
  st "ERROR: torchvision failed to install (see /root/pip.log)"; exit 1; }

while [ ! -f /root/dl_done ]; do
  st "downloading Qwen-Image 2.1 and $NAME ($(du -sm models 2>/dev/null | cut -f1) of ~$TOTAL MB)"
  sleep 5
done
[ -f /root/dl_fail ] && { st "ERROR: model download failed: $(cat /root/dl_fail)"; exit 1; }

# after READY: one group of files into /root/extra (outside models/, so ComfyUI never sees half a file), then moved
# into place together and flagged
extra() {  # extra <flag> <url> <models/subdir> [<url> <models/subdir>]...
  local flag=$1 pids="" files=""
  shift
  mkdir -p /root/extra
  while [ $# -gt 1 ]; do
    local name=$(basename "$1")
    get "$1" "/root/extra/$name" &
    pids="$pids $!"
    files="$files $name:$2"
    shift 2
  done
  wait $pids  # not a bare wait: ComfyUI is a child of this shell too and never ends
  for f in $files; do
    [ -s "/root/extra/${f%%:*}" ] || return 1
  done
  for f in $files; do
    mv "/root/extra/${f%%:*}" "${f#*:}/" || return 1
  done
  touch "/root/done_$flag"
}

st "starting ComfyUI"
setsid nohup python main.py --listen 127.0.0.1 --port 8188 > /root/comfy.log 2>&1 < /dev/null &
for i in $(seq 1 60); do
  if curl -s -m 3 -o /dev/null http://127.0.0.1:8188/system_stats; then
    st "READY"
    extra qwen_hq $Q/diffusion_models/qwen_image_2.1_bf16.safetensors models/diffusion_models \
      $Q/text_encoders/qwen3vl_8b_bf16.safetensors models/text_encoders
    extra ideogram $I/diffusion_models/ideogram4_int8_convrot.safetensors models/diffusion_models \
      $I/diffusion_models/ideogram4_unconditional_int8_convrot.safetensors models/diffusion_models \
      $I/text_encoders/qwen3vl_8b_fp8_scaled.safetensors models/text_encoders \
      $I/vae/flux2-vae.safetensors models/vae
    exit 0
  fi
  sleep 5
done
st "ERROR: ComfyUI did not start (see /root/comfy.log)"

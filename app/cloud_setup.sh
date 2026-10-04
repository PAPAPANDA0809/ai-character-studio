#!/bin/bash
# Reelmint fast-GPU pod (RunPod, runpod/pytorch image with torch 2.9.1 for CUDA 13): ComfyUI v0.38.0 (the version on the PC)
# with Qwen-Image 2.1. Progress goes to /root/status, ending in READY or ERROR: ...; trained characters are
# copied into /root/loras by the app.
# Lessons from the first pods: pip resolving the full requirements tried to replace torch (gigabytes), and
# pypi.org downloads stalled in some data centres, so packages are pinned and installed without resolving,
# with short timeouts, retries and a mirror as fallback; Hugging Face downloads are fast everywhere.
st() { echo "$*" > /root/status; }
cd /root

# community hosts can hand out a card PyTorch can't use ("CUDA unknown error"): find out before downloading anything
st "testing the GPU"
python3 -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || { st "ERROR: FAULTY_GPU"; exit 1; }

st "installing ComfyUI"
[ -d ComfyUI ] || git clone -q --depth 1 --branch v0.38.0 https://github.com/comfyanonymous/ComfyUI || { st "ERROR: could not download ComfyUI"; exit 1; }
cd ComfyUI
mkdir -p /root/loras
printf 'reelmint:\n  base_path: /root\n  loras: loras\n' > extra_model_paths.yaml

H=https://huggingface.co
get() { curl -sS -L --retry 20 --retry-all-errors -C - -o "$2" "$1" || echo "$2" >> /root/dl_fail; }
(
  get $H/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors &
  get $H/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors models/text_encoders/qwen3vl_8b_int8_convrot.safetensors &
  get $H/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors models/vae/qwen_image_2.1_vae_bf16.safetensors &
  get $H/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors models/upscale_models/4x-UltraSharp.safetensors &
  wait
  touch /root/dl_done
) &

st "installing Python packages"
PKGS="torchsde==0.2.6 trampoline==0.1.2 numpy==2.5.3 einops==0.8.2 transformers==5.17.0 tokenizers==0.23.2
 sentencepiece==0.2.2 safetensors==0.8.0 huggingface-hub==1.33.0 hf-xet==1.6.0 regex==2026.9.29 packaging==26.3 aiohttp==3.14.3
 aiohappyeyeballs==2.7.1 aiosignal==1.4.0 attrs==26.1.0 frozenlist==1.8.0 multidict==6.9.1 propcache==0.5.4 yarl==1.25.1 idna==3.20
 pyyaml==6.0.3 pillow==12.3.0 scipy==1.18.1 tqdm==4.70.1 psutil==7.2.2 alembic==1.20.0 mako==1.4.3 markupsafe==3.0.3
 sqlalchemy==2.1.1 typing-extensions==4.16.0 filelock==4.0.6 av==19.0.0 comfy-kitchen==0.2.36 comfy-aimdo==0.5.5 requests==2.34.2
 charset-normalizer==3.5.1 urllib3==2.8.0 certifi==2026.7.22 simpleeval==1.0.8 blake3==1.0.10 kornia==0.8.3 kornia-rs==0.2.0
 spandrel==0.4.2 pydantic==2.13.5 pydantic-core==2.46.5 annotated-types==0.8.0 typing-inspection==0.4.4 pydantic-settings==2.15.0
 python-dotenv==1.2.3 fsspec==2026.9.0 httpx==0.28.1 httpcore==1.0.9 anyio==4.15.1 sniffio==1.3.1 h11==0.16.0 click==8.5.0
 comfyui-frontend-package==1.53.6"
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
  st "downloading Qwen-Image 2.1 ($(du -sm models 2>/dev/null | cut -f1) of ~17,500 MB)"
  sleep 5
done
[ -f /root/dl_fail ] && { st "ERROR: model download failed: $(cat /root/dl_fail)"; exit 1; }

st "starting ComfyUI"
setsid nohup python main.py --listen 127.0.0.1 --port 8188 > /root/comfy.log 2>&1 < /dev/null &
for i in $(seq 1 60); do
  curl -s -m 3 -o /dev/null http://127.0.0.1:8188/system_stats && { st "READY"; exit 0; }
  sleep 5
done
st "ERROR: ComfyUI did not start (see /root/comfy.log)"

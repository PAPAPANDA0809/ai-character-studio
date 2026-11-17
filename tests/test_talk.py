"""Talking videos (LongCat-Video-Avatar 1.6 on the rented card): voice upload, the job against a fake pod, the pod
agent's job endpoints, the runner's maths and its flash_attn stand-in.

Run with: python -m unittest discover -s tests -v   (needs ffmpeg; the attention check needs torch)
"""
import base64
import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import cloudgpu
import comfy
import pipeline

FF = pipeline.find_ffmpeg(pipeline.load_config()["ffmpeg"])


def load_runner():
    spec = importlib.util.spec_from_file_location("cloud_longcat", ROOT / "app" / "cloud_longcat.py")
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x", "job"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


class TalkTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp, True)
        p = patch.object(pipeline, "PROJECTS", str(self.temp / "projects"))
        p.start()
        self.addCleanup(p.stop)
        self.pid = "u--20261117-000000-talk"
        os.makedirs(pipeline.pdir(self.pid, "keyframes"))
        os.makedirs(pipeline.pdir(self.pid, "clips"))
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=368x640", "-frames:v", "1",
                        pipeline.pdir(self.pid, "keyframes", "kf_00.png")], check=True)
        self.p = {"id": self.pid, "plan": {"title": "t", "keyframes": [{"image_prompt": "a woman", "subtitle": ""}]},
                  "status": "planned", "log": [], "seed": 5, "aspect": "portrait", "idea": "a woman talking"}
        pipeline.save(self.p)

    def voice(self, seconds, ext=".mp3"):
        f = self.temp / ("voice" + ext)
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
                        "-ac", "2", "-ar", "44100", str(f)], check=True)
        return f.read_bytes()

    def test_voice_upload_is_kept_and_converted(self):
        r = pipeline.save_talk_audio(self.pid, 0, "memo.mp3", self.voice(3))
        self.assertAlmostEqual(r["seconds"], 3, delta=0.2)
        self.assertFalse(r["capped"])
        import wave
        with wave.open(pipeline.pdir(self.pid, "clips", "talk_00.wav")) as w:
            self.assertEqual((w.getnchannels(), w.getframerate(), w.getsampwidth()), (1, 16000, 2))
        self.assertTrue(os.path.exists(pipeline.pdir(self.pid, "clips", "talk_src_00.mp3")))

    def test_voice_upload_refuses_bad_files(self):
        with self.assertRaises(ValueError):
            pipeline.save_talk_audio(self.pid, 0, "notes.txt", b"hello")
        with self.assertRaises(ValueError):
            pipeline.save_talk_audio(self.pid, 0, "memo.mp3", b"not really audio")
        with self.assertRaises(ValueError):
            pipeline.save_talk_audio(self.pid, 3, "memo.mp3", self.voice(2))

    def test_long_voice_is_cut_at_the_limit(self):
        with patch.object(pipeline, "TALK_MAX_SECONDS", 2):
            r = pipeline.save_talk_audio(self.pid, 0, "memo.wav", self.voice(4, ".wav"))
        self.assertTrue(r["capped"])
        self.assertAlmostEqual(r["seconds"], 2, delta=0.1)

    def test_talk_job_against_a_fake_pod(self):
        pipeline.save_talk_audio(self.pid, 0, "memo.mp3", self.voice(3))
        raw = self.temp / "raw.mp4"  # what LongCat sends back: 480x832, 25 fps, no sound
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=blue:s=480x832:r=25", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(raw)], check=True)
        started, polls = {}, iter([{"status": "loading the models"}, {"status": "part 1 of 2"},
                                   {"status": "DONE", "size": raw.stat().st_size}])

        def start(jid, image, wav, job):
            started.update(jid=jid, image=image[:8], wav=len(wav), **job)

        state = {**cloudgpu.state, "status": "ready", "gpu": "L40S", "extras": {"longcat": True}}
        with patch.dict(cloudgpu.state, state), \
                patch.object(cloudgpu, "info", return_value={"label": "L40S", "talk": True, "talk_720": False, "mem": 48}), \
                patch.object(cloudgpu, "url", return_value="http://127.0.0.1:1"), \
                patch.object(comfy.Comfy, "free") as free, patch.object(cloudgpu, "talk_start", start), \
                patch.object(cloudgpu, "talk_status", lambda jid: next(polls)), \
                patch.object(cloudgpu, "talk_fetch", lambda jid, size, dest: shutil.copy(raw, dest)), \
                patch.object(pipeline.time, "sleep"):
            pipeline.do_talk(self.p, 0, "she smiles", "720p")
        free.assert_called_once()
        self.assertEqual(started["image"], b"\x89PNG\r\n\x1a\n")
        self.assertEqual((started["prompt"], started["res"], started["int8"]), ("she smiles", "480p", True))
        self.assertAlmostEqual(started["seconds"], 3, delta=0.1)
        out = pipeline.pdir(self.pid, "clips", "img_00.mp4")
        probe = subprocess.run([FF, "-i", out], capture_output=True, text=True).stderr
        self.assertIn("Audio:", probe)
        self.assertIn("Video:", probe)
        self.assertIn("0", pipeline.load(self.pid)["image_clips"])
        self.assertFalse(os.path.exists(pipeline.pdir(self.pid, "clips", "talk_raw_00.mp4")))

    def test_talk_needs_a_big_enough_card(self):
        pipeline.save_talk_audio(self.pid, 0, "memo.mp3", self.voice(2))
        with patch.dict(cloudgpu.state, {"status": "off"}), self.assertRaises(RuntimeError):
            pipeline.do_talk(self.p, 0)
        with patch.dict(cloudgpu.state, {"status": "ready", "gpu": "RTX 5090"}), \
                patch.object(cloudgpu, "info", return_value={"talk": False}), self.assertRaisesRegex(RuntimeError, "48 GB"):
            pipeline.do_talk(self.p, 0)

    def test_failed_pod_job_is_stopped_and_reported(self):
        pipeline.save_talk_audio(self.pid, 0, "memo.mp3", self.voice(2))
        state = {"status": "ready", "gpu": "H100", "extras": {"longcat": True}}
        with patch.dict(cloudgpu.state, state), \
                patch.object(cloudgpu, "info", return_value={"label": "H100", "talk": True, "talk_720": True, "mem": 80}), \
                patch.object(cloudgpu, "url", return_value="http://127.0.0.1:1"), patch.object(comfy.Comfy, "free"), \
                patch.object(cloudgpu, "talk_start"), patch.object(cloudgpu, "talk_stop") as stop, \
                patch.object(cloudgpu, "talk_status", return_value={"status": "ERROR: CUDA out of memory", "log": ""}):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                pipeline.do_talk(self.p, 0)
        stop.assert_called_once()

    def test_cards_and_disk(self):
        l40s = next(c for c in cloudgpu.CATALOG if c["label"] == "L40S")
        h100 = next(c for c in cloudgpu.CATALOG if c["label"] == "H100")
        r5090 = next(c for c in cloudgpu.CATALOG if c["label"] == "RTX 5090")
        self.assertTrue(cloudgpu.longcat_int8(l40s))
        self.assertFalse(cloudgpu.longcat_int8(h100))
        with patch.object(cloudgpu, "comfy_te", return_value="qwen3vl_32b_minimax_h3_int8_convrot.safetensors"):
            small = cloudgpu.disk_gb(r5090, "ltx25")
            with patch.object(cloudgpu, "LONGCAT_MIN_MEM", 999):
                without = cloudgpu.disk_gb(l40s, "ltx25")
            self.assertGreater(cloudgpu.disk_gb(l40s, "ltx25") - without, 50)
        self.assertGreater(small, 0)

    def test_runner_parts(self):
        lc = load_runner()
        self.assertEqual(lc.segments_for(3.72), 1)    # 93 frames
        self.assertEqual(lc.segments_for(5), 2)
        self.assertEqual(lc.segments_for(6.92), 2)    # 93 + 80 = 173 frames
        self.assertEqual(lc.segments_for(120), 38)

    def test_flash_attn_stand_in_matches_plain_attention(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        lc = load_runner()
        saved = sys.modules.get("flash_attn")
        lc.install_flash_attn_shim()
        self.addCleanup(lambda: sys.modules.__setitem__("flash_attn", saved) if saved else sys.modules.pop("flash_attn"))
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 7, 4, 16) for _ in range(3))  # [B, S, H, D]
        scale = 0.3
        ref = torch.softmax(torch.einsum("bshd,bthd->bhst", q, k) * scale, -1)
        ref = torch.einsum("bhst,bthd->bshd", ref, v)
        self.assertTrue(torch.allclose(flash_attn_func(q, k, v, softmax_scale=scale), ref, atol=1e-5))
        # packed: two queries of 5 against keys of 3 and 6
        q2, k2, v2 = torch.randn(10, 4, 16), torch.randn(9, 4, 16), torch.randn(9, 4, 16)
        out = flash_attn_varlen_func(q2, k2, v2, torch.tensor([0, 5, 10]), torch.tensor([0, 3, 9]), 5, 6)
        for (a, b), (c, d) in (((0, 5), (0, 3)), ((5, 10), (3, 9))):
            w = torch.softmax(torch.einsum("shd,thd->hst", q2[a:b], k2[c:d]) / 4, -1)  # default scale 1/sqrt(16)
            self.assertTrue(torch.allclose(out[a:b], torch.einsum("hst,thd->shd", w, v2[c:d]), atol=1e-5))


class AgentTalkTests(unittest.TestCase):
    """The pod agent's job endpoints: paths moved to a temp dir and the LongCat runner swapped for a tiny script."""

    def test_job_round_trip(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        src = (ROOT / "app" / "cloud_agent.py").read_text(encoding="utf-8")
        runner = tmp / "runner.py"
        runner.write_text("import sys, os\nd = sys.argv[1]\nopen(os.path.join(d, 'out.mp4'), 'wb').write(b'x' * 20000)\n"
                          "open(os.path.join(d, 'status'), 'w').write('DONE')\n")
        (tmp / "done_longcat").write_text("")
        src = (src.replace('"/root/talk"', repr(str(tmp / "talk")))
                  .replace('"/root/done_longcat"', repr(str(tmp / "done_longcat")))
                  .replace('["python3", "/root/longcat_run.py", d]', f'[{sys.executable!r}, {str(runner)!r}, d]')
                  .replace('ThreadingHTTPServer(("0.0.0.0", 8189), H).serve_forever()', ''))
        ns = {"__name__": "agent"}
        with patch.dict(os.environ, {"RM_KEY": "k"}):
            exec(compile(src, "cloud_agent.py", "exec"), ns)
        srv = ns["ThreadingHTTPServer"](("127.0.0.1", 0), ns["H"])
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)

        def call(method, path, body=b""):
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=30)
            c.request(method, path, body=body, headers={"X-RM-Key": "k", "Content-Length": str(len(body))})
            r = c.getresponse()
            return r.status, r.read()

        jid = "0123456789abcdef"
        self.assertEqual(call("POST", f"/rm/talk?id={jid}", b"{}")[0], 400)  # no picture / voice yet
        self.assertEqual(call("PUT", f"/rm/talk/in?id={jid}&name=../x", b"a")[0], 400)
        self.assertEqual(call("PUT", f"/rm/talk/in?id=../../x&name=image.png", b"a")[0], 400)
        self.assertEqual(call("PUT", f"/rm/talk/in?id={jid}&name=image.png", b"png")[0], 200)
        self.assertEqual(call("PUT", f"/rm/talk/in?id={jid}&name=audio.wav", b"wav")[0], 200)
        self.assertEqual(call("POST", f"/rm/talk?id={jid}", b'{"seconds": 3}')[0], 200)
        for _ in range(100):
            st = json.loads(call("GET", f"/rm/talk?id={jid}")[1])
            if st["status"] == "DONE":
                break
            time.sleep(0.1)
        self.assertEqual((st["status"], st["size"]), ("DONE", 20000))
        code, body = call("GET", f"/rm/talk/out?id={jid}&offset=15000&size=8000")
        self.assertEqual((code, len(body)), (200, 5000))
        self.assertEqual(call("GET", "/rm/talk?id=ffffffffffffffff")[0], 404)
        self.assertTrue(json.loads(call("GET", "/rm/extras")[1])["longcat"] is False)  # the real /root path


if __name__ == "__main__":
    unittest.main()

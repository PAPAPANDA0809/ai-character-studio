"""Regression checks for account setup, paid GPU cleanup and cloud image routing.

Run with: python -m unittest discover -s tests -v
All cloud calls and GPU commands are mocked; storage uses temporary fixtures.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import auth
import cloudgpu
import comfy
import llm
import pipeline


class ImmediateThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix=".fixture-", dir=ROOT / "tests")))
        for module, attr, value in (
            (auth, "USERS", str(self.temp / "users.json")),
            (auth, "SESSIONS", str(self.temp / "sessions.json")),
            (pipeline, "PROJECTS", str(self.temp / "projects")),
            (pipeline, "CHARACTERS", str(self.temp / "characters")),
            (cloudgpu, "STATE_FILE", str(self.temp / "cloud_gpu.json")),
        ):
            self.stack.enter_context(patch.object(module, attr, value))
        Path(pipeline.PROJECTS).mkdir()
        Path(pipeline.CHARACTERS).mkdir()
        self.state = copy.deepcopy(cloudgpu.state)
        self.generation = cloudgpu._generation[0]
        self.addCleanup(self.restore_cloud)
        cloudgpu.state.update(status="off", pod=None, key="", started=0, last_used=0, gpu="")
        self.card = cloudgpu.CATALOG[0]
        self.run = self.stack.enter_context(patch.object(cloudgpu, "_run", return_value=(0, "", "")))
        self.stack.enter_context(patch.object(cloudgpu, "_call", side_effect=AssertionError("unexpected cloud call")))
        self.stack.enter_context(patch.object(cloudgpu.time, "sleep", return_value=None))

    def restore_cloud(self):
        cloudgpu.state.clear()
        cloudgpu.state.update(self.state)
        cloudgpu._generation[0] = self.generation

    def immediate(self):
        return patch.object(cloudgpu.threading, "Thread", ImmediateThread)

    def project(self, engine="gpt-image", refs=False):
        p = {"id": "viewer--fixture", "image_engine": engine, "mode": "images", "quality": "sharp",
             "aspect": "portrait", "seed": 1, "keyframe_seeds": {}, "references": [], "reference": "",
             "plan": {"keyframes": [{"image_prompt": "woman in a park"}], "character": "", "setting": "",
                      "caption": "Fixture", "hashtags": []}}
        Path(pipeline.pdir(p["id"], "keyframes")).mkdir(parents=True)
        if refs:
            Path(pipeline.char_dir("viewer"), "fixture.png").write_bytes(b"fixture image")
            p.update(references=["fixture.png"], reference="fixture.png")
        pipeline.save(p)
        return p

    def test_retry_cannot_replace_undeleted_gpu(self):
        cloudgpu.state.update(status="error", pod="old-pod")
        with self.immediate(), patch.object(cloudgpu, "_create") as create:
            self.assertFalse(cloudgpu.start(self.card["id"], "COMMUNITY"))
        self.assertEqual(cloudgpu.state["pod"], "old-pod")
        create.assert_not_called()

    def test_stop_during_placement_deletes_registered_pod(self):
        cloudgpu.state.update(status="starting", gpu=self.card["label"])
        def stop_during_place(pod):
            self.assertEqual(cloudgpu.state["pod"], pod)
            cloudgpu.stop()
            return "CA"
        with self.immediate(), \
             patch.object(cloudgpu, "_create", return_value=({"id": "new-pod", "costPerHr": 1}, "")), \
             patch.object(cloudgpu, "_place", side_effect=stop_during_place):
            cloudgpu._start(self.card, "COMMUNITY")
        self.assertEqual(cloudgpu.state["status"], "off")
        self.assertIsNone(cloudgpu.state["pod"])
        self.assertTrue(any(c.args[0][-1] == "new-pod" for c in self.run.call_args_list))

    def test_failed_cancellation_cleanup_keeps_pod_for_retry(self):
        cloudgpu.state.update(status="starting", gpu=self.card["label"])
        def create_then_stop(*args, **kwargs):
            cloudgpu.stop()
            return {"id": "cleanup-pod", "costPerHr": 1}, ""
        self.run.return_value = (1, "", "simulated delete failure")
        with self.immediate(), patch.object(cloudgpu, "_create", side_effect=create_then_stop):
            cloudgpu._start(self.card, "COMMUNITY")
        self.assertEqual(cloudgpu.state["status"], "error")
        self.assertEqual(cloudgpu.state["pod"], "cleanup-pod")
        self.assertEqual(self.run.call_count, 3)
        self.assertEqual(json.loads(Path(cloudgpu.STATE_FILE).read_text())["pod"], "cleanup-pod")

    def test_delete_timeout_retains_recovery_record(self):
        cloudgpu._set(status="stopping", pod="timeout-pod")
        self.run.side_effect = subprocess.TimeoutExpired("fixture", 1)
        cloudgpu._delete("Stopped")
        self.assertEqual(cloudgpu.state["status"], "error")
        self.assertEqual(cloudgpu.state["pod"], "timeout-pod")
        self.assertEqual(self.run.call_count, 3)

    def test_stale_delete_worker_cannot_delete_new_rental(self):
        cloudgpu._generation[0] = 2
        cloudgpu.state.update(status="starting", pod="new-rental")
        cloudgpu._delete("old cancellation", generation=1)
        self.run.assert_not_called()
        self.assertEqual(cloudgpu.state["pod"], "new-rental")

    def test_stop_during_creation_with_real_threads(self):
        entered, release, deleted = threading.Event(), threading.Event(), threading.Event()
        real_thread, workers = threading.Thread, []
        def tracked_thread(*args, **kwargs):
            worker = real_thread(*args, **kwargs)
            workers.append(worker)
            return worker
        def create(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("creation fixture was not released")
            return {"id": "threaded-pod", "costPerHr": 1}, ""
        def run(*args, **kwargs):
            deleted.set()
            return 0, "", ""
        self.run.side_effect = run
        with patch.object(cloudgpu, "_create", side_effect=create), \
             patch.object(cloudgpu.threading, "Thread", side_effect=tracked_thread):
            self.assertTrue(cloudgpu.start(self.card["id"], "COMMUNITY"))
            self.assertTrue(entered.wait(5))
            self.assertTrue(cloudgpu.stop())
            self.assertFalse(cloudgpu.start(self.card["id"], "COMMUNITY"))
            release.set()
            self.assertTrue(deleted.wait(5))
            for worker in workers:
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
        self.assertIsNone(cloudgpu.state["pod"])

    def test_successful_start_reaches_ready(self):
        def call(method, path, *args, **kwargs):
            return (200, b"READY") if path == "/rm/status" else (200, b"started")
        with self.immediate(), \
             patch.object(cloudgpu, "_create", return_value=({"id": "ready-pod", "costPerHr": 1}, "")), \
             patch.object(cloudgpu, "_place", return_value="CA"), \
             patch.object(cloudgpu, "_wait_agent", return_value=True), \
             patch.object(cloudgpu, "_call", side_effect=call), \
             patch.object(cloudgpu, "_send_loras"), patch.object(cloudgpu, "_open_relay"), patch.object(cloudgpu, "_free_local"):
            self.assertTrue(cloudgpu.start(self.card["id"], "COMMUNITY"))
        self.assertEqual(cloudgpu.state["status"], "ready")
        self.assertEqual(cloudgpu.state["pod"], "ready-pod")

    def test_cancel_at_relay_cannot_become_ready_again(self):
        cloudgpu.state.update(status="starting", gpu=self.card["label"])
        with self.immediate(), \
             patch.object(cloudgpu, "_create", return_value=({"id": "cancel-pod", "costPerHr": 1}, "")), \
             patch.object(cloudgpu, "_place", return_value="CA"), \
             patch.object(cloudgpu, "_wait_agent", return_value=True), \
             patch.object(cloudgpu, "_call", return_value=(200, b"READY")), \
             patch.object(cloudgpu, "_send_loras"), patch.object(cloudgpu, "_open_relay", side_effect=cloudgpu.stop):
            cloudgpu._start(self.card, "COMMUNITY")
        self.assertEqual(cloudgpu.state["status"], "off")
        self.assertIsNone(cloudgpu.state["pod"])

    def test_concurrent_setup_creates_only_one_admin(self):
        barrier = threading.Barrier(2)
        def setup(name):
            barrier.wait(timeout=5)
            try:
                return auth.create_admin(name, "fixture-password")
            except ValueError as exc:
                return str(exc)
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(setup, ("alpha", "beta")))
        self.assertEqual(result.count("already set up"), 1)
        self.assertEqual(len(auth.users()), 1)
        self.assertTrue(all(u["admin"] for u in auth.users().values()))

    def test_setup_cannot_overwrite_existing_admin(self):
        auth.create_admin("alpha", "fixture-password")
        with self.assertRaisesRegex(ValueError, "already set up"):
            auth.create_admin("alpha", "different-password")
        self.assertTrue(auth.check("alpha", "fixture-password"))

    def test_gpt_runs_without_comfyui_and_keeps_reference(self):
        p = self.project(refs=True)
        with patch.object(comfy, "Comfy", side_effect=AssertionError("GPT must not use ComfyUI")), \
             patch.object(pipeline.cloud, "gpt_image", return_value=b"fixture result") as generate:
            pipeline._keyframes(p)
        self.assertEqual(Path(pipeline.pdir(p["id"], "keyframes", "kf_00.png")).read_bytes(), b"fixture result")
        self.assertEqual(generate.call_args.args[4], [str(Path(pipeline.char_dir("viewer"), "fixture.png"))])
        self.assertIn("reference image", generate.call_args.args[1])

    def test_gpt_override_bypasses_local_engine_preparation(self):
        p = self.project("qwen-image-21")
        with patch.object(comfy, "Comfy", side_effect=AssertionError("GPT must not use ComfyUI")), \
             patch.object(pipeline.cloud, "gpt_image", return_value=b"fixture result") as generate:
            pipeline._keyframes(p, engine="gpt-image")
        generate.assert_called_once()

    def test_reference_update_preserves_gpt_engine(self):
        p = self.project(refs=True)
        self.assertEqual(pipeline.set_reference(p["id"], ["fixture.png"])["image_engine"], "gpt-image")

    def test_nonreference_engine_still_switches_to_reference_engine(self):
        p = self.project("z-image", refs=True)
        self.assertEqual(pipeline.set_reference(p["id"], ["fixture.png"])["image_engine"], "qwen-image-21")

    def test_local_failure_gpt_fallback_keeps_reference(self):
        p = self.project("qwen-image-21", refs=True)
        with patch.object(comfy, "Comfy") as fake, patch.object(pipeline.cloud, "available", return_value=True), \
             patch.object(pipeline.cloud, "gpt_image", return_value=b"fixture result") as generate:
            fake.return_value.alive.return_value = True
            fake.return_value.upload_image.return_value = "fixture.png"
            fake.return_value.run.side_effect = RuntimeError("simulated local failure")
            pipeline._keyframes(p)
        self.assertEqual(generate.call_args.args[4], [str(Path(pipeline.char_dir("viewer"), "fixture.png"))])
        self.assertIn("reference image", generate.call_args.args[1])

    def test_edit_still_requires_comfyui(self):
        p = self.project(refs=True)
        with patch.object(comfy, "Comfy") as fake, patch.object(pipeline.cloud, "gpt_image") as generate:
            fake.return_value.alive.return_value = False
            with self.assertRaisesRegex(RuntimeError, "ComfyUI is not running"):
                pipeline._keyframes(p, edit="make the background blue")
        generate.assert_not_called()

    def test_manifest_icons_exist_and_are_not_ignored(self):
        paths = ["app/static/reelmint-32.png", "app/static/reelmint-192.png", "app/static/reelmint-512.png",
                 "app/static/reelmint-maskable-512.png", "myapp/icon-192.png", "myapp/icon-512.png",
                 "myapp/icon-maskable-512.png", "myapp/reelmint-192.png"]
        for path in paths:
            self.assertTrue((ROOT / path).is_file(), path)
        git = subprocess.run(["git", "check-ignore", *paths], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(git.returncode, 1, git.stdout + git.stderr)

    def test_krea_loras_are_sent_to_the_pod(self):
        cfg = {"loras": [{"file": "q.safetensors", "engine": "qwen-image-21"},
                         {"file": "k.safetensors", "engine": "krea2-turbo"},
                         {"file": "z.safetensors", "engine": "z-image"}]}
        cloudgpu.state.update(pod="lora-pod", loras=[])
        with patch.object(cloudgpu.json, "load", return_value=cfg), patch("builtins.open"), \
             patch.object(cloudgpu.os.path, "exists", return_value=True), \
             patch.object(cloudgpu, "_send_lora", return_value=True) as send:
            cloudgpu._send_loras()
        self.assertEqual([c.args[1] for c in send.call_args_list], ["q.safetensors", "k.safetensors"])
        self.assertEqual(cloudgpu.state["loras"], ["q.safetensors", "k.safetensors"])

    def test_pod_files_take_the_best_precision_each_card_holds(self):
        card = {c["label"]: c for c in cloudgpu.CATALOG}
        f4090, f5090, f80, f141 = (cloudgpu.pod_files(card[k]) for k in ("RTX 4090", "RTX 5090", "A100 80 GB", "H200"))
        self.assertEqual(f4090["KREA_UNET"], "krea2_turbo_fp8_scaled.safetensors")  # 24.5 GB bf16 needs 32 GB
        self.assertEqual(f5090["KREA_UNET"], "krea2_turbo_bf16.safetensors")
        self.assertTrue(all(f["MOSS_UNET"].endswith("bf16.safetensors") for f in (f4090, f141)))
        self.assertIn("int8", f5090["LTX_UNET"])
        self.assertIn("bf16", f80["LTX_UNET"])
        self.assertIn("int8", f4090["LTX_TE"])
        self.assertIn("bf16", f5090["LTX_TE"])
        self.assertEqual(cloudgpu.h3_plan(card["H200"])[0], cloudgpu.H3_BF16)
        self.assertEqual(cloudgpu.h3_plan(card["RTX PRO 6000"])[0], cloudgpu.H3_FULL)  # 96 GB: frames need room
        self.assertEqual(cloudgpu.h3_plan(card["RTX 5090"])[0], cloudgpu.H3_PRUNED)
        self.assertIn(cloudgpu.H3_BF16, cloudgpu.H3_REF)
        self.assertEqual(cloudgpu.H3_REF[cloudgpu.H3_PRUNED], cloudgpu.OMNI_INT8)  # FastH3 Omni: 8 steps
        self.assertEqual(cloudgpu.H3_REF[cloudgpu.H3_BF16], cloudgpu.OMNI_BF16)
        script = (ROOT / "app" / "cloud_setup.sh").read_text(encoding="utf-8")
        self.assertIn("FastVideo/FastH3-Omni-Comfy/resolve/82eff2a6", script)
        self.assertIn("$F/diffusion_models/$H3_REF", script)
        self.assertIn("bf16", cloudgpu.comfy_te(card["H100"]))
        for c in cloudgpu.CATALOG:  # every planned file has a size, and the disk is never below the old fixed sizes
            plan = cloudgpu.h3_plan(c)
            gb = cloudgpu.disk_gb(c, "h3", plan[0]) if plan else cloudgpu.disk_gb(c, "ltx25")
            self.assertGreaterEqual(gb, 170, c["label"])
            self.assertLess(gb, 600, c["label"])

    def test_balance_is_read_in_the_background_and_keeps_only_the_number(self):
        reply = json.dumps({"clientBalance": 2.8295, "email": "someone@example.com", "spendLimit": 80})
        self.run.return_value = (0, reply, "")
        with patch.object(cloudgpu, "_balance", [0.0, None, False]), self.immediate():
            self.assertEqual(cloudgpu.balance(), 2.83)  # the immediate thread already read it
            self.assertNotIn("someone@example.com", json.dumps(cloudgpu.info()))
            self.run.return_value = (1, "", "network down")
            cloudgpu._balance[0] = 0.0
            self.assertEqual(cloudgpu.balance(), 2.83)  # a failed read keeps the last value
            self.run.reset_mock()
            cloudgpu.balance()
            self.run.assert_not_called()  # read at most once a minute

    def test_pod_without_a_saved_plan_uses_the_setup_scripts_defaults(self):
        script = (ROOT / "app" / "cloud_setup.sh").read_text(encoding="utf-8")
        for var, name in cloudgpu.POD_DEFAULT_FILES.items():  # the fallback names are the script's ${VAR:-default}s
            self.assertIn("${%s:-%s}" % (var, name), script)
        cloudgpu.state["files"] = {}
        self.assertEqual(cloudgpu.files_on_pod(), cloudgpu.POD_DEFAULT_FILES)
        plan = cloudgpu.pod_files(cloudgpu.CATALOG[1])
        cloudgpu.state["files"] = plan
        self.assertEqual(cloudgpu.files_on_pod(), plan)

    def test_moss_runs_on_the_pod_once_it_has_moss_and_its_lora(self):
        p = self.project("moss-image")
        p["lora"] = "m.safetensors"
        cfg = {"loras": [{"file": "m.safetensors", "label": "Fixture", "engine": "moss-image"}]}
        with patch.object(cloudgpu, "url", return_value="http://pod.fixture"), \
             patch.object(pipeline, "load_config", return_value=cfg), \
             patch.object(cloudgpu, "using"), patch.object(cloudgpu, "info", return_value={"label": "card"}), \
             patch.object(pipeline, "_keyframes") as keyframes:
            cloudgpu.state.update(extras={**cloudgpu.no_extras(), "moss": False}, loras=["m.safetensors"])
            pipeline.do_keyframes(p)  # the pod has no MOSS yet: this PC
            self.assertEqual(len(keyframes.call_args.args), 5)
            cloudgpu.state.update(extras={**cloudgpu.no_extras(), "moss": True}, loras=[])
            pipeline.do_keyframes(p)  # MOSS is in but the LoRA is still uploading: this PC
            self.assertEqual(len(keyframes.call_args.args), 5)
            cloudgpu.state["loras"] = ["m.safetensors"]
            pipeline.do_keyframes(p)
            self.assertEqual(keyframes.call_args.args[-1], "http://pod.fixture")

    def test_krea_waits_for_its_lora_on_the_pod(self):
        p = self.project("krea2-turbo")
        p["lora"] = "k.safetensors"
        cfg = {"loras": [{"file": "k.safetensors", "label": "Fixture", "engine": "krea2-turbo"}]}
        cloudgpu.state.update(extras={**cloudgpu.state["extras"], "krea2": True}, loras=[])
        with patch.object(cloudgpu, "url", return_value="http://pod.fixture"), \
             patch.object(pipeline, "load_config", return_value=cfg), \
             patch.object(pipeline, "_keyframes") as keyframes:
            with self.assertRaisesRegex(RuntimeError, "still uploading"):
                pipeline.do_keyframes(p)
            keyframes.assert_not_called()
            cloudgpu.state["loras"] = ["k.safetensors"]
            with patch.object(cloudgpu, "using"):
                pipeline.do_keyframes(p)
            keyframes.assert_called_once()


if __name__ == "__main__":
    unittest.main()



class BatchQueueTests(unittest.TestCase):
    """One press queues 1, 2, 4 or 8 pictures; the project stays "queued" until the last one is done."""
    restore_cloud, project = RegressionTests.restore_cloud, RegressionTests.project

    def setUp(self):
        RegressionTests.setUp(self)
        with pipeline.jobs.mutex:
            pipeline.jobs.queue.clear()
        self.addCleanup(lambda: pipeline.jobs.queue.clear())

    def test_count_queues_that_many_jobs_with_fresh_seeds_after_the_first(self):
        p = self.project()
        pipeline.enqueue(p["id"], "keyframes", count=4, indices=[0], reseed=False)
        jobs = list(pipeline.jobs.queue)
        self.assertEqual(len(jobs), 4)
        self.assertFalse(jobs[0][2]["reseed"])
        self.assertTrue(all(j[2]["reseed"] for j in jobs[1:]))
        self.assertEqual([j[2]["_batch"] for j in jobs], [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertEqual(pipeline.load(p["id"])["batch"], {"n": 1, "of": 4})

    def test_count_is_limited_to_the_offered_sizes_and_to_pictures(self):
        p = self.project()
        with self.assertRaises(ValueError):
            pipeline.enqueue(p["id"], "keyframes", count=3)
        pipeline.enqueue(p["id"], "animate", count=8)
        self.assertEqual(len(pipeline.jobs.queue), 1)
        self.assertIsNone(pipeline.load(p["id"])["batch"])

    def run_queue_once(self):
        """Run the worker for exactly one job."""
        got = pipeline.jobs.get
        def once(*a, **k):
            once.calls += 1
            if once.calls > 1:
                raise StopIteration
            return got()
        once.calls = 0
        with patch.object(pipeline.jobs, "get", once), self.assertRaises(StopIteration):
            pipeline.worker()

    def test_project_stays_queued_between_pictures_of_a_batch(self):
        p = self.project()
        pipeline.enqueue(p["id"], "keyframes", count=2, indices=[0])
        with patch.object(pipeline, "do_keyframes", lambda p, **kw: None):
            self.run_queue_once()
            self.assertEqual(pipeline.load(p["id"])["status"], "queued:keyframes")
            self.assertEqual(pipeline.load(p["id"])["batch"], {"n": 1, "of": 2})
            self.run_queue_once()
        done = pipeline.load(p["id"])
        self.assertEqual(done["status"], "done:keyframes")
        self.assertIsNone(done["batch"])

    def test_a_failed_picture_drops_the_rest_of_its_batch(self):
        p = self.project()
        pipeline.enqueue(p["id"], "keyframes", count=8, indices=[0])
        def boom(p, **kw):
            raise RuntimeError("engine down")
        with patch.object(pipeline, "do_keyframes", boom), patch.object(pipeline, "_announce"):
            self.run_queue_once()
        self.assertEqual(pipeline.load(p["id"])["status"], "error:keyframes")
        self.assertTrue(pipeline.jobs.empty())


class FastH3OmniTests(unittest.TestCase):
    """Character videos on the rented card use FastH3 Omni (8 steps, sigma shifts); other H3 clips stay as they were."""

    def test_reference_clips_run_the_omni_recipe(self):
        wf = comfy.h3_workflow("p", None, None, 768, 1344, 7, "t", refs=["a.png", "b.png"], ref_unet=cloudgpu.OMNI_INT8)
        self.assertEqual(wf["1"]["inputs"]["unet_name"], cloudgpu.OMNI_INT8)
        self.assertEqual(wf["17"]["class_type"], "MiniMaxH3SigmaShift")
        self.assertEqual((wf["17"]["inputs"]["shift_video"], wf["17"]["inputs"]["shift_audio"]), (12.0, 3.0))
        self.assertEqual(wf["8"]["inputs"]["model"], ["17", 0])
        self.assertEqual(wf["11"]["inputs"]["model"], ["17", 0])
        self.assertEqual((wf["8"]["inputs"]["steps"], wf["9"]["inputs"]["sampler_name"]), (8, "euler"))
        self.assertEqual(wf["6"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertTrue(wf["6"]["inputs"]["prompt"].startswith("Use <Picture 1>"))
        self.assertEqual(wf["6"]["inputs"]["ref_images.ref_image_1"], ["r1", 0])

    def test_formatted_prompts_get_no_lead_in_and_other_clips_keep_the_base_recipe(self):
        wf = comfy.h3_workflow("subject_definitions:\n...", None, None, 768, 1344, 7, "t", refs=["a.png"], formatted=True)
        self.assertTrue(wf["6"]["inputs"]["prompt"].startswith("subject_definitions:"))
        wf = comfy.h3_workflow("p", "a.png", "b.png", 768, 1344, 7, "t")
        self.assertNotIn("17", wf)
        self.assertEqual((wf["8"]["inputs"]["steps"], wf["9"]["inputs"]["sampler_name"]), (20, "res_multistep"))
        self.assertEqual(wf["1"]["inputs"]["unet_name"], comfy.H3["unet"])


class H3PromptTests(unittest.TestCase):
    """Claude rewrites H3 clip prompts into MiniMax's format; the plain prompt is sent when it can't."""

    def test_request_names_the_task_length_and_scene(self):
        u = llm.h3_request("fl2va", "she turns and smiles", 5.0, character="a woman in a red coat", style="film", motion="slow push in")
        self.assertIn("First-and-last-frame", u)
        self.assertIn("Clip length: 5.00 seconds", u)
        self.assertIn("Scene: she turns and smiles", u)
        self.assertIn("Motion and camera for this clip: slow push in", u)
        self.assertIn("<Picture 1>..<Picture 2>", llm.h3_request("ref2va", "s", 10, refs=2))

    def test_prompt_is_none_without_claude_or_without_the_format(self):
        with patch.object(llm, "_claude", return_value=(None, "")):
            self.assertIsNone(llm.h3_prompt({}, "t2va", "s", 5))
        with patch.object(llm, "_claude", return_value=("just a sentence", "Claude")):
            self.assertIsNone(llm.h3_prompt({}, "t2va", "s", 5))
        with patch.object(llm, "_claude", side_effect=RuntimeError("rate-limited")):
            self.assertIsNone(llm.h3_prompt({}, "t2va", "s", 5))
        good = "integrated_multimodal_description: [Shot 1] ...\n\noverall_soundscape: wind\n\nnon_diegetic_music: N/A"
        with patch.object(llm, "_claude", return_value=(good, "Claude")):
            self.assertEqual(llm.h3_prompt({}, "t2va", "s", 5), good)

    def test_local_model_writes_when_claude_declines(self):
        cfg = {"uncensored_fallback": True, "provider": "ollama", "ollama_model": "q"}
        good = "integrated_multimodal_description: [Shot 1] ...\n\noverall_soundscape: wind\n\nnon_diegetic_music: N/A"
        with patch.object(llm, "_claude", return_value=("I'm sorry, I can't help with that.", "Claude")),                 patch.object(llm, "chat", return_value="<think>x</think>" + good):
            self.assertEqual(llm.h3_prompt(cfg, "t2va", "s", 5), good)
        with patch.object(llm, "_claude", side_effect=RuntimeError("Claude declined this prompt")),                 patch.object(llm, "chat", return_value="a woman in a red dress on a beach at sunset"):
            text, label = llm.enhance(cfg, "beach photo", "images", count=1)
            self.assertEqual((text, label), ("a woman in a red dress on a beach at sunset", "local q"))
        with patch.object(llm, "_claude", return_value=("Sorry, I won't write that.", "Claude")),                 patch.object(llm, "chat", return_value="rewritten"):
            self.assertEqual(llm.enhance(cfg, "idea", "images", count=1), ("rewritten", "local q (Claude declined)"))
        with patch.object(llm, "_claude", return_value=("Sorry, no.", "Claude")):  # fallback off: Claude's word stands
            self.assertIsNone(llm.h3_prompt({}, "t2va", "s", 5))

    def test_local_fallback_keeps_the_minors_check(self):
        cfg = {"uncensored_fallback": True, "provider": "ollama", "ollama_model": "q"}
        with patch.object(llm, "_claude", return_value=(None, "")), patch.object(llm, "chat", return_value="x") as chat:
            with self.assertRaises(ValueError):
                llm.enhance(cfg, "a nude 15 year old", "images", count=1)
            chat.assert_not_called()  # never reaches the model
        with patch.object(llm, "_claude", return_value=(None, "")),                 patch.object(llm, "chat", return_value="a naked schoolgirl"):
            with self.assertRaises(ValueError):
                llm.enhance(cfg, "a woman at school", "images", count=1)
            self.assertIsNone(llm.h3_prompt(cfg, "t2va", "a woman at school", 5))

    def test_clip_workflow_sends_claudes_prompt_once_and_remembers_it(self):
        p = {"id": "x", "video_res": "480p", "h3_prompts": {}}
        good = "integrated_multimodal_description: [Shot 1] a\n\noverall_soundscape: b\n\nnon_diegetic_music: N/A"
        with patch.object(llm, "h3_prompt", return_value=good) as h3, patch.object(pipeline, "save"), \
                patch.object(pipeline, "log"), patch.object(cloudgpu, "info", return_value={"h3_max": {"480p": 15}}), \
                patch.dict(cloudgpu.state, {"h3_unet": cloudgpu.H3_PRUNED, "h3_te": "te.safetensors"}):
            wf = pipeline.clip_workflow({"llm": {}}, "h3", "she waves", "a.png", None, 480, 832, 1, "t", p)
            self.assertEqual(wf["6"]["inputs"]["prompt"], good)
            self.assertEqual(h3.call_args.args[1:3], ("i2va", "she waves"))
            self.assertAlmostEqual(h3.call_args.args[3], 124 / 24)  # 5 s on H3's 17k+5 frame grid
            pipeline.clip_workflow({"llm": {}}, "h3", "she waves", "a.png", None, 480, 832, 1, "t", p)
            self.assertEqual(h3.call_count, 1)  # remembered in the project
            self.assertEqual(list(p["h3_prompts"].values()), [good])
            h3.return_value = None
            wf = pipeline.clip_workflow({"llm": {}}, "h3", "he nods", None, None, 480, 832, 1, "t", p, refs=["r.png"])
            self.assertTrue(wf["6"]["inputs"]["prompt"].startswith("Use <Picture 1>"))  # plain prompt with the lead-in
            self.assertEqual(h3.call_args.args[1], "ref2va")


class LoraSettingsTests(unittest.TestCase):
    """A LoRA entry may carry its own CFG and negative prompt (e.g. a Qwen-Image 2.1 LoRA that wants CFG 4, " ")."""

    def test_lora_cfg_and_negative_reach_the_qwen21_sampler(self):
        cfg = pipeline.load_config()
        wf = comfy.qwen_image21_workflow(cfg, "a woman in a park", 1024, 1024, 7, "t")
        self.assertEqual(wf["7"]["inputs"]["cfg"], cfg["qwen_image_21"]["cfg"])
        wf = comfy.apply_lora_settings(comfy.add_lora(wf, "x.safetensors"), {"cfg": 4.0, "negative": " "})
        self.assertEqual(wf["7"]["inputs"]["cfg"], 4.0)
        self.assertEqual(wf["5"]["inputs"]["negative_prompt"], " ")
        self.assertEqual(wf["2"]["inputs"]["model"], ["45", 0])  # the LoRA sits between the loader and the cache

    def test_lora_sampler_and_scheduler_reach_the_sampler(self):
        cfg = pipeline.load_config()
        wf = comfy.apply_lora_settings(comfy.qwen_image21_workflow(cfg, "p", 1024, 1024, 7, "t"),
                                       {"sampler": "er_sde", "scheduler": "beta"})
        self.assertEqual((wf["7"]["inputs"]["sampler_name"], wf["7"]["inputs"]["scheduler"]), ("er_sde", "beta"))

    def test_a_model_entry_swaps_the_diffusion_model_instead_of_adding_a_lora(self):
        cfg = pipeline.load_config()
        wf = comfy.set_unet(comfy.qwen_image21_workflow(cfg, "p", 1024, 1024, 7, "t"), "NoctQ_V4_int8_convrot.safetensors")
        self.assertEqual(wf["1"]["inputs"]["unet_name"], "NoctQ_V4_int8_convrot.safetensors")
        self.assertEqual(wf["1"]["class_type"], "UNETLoader")
        self.assertNotIn("45", wf)
        self.assertEqual(wf["2"]["inputs"]["model"], ["1", 0])
        self.assertEqual(pipeline.lora_file({"file": "n.safetensors", "model": True}), "diffusion_models/n.safetensors")
        self.assertEqual(pipeline.lora_file({"file": "x.safetensors"}), "loras/x.safetensors")

    def test_a_sigma_schedule_replaces_ksampler_with_a_custom_sampler(self):
        cfg = pipeline.load_config()
        turbo = pipeline.lora_entry(cfg, "qwen_image_2.1_turbo_int8_convrot.safetensors", "qwen-image-21")
        wf = comfy.qwen_image21_workflow(cfg, "p", 1024, 1024, 7, "t")
        wf = comfy.apply_lora_settings(comfy.set_unet(wf, turbo["file"]), turbo)
        self.assertEqual(wf["7"]["class_type"], "SamplerCustomAdvanced")
        self.assertEqual(wf["7s"]["inputs"]["sigmas"].split(", ")[-2:], ["0.414568", "0"])
        self.assertEqual(wf["7n"]["inputs"]["noise_seed"], 7)
        self.assertEqual((wf["7g"]["inputs"]["cfg"], wf["7g"]["inputs"]["model"]), (cfg["qwen_image_21"]["cfg"], ["2", 0]))
        self.assertEqual(wf["7"]["inputs"]["latent_image"], ["6", 0])
        self.assertEqual(wf["8"]["inputs"]["samples"], ["7", 0])  # the decoder still reads the sampler's latent
        self.assertEqual(wf["1"]["inputs"]["unet_name"], turbo["file"])
        self.assertEqual(turbo["pod"], {"extra": "qwen_turbo", "file": "qwen_image_2.1_turbo_bf16.safetensors"})

    def test_a_swapped_model_runs_on_the_pod_once_its_copy_is_there(self):
        turbo = {"file": "t.safetensors", "model": True, "pod": {"extra": "qwen_turbo", "file": "t_bf16.safetensors"}}
        with patch.dict(cloudgpu.state, {"status": "ready", "extras": {**cloudgpu.state["extras"], "qwen_turbo": False}}):
            self.assertFalse(pipeline.on_pod(turbo))
        with patch.dict(cloudgpu.state, {"status": "ready", "extras": {**cloudgpu.state["extras"], "qwen_turbo": True}}):
            self.assertTrue(pipeline.on_pod(turbo))
            self.assertFalse(pipeline.on_pod({"file": "n.safetensors", "model": True}))
        self.assertIn("qwen_turbo", cloudgpu.EXTRAS)
        script = (ROOT / "app" / "cloud_setup.sh").read_text(encoding="utf-8")
        self.assertIn("extra qwen_turbo $Q/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors", script)
        self.assertIn('"qwen_turbo"', (ROOT / "app" / "cloud_agent.py").read_text(encoding="utf-8"))

    def test_lora_without_settings_leaves_the_recipe_alone(self):
        cfg = pipeline.load_config()
        wf = comfy.apply_lora_settings(comfy.qwen_image21_workflow(cfg, "p", 1024, 1024, 7, "t"), {"strength": 1.0})
        self.assertEqual(wf["7"]["inputs"]["cfg"], cfg["qwen_image_21"]["cfg"])
        self.assertEqual(wf["5"]["inputs"]["negative_prompt"], "")


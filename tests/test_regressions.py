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


if __name__ == "__main__":
    unittest.main()

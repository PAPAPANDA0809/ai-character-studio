"""Krea API contracts and engine routing, without paid calls or GPU generation."""
import copy
import io
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import cloudgpu
import comfy
import krea
import pipeline
import server


def png():
    from PIL import Image
    out = io.BytesIO()
    Image.new('RGB', (32, 48), '#326789').save(out, 'PNG')
    return out.getvalue()


class KreaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='.fixture-krea-')
        self.addCleanup(self.temp.cleanup)
        self.secrets = Path(self.temp.name) / 'secrets.json'
        self.secret_patch = patch.object(krea, 'SECRETS', str(self.secrets))
        self.secret_patch.start()
        self.addCleanup(self.secret_patch.stop)
        self.environment_patch = patch.dict(os.environ, {'KREA_API_KEY': ''})
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)

    def test_key_storage_preserves_existing_secrets_and_never_returns_key(self):
        self.secrets.write_text(json.dumps({'openai_api_key': 'keep-this'}))
        krea.save_key('krea-test-secret')
        self.assertEqual(json.loads(self.secrets.read_text())['openai_api_key'], 'keep-this')
        self.assertTrue(krea.available())
        self.assertNotIn('krea-test-secret', json.dumps(krea.entries()))
        with self.assertRaises(ValueError):
            krea.save_key('bad\nkey-data')
        with patch.dict(os.environ, {'KREA_API_KEY': 'environment-key'}):
            with self.assertRaises(ValueError):
                krea.save_key('new-secret-key')
            self.assertEqual(krea.api_key(), 'environment-key')

    def test_missing_key_and_cancel_before_submit_do_not_send_requests(self):
        with patch.object(krea, 'request') as submit:
            with self.assertRaisesRegex(RuntimeError, 'API key'):
                krea.generate('krea-api-turbo', 'A cabin', 'portrait', 7, (48, 80))
            submit.assert_not_called()
            krea.save_key('test-secret')
            with self.assertRaisesRegex(RuntimeError, 'Stopped before'):
                krea.generate('krea-api-turbo', 'A cabin', 'portrait', 7, (48, 80), lambda: True)
            submit.assert_not_called()

    def test_child_sexual_prompts_are_rejected_before_submission(self):
        with self.assertRaisesRegex(ValueError, 'minors'):
            krea.validate_prompt('An underage person in a sexual scene')
        krea.validate_prompt('A child reading a book in a classroom')
        krea.validate_prompt('An adult woman in an evening dress')
        with patch.object(krea, 'request') as submit:
            with self.assertRaises(ValueError):
                krea.generate('krea-api-turbo', 'An underage person in a sexual scene', 'portrait', 1, None)
            submit.assert_not_called()

    def test_all_models_use_documented_routes_poll_and_save_requested_size(self):
        from PIL import Image
        krea.save_key('test-secret')
        for engine, (route, _, _) in krea.API_ENGINES.items():
            with self.subTest(engine=engine), patch.object(krea, 'request', side_effect=[
                {'job_id': 'test-job', 'status': 'queued'},
                {'job_id': 'test-job', 'status': 'completed', 'result': {'urls': ['https://gen.krea.ai/test.png']}}
            ]) as request, patch.object(krea.time, 'sleep'), patch.object(krea.urllib.request, 'urlopen', return_value=io.BytesIO(png())) as download:
                result = krea.generate(engine, 'A cabin', 'portrait', 7, (48, 80))
                self.assertEqual(request.call_args_list[0].args[0], '/generate/image/krea/krea-2/' + route)
                body = request.call_args_list[0].args[2]
                self.assertEqual(body, {'prompt': 'A cabin', 'aspect_ratio': '9:16', 'resolution': '1K', 'seed': 7, 'creativity': 'raw'})
                self.assertEqual(request.call_args_list[1].args[0], '/jobs/test-job')
                self.assertEqual(Image.open(io.BytesIO(result)).size, (48, 80))
                self.assertIsInstance(download.call_args.args[0], str)  # no bearer header sent to image host

    def test_failed_cancelled_and_timeout_jobs_do_not_resubmit(self):
        krea.save_key('test-secret')
        for status in ('failed', 'cancelled'):
            with patch.object(krea, 'request', return_value={'job_id': 'job', 'status': status}) as request:
                with self.assertRaisesRegex(RuntimeError, status):
                    krea.generate('krea-api-turbo', 'A cabin', 'square', 7, (32, 32))
                self.assertEqual(request.call_count, 1)
        with patch.object(krea, 'request', return_value={'job_id': 'job', 'status': 'processing'}) as request, patch.object(krea.time, 'monotonic', side_effect=[0, 601]):
            with self.assertRaisesRegex(RuntimeError, 'longer than expected'):
                krea.generate('krea-api-turbo', 'A cabin', 'square', 7, None)
            self.assertEqual(request.call_count, 1)

    def test_http_errors_are_actionable_and_secrets_are_not_echoed(self):
        for code, expected in ((401, 'rejected'), (402, 'balance'), (429, 'limit'), (500, 'failed')):
            error = urllib.error.HTTPError('https://api.krea.ai', code, 'err', {}, io.BytesIO(b'private-secret prompt'))
            with patch.object(krea.urllib.request, 'urlopen', side_effect=error):
                with self.assertRaisesRegex(RuntimeError, expected) as caught:
                    krea.request('/jobs/id', 'private-secret')
                self.assertNotIn('private-secret', str(caught.exception))

    def test_invalid_result_and_stop_after_submit_keep_request_single(self):
        krea.save_key('test-secret')
        with patch.object(krea, 'request', return_value={'job_id': 'job', 'status': 'processing'}) as request:
            with self.assertRaisesRegex(RuntimeError, 'may still complete'):
                krea.generate('krea-api-turbo', 'A cabin', 'square', 7, None, iter([False, True]).__next__)
            self.assertEqual(request.call_count, 1)
        for url in ('http://gen.krea.ai/image.png', 'https://127.0.0.1/internal', 'https://name:secret@gen.krea.ai/image'):
            with patch.object(krea, 'request', return_value={'job_id': 'job', 'status': 'completed', 'result': {'urls': [url]}}), patch.object(krea.urllib.request, 'urlopen') as fetch:
                with self.assertRaisesRegex(RuntimeError, 'invalid image URL'):
                    krea.generate('krea-api-turbo', 'A cabin', 'square', 7, None)
                fetch.assert_not_called()

    def test_gpu_workflow_uses_native_krea_nodes_and_all_links_resolve(self):
        workflow = comfy.image_workflow(pipeline.load_config(), 'krea2-turbo', 'A cabin', 1080, 1920, 31, 'test/krea')
        self.assertEqual(workflow['2']['inputs']['type'], 'krea2')
        self.assertEqual(workflow['6']['inputs']['steps'], 8)
        self.assertEqual(workflow['6']['inputs']['sampler_name'], 'euler')
        self.assertEqual(workflow['6']['inputs']['cfg'], 1)
        self.assertEqual(workflow['4']['class_type'], 'ConditioningZeroOut')
        for node in workflow.values():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    self.assertIn(value[0], workflow)

    def test_engine_catalog_marks_missing_and_remote_models_correctly(self):
        state = copy.deepcopy(cloudgpu.state)
        self.addCleanup(lambda: (cloudgpu.state.clear(), cloudgpu.state.update(state)))
        cloudgpu.state.update(status='off')
        with patch.object(pipeline, 'model_path', return_value=None), patch.object(pipeline.cloud, 'available', return_value=False):
            catalog = pipeline.engines()
            self.assertFalse(next(e for e in catalog['image'] if e['id'] == 'krea2-turbo')['installed'])
            self.assertFalse(next(e for e in catalog['image'] if e['id'] == 'krea-api-turbo')['installed'])
            self.assertNotIn('krea2-turbo', catalog['reference_engines'])
            cloudgpu.state.update(status='ready', extras={**cloudgpu.state['extras'], 'krea2': True})
            self.assertTrue(next(e for e in pipeline.engines()['image'] if e['id'] == 'krea2-turbo')['installed'])

    def test_api_routing_does_not_open_comfy_even_with_cloud_connected(self):
        project = {'id': 'andy--test', 'image_engine': 'krea-api-large', 'references': []}
        with patch.object(pipeline, '_keyframes') as generate, patch.object(cloudgpu, 'url', return_value='http://rented') as remote:
            pipeline.do_keyframes(project)
            remote.assert_not_called()
            self.assertIsNone(generate.call_args.kwargs.get('remote'))

    def test_gpu_routing_uses_rental_and_does_not_silently_fall_back_to_local(self):
        project = {'id': 'andy--test', 'image_engine': 'krea2-turbo', 'references': []}
        with patch.object(cloudgpu, 'url', return_value='http://rented'), patch.dict(cloudgpu.state, {'extras': {'krea2': False}}), patch.object(pipeline, '_keyframes') as generate:
            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                pipeline.do_keyframes(project)
            generate.assert_not_called()
        with patch.object(cloudgpu, 'url', return_value='http://rented'), patch.dict(cloudgpu.state, {'extras': {'krea2': True}}), patch.object(pipeline, '_keyframes') as generate:
            pipeline.do_keyframes(project)
            self.assertEqual(generate.call_args.args[-1], 'http://rented')

    def test_paid_keyframe_pipeline_preserves_versions_without_local_engine(self):
        cfg = pipeline.load_config()
        project = {'id': 'andy--test', 'image_engine': 'krea-api-turbo', 'quality': 'high', 'mode': 'images',
                   'aspect': 'portrait', 'style': '', 'references': [], 'seed': 4, 'keyframe_seeds': {},
                   'plan': {'keyframes': [{'image_prompt': 'A cabin'}], 'character': '', 'setting': ''}, 'log': []}
        with patch.object(pipeline, 'list_characters', return_value=[]), patch.object(pipeline, 'log'), patch.object(pipeline, 'write_caption'), patch.object(pipeline, 'save_keyframe') as save, patch.object(krea, 'generate', return_value=b'png') as generate, patch.object(comfy, 'Comfy', side_effect=AssertionError('API must not open ComfyUI')):
            pipeline._keyframes(project)
            self.assertEqual(generate.call_args.args[0], 'krea-api-turbo')
            self.assertEqual(save.call_args.args[3]['engine'], 'krea-api-turbo')
            self.assertEqual(project['keyframe_engines']['0'], 'krea-api-turbo')

    def test_settings_http_requires_admin_and_never_exposes_key(self):
        http_server = server.Server(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        thread.start()
        def send(method, path, body=None, role='', origin=None):
            connection = http.client.HTTPConnection('127.0.0.1', http_server.server_port, timeout=5)
            headers = {'Content-Type': 'application/json', 'X-Test-User': role}
            if origin:
                headers['Origin'] = origin
            connection.request(method, path, json.dumps(body) if body is not None else None, headers)
            response = connection.getresponse()
            status, data = response.status, json.loads(response.read())
            connection.close()
            return status, data
        try:
            with patch.object(server.Handler, 'user', lambda handler: handler.headers.get('X-Test-User')), patch.object(server.auth, 'is_admin', lambda user: user == 'admin'), patch.object(server.Handler, 'log_message'):
                self.assertEqual(send('GET', '/api/krea')[0], 401)
                self.assertEqual(send('POST', '/api/krea/key', {'key': 'test-api-key'}, 'member')[0], 403)
                self.assertEqual(send('POST', '/api/krea/key', {'key': 'test-api-key'}, 'admin', 'https://other.example')[0], 403)
                self.assertFalse(self.secrets.exists())
                status, result = send('POST', '/api/krea/key', {'key': 'test-api-key'}, 'admin')
                self.assertEqual(status, 200)
                self.assertEqual(result, {'api_ready': True})
                status, result = send('GET', '/api/krea', role='member')
                self.assertEqual(status, 200)
                self.assertTrue(result['api_ready'])
                self.assertNotIn('test-api-key', json.dumps(result))
        finally:
            http_server.shutdown()
            http_server.server_close()
            thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()

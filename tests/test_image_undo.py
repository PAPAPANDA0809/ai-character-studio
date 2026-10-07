"""Image recovery tests use temporary files; no live projects or generation services."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import imageundo
import pipeline


class ImageUndoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.undo-fixture-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for key, value in [('PROJECTS', root / 'projects'), ('TRASH', root / 'trash')]:
            mock = patch.object(pipeline, key, str(value))
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch.dict(pipeline.current, pid=None, task=None)
        mock.start()
        self.addCleanup(mock.stop)
        self.pid = 'viewer--fixture'
        self.folder = Path(pipeline.pdir(self.pid))
        (self.folder / 'keyframes/history').mkdir(parents=True)
        self.current = self.folder / 'keyframes/kf_00.png'
        self.old = self.folder / 'keyframes/history/kf_00_123.png'
        self.current.write_bytes(b'current image')
        self.old.write_bytes(b'older image')
        self.project = dict(id=self.pid, status='done', versions={'0':['kf_00_123.png']},
                            pic_info={'0':{'prompt':'current prompt'}},
                            version_info={'kf_00_123.png':{'prompt':'old prompt'}},
                            plan={'keyframes':[{'image_prompt':'plan prompt'}]}, keyframes_version=1, clips_version=1)
        pipeline.save(self.project)

    def delete(self, rel='keyframes/kf_00.png'):
        return imageundo.delete('viewer', self.pid, rel)['undo']

    def test_current_restores_image_promotion_and_prompt_records(self):
        token = self.delete()
        self.assertEqual(self.current.read_bytes(), b'older image')
        self.assertFalse(self.old.exists())
        imageundo.restore('viewer', token)
        self.assertEqual(self.current.read_bytes(), b'current image')
        self.assertEqual(self.old.read_bytes(), b'older image')
        p = pipeline.load(self.pid)
        for key in ('versions', 'pic_info', 'version_info', 'plan'):
            self.assertEqual(p[key], self.project[key])

    def test_older_version_restores_exact_order_and_prompt(self):
        token = self.delete('keyframes/history/kf_00_123.png')
        self.assertFalse(self.old.exists())
        imageundo.restore('viewer', token)
        self.assertEqual(self.old.read_bytes(), b'older image')
        self.assertEqual(pipeline.load(self.pid)['version_info'], self.project['version_info'])

    def test_restore_without_previous_version(self):
        p = copy.deepcopy(self.project)
        p['versions'] = {}
        pipeline.save(p)
        token = self.delete()
        self.assertFalse(self.current.exists())
        imageundo.restore('viewer', token)
        self.assertEqual(self.current.read_bytes(), b'current image')

    def test_tokens_are_account_scoped_and_single_use(self):
        token = self.delete()
        with self.assertRaises(ValueError):
            imageundo.restore('someone', token)
        imageundo.restore('viewer', token)
        with self.assertRaises(ValueError):
            imageundo.restore('viewer', token)

    def test_path_traversal_and_wrong_owner_are_rejected(self):
        for user, rel in [('someone','keyframes/kf_00.png'), ('viewer','../project.json'), ('viewer','/keyframes/kf_00.png')]:
            with self.assertRaises(ValueError):
                imageundo.delete(user, self.pid, rel)
        with self.assertRaises(ValueError):
            imageundo.restore('viewer', '../receipt')
        self.assertEqual(self.current.read_bytes(), b'current image')

    def test_newer_file_is_never_overwritten(self):
        token = self.delete()
        self.current.write_bytes(b'newer generation')
        with self.assertRaisesRegex(ValueError, 'newer images'):
            imageundo.restore('viewer', token)
        self.assertEqual(self.current.read_bytes(), b'newer generation')

    def test_expired_record_and_running_project_keep_data(self):
        token = self.delete()
        receipt = Path(pipeline.TRASH) / 'viewer/undo' / token / 'receipt.json'
        record = json.loads(receipt.read_text())
        record['expires'] = 0
        receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'expired'):
            imageundo.restore('viewer', token)
        self.assertTrue(receipt.with_name('deleted.png').exists())
        p = pipeline.load(self.pid)
        p['status'] = 'queued'
        pipeline.save(p)
        with self.assertRaisesRegex(ValueError, 'generation'):
            self.delete()

    def test_failed_save_rolls_back_file_moves(self):
        with patch.object(pipeline, 'save', side_effect=OSError('fixture save failure')):
            with self.assertRaises(OSError):
                self.delete()
        self.assertEqual(self.current.read_bytes(), b'current image')
        self.assertEqual(self.old.read_bytes(), b'older image')

    def test_prompt_edits_survive_restore(self):
        token = self.delete()
        p = pipeline.load(self.pid)
        p['plan']['keyframes'][0]['image_prompt'] = 'updated prompt'
        pipeline.save(p)
        imageundo.restore('viewer', token)
        self.assertEqual(pipeline.load(self.pid)['plan']['keyframes'][0]['image_prompt'], 'updated prompt')

    def test_authenticated_http_delete_and_restore(self):
        import http.client
        import threading
        import server
        srv = server.Server(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        def request(path, body):
            conn = http.client.HTTPConnection('127.0.0.1', srv.server_port, timeout=5)
            conn.request('POST', path, json.dumps(body), {'Content-Type':'application/json'})
            response = conn.getresponse()
            result = response.status, json.loads(response.read())
            conn.close()
            return result
        try:
            with patch.object(server.Handler, 'log_message'), patch.object(server.Handler, 'user', return_value='viewer'):
                code, body = request('/api/delete', {'undo':True,'files':[{'project':self.pid,'path':'keyframes/kf_00.png'}]})
                self.assertEqual(code, 200)
                token = body['undo']
                with patch.object(server.Handler, 'user', return_value=None):
                    self.assertEqual(request('/api/restore_image', {'token':token})[0], 401)
                with patch.object(server.Handler, 'user', return_value='someone'):
                    self.assertEqual(request('/api/restore_image', {'token':token})[0], 400)
                self.assertEqual(request('/api/restore_image', {'token':token})[0], 200)
                self.assertEqual(self.current.read_bytes(), b'current image')
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()

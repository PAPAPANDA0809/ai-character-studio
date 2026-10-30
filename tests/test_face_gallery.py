"""Local gallery import, private selection and authenticated API regression checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import facegallery
import server


class FaceGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.records = []
        receipts = {}
        for identifier, section in (("face1", "face"), ("style1", "styling")):
            path = self.source / (identifier + ".png")
            Image.new("RGB", (30, 40), "navy").save(path)
            self.records.append({"id": identifier, "section": section, "hair": "ショート", "bytes": path.stat().st_size})
            receipts[identifier] = {"file": path.name}
        (self.source / "gallery-manifest.json").write_text(json.dumps(self.records), encoding="utf-8")
        (self.source / "download-receipts.json").write_text(json.dumps(receipts), encoding="utf-8")
        self.patch = patch.multiple(facegallery, ROOT=self.root / "cache", _cache=None)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_import_indexes_both_sections_and_keeps_original_paths_private(self):
        result = facegallery.import_local(self.source)
        self.assertEqual((result["total"], result["portraits"], result["styling"]), (2, 1, 1))
        items = facegallery.listing()["items"]
        self.assertEqual([item["cat"] for item in items], ["Face portraits", "Styling"])
        self.assertTrue(all(item["sub"] == "Short" and "original" not in item for item in items))
        self.assertNotIn(str(self.source), json.dumps(items))
        self.assertTrue(Path(facegallery.thumb_file("face:face1")).is_file())
        self.assertIsNone(facegallery.thumb_file("face:../../private"))

    def test_selection_is_private_repeatable_and_preserves_existing_pictures(self):
        facegallery.import_local(self.source)
        first_user = self.root / "users" / "ann"
        second_user = self.root / "users" / "bob"
        first_user.mkdir(parents=True)
        existing = first_user / "gallery-face1.png"
        existing.write_bytes(b"my existing photo")
        name = facegallery.select("face:face1", first_user)
        self.assertEqual(name, "gallery-face1-2.png")
        self.assertEqual(existing.read_bytes(), b"my existing photo")
        self.assertEqual(facegallery.select("face:face1", first_user), name)
        self.assertFalse(second_user.exists())
        self.assertEqual((first_user / name).read_bytes(), (self.source / "face1.png").read_bytes())
        with self.assertRaises(ValueError):
            facegallery.select("face:../../private", first_user)

    def test_import_rejects_a_manifest_path_outside_the_library(self):
        outside = self.root / "outside.png"
        outside.write_bytes((self.source / "face1.png").read_bytes())
        (self.source / "download-receipts.json").write_text(json.dumps({"face1": {"file": "../outside.png"}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            facegallery.import_local(self.source)

    def test_face_selection_endpoint_requires_signin(self):
        handler = object.__new__(server.Handler)
        handler.path = "/api/library/face/select"
        replies = []
        handler.send_json = lambda body, code=200: replies.append((body, code))
        handler.user = lambda: None
        with patch.object(facegallery, "select") as select:
            handler.do_POST()
        self.assertEqual(replies[0][1], 401)
        select.assert_not_called()

    def test_face_selection_endpoint_uses_current_users_characters(self):
        facegallery.import_local(self.source)
        handler = object.__new__(server.Handler)
        handler.path = "/api/library/face/select"
        replies = []
        handler.send_json = lambda body, code=200: replies.append((body, code))
        handler.user = lambda: "ann"
        handler.body = lambda: {"id": "face:style1"}
        destination = self.root / "ann"
        with patch.object(server.pipeline, "char_dir", return_value=str(destination)) as char_dir:
            handler.do_POST()
        char_dir.assert_called_once_with("ann")
        self.assertEqual(replies, [({"name": "gallery-style1.png"}, 200)])
        self.assertTrue((destination / "gallery-style1.png").is_file())


if __name__ == "__main__":
    unittest.main()

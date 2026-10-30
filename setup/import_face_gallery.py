"""Add an existing AI Face Reference Library to Reelmint's face picker."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import facegallery

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python setup/import_face_gallery.py "path/to/AI-Face-Reference-Library"')
    print(json.dumps(facegallery.import_local(sys.argv[1]), ensure_ascii=True), flush=True)

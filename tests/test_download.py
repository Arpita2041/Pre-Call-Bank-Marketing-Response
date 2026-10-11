import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from project.common import download_member, sha256
from project.data import URL

class DownloadTests(unittest.TestCase):
    def test_named_member_cache_hash_and_archive_layout(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("nested/member.csv", "x,y\n1,2\n")
            archive.writestr("unrelated.txt", "not selected")
        inner = buffer.getvalue()
        for nested in (False, True):
            if nested:
                outer = io.BytesIO()
                with zipfile.ZipFile(outer, "w") as archive:
                    archive.writestr("folder/bank.zip", inner)
                payload = outer.getvalue()
            else:
                payload = inner
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "result.csv"
                with patch("project.common.urlopen", return_value=io.BytesIO(payload)) as mocked:
                    download_member(URL, destination, "member.csv",
                                    inner_archive="bank.zip" if nested else None)
                    self.assertEqual(destination.read_text(), "x,y\n1,2\n")
                    metadata = json.loads(destination.with_suffix(".csv.source.json").read_text())
                    self.assertEqual(metadata["sha256"], sha256(destination))
                    download_member(URL, destination, "member.csv")
                    self.assertEqual(mocked.call_count, 1)
                    self.assertFalse(destination.with_name("result.csv.download").exists())

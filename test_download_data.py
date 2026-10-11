import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from download_data import download


class DownloadTest(unittest.TestCase):
    def test_verified_download_and_failed_replacement(self):
        content = b"test data"
        checksum = hashlib.md5(content).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            target = folder / "test.npy"
            with patch("download_data.urlopen", return_value=io.BytesIO(content)):
                download(target.name, len(content), checksum, folder)
            self.assertEqual(target.read_bytes(), content)
            with patch("download_data.urlopen") as request:
                download(target.name, len(content), checksum, folder)
                request.assert_not_called()
            with patch("download_data.urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(ValueError):
                    download(target.name, 4, checksum, folder)
            self.assertEqual(target.read_bytes(), content)
            self.assertFalse(target.with_name("test.npy.part").exists())


if __name__ == "__main__":
    unittest.main()

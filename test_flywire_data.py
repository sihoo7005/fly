import struct
import tempfile
import unittest
from pathlib import Path

from flywire_data import FlatBuffer, numbers, read_root_ids, unpack_buffer


class DataTests(unittest.TestCase):
    def test_npy_preserves_uint64_and_rejects_duplicates_and_truncation(self):
        header = b"{'descr': '<u8', 'fortran_order': False, 'shape': (2,), }\n"
        prefix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
        ids = (720575940621014292, 720575940621939071)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roots.npy"
            path.write_bytes(prefix + struct.pack("<QQ", *ids))
            self.assertEqual(list(read_root_ids(path)), list(ids))
            path.write_bytes(prefix + struct.pack("<QQ", ids[0], ids[0]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_root_ids(path)
            path.write_bytes(prefix + struct.pack("<Q", ids[0]))
            with self.assertRaisesRegex(ValueError, "Truncated"):
                read_root_ids(path)

    def test_numeric_and_uncompressed_arrow_buffers(self):
        raw = struct.pack("<qqq", -1, 0, 720575940621014292)
        self.assertEqual(list(numbers(raw, "q")), [-1, 0, 720575940621014292])
        self.assertEqual(unpack_buffer(struct.pack("<q", -1) + raw, 1), raw)
        self.assertEqual(unpack_buffer(raw, None), raw)
        with self.assertRaises(ValueError):
            unpack_buffer(b"short", 1)

    def test_zstandard_using_standard_library(self):
        try:
            from compression import zstd
        except ImportError:
            self.skipTest("Python 3.14 standard-library compression.zstd is required")
        raw = struct.pack("<qq", 123, 720575940621014292)
        compressed = zstd.compress(raw)
        self.assertEqual(unpack_buffer(struct.pack("<q", len(raw)) + compressed, 1), raw)
        with self.assertRaisesRegex(ValueError, "size"):
            unpack_buffer(struct.pack("<q", len(raw) + 1) + compressed, 1)

    def test_flatbuffer_bounds(self):
        fb = FlatBuffer(b"\0\0\0\0")
        for position in (-1, 1):
            with self.assertRaises(ValueError):
                fb.number("I", position)


if __name__ == "__main__":
    unittest.main()

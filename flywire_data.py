"""Readers for FlyWire v783 NPY and Feather V2, using the standard library."""

import ast
import struct
import sys
from array import array
from pathlib import Path


class FlatBuffer:
    def __init__(self, data):
        self.data = data

    def number(self, fmt, offset):
        if offset < 0 or offset + struct.calcsize("<" + fmt) > len(self.data):
            raise ValueError("Invalid metadata offset")
        return struct.unpack_from("<" + fmt, self.data, offset)[0]

    def field(self, table, index):
        vtable = table - self.number("i", table)
        entry = vtable + 4 + 2 * index
        if entry >= vtable + self.number("H", vtable):
            return None
        relative = self.number("H", entry)
        return table + relative if relative else None

    def scalar(self, table, index, fmt, default=0):
        position = self.field(table, index)
        return default if position is None else self.number(fmt, position)

    def reference(self, table, index):
        position = self.field(table, index)
        return None if position is None else position + self.number("I", position)

    def string(self, position):
        length = self.number("I", position)
        start = position + 4
        if start + length > len(self.data):
            raise ValueError("Invalid metadata string")
        return self.data[start:start + length].decode("utf-8")

    def tables(self, vector):
        if vector is None:
            return []
        count = self.number("I", vector)
        return [p + self.number("I", p)
                for p in range(vector + 4, vector + 4 + count * 4, 4)]


def read_exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("Truncated data file")
    return data


def numbers(data, code):
    result = array(code)
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def read_root_ids(path):
    with Path(path).open("rb") as stream:
        if read_exact(stream, 8) != b"\x93NUMPY\x01\x00":
            raise ValueError("Expected NPY version 1.0")
        length = struct.unpack("<H", read_exact(stream, 2))[0]
        header = ast.literal_eval(read_exact(stream, length).decode("latin1").strip())
        if header["descr"] != "<u8" or len(header["shape"]) != 1:
            raise ValueError("Expected a one-dimensional uint64 neuron list")
        ids = numbers(read_exact(stream, header["shape"][0] * 8), "Q")
        if stream.read(1) or len(set(ids)) != len(ids) or 0 in ids:
            raise ValueError("Invalid neuron list: extra data, duplicate or zero ID")
        return ids


def unpack_buffer(data, codec):
    if codec is None or not data:
        return data
    if len(data) < 8:
        raise ValueError("Missing compressed buffer length")
    expected = struct.unpack_from("<q", data)[0]
    if expected == -1:
        return data[8:]
    if expected < 0 or codec != 1:
        raise ValueError("Only Arrow Zstandard buffer compression is supported")
    try:
        from compression import zstd
    except ImportError as error:
        raise RuntimeError("Reading FlyWire Feather requires Python 3.14+ with "
                           "standard-library compression.zstd. No pip packages are needed.") from error
    decoded = zstd.decompress(data[8:])
    if len(decoded) != expected:
        raise ValueError("Decompressed buffer size mismatch")
    return decoded


class FeatherReader:
    """Stream flat int64, float64 and UTF-8 columns from these four data files."""

    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            if read_exact(stream, 6) != b"ARROW1" or size < 16:
                raise ValueError("Expected Feather V2 / Arrow file")
            stream.seek(size - 10)
            trailer = read_exact(stream, 10)
            if trailer[4:] != b"ARROW1":
                raise ValueError("Missing Arrow footer")
            length = struct.unpack_from("<I", trailer)[0]
            if length > size - 16:
                raise ValueError("Invalid footer length")
            stream.seek(size - 10 - length)
            fb = FlatBuffer(read_exact(stream, length))
        root = fb.number("I", 0)
        schema = fb.reference(root, 1)
        if fb.scalar(schema, 0, "h") != 0:
            raise ValueError("Only little-endian Arrow files are supported")
        self.columns = []
        buffer_index = 0
        for field in fb.tables(fb.reference(schema, 1)):
            name = fb.string(fb.reference(field, 0))
            kind = fb.scalar(field, 2, "B")
            detail = fb.reference(field, 3)
            if fb.reference(field, 4) is not None or fb.tables(fb.reference(field, 5)):
                raise ValueError("Dictionary and nested columns are not supported")
            if kind == 2 and fb.scalar(detail, 0, "i") == 64:
                dtype = "q" if fb.scalar(detail, 1, "?") else "Q"
            elif kind == 3 and fb.scalar(detail, 0, "h") == 2:
                dtype = "d"
            elif kind == 5:
                dtype = "utf8"
            else:
                raise ValueError(f"Unsupported column type: {name}")
            self.columns.append((name, dtype, buffer_index))
            buffer_index += 3 if dtype == "utf8" else 2
        self.buffer_count = buffer_index
        dictionaries = fb.reference(root, 2)
        if dictionaries is not None and fb.number("I", dictionaries):
            raise ValueError("Dictionary batches are not supported")
        vector = fb.reference(root, 3)
        self.blocks = [struct.unpack_from("<qi4xq", fb.data, vector + 4 + i * 24)
                       for i in range(fb.number("I", vector))]
        for offset, metadata_length, body_length in self.blocks:
            if min(offset, metadata_length, body_length) < 0 or offset + metadata_length + body_length > size - 10 - length:
                raise ValueError("Record batch outside file")

    def batches(self, columns=None):
        requested = set(columns) if columns is not None else {c[0] for c in self.columns}
        if requested - {c[0] for c in self.columns}:
            raise ValueError("Requested column is missing")
        with self.path.open("rb") as stream:
            for offset, metadata_length, body_length in self.blocks:
                stream.seek(offset)
                raw = read_exact(stream, metadata_length)
                prefix = 8 if raw[:4] == b"\xff\xff\xff\xff" else 4
                message = FlatBuffer(raw[prefix:])
                root = message.number("I", 0)
                if message.scalar(root, 1, "B") != 3:
                    raise ValueError("Expected Arrow RecordBatch")
                batch = message.reference(root, 2)
                count = message.scalar(batch, 0, "q")
                nodes = message.reference(batch, 1)
                buffers = message.reference(batch, 2)
                if count < 0 or message.number("I", nodes) != len(self.columns) or message.number("I", buffers) != self.buffer_count:
                    raise ValueError("Unexpected batch layout")
                compression = message.reference(batch, 3)
                codec = None if compression is None else message.scalar(compression, 0, "B")
                if compression is not None and message.scalar(compression, 1, "B") != 0:
                    raise ValueError("Expected per-buffer compression")

                def buffer(index):
                    start = message.number("q", buffers + 4 + index * 16)
                    size = message.number("q", buffers + 12 + index * 16)
                    if min(start, size) < 0 or start + size > body_length:
                        raise ValueError("Buffer outside record batch")
                    stream.seek(offset + metadata_length + start)
                    return unpack_buffer(read_exact(stream, size), codec)

                result = {}
                for i, (name, dtype, index) in enumerate(self.columns):
                    if name not in requested:
                        continue
                    if message.number("q", nodes + 4 + i * 16) != count:
                        raise ValueError("Column length differs from batch")
                    null_count = message.number("q", nodes + 12 + i * 16)
                    validity = buffer(index) if null_count else b""
                    if null_count and len(validity) < (count + 7) // 8:
                        raise ValueError("Truncated validity bitmap")
                    if dtype == "utf8":
                        offsets = numbers(buffer(index + 1), "i")
                        strings = buffer(index + 2)
                        if len(offsets) != count + 1 or offsets[0] != 0 or offsets[-1] != len(strings):
                            raise ValueError("Invalid string offsets")
                        if any(a > b for a, b in zip(offsets, offsets[1:])):
                            raise ValueError("Nonmonotonic string offsets")
                        values = [strings[offsets[j]:offsets[j + 1]].decode("utf-8") for j in range(count)]
                    else:
                        values = numbers(buffer(index + 1), dtype)
                        if len(values) != count:
                            raise ValueError("Invalid numeric column length")
                    if null_count:
                        values = [value if validity[j // 8] & (1 << (j % 8)) else None
                                  for j, value in enumerate(values)]
                    result[name] = values
                yield result

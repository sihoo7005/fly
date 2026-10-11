"""Download the two core FlyWire v783 files with Python's standard library."""

import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://zenodo.org/api/records/10676866/files/"
FILES = {
    "proofread_connections_783.feather": (852022274, "f48f972d262323a102aed49af1396b8a"),
    "proofread_root_ids_783.npy": (1114168, "e0e6c19732fd8c7a4e39a2d170105421"),
}


def valid(path, size, checksum):
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == checksum


def download(name, size, checksum, folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if valid(target, size, checksum):
        print(f"Already verified: {target}", flush=True)
        return
    partial = target.with_name(target.name + ".part")
    try:
        print(f"Downloading {name} ({size / 1_000_000:.1f} MB)...", flush=True)
        with urlopen(BASE_URL + name + "/content", timeout=60) as source:
            with partial.open("wb") as output:
                received = 0
                last_report = 0
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
                    received += len(chunk)
                    if received - last_report >= 50_000_000:
                        print(f"  {received / size:.0%}", flush=True)
                        last_report = received
        if not valid(partial, size, checksum):
            raise ValueError(f"Size or checksum mismatch: {name}")
        partial.replace(target)
        print(f"Verified: {target}", flush=True)
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "data" / "raw"
    try:
        for name, (size, checksum) in FILES.items():
            download(name, size, checksum, folder)
    except (OSError, ValueError) as error:
        sys.exit(f"Download failed: {error}\nRun the command again to retry.")

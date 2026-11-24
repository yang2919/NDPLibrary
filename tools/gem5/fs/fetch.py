"""Fetch official F0 resources and verify checksums of uncompressed files."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
manifest = json.loads(Path(__file__).with_name("resources.json").read_text())

def checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for resource in manifest["resources"]:
    target = destination / resource["file"]
    if target.exists() and checksum(target) == resource["md5"]:
        print(f"Verified {target}", flush=True)
        continue
    archive = target.with_name(target.name + ".download")
    print(f"Downloading {resource['url']}", flush=True)
    urllib.request.urlretrieve(resource["url"], archive)
    temporary = target.with_name(target.name + ".partial")
    if resource["gzip"]:
        with gzip.open(archive, "rb") as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
    else:
        archive.replace(temporary)
    if checksum(temporary) != resource["md5"]:
        raise RuntimeError(f"Checksum mismatch: {temporary}")
    temporary.replace(target)
    archive.unlink(missing_ok=True)
    print(f"Verified {target}", flush=True)

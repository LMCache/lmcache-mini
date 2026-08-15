# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

from lmcache_mini.l2.base import L2Adapter


def filename(key):
    return (f"{key.model.replace('/', '-')}@{key.rank:x}"
            f"@{key.chunk_hash.hex()}.data")


class FSAdapter(L2Adapter):
    def __init__(self, base_path):
        self.base = Path(base_path)
        self.tmp = self.base / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        super().__init__()

    def store(self, keys, objs):
        for key, obj in zip(keys, objs):
            final = self.base / filename(key)
            if final.exists():
                continue
            tmp = self.tmp / final.name
            tmp.write_bytes(obj.byte_array)
            os.replace(tmp, final)

    def lookup(self, keys):
        hits = 0
        for key in keys:
            if not (self.base / filename(key)).exists():
                break
            hits += 1
        return hits

    def load(self, keys, objs):
        loaded = 0
        for key, obj in zip(keys, objs):
            path = self.base / filename(key)
            if not path.exists():
                break
            with open(path, "rb") as f:
                if f.readinto(obj.byte_array) != obj.nbytes():
                    break
            loaded += 1
        return loaded

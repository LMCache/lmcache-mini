# SPDX-License-Identifier: Apache-2.0
from lmcache_mini.l2.base import L2Adapter


class MockAdapter(L2Adapter):
    def __init__(self):
        self.chunks = {}
        super().__init__()

    def store(self, keys, objs):
        for key, obj in zip(keys, objs):
            self.chunks[key] = bytes(obj.byte_array)

    def lookup(self, keys):
        hits = 0
        for key in keys:
            if key not in self.chunks:
                break
            hits += 1
        return hits

    def load(self, keys, objs):
        loaded = 0
        for key, obj in zip(keys, objs):
            data = self.chunks.get(key)
            if data is None:
                break
            obj.byte_array[:] = data
            loaded += 1
        return loaded

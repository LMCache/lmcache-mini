# SPDX-License-Identifier: Apache-2.0
import queue
import threading
import time


class PrefetchController:
    def __init__(self, l1, l2s):
        self.l1 = l1
        self.l2s = l2s
        self.lock = threading.Lock()
        self.jobs = queue.SimpleQueue()
        self.hits: dict[str, tuple | None] = {}
        self.held: dict[str, dict] = {}
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            self.prefetch(*self.jobs.get())

    def prefetch(self, request_id, keys, chunk_nbytes):
        l1_hits = self.l1.reserve_read_prefix(keys)
        missing = keys[l1_hits:]
        l2_hits = self.lookup_l2s(missing)
        load_keys, objs = self.reserve_load(missing, max(l2_hits, default=0),
                                            chunk_nbytes)
        start = time.perf_counter()
        loaded = self.load_from_l2s(load_keys, objs, l2_hits)
        elapsed = time.perf_counter() - start
        self.l1.finish_write_and_reserve_read(load_keys[:loaded])
        self.l1.delete(load_keys[loaded:], force=True)
        gbps = loaded * chunk_nbytes / elapsed / 1e9 if loaded else 0.0
        self.resolve(request_id, keys[:l1_hits], load_keys[:loaded], gbps)

    def lookup_l2s(self, missing):
        if not missing:
            return []
        futures = [l2.submit_lookup(missing) for l2 in self.l2s]
        return [future.result() for future in futures]

    def reserve_load(self, missing, l2_hits, chunk_nbytes):
        load_keys, objs = [], []
        for key in missing[:l2_hits]:
            obj = self.l1.reserve_write([key], chunk_nbytes,
                                        is_temporary=True)[key]
            if obj is None:
                break
            load_keys.append(key)
            objs.append(obj)
        return load_keys, objs

    def load_from_l2s(self, load_keys, objs, l2_hits):
        if not load_keys:
            return 0
        loaded = 0
        for i in sorted(range(len(self.l2s)), key=lambda i: -l2_hits[i]):
            if loaded == len(load_keys) or l2_hits[i] <= loaded:
                break
            loaded += self.l2s[i].submit_load(load_keys[loaded:],
                                              objs[loaded:]).result()
        return loaded

    def resolve(self, request_id, l1_keys, loaded_keys, gbps):
        hit_keys = l1_keys + loaded_keys
        with self.lock:
            if request_id in self.hits:
                self.hits[request_id] = (len(l1_keys), len(loaded_keys), gbps)
                self.held[request_id] = dict.fromkeys(hit_keys)
                return
        self.l1.finish_read(hit_keys)

    def start_session(self, request_id, keys, chunk_nbytes):
        with self.lock:
            self.hits[request_id] = None
        self.jobs.put((request_id, keys, chunk_nbytes))

    def query(self, request_id):
        with self.lock:
            return self.hits.get(request_id)

    def release(self, request_id, keys=None):
        with self.lock:
            held = self.held.get(request_id, {})
            if keys is None:
                keys = list(held)
            released = [key for key in keys if key in held]
            for key in released:
                del held[key]
        self.l1.finish_read(released)

    def release_first(self, request_id, count):
        with self.lock:
            keys = list(self.held.get(request_id, {}))[:count]
        self.release(request_id, keys)

    def end_session(self, request_id):
        self.release(request_id)
        with self.lock:
            self.hits.pop(request_id, None)
            self.held.pop(request_id, None)

# SPDX-License-Identifier: Apache-2.0
import queue
import threading

from lmcache_mini.l1.manager import Listener


class StoreController(Listener):
    def __init__(self, l1, l2s):
        self.l1 = l1
        self.l2s = l2s
        self.pending = queue.SimpleQueue()
        l1.register_listener(self)
        threading.Thread(target=self.run, daemon=True).start()

    def on_write_finished(self, keys):
        self.pending.put(list(keys))

    def run(self):
        while True:
            keys = self.pending.get()
            reserved = self.l1.reserve_read(keys)
            locked = [key for key in keys if reserved[key] is not None]
            if not locked:
                continue
            objs = [reserved[key] for key in locked]
            for future in [l2.submit_store(locked, objs) for l2 in self.l2s]:
                future.result()
            self.l1.finish_read(locked)

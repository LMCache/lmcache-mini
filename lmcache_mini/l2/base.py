# SPDX-License-Identifier: Apache-2.0
import queue
import threading
from concurrent.futures import Future


class L2Adapter:
    def __init__(self):
        self.tasks = queue.SimpleQueue()
        threading.Thread(target=self.run, daemon=True).start()

    def submit_store(self, keys, objs) -> Future:
        return self.submit(self.store, keys, objs)

    def submit_lookup(self, keys) -> Future:
        return self.submit(self.lookup, keys)

    def submit_load(self, keys, objs) -> Future:
        return self.submit(self.load, keys, objs)

    def submit(self, fn, *args) -> Future:
        future = Future()
        self.tasks.put((future, fn, args))
        return future

    def run(self):
        while True:
            future, fn, args = self.tasks.get()
            future.set_result(fn(*args))

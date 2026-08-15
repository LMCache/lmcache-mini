# SPDX-License-Identifier: Apache-2.0
import threading
import time
from collections import OrderedDict

from lmcache_mini.l1.manager import Listener

WATERMARK = 0.8
EVICTION_RATIO = 0.2
INTERVAL = 1.0


class LRUPolicy(Listener):
    def __init__(self):
        self.lock = threading.Lock()
        self.order = OrderedDict()

    def on_created(self, keys):
        with self.lock:
            for key in reversed(keys):
                self.order[key] = None
                self.order.move_to_end(key)

    on_touched = on_created

    def on_removed(self, keys):
        with self.lock:
            for key in keys:
                self.order.pop(key, None)

    def get_victims(self, ratio, eligible):
        with self.lock:
            target = max(1, int(len(self.order) * ratio))
            victims = []
            for key in self.order:
                if len(victims) >= target:
                    break
                if eligible(key):
                    victims.append(key)
            return victims


class EvictionController:
    def __init__(self, l1, policy):
        self.l1 = l1
        self.policy = policy
        l1.register_listener(policy)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            time.sleep(INTERVAL)
            used, total = self.l1.usage()
            if used / total < WATERMARK:
                continue
            victims = self.policy.get_victims(EVICTION_RATIO, self.l1.is_evictable)
            self.l1.delete(victims)

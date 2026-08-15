# SPDX-License-Identifier: Apache-2.0
from lmcache_mini.l1.eviction import EvictionController, LRUPolicy
from lmcache_mini.l1.manager import L1Manager
from lmcache_mini.l1.memory import PoolAllocator
from lmcache_mini.l1.prefetch_controller import PrefetchController
from lmcache_mini.l1.store_controller import StoreController


class StorageManager:
    def __init__(self, capacity_bytes, l2s=()):
        self.l1 = L1Manager(PoolAllocator(capacity_bytes))
        self.l2s = l2s
        self.store = StoreController(self.l1, l2s) if l2s else None
        self.prefetch = PrefetchController(self.l1, l2s)
        self.eviction = EvictionController(self.l1, LRUPolicy())

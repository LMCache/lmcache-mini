# SPDX-License-Identifier: Apache-2.0
import threading
from dataclasses import dataclass

from lmcache_mini.l1.memory import MemoryObj
from lmcache_mini.protocol import ChunkKey


class Listener:
    def on_created(self, keys):
        pass

    def on_write_finished(self, keys):
        pass

    def on_touched(self, keys):
        pass

    def on_removed(self, keys):
        pass


@dataclass
class Entry:
    obj: MemoryObj
    write_locked: bool = False
    read_count: int = 0
    is_temporary: bool = False

    def readable(self):
        return not self.write_locked

    def unlocked(self):
        return not self.write_locked and self.read_count == 0


class L1Manager:
    def __init__(self, allocator):
        self.allocator = allocator
        self.lock = threading.Lock()
        self.entries: dict[ChunkKey, Entry] = {}
        self.listeners = []

    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self, event, keys):
        for listener in self.listeners:
            getattr(listener, event)(keys)

    def reserve_write(self, keys, chunk_nbytes, is_temporary=False):
        with self.lock:
            result = dict.fromkeys(keys)
            missing = [key for key in keys if key not in self.entries]
            objs = self.allocator.allocate(len(missing), chunk_nbytes)
            if objs is None:
                return result
            for key, obj in zip(missing, objs):
                self.entries[key] = Entry(obj, write_locked=True,
                                          is_temporary=is_temporary)
                result[key] = obj
            return result

    def finish_write(self, keys):
        with self.lock:
            for key in keys:
                self.entries[key].write_locked = False
            self.notify("on_created", keys)
            self.notify("on_write_finished", keys)

    def finish_write_and_reserve_read(self, keys):
        with self.lock:
            for key in keys:
                entry = self.entries[key]
                entry.write_locked = False
                entry.read_count += 1
            self.notify("on_created", keys)

    def reserve_read(self, keys):
        with self.lock:
            result = {}
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or not entry.readable():
                    result[key] = None
                else:
                    entry.read_count += 1
                    result[key] = entry.obj
            return result

    def reserve_read_prefix(self, keys):
        with self.lock:
            hit = 0
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or not entry.readable():
                    break
                hit += 1
            for key in keys[:hit]:
                self.entries[key].read_count += 1
            return hit

    def read(self, keys):
        with self.lock:
            return [self.entries[key].obj for key in keys]

    def finish_read(self, keys):
        freed, touched = [], []
        with self.lock:
            for key in keys:
                entry = self.entries[key]
                entry.read_count -= 1
                if entry.read_count == 0 and entry.is_temporary:
                    self.allocator.free([entry.obj])
                    del self.entries[key]
                    freed.append(key)
                else:
                    touched.append(key)
            self.notify("on_touched", touched)
            self.notify("on_removed", freed)

    def is_evictable(self, key):
        entry = self.entries.get(key)
        return entry is not None and entry.unlocked()

    def delete(self, keys, force=False):
        with self.lock:
            deleted = []
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or (not entry.unlocked() and not force):
                    continue
                self.allocator.free([entry.obj])
                del self.entries[key]
                deleted.append(key)
            self.notify("on_removed", deleted)
            return len(deleted)

    def usage(self):
        with self.lock:
            return self.allocator.usage()

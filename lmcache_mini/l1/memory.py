# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch


@dataclass
class MemoryObj:
    tensor: torch.Tensor
    offset: int

    @property
    def byte_array(self):
        return memoryview(self.tensor.numpy())

    def nbytes(self):
        return self.tensor.numel()


class PoolAllocator:
    def __init__(self, capacity_bytes):
        self.pool = torch.empty(capacity_bytes, dtype=torch.uint8,
                                pin_memory=True)
        self.brk = 0
        self.free_segments: dict[int, list[int]] = {}
        self.used_bytes = 0

    def allocate(self, count, nbytes):
        free = self.free_segments.setdefault(nbytes, [])
        offsets = []
        while len(offsets) < count and free:
            offsets.append(free.pop())
        while len(offsets) < count and self.brk + nbytes <= self.pool.numel():
            offsets.append(self.brk)
            self.brk += nbytes
        if len(offsets) < count:
            free.extend(offsets)
            return None
        self.used_bytes += count * nbytes
        return [MemoryObj(self.pool[o:o + nbytes], o) for o in offsets]

    def free(self, objs):
        for obj in objs:
            self.used_bytes -= obj.nbytes()
            self.free_segments[obj.nbytes()].append(obj.offset)

    def usage(self):
        return self.used_bytes, self.pool.numel()

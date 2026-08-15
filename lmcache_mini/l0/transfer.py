# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import Future

import torch

from lmcache_mini.l1.memory import MemoryObj


def export_kv_caches(tensors):
    from torch.multiprocessing.reductions import reduce_tensor

    return [reduce_tensor(t) for t in tensors]

def import_kv_caches(ipc_handles):
    return [rebuild(*args) for rebuild, args in ipc_handles]


def export_event(event):
    return bytes(event.ipc_handle())

def import_event(device_index, handle):
    return torch.cuda.Event.from_ipc_handle(device_index, handle)


class DeviceFuture:
    def __init__(self, future: Future, device_index: int):
        self.future = future
        self.device_index = device_index
        self.completion = None

    def done(self):
        if not self.future.done():
            return False
        if self.completion is None:
            self.completion = import_event(self.device_index,
                                           self.future.result())
        return self.completion.query()


class Pipeline:
    def __init__(self, kv_caches, blocks_per_chunk, block_nbytes,
                 chunk_nbytes):
        device = kv_caches[0].device
        self.lock = threading.Lock()
        self.kernel_stream = torch.cuda.Stream(device)
        self.copy_stream = torch.cuda.Stream(device)
        self.staging = torch.empty(3, chunk_nbytes, dtype=torch.uint8,
                                   device=device)
        self.ready = [torch.cuda.Event() for _ in range(3)]
        self.free = [torch.cuda.Event() for _ in range(3)]
        self.layer_views = []
        for buf in self.staging:
            chunk = buf.view(len(kv_caches), blocks_per_chunk, block_nbytes)
            views = []
            for layer, kv in enumerate(kv_caches):
                per_block_shape = kv.shape[1:]
                views.append(chunk[layer].view(kv.dtype)
                             .view(blocks_per_chunk, *per_block_shape))
            self.layer_views.append(views)


class KVTransfer:
    def __init__(self, ipc_handles, block_size, chunk_size, rank):
        self.rank = rank
        self.kv_caches = import_kv_caches(ipc_handles)
        self.device = self.kv_caches[0].device
        self.num_layers = len(self.kv_caches)
        self.blocks_per_chunk = chunk_size // block_size
        self.block_nbytes = self.kv_caches[0][0].nbytes
        self.chunk_nbytes = (self.num_layers * self.blocks_per_chunk
                             * self.block_nbytes)
        self.d2h = Pipeline(self.kv_caches, self.blocks_per_chunk,
                            self.block_nbytes, self.chunk_nbytes)
        self.h2d = Pipeline(self.kv_caches, self.blocks_per_chunk,
                            self.block_nbytes, self.chunk_nbytes)

    def store(self, block_ids, objs: list[MemoryObj], producer_event_handle):
        block_ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        pipe = self.d2h
        with pipe.lock, torch.cuda.device(self.device):
            producer = import_event(self.device.index, producer_event_handle)
            pipe.kernel_stream.wait_event(producer)
            begin = torch.cuda.Event(enable_timing=True)
            begin.record(pipe.kernel_stream)
            for i, obj in enumerate(objs):
                buf = i % 3
                block_ids_in_chunk = block_ids[
                    i * self.blocks_per_chunk:(i + 1) * self.blocks_per_chunk]
                with torch.cuda.stream(pipe.kernel_stream):
                    pipe.kernel_stream.wait_event(pipe.free[buf])
                    for layer, kv_cache in enumerate(self.kv_caches):
                        torch.index_select(kv_cache, 0, block_ids_in_chunk,
                                           out=pipe.layer_views[buf][layer])
                    pipe.ready[buf].record(pipe.kernel_stream)
                with torch.cuda.stream(pipe.copy_stream):
                    pipe.copy_stream.wait_event(pipe.ready[buf])
                    obj.tensor.copy_(pipe.staging[buf], non_blocking=True)
                    pipe.free[buf].record(pipe.copy_stream)
            end = torch.cuda.Event(enable_timing=True)
            end.record(pipe.copy_stream)
            completion = torch.cuda.Event(interprocess=True)
            completion.record(pipe.copy_stream)
        completion.synchronize()
        return completion, begin.elapsed_time(end) / 1e3

    def load(self, block_ids, objs: list[MemoryObj], producer_event_handle,
             skip_blocks=0):
        block_ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        pipe = self.h2d
        with pipe.lock, torch.cuda.device(self.device):
            producer = import_event(self.device.index, producer_event_handle)
            pipe.copy_stream.wait_event(producer)
            begin = torch.cuda.Event(enable_timing=True)
            begin.record(pipe.copy_stream)
            for i, obj in enumerate(objs):
                buf = i % 3
                skip = skip_blocks if i == 0 else 0
                selected = self.blocks_per_chunk - skip
                if selected <= 0:
                    continue
                block_ids_in_chunk = block_ids[
                    i * self.blocks_per_chunk + skip:
                    (i + 1) * self.blocks_per_chunk]
                with torch.cuda.stream(pipe.copy_stream):
                    pipe.copy_stream.wait_event(pipe.free[buf])
                    pipe.staging[buf].copy_(obj.tensor, non_blocking=True)
                    pipe.ready[buf].record(pipe.copy_stream)
                with torch.cuda.stream(pipe.kernel_stream):
                    pipe.kernel_stream.wait_event(pipe.ready[buf])
                    for layer, kv_cache in enumerate(self.kv_caches):
                        kv_cache.index_copy_(0, block_ids_in_chunk,
                                             pipe.layer_views[buf][layer][skip:])
                    pipe.free[buf].record(pipe.kernel_stream)
            end = torch.cuda.Event(enable_timing=True)
            end.record(pipe.kernel_stream)
            completion = torch.cuda.Event(interprocess=True)
            completion.record(pipe.kernel_stream)
        completion.synchronize()
        return completion, begin.elapsed_time(end) / 1e3

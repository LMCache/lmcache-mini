# SPDX-License-Identifier: Apache-2.0
import uuid
from dataclasses import dataclass, field

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

from lmcache_mini.l0 import kv_format
from lmcache_mini.l0.transfer import (
    DeviceFuture,
    export_event,
    export_kv_caches,
)
from lmcache_mini.mq import MQClient
from lmcache_mini.protocol import (
    FreeLocksPayload,
    LoadStoreOp,
    LookupPayload,
    QueryPayload,
    RegisterPayload,
    Req,
    ReqMeta,
    TransferPayload,
)


@dataclass
class RequestTracker:
    token_ids: list[int]
    lmcache_hits: int = 0
    vllm_hits: int = 0
    block_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    num_stored_tokens: int = 0
    lookup_resolved: bool = False
    load_pending: bool = False
    alloc_seen: bool = False


@dataclass
class MiniConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]


class MiniConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.vllm_config = vllm_config
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.client = MQClient("tcp://{}:{}".format(
            extra.get("mini.host", "localhost"), extra.get("mini.port", 5555)))
        self.chunk_size = self.client.call(Req.GET_CHUNK_SIZE)
        self.block_size = vllm_config.cache_config.block_size
        assert self.chunk_size % self.block_size == 0
        self.model = vllm_config.model_config.model
        self.world_size = vllm_config.parallel_config.tensor_parallel_size
        self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.trackers: dict[str, RequestTracker] = {}
        self.store_futures: dict[str, list[DeviceFuture]] = {}
        self.load_futures: dict[str, DeviceFuture] = {}
        self.pending_sends: set[str] = set()

    def generate_retrieve_op(self, tracker) -> LoadStoreOp | None:
        if not tracker.load_pending:
            return None
        tracker.load_pending = False
        start = tracker.vllm_hits // self.chunk_size * self.chunk_size
        end = tracker.lmcache_hits
        return LoadStoreOp(
            tracker.token_ids[:end],
            tracker.block_ids[start // self.block_size:end // self.block_size],
            start, end, tracker.vllm_hits - start)

    def generate_store_op(self, tracker) -> LoadStoreOp | None:
        available = min(len(tracker.token_ids),
                        len(tracker.block_ids) * self.block_size,
                        tracker.num_computed_tokens)
        num_chunks = (available - tracker.num_stored_tokens) // self.chunk_size
        if num_chunks <= 0:
            return None
        start = tracker.num_stored_tokens
        end = start + num_chunks * self.chunk_size
        tracker.num_stored_tokens = end
        return LoadStoreOp(
            tracker.token_ids[:end],
            tracker.block_ids[start // self.block_size:end // self.block_size],
            start, end)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        tracker = self.trackers.get(request.request_id)
        if tracker is None:
            tracker = RequestTracker(request.all_token_ids)
            self.trackers[request.request_id] = tracker
            prompt = request.prompt_token_ids
            num_lookup = (len(prompt) - 1) // self.chunk_size * self.chunk_size
            self.client.submit(Req.LOOKUP, LookupPayload(
                request.request_id, prompt[:num_lookup],
                self.model, self.world_size))
        elif tracker.lookup_resolved:
            return 0, False
        hits = self.client.call(Req.QUERY_PREFETCH_STATUS, QueryPayload(
            request.request_id, self.world_size))
        if hits is None:
            return None, True
        tracker.lookup_resolved = True
        tracker.lmcache_hits = hits
        tracker.vllm_hits = num_computed_tokens
        tracker.num_stored_tokens = hits
        if hits > num_computed_tokens:
            return hits - num_computed_tokens, True
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if not (tracker := self.trackers.get(request.request_id)):
            return
        group_blocks = blocks.get_block_ids()[0]
        tracker.block_ids.extend(group_blocks[len(tracker.block_ids):])
        if num_external_tokens > 0:
            tracker.load_pending = True
        if tracker.alloc_seen:
            return
        tracker.alloc_seen = True
        lmcache_chunks = tracker.lmcache_hits // self.chunk_size
        free_until = min(tracker.vllm_hits // self.chunk_size, lmcache_chunks) \
            if tracker.load_pending else lmcache_chunks
        if free_until:
            self.client.submit(Req.FREE_LOOKUP_LOCKS, FreeLocksPayload(
                request.request_id, free_until, self.world_size))

    def build_connector_meta(self, scheduler_output):
        metas = []
        for req_id, tracker in self.trackers.items():
            op = self.generate_retrieve_op(tracker)
            if op is not None:
                metas.append(ReqMeta(req_id, Req.RETRIEVE, op))
        for new_req in scheduler_output.scheduled_new_reqs:
            tracker = self.trackers.get(new_req.req_id)
            if tracker is not None:
                tracker.block_ids = list(new_req.block_ids[0])
                tracker.num_computed_tokens = new_req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            tracker = self.trackers.get(req_id)
            if tracker is None:
                continue
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                if req_id in cached.resumed_req_ids:
                    tracker.block_ids = list(new_blocks[0])
                else:
                    tracker.block_ids.extend(new_blocks[0])
            tracker.num_computed_tokens = cached.num_computed_tokens[i]
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            tracker = self.trackers.get(req_id)
            if tracker is None:
                continue
            tracker.num_computed_tokens += scheduled
            op = self.generate_store_op(tracker)
            if op is not None:
                metas.append(ReqMeta(req_id, Req.STORE, op))
        return MiniConnectorMetadata(metas)

    def request_finished(self, request, block_ids):
        self.trackers.pop(request.request_id, None)
        self.client.submit(Req.END_SESSION, request.request_id)
        return True, None

    def register_kv_caches(self, kv_caches):
        layers = [t for _, t in sorted(kv_caches.items())]
        views = kv_format.normalize(
            layers, self.vllm_config.cache_config.num_gpu_blocks)
        self.client.call(Req.REGISTER_KV_CACHE, RegisterPayload(
            instance_id=self.instance_id,
            model=self.model,
            rank=get_tensor_model_parallel_rank(),
            world_size=self.world_size,
            block_size=self.block_size,
            ipc_handles=export_kv_caches(views),
        ))

    def submit_transfers(self, req):
        ops = [m for m in self._get_connector_metadata().requests
               if m.req == req]
        if not ops:
            return {}
        producer = torch.cuda.Event(interprocess=True)
        producer.record()
        handle = export_event(producer)
        device = torch.cuda.current_device()
        return {
            m.request_id: DeviceFuture(self.client.submit(req, TransferPayload(
                m.request_id, self.instance_id, m.op, handle)), device)
            for m in ops
        }

    def start_load_kv(self, forward_context, **kwargs):
        self.load_futures.update(self.submit_transfers(Req.RETRIEVE))

    def wait_for_save(self):
        for req_id, future in self.submit_transfers(Req.STORE).items():
            self.store_futures.setdefault(req_id, []).append(future)

    def get_finished(self, finished_req_ids):
        self.pending_sends |= finished_req_ids
        for req_id, futures in list(self.store_futures.items()):
            live = [f for f in futures if not f.done()]
            if live:
                self.store_futures[req_id] = live
            else:
                del self.store_futures[req_id]
        finished_sending = {req_id for req_id in self.pending_sends
                            if req_id not in self.store_futures}
        self.pending_sends -= finished_sending
        finished_recving = {req_id for req_id, future
                            in self.load_futures.items() if future.done()}
        for req_id in finished_recving:
            del self.load_futures[req_id]
        return finished_sending, finished_recving

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def shutdown(self):
        self.client.call(Req.UNREGISTER_KV_CACHE, self.instance_id)
        self.client.close()

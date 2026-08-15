# SPDX-License-Identifier: Apache-2.0
import enum
from dataclasses import dataclass, field


class Req(enum.IntEnum):
    GET_CHUNK_SIZE = 1
    REGISTER_KV_CACHE = 2
    UNREGISTER_KV_CACHE = 3
    LOOKUP = 4
    QUERY_PREFETCH_STATUS = 5
    RETRIEVE = 6
    STORE = 7
    FREE_LOOKUP_LOCKS = 8
    END_SESSION = 9


@dataclass(frozen=True)
class ChunkKey:
    chunk_hash: bytes
    model: str
    rank: int


@dataclass
class LoadStoreOp:
    token_ids: list[int]
    block_ids: list[int]
    start: int
    end: int
    skip_first_n_tokens: int = 0


@dataclass
class ReqMeta:
    request_id: str
    req: Req
    op: LoadStoreOp


@dataclass
class RegisterPayload:
    instance_id: int
    model: str
    rank: int
    world_size: int
    block_size: int
    ipc_handles: list = field(default_factory=list)


@dataclass
class LookupPayload:
    request_id: str
    token_ids: list[int]
    model: str
    world_size: int


@dataclass
class TransferPayload:
    request_id: str
    instance_id: int
    op: LoadStoreOp
    event_handle: bytes


@dataclass
class QueryPayload:
    request_id: str
    world_size: int


@dataclass
class FreeLocksPayload:
    request_id: str
    num_chunks: int
    world_size: int

# SPDX-License-Identifier: Apache-2.0
import struct

import blake3


def hash_one(prefix_hash, tokens):
    h = blake3.blake3(prefix_hash)
    h.update(struct.pack(f">{len(tokens)}I", *tokens))
    return h.digest()


NONE_HASH = hash_one((0).to_bytes(8, byteorder="big", signed=True), (0,))


def chunk_hashes(token_ids, chunk_size):
    hashes = []
    prefix_hash = NONE_HASH
    for i in range(0, len(token_ids) - len(token_ids) % chunk_size, chunk_size):
        prefix_hash = hash_one(prefix_hash, token_ids[i : i + chunk_size])
        hashes.append(prefix_hash)
    return hashes

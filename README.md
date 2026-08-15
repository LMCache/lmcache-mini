# lmcache-mini

main project: [LMCache](https://github.com/LMCache/LMCache)

# quickstart

```bash
python -m lmcache_mini.server --port 45881 --l1-size-gb 8 \
    --l2-adapter '{"type": "fs", "base_path": "/tmp/mini-l2"}'
```

```bash
vllm serve Qwen/Qwen3-0.6B --enforce-eager \
    --max-model-len 4096 --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "MiniConnector",
                           "kv_connector_module_path": "lmcache_mini.integration.vllm_connector",
                           "kv_role": "kv_both",
                           "kv_connector_extra_config": {"mini.port": 45881}}'
```

```bash
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "{
    \"model\": \"Qwen/Qwen3-0.6B\", \"temperature\": 0, \"max_tokens\": 24,
    \"prompt\": \"$(python3 -c "print('A field guide to the birds of North America. ' * 80)")\"}"
```

cache server output:

```
mini cache server up and ready on tcp://127.0.0.1:45881 (chunk_size=256, 8 GB L1, 1 L2 adapters)
REGISTER Qwen/Qwen3-0.6B rank 0/1 (chunk=28 MiB, 1 engines)
STORE rid=cmpl-...-0 tokens [0, 768) L0->L1 38.1 GB/s
RETRIEVE rid=cmpl-...-0 tokens [0, 768) hit L1=3 L2=0 | L2->L1 0.0 GB/s | L1->L0 35.7 GB/s
```

The first STORE and first RETRIEVE need a warmup (30-40x slower).

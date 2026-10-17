# Phase 5 targeted follow-up: RAM counters and ubatch-8 wave isolation

The Oct 17 runtime results in `PHASE5-CORRECTED-BUILD-RESULTS-TO-CHATGPT.zip` established successful bit-for-bit *generated text* comparisons for `-ub 1` with 0, 1024, and 2048 MiB RAM cache. Normal inference and direct Windows GGUF streaming work with `--moe-stream-cache 8s`.

The `-ub 8 --moe-stream-cache 24s` run diverged from its matching normal `-ub 8` control **after 379 characters**, while still producing coherent text and exiting normally. The normal `-ub 1` and `-ub 8` outputs *also differ*, starting at character 330. A mismatch under greedy decoding is not sufficient evidence of weight corruption: changed floating-point reduction order and near-equal argmax logits can cause differences. But neither is it proof of numerical harmlessness. The multi-wave path remains unvalidated.

## Changes in next build

- At destruction of a streaming model, `llama_moe_stream::~llama_moe_stream` prints cumulative streaming counters even if the caller does not invoke `llama_perf_context_print`. Logs include `MOE_STREAM_CACHE_COUNTERS budget_bytes=... resident_bytes=... hits=... misses=... evictions=... gguf_payload_read_bytes=...`.
- Streaming source and router are unchanged by this telemetry patch.
- Do **not** conflate `gguf_payload_read_bytes` with physical NVMe device-read bytes: alignment padding and the controller cache are excluded.

## Controlled next validation: same GGUF and same build/DLLs

**First:** normal `-ub 1`, streamed `-ub 1 --moe-stream-cache 8s` at 0, 1024 and 2048 MiB RAM cache. Confirm output equality and collect accurate RAM cache counters (now printed at process/model teardown). The model must exit gracefully to print final counters.

**Then isolate ubatch-8:**

| Name | Flags / environment | Purpose |
|---|---|---|
| normal-ub8-A and B | `-ub 8`, no streaming | Reproducibility of normal greedy output |
| wave-ub8-preload | `-ub 8 --moe-stream --moe-stream-cache 24s --moe-stream-ram-cache-mib 0`, default environment | Multi-wave behavior with next-wave preload |
| wave-ub8-no-preload | same flags, but process environment `LLAMA_MOE_STREAM_NO_PRELOAD=1` | Determine whether wave preloading contributes to divergence |
| nowave-ub8 | `-ub 8 --moe-stream --moe-stream-cache 64s --moe-stream-ram-cache-mib 0` | Suppress waves at 8 tokens × 8 experts/token to isolate the multi-wave path |

Use identical `-m`, `-ngl 99 -c 2048 -b 128 -t 6 -n 96 --temp 0 --seed 42` and prompt. Verify the actual `n_ubatch`, slot count, wave counters, generated token count, and GGUF read counters in each log. If the model has special prompt or batch scheduling affecting effective tokens per microbatch, use the debug information and report that limitation.

The 64-slot configuration is **a targeted diagnostic only**, not a proposed VRAM budget; it requires roughly 3.52 GiB of expert tensors for this model and may increase total VRAM. Do not run cache sweeps.

1. If normal-ub8-A and B differ, report normal-path non-determinism before drawing conclusions from exact text equality.
2. If 64s matches normal, while 24s does not, focus on wave masking, slot parking, ordering and accumulation.
3. If 24s no-preload matches normal while preload differs, focus on copy/compute synchronization, slot lifetime and I/O worker prefetch.
4. If neither matches, compare selected expert IDs and per-token/logit differences if a reliable non-invasive supported dump can be enabled. Do not assume all differences imply corrupted weights.
5. Run longer numerical probes only after a basic working result. Do not change router, weights, GGUF, system settings or GPU cache implementation.

Because the current Phase 5 `-ub 1` speed was 1.97–4.31 tok/s versus ~124 tok/s normal GPU decode, correctness and I/O accounting come **before** performance optimization. Avoid rerunning unrelated cache-size sweeps.

The next architecture research question is whether we can safely reuse device expert scratch across layers with the lowest practical memory footprint. Do not attempt this until the current multi-wave behavior has been isolated and corrected or qualified.

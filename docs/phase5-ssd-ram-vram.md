# Phase 5 — Experimental hierarchical MoE expert streaming (Windows RTX 3060)

**Branch:** `phase5-ssd-ram-vram`. **Upstream foundation:** [ggml-org/llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294). Development is kept separate from `master`.

## Design

1. GGUF file remains on NVMe. Expert tensors are **not fully materialized in system RAM**.
2. A global, byte-capped **RAM expert LRU** stores complete (layer, expert) gate/up/down weight groups on demand, regardless of the current GPU cache state.
3. The upstream per-layer **device expert cache** stores frequently selected experts in VRAM, with a separate configurable GPU memory budget.
4. On a GPU expert cache miss, the worker consults RAM. RAM hit: copy from RAM to GPU without SSD read. RAM miss: read original GGUF weights from storage, install a complete entry in the RAM LRU (if it fits), and copy to GPU.
5. The router, expert IDs and expert-weight values are not modified or approximated. The upstream implementation's wave-prefill and CPU-side ID remapping remain unchanged.

The GPU cache and RAM cache have **independent** capacity knobs. RAM-cache payload accounting excludes bounded per-worker staging and metadata. Additional shared tensors, KV context, compute buffers, OS memory, and CUDA allocations are separate from these two budgets. Each worker can hold a temporary expert-sized buffer during a miss or hit.

### CLI knobs

- `--moe-stream`: enable SSD expert streaming.
- `--moe-stream-cache 512m`: combined model-wide GPU expert-cache budget of 512 MiB, resolved to per-layer slots (rounding means actual allocation may differ).
- `--moe-stream-ram-cache-mib 1024`: RAM expert LRU payload budget of 1024 MiB (0 disables the RAM cache).
- `--moe-stream-io-threads 8`: number of concurrent I/O workers (default 9, max 18).
- `--moe-stream-direct`: request Windows unbuffered overlapped positional I/O; falls back to buffered overlapped reads if unbuffered open/read fails.

The RAM option may also be supplied as environment variable `LLAMA_ARG_MOE_STREAM_RAM_CACHE_MIB`.

**Do not combine `--cpu-moe` or the old `--moe-cache-mib` path with `--moe-stream` during initial validation.** Those options belong to a different caching implementation.

The upstream PR uses per-layer GPU slot caches. In this phase, GPU caching is **not the older shared cross-layer LRU from Phase 2**. Both modes remain in the source tree, but the stream path uses the upstream streaming cache.

## Windows disk I/O adaptation

The upstream Windows code serialized file.seek/file.read behind a process-wide mutex. This port instead opens read-only Windows `CreateFileW` handles with `FILE_FLAG_OVERLAPPED`, adds `FILE_FLAG_NO_BUFFERING` when requested, and uses a per-operation `OVERLAPPED` with byte offsets and worker-local aligned staging buffers.

When direct I/O is requested, reads align file offsets and read sizes to 4096 bytes and return a pointer into the aligned result. No modifications or writes to the GGUF occur. A probe read tests direct-I/O support. Device/driver caches can still affect measured SSD bandwidth.

## Initial validation on the user's PC

First preserve all prior binaries and model files. Extract the experimental Windows CUDA artifact to a new folder.

Choose a deterministic short prompt and generate 64–128 tokens. Check the output and exit status using `--moe-stream-cache 512m`, `--moe-stream-ram-cache-mib 0`, `--moe-stream-direct` and `-ub 1`. Read logs to confirm *streamed* layers are detected and that read-from-disk counters increase.

Then run the same prompt at exactly the same settings with `--moe-stream-ram-cache-mib 1024`. Confirm RAM cache hits rise on repeated expert usage, the resident payload size never exceeds the configured cap, SSD read bytes fall or stay equal, and output remains identical.

Then increase `-ub` to 8 to validate original PR prefill-wave behavior, and compare with a normal full-GPU run with identical sampling and token count.

Record separately: decode tokens/s, prefill tokens/s, peak process resident/private RAM, peak process VRAM, RAM cache hits/misses/evictions/bytes, total GGUF read bytes, streamed GPU cache misses, and exception/error logs. Do not conflate page cache or total RAM with the explicit RAM expert LRU payload.

**This is experimental and has not been validated on the user's GGUF or NVIDIA hardware until those runtime tests complete.** Upstream's single-context restriction remains: do not use concurrent distinct llama_context instances sharing the same streamed model.

## Performance / safety cautions

- The RAM LRU currently holds complete expert slabs and copies an entry into a worker-local buffer on a hit. This prioritizes correctness and eviction safety; it is not zero-copy or maximum-performance.
- A single GPU cache miss can involve several GGUF weight slabs. On an SSD cache miss, the worker performs the needed positional reads and synchronous GPU buffer updates.
- The read count represents application-level GGUF bytes requested. Physical disk I/O may differ due to filesystem alignment or controller caching.
- The I/O workers are asynchronous with respect to the remap queue, but the target remap still waits for required experts. **No speculative expert prediction is implemented.**
- Validate model outputs before performance measurements. No streaming throughput guarantee is implied by standalone Phase 4 hardware benchmarks.
- A 512 MiB GPU cache setting may yield an actual size slightly below the nominal budget because slots are resolved discretely for every layer.

## What to report

Generate a separate Phase 5 ZIP with raw logs, deterministic output comparisons, captured RAM/VRAM data, cache metadata, actual read counters and time spent waiting for loads. Compare RAM cache disabled and enabled at a constant GPU expert cache size. Highlight correctness errors immediately rather than proceeding to performance tuning.

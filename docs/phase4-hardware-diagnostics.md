# Phase 4 — Native bottleneck measurements (Windows / CUDA)

This phase measures CPU waiting and copy submission in the real llama.cpp expert-cache path, and measures Windows NVMe / RAM-to-GPU throughput independently. **No prediction is implemented.** No claim of SSD streaming inside llama.cpp.

## What the new GitHub Actions artifact contains

- llama-cli.exe, llama-server.exe, and required runtime DLLs.
- moe-phase4-probe.exe: standalone read-only Windows unbuffered-I/O and CUDA transfer benchmark.

Use binaries from the **new successful** Windows CUDA RTX 3060 workflow run; older ZIPs do not contain the probe.

Download the artifact in your browser and leave it in Downloads for the local agent. There is no need to install GitHub CLI, CUDA Toolkit, MSVC or a local compiler. Extract into a new versioned binary folder; preserve all earlier benchmarks and model files.

## Experiment A: llama.cpp, one fixed GPU expert cache

Disable MOE_TRACE_PATH (old JSONL). Use one fixed cache budget, for example 2048 MiB. Use the same command and prompt in three runs.

**A1: baseline**: unset both MOE_PHASE4_PROFILE_PATH and MOE_PHASE4_FORCE_SYNC. Capture stdout/stderr and decode tokens/s.

**A2: passive**: set MOE_PHASE4_PROFILE_PATH to a unique writable absolute path ending in .csv. Do not set MOE_PHASE4_FORCE_SYNC. Per cache callback the CSV records layer, number of selected IDs (flattened), hits, misses, bytes submitted, CPU readback+sync wall time, CPU LRU planning time, and CPU submission time. File I/O is buffered.

**A3: intrusive**: set MOE_PHASE4_PROFILE_PATH to another CSV file and MOE_PHASE4_FORCE_SYNC=1. This deliberately forces a backend synchronization after upload submissions and records drain_wait_us. It **alters inference scheduling and performance**; never treat its throughput as normal. drain_wait_us includes any outstanding queued GPU work, not purely expert copies.

Example PowerShell commands (change the model path, run from the new binary folder):

~~~powershell
$model = "D:\Models\Ling-3.0-tiny-Q8_0.gguf"
Remove-Item Env:\MOE_TRACE_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\MOE_PHASE4_PROFILE_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\MOE_PHASE4_FORCE_SYNC -ErrorAction SilentlyContinue
.\llama-cli.exe -m $model -ngl 99 -cmoe --moe-cache-mib 2048 --load-mode none -c 2048 -b 128 -ub 8 -n 256 -t 6 --temp 0 --seed 42 -p "Explain the role of an electrical transformer."

$env:MOE_PHASE4_PROFILE_PATH = "$PWD\phase4-passive.csv"
.\llama-cli.exe -m $model -ngl 99 -cmoe --moe-cache-mib 2048 --load-mode none -c 2048 -b 128 -ub 8 -n 256 -t 6 --temp 0 --seed 42 -p "Explain the role of an electrical transformer."
Remove-Item Env:\MOE_PHASE4_PROFILE_PATH

$env:MOE_PHASE4_PROFILE_PATH = "$PWD\phase4-forced-sync.csv"
$env:MOE_PHASE4_FORCE_SYNC = "1"
.\llama-cli.exe -m $model -ngl 99 -cmoe --moe-cache-mib 2048 --load-mode none -c 2048 -b 128 -ub 8 -n 256 -t 6 --temp 0 --seed 42 -p "Explain the role of an electrical transformer."
Remove-Item Env:\MOE_PHASE4_PROFILE_PATH
Remove-Item Env:\MOE_PHASE4_FORCE_SYNC
~~~

Existing Phase 2A JSONL traces are independent. The new CSV avoids heavyweight expert-ID JSON and its metadata is deliberately minimal.

Meaning of the new measurements:
- readback_sync_us: read selected IDs **and synchronize backend**. Time may include necessary preceding GPU computation.
- plan_us: CPU cache LRU plan time.
- submit_us: CPU submission time for async expert and slot-map copies; **not actual GPU transfer time**.
- drain_wait_us: -1 normally; if forced sync, CPU backend-wait time for ALL outstanding GPU operations. **Not a pure DMA duration**.

## Experiment B: read-only storage and transfer microbenchmark

The probe opens the existing Ling Tiny GGUF read-only. It runs:
- Windows overlapped no-buffering sequential/random file reads using 256 KiB, 1 MiB and 4 MiB blocks, and queue depths of 1, 4 and 16.
- Pageable versus CUDA pinned-host RAM-to-GPU copies using CUDA events and CPU wall clock.
- A deliberately serialized direct-file-I/O to CPU RAM to pinned RAM to GPU path (no overlap).

~~~powershell
.\moe-phase4-probe.exe "D:\Models\Ling-3.0-tiny-Q8_0.gguf" 1> phase4-probe.csv 2> phase4-probe-stderr.txt
~~~

The probe **never writes to the model** and does not create a synthetic test file. Its unbuffered reads bypass the regular Windows filesystem cache but do not rule out SSD/controller caches. CUDA event times are from its independent GPU stream and do **not** time llama.cpp expert-weight copies.

The CSV contains different row categories (disk, h2d, pipeline). Interpret their last columns according to the row type. Do not compare a GPU event duration to a disk duration without labeling units and source.

If Windows rejects no-buffering operations, preserve the exact error. Do not alter drive security settings or install large toolchains to work around it without approval.

## Limitations and outputs

This phase is designed to test whether storage bandwidth, H2D bandwidth, or host-side scheduling is an obvious priority. It cannot accurately measure GPU-kernel idle periods, true GPU completion of llama.cpp's copies, or compute-copy overlap. That would require CUDA backend instrumentation tied to the actual llama.cpp streams, or a GPU execution trace.

Save baseline/passive/forced-sync logs, CSVs, PCIe link information, driver version, and a report with clear separation between measured values, hypothetical ceilings, and unknowns. Preserve every Phase 1–3 result. Do not upload user traces or local file paths to the public repository.

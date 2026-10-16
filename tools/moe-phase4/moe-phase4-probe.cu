// Phase 4 hardware probe: read-only Windows unbuffered file I/O and CUDA H2D timings.
// Standalone diagnostic, not part of llama.cpp inference or its CUDA streams.
// Usage: moe-phase4-probe.exe "D:\Models\Ling-3.0-tiny-Q8_0.gguf"
// Writes CSV to stdout. No writes to the model file or disk.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double millis(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void fail(const char * message) {
    std::cerr << "ERROR: " << message << " (win32=" << GetLastError() << ")\n";
    std::exit(2);
}

static void cuda_check(cudaError_t err, const char * action) {
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << action << ": " << cudaGetErrorString(err) << "\n";
        std::exit(3);
    }
}

struct slot {
    OVERLAPPED ov{};
    HANDLE event = nullptr;
    void * buffer = nullptr;
};

static HANDLE open_direct(const wchar_t * path, uint64_t & file_size) {
    HANDLE f = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (f == INVALID_HANDLE_VALUE) fail("CreateFileW direct/unbuffered");
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) fail("GetFileSizeEx");
    file_size = static_cast<uint64_t>(sz.QuadPart);
    return f;
}

static double mbps(uint64_t n, double elapsed_ms) {
    return elapsed_ms > 0.0 ? (double(n) / 1048576.0) / (elapsed_ms / 1000.0) : 0.0;
}

static void disk_test(const wchar_t * path, size_t block, unsigned qd, bool random_io, unsigned operations) {
    uint64_t file_size = 0;
    HANDLE f = open_direct(path, file_size);
    const uint64_t n_blocks = file_size / block;
    if (n_blocks < 2) fail("File is too small for the unbuffered test");

    std::vector<slot> slots(qd);
    for (auto & s : slots) {
        // VirtualAlloc gives page-aligned buffers; 4 KiB block alignment is intentional.
        s.buffer = VirtualAlloc(nullptr, block, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        s.event  = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!s.buffer || !s.event) fail("VirtualAlloc/CreateEvent");
    }

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, n_blocks - 1);
    uint64_t actual = 0;
    const auto start = clock_type::now();
    for (unsigned i = 0; i < operations; i += qd) {
        const unsigned current = std::min(qd, operations - i);
        for (unsigned j = 0; j < current; ++j) {
            auto & s = slots[j];
            std::memset(&s.ov, 0, sizeof(s.ov));
            ResetEvent(s.event);
            const uint64_t index = random_io ? dist(rng) : ((i + j) % n_blocks);
            const uint64_t offset = index * block;
            s.ov.Offset     = static_cast<DWORD>(offset);
            s.ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
            s.ov.hEvent = s.event;
            const BOOL ok = ReadFile(f, s.buffer, static_cast<DWORD>(block), nullptr, &s.ov);
            if (!ok && GetLastError() != ERROR_IO_PENDING) fail("ReadFile unbuffered");
        }
        for (unsigned j = 0; j < current; ++j) {
            DWORD done = 0;
            if (!GetOverlappedResult(f, &slots[j].ov, &done, TRUE)) fail("GetOverlappedResult");
            if (done != block) fail("Unexpected short unbuffered read");
            actual += done;
        }
    }
    const double elapsed = millis(start, clock_type::now());
    std::cout << "disk," << (random_io ? "random" : "sequential") << ","
              << block << "," << qd << "," << operations << ","
              << actual << "," << std::fixed << std::setprecision(4)
              << elapsed << "," << mbps(actual, elapsed) << ",NA,NA\n";
    for (auto & s : slots) {
        CloseHandle(s.event);
        VirtualFree(s.buffer, 0, MEM_RELEASE);
    }
    CloseHandle(f);
}

static void h2d_test(size_t block, bool pinned, unsigned operations) {
    void * host = nullptr;
    void * gpu = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t a = nullptr, b = nullptr;
    if (pinned) {
        cuda_check(cudaMallocHost(&host, block), "cudaMallocHost");
    } else {
        host = VirtualAlloc(nullptr, block, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!host) fail("VirtualAlloc pageable host");
    }
    std::memset(host, 0x42, block);
    cuda_check(cudaMalloc(&gpu, block), "cudaMalloc device");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    cuda_check(cudaEventCreate(&a), "cudaEventCreate A");
    cuda_check(cudaEventCreate(&b), "cudaEventCreate B");

    for (int i = 0; i < 4; ++i) {
        cuda_check(cudaMemcpyAsync(gpu, host, block, cudaMemcpyHostToDevice, stream), "warmup H2D");
    }
    cuda_check(cudaStreamSynchronize(stream), "warmup sync");

    const auto start = clock_type::now();
    cuda_check(cudaEventRecord(a, stream), "start event");
    for (unsigned i = 0; i < operations; ++i) {
        cuda_check(cudaMemcpyAsync(gpu, host, block, cudaMemcpyHostToDevice, stream), "H2D");
    }
    cuda_check(cudaEventRecord(b, stream), "stop event");
    cuda_check(cudaEventSynchronize(b), "H2D stop sync");
    const double elapsed = millis(start, clock_type::now());
    float gpu_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&gpu_ms, a, b), "cudaEventElapsedTime");
    const uint64_t bytes = uint64_t(block)*operations;
    std::cout << "h2d," << (pinned ? "pinned" : "pageable") << ","
              << block << ",1," << operations << "," << bytes << ","
              << std::fixed << std::setprecision(4) << elapsed << ","
              << mbps(bytes, elapsed) << "," << gpu_ms << ","
              << mbps(bytes, gpu_ms) << "\n";
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaStreamDestroy(stream);
    cudaFree(gpu);
    if (pinned) cudaFreeHost(host);
    else VirtualFree(host, 0, MEM_RELEASE);
}

static void serial_disk_to_gpu(const wchar_t * path, unsigned operations) {
    constexpr size_t block = 1024*1024;
    uint64_t file_size = 0;
    HANDLE f = open_direct(path, file_size);
    const uint64_t n_blocks = file_size / block;
    if (n_blocks < 2) fail("File is too small for disk-to-GPU test");

    void * input = VirtualAlloc(nullptr, block, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    void * stage = nullptr;
    void * gpu = nullptr;
    if (!input) fail("VirtualAlloc read buffer");
    cuda_check(cudaMallocHost(&stage, block), "cudaMallocHost stage");
    cuda_check(cudaMalloc(&gpu, block), "cudaMalloc GPU");
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    OVERLAPPED ov{};
    ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ov.hEvent) fail("CreateEvent serial");

    double disk_ms = 0, cpu_copy_ms = 0, gpu_wait_ms = 0;
    const auto start = clock_type::now();
    for (unsigned i = 0; i < operations; ++i) {
        ResetEvent(ov.hEvent);
        const uint64_t offset = uint64_t(i % n_blocks)*block;
        ov.Offset = static_cast<DWORD>(offset);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        const auto d0 = clock_type::now();
        BOOL ok = ReadFile(f, input, static_cast<DWORD>(block), nullptr, &ov);
        if (!ok && GetLastError() != ERROR_IO_PENDING) fail("serial ReadFile");
        DWORD done = 0;
        if (!GetOverlappedResult(f, &ov, &done, TRUE) || done != block) fail("serial read completion");
        const auto d1 = clock_type::now();
        std::memcpy(stage, input, block);
        const auto d2 = clock_type::now();
        cuda_check(cudaMemcpyAsync(gpu, stage, block, cudaMemcpyHostToDevice, stream), "serial H2D");
        cuda_check(cudaStreamSynchronize(stream), "serial H2D sync");
        const auto d3 = clock_type::now();
        disk_ms += millis(d0, d1);
        cpu_copy_ms += millis(d1, d2);
        gpu_wait_ms += millis(d2, d3);
    }
    const double elapsed = millis(start, clock_type::now());
    const uint64_t bytes = uint64_t(block)*operations;
    std::cout << "pipeline,serial_direct_io_then_H2D," << block
              << ",1," << operations << "," << bytes << ","
              << std::fixed << std::setprecision(4)
              << elapsed << "," << mbps(bytes, elapsed) << ","
              << disk_ms << "," << cpu_copy_ms << "\n";
    std::cerr << "PIPELINE_DIAGNOSTIC_MS disk=" << disk_ms << " cpu_stage=" << cpu_copy_ms
              << " h2d_submit_and_sync=" << gpu_wait_ms << " total=" << elapsed << "\n";
    CloseHandle(ov.hEvent);
    CloseHandle(f);
    cudaStreamDestroy(stream);
    cudaFree(gpu);
    cudaFreeHost(stage);
    VirtualFree(input, 0, MEM_RELEASE);
}

int wmain(int argc, wchar_t ** argv) {
    if (argc != 2) {
        std::wcerr << L"Usage: moe-phase4-probe.exe <existing-model-or-large-file-path>\n";
        return 1;
    }
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::cerr << "GPU: " << prop.name << ", CUDA compute " << prop.major
              << "." << prop.minor << "\n";
    std::cout << "category,mode,block_bytes,queue_depth,operations,bytes,wall_ms,wall_MiBps,gpu_event_or_disk_ms,gpu_MiBps_or_cpu_stage_ms\n";
    for (const size_t block : {size_t(256*1024),size_t(1024*1024),size_t(4*1024*1024)}) {
        for (const unsigned qd : {1u, 4u, 16u}) {
            disk_test(argv[1], block, qd, false, 128);
            disk_test(argv[1], block, qd, true, 128);
        }
    }
    for (const size_t block : {size_t(256*1024),size_t(1024*1024),size_t(4*1024*1024)}) {
        h2d_test(block, false, 128);
        h2d_test(block, true, 128);
    }
    serial_disk_to_gpu(argv[1], 128);
    std::cerr << "NOTE: This independent probe does not measure llama.cpp kernel execution or overlapping I/O."
                 " Unbuffered Win32 reads bypass the Windows filesystem data cache but cannot"
                 " rule out SSD/controller caches. CUDA event times are device-stream timings.\n";
    return 0;
}

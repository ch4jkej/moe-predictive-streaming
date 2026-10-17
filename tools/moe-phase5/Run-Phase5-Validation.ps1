# Phase 5: reproduce correctness and RAM-cache benefit without compiling locally.
# Run on Windows PowerShell 5.1+ with a successfully built phase5 artifact.
param(
    [Parameter(Mandatory=$true)][string]$BinDir,
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [int]$GenerateTokens = 96,
    [int]$GpuCacheSlots = 8  # minimum for Ling Tiny (8 routed experts per token) when -ub 1
)
if ($GpuCacheSlots -lt 8) { throw 'Ling Tiny uses 8 routed experts/token; streamed GPU staging requires at least 8 slots per layer at -ub 1.' }
$ErrorActionPreference = 'Stop'
$exe = Join-Path $BinDir 'llama-cli.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing llama-cli.exe at $exe" }
if (-not (Test-Path -LiteralPath $ModelPath)) { throw "Missing GGUF model at $ModelPath" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$BinDir = (Resolve-Path -LiteralPath $BinDir).Path
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$log = Join-Path $OutputDir 'phase5-run-manifest.csv'
'mode,ram_cache_mib,gpu_cache_slots_per_layer,exit_code,start_time,end_time,stdout,stderr' | Set-Content -LiteralPath $log -Encoding Ascii

$env:MOE_PHASE4_PROFILE_PATH = ''
$env:MOE_PHASE4_FORCE_SYNC = ''
$env:MOE_TRACE_PATH = ''

# This deliberately uses ubatch=1 first and the minimum 8 expert staging slots per MoE layer.
# 0 does NOT mean zero GPU allocations: 0 triggers automatic cache sizing.
# At -ub 8, the upstream wave-prefill implementation may need 24+ slots/layer.
# That is a separate correctness test, NOT a reason to inflate the minimal-footprint baseline.
$baseArgs = @('-m',$ModelPath,'-ngl','99','-c','2048','-b','128','-ub','1',
              '-t','6','-n',[string]$GenerateTokens,'--temp','0','--seed','42',
              '-p','Explain how an electrical transformer works, in a concise paragraph.')

$runs = @(
    @{ name='gpu-reference'; ram=0; streamed=$false },
    @{ name='ssd-only'; ram=0; streamed=$true },
    @{ name='ram-1024'; ram=1024; streamed=$true },
    @{ name='ram-2048'; ram=2048; streamed=$true }
)
Push-Location -LiteralPath $BinDir
try {
    foreach ($run in $runs) {
        $cliArgs = @($baseArgs)
        if ($run.streamed) {
            $cliArgs += @('--moe-stream','--moe-stream-cache',([string]$GpuCacheSlots + 's'),
                       '--moe-stream-ram-cache-mib',[string]$run.ram,
                       '--moe-stream-io-threads','8','--moe-stream-direct')
        }
        $name = $run.name
        $stdout = Join-Path $OutputDir "$name-stdout.txt"
        $stderr = Join-Path $OutputDir "$name-stderr.txt"
        $start = (Get-Date).ToString('o')
        Write-Host ("Running {0} (RAM cache: {1} MiB)" -f $name, $run.ram)
        $ErrorActionPreference = 'Continue'
        & $exe @cliArgs 1> $stdout 2> $stderr
        $ErrorActionPreference = 'Stop'
        $exit = $LASTEXITCODE
        $end = (Get-Date).ToString('o')
        ('"{0}",{1},{2},{3},"{4}","{5}","{6}","{7}"' -f
            $name,$run.ram,$GpuCacheSlots,$exit,$start,$end,$stdout,$stderr) |
            Add-Content -LiteralPath $log -Encoding Ascii
        if ($exit -ne 0) {
            Write-Warning "Run $name failed with exit code $exit. Review the error log; skipping subsequent runs."
            break
        }
    }
} finally {
    Pop-Location
}
Write-Host "Completed/attempted runs, manifest: $log"
Write-Host "Inspect output token equivalence, GPU residency, RAM-cache totals, disk-read counters and real process memory with the local agent."

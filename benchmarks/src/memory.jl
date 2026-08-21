# Memory instrumentation: what a solve actually holds, on the host and the device.
#
# Two decisions shape this file.
#
# First, memory is measured by its own stage and never read off a timing sweep.
# The two questions want opposite things from a run — a timing must not be
# perturbed and so cannot afford a sampler, while a peak needs one and does not
# care what it costs. Separating them is what lets each be measured properly
# instead of both being measured badly.
#
# Second, the host figure is the process resident set, sampled, in both
# languages. `benchkit/memory.py` samples the same operating-system quantity
# through `psutil`, which is what makes a Julia number and a Python number
# comparable at all. Julia's own `Base.gc_live_bytes` would count only the parts
# of a solve that Julia's collector manages, and PuMA's solver is C.

using CUDA

const PAGESIZE = Ref{Int}(0)

"""Resident set of this process, in bytes.

The resident set is the right quantity for "how much memory did this need": it
counts what is actually in physical memory, includes allocations made by C
libraries, and — unlike a peak counter — can be sampled repeatedly.

Read straight from the OS on Linux and Windows. On any other platform this falls
back to `Sys.maxrss`, which is a high-water mark that never falls, so a sampled
peak there degrades to "the largest this process ever was".
"""
function current_rss()
    @static if Sys.islinux()
        return _rss_linux()
    elseif Sys.iswindows()
        return _rss_windows()
    else
        return Int(Sys.maxrss())
    end
end

function _rss_linux()
    PAGESIZE[] == 0 && (PAGESIZE[] = Int(ccall(:getpagesize, Cint, ())))
    fields = split(read("/proc/self/statm", String))
    length(fields) >= 2 || return Int(Sys.maxrss())
    return parse(Int, fields[2]) * PAGESIZE[]
end

# PROCESS_MEMORY_COUNTERS on 64-bit Windows: two DWORDs then eight SIZE_Ts, so
# 72 bytes with WorkingSetSize — the resident set — at offset 16.
const _PMC_BYTES = 72
const _PMC_WORKINGSET_OFFSET = 16

function _rss_windows()
    buf = zeros(UInt8, _PMC_BYTES)
    handle = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
    rss = GC.@preserve buf begin
        p = pointer(buf)
        unsafe_store!(Ptr{UInt32}(p), UInt32(_PMC_BYTES))
        ok = ccall((:GetProcessMemoryInfo, "psapi"), Cint,
                   (Ptr{Cvoid}, Ptr{UInt8}, UInt32), handle, p, UInt32(_PMC_BYTES))
        ok == 0 ? -1 : Int(unsafe_load(Ptr{UInt64}(p + _PMC_WORKINGSET_OFFSET)))
    end
    return rss < 0 ? Int(Sys.maxrss()) : rss
end

"""Device bytes actually allocated, excluding blocks the pool merely caches.

This is the figure that answers "how much device memory does this solve need".
Its counterpart [`device_pool_bytes`](@ref) is recorded alongside for context but
must not be compared between configurations: an idle session can show tens of
gigabytes of pool against half a megabyte live, and under pressure the pool
saturates at the card's capacity and reports the same number for every
configuration — precisely where the comparison matters most.
"""
device_live_bytes() = CUDA.functional() ? Int(CUDA.memory_stats().live) : 0

"""Device bytes the driver reports as unavailable: CUDA.jl's pool footprint."""
device_pool_bytes() = CUDA.functional() ? Int(CUDA.total_memory() - CUDA.available_memory()) : 0

"""What a sampled run held, alongside whatever the measured code returned."""
struct PeakUsage
    value::Any
    elapsed::Float64
    baseline_rss::Int
    peak_rss::Int
    peak_device::Int
    peak_pool::Int
    samples::Int
end

"""Run `f` while a background task samples memory, and report the peaks.

The sampler is placed on Julia's interactive thread pool when one exists, so it
keeps sampling while the measured code saturates the default pool. Start the
process with `-t <n>,1` to get one. With no spare thread the peaks degrade to
readings taken either side of `f`, which for a Krylov solve — whose workspace is
allocated up front and then reused — is close but not guaranteed to catch a
transient during setup.
"""
function with_peak_sampling(f; interval_ms::Real=10, gpu::Bool=false)
    GC.gc(true)
    gpu && CUDA.functional() && CUDA.reclaim()

    baseline_rss = current_rss()
    peak_rss = Threads.Atomic{Int}(baseline_rss)
    peak_device = Threads.Atomic{Int}(gpu ? device_live_bytes() : 0)
    peak_pool = Threads.Atomic{Int}(gpu ? device_pool_bytes() : 0)
    samples = Threads.Atomic{Int}(0)
    running = Threads.Atomic{Bool}(true)

    take_sample!() = begin
        Threads.atomic_max!(peak_rss, current_rss())
        if gpu
            Threads.atomic_max!(peak_device, device_live_bytes())
            Threads.atomic_max!(peak_pool, device_pool_bytes())
        end
        Threads.atomic_add!(samples, 1)
    end

    sampler = _spawn_sampler(running, take_sample!, interval_ms / 1000)
    local value, elapsed
    try
        elapsed = @elapsed (value = f())
        # A last reading with everything the solve allocated still reachable: the
        # caller drops its references only after this returns, so the end state is
        # always represented even if the sampler was never scheduled.
        take_sample!()
    finally
        running[] = false
        sampler === nothing || wait(sampler)
    end

    return PeakUsage(value, elapsed, baseline_rss, peak_rss[], peak_device[],
                     peak_pool[], samples[])
end

function _spawn_sampler(running, take_sample!, interval_s)
    loop() = begin
        while running[]
            take_sample!()
            sleep(interval_s)
        end
    end
    if Threads.nthreads(:interactive) >= 1
        return Threads.@spawn :interactive loop()
    elseif Threads.nthreads() >= 2
        return Threads.@spawn loop()
    end
    @warn "no spare thread for the memory sampler — peaks fall back to readings either side " *
          "of the solve; start Julia with `-t $(Threads.nthreads()),1`" maxlog = 1
    return nothing
end

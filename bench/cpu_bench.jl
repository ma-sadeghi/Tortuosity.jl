# Benchmark: CPU loops (old path) vs KA kernels on CPU backend (new GPU-ready path)
#
# The CPU loop implementations ARE the old code (unchanged algorithms).
# The KA kernels are the new backend-agnostic path, here tested on CPU.
using Tortuosity
using Tortuosity:
    _create_connectivity_list_cpu,
    _create_connectivity_list_ka,
    create_adjacency_matrix,
    laplacian,
    spdiagm,
    find_boundary_nodes,
    interpolate_edge_values,
    apply_dirichlet_bc_fast!,
    PortableSparseCSC,
    exclusive_scan!,
    fill_idx_kernel!,
    histogram_connections_kernel!,
    write_connections_offset_kernel!,
    _spmv_kernel!,
    _laplacian_colptr_kernel!,
    _laplacian_entries_kernel!,
    _histogram_cols_kernel!,
    _scatter_coo_to_csc_kernel!,
    _build_colptr_kernel!,
    set_diag!,
    get_diag,
    zero_rows_cols!,
    dropzeros!,
    compact_and_count_kernel!
using BenchmarkTools
using SparseArrays
using LinearAlgebra
using LinearSolve
using KernelAbstractions
using Atomix
using Statistics

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
img_small  = ones(Bool, 16, 16, 16)      # 4 096 voxels
img_medium = ones(Bool, 32, 32, 32)      # 32 768 voxels
img_blob   = Array{Bool}(Tortuosity.Imaginator.blobs(; shape=(32, 32, 32), porosity=0.6, blobiness=1, seed=42))

# Pre-build connectivity lists for adjacency / laplacian benchmarks
conns_small  = _create_connectivity_list_cpu(img_small)
conns_medium = _create_connectivity_list_cpu(img_medium)
conns_blob   = _create_connectivity_list_cpu(img_blob)
nnodes_small  = sum(img_small)
nnodes_medium = sum(img_medium)
nnodes_blob   = sum(img_blob)

# Pre-build adjacency matrices (Float64 weights to exercise SpMV/dropzeros)
w_small  = ones(Float64, size(conns_small, 1))
w_medium = ones(Float64, size(conns_medium, 1))
w_blob   = ones(Float64, size(conns_blob, 1))
am_small  = create_adjacency_matrix(conns_small;  n=nnodes_small,  weights=w_small)
am_medium = create_adjacency_matrix(conns_medium; n=nnodes_medium, weights=w_medium)
am_blob   = create_adjacency_matrix(conns_blob;   n=nnodes_blob,   weights=w_blob)

# Pre-build Laplacians (SparseMatrixCSC)
L_small  = laplacian(am_small)
L_medium = laplacian(am_medium)

# Build a PortableSparseCSC (CPU vectors) for KA kernel benchmarks
function sparse_to_portable(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    PortableSparseCSC(
        size(A, 1), size(A, 2),
        copy(SparseArrays.getcolptr(A)),
        copy(rowvals(A)),
        copy(nonzeros(A)),
    )
end

am_port_small  = sparse_to_portable(am_small)
am_port_medium = sparse_to_portable(am_medium)

L_port_small  = sparse_to_portable(L_small)
L_port_medium = sparse_to_portable(L_medium)

# ---------------------------------------------------------------------------
# Helper: pretty-print a benchmark pair
# ---------------------------------------------------------------------------
struct BenchRow
    name::String
    t_old_ns::Float64
    t_new_ns::Float64
end

function format_time(ns)
    if ns < 1_000
        return @sprintf("%.0f ns", ns)
    elseif ns < 1_000_000
        return @sprintf("%.1f us", ns / 1_000)
    elseif ns < 1_000_000_000
        return @sprintf("%.2f ms", ns / 1_000_000)
    else
        return @sprintf("%.3f s", ns / 1_000_000_000)
    end
end

using Printf

function print_table(title, rows::Vector{BenchRow})
    println("\n", "="^76)
    println(title)
    println("="^76)
    hdr = @sprintf("%-40s %12s %12s %10s", "Operation", "Old (CPU)", "New (KA/CPU)", "Speedup")
    println(hdr)
    println("-"^76)
    for r in rows
        speedup = r.t_old_ns / r.t_new_ns
        s = if speedup >= 1.0
            @sprintf("%.2fx", speedup)
        else
            @sprintf("%.2fx", speedup)
        end
        println(@sprintf("%-40s %12s %12s %10s",
            r.name, format_time(r.t_old_ns), format_time(r.t_new_ns), s))
    end
    println("="^76)
end

# ---------------------------------------------------------------------------
# 1.  OVERALL WORKFLOW
# ---------------------------------------------------------------------------
println("\n>>> Warming up (first call)...")
warmup_sim = TortuositySimulation(img_small; axis=:x, gpu=false)
solve(warmup_sim.prob, KrylovJL_CG(); reltol=1e-6)

println(">>> Benchmarking overall workflow...")

workflow_rows = BenchRow[]

for (label, img) in [("16^3 open", img_small), ("32^3 open", img_medium), ("32^3 blob (eps=0.6)", img_blob)]
    b = @benchmark begin
        sim = TortuositySimulation($img; axis=:x, gpu=false)
        solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
    end
    t = median(b).time  # nanoseconds
    push!(workflow_rows, BenchRow("Full workflow: $label", t, t))
end

println("\n", "="^76)
println("OVERALL WORKFLOW  (construction + CG solve, CPU)")
println("="^76)
hdr = @sprintf("%-46s %12s %10s", "Workflow", "Time", "Samples")
println(hdr)
println("-"^76)
for r in workflow_rows
    println(@sprintf("%-46s %12s %10d",
        r.name, format_time(r.t_old_ns), 50))
end
println("="^76)

# ---------------------------------------------------------------------------
# 2.  COMPONENT BENCHMARKS: CPU loops vs KA kernels on CPU
# ---------------------------------------------------------------------------
println("\n>>> Benchmarking individual components...")

comp_rows = BenchRow[]

# -- create_connectivity_list --
for (label, img) in [("16^3 open", img_small), ("32^3 blob", img_blob)]
    b_old = @benchmark _create_connectivity_list_cpu($img)
    b_new = @benchmark _create_connectivity_list_ka($img)
    push!(comp_rows, BenchRow("connectivity_list ($label)", median(b_old).time, median(b_new).time))
end

# -- create_adjacency_matrix: CPU (SparseMatrixCSC) vs KA (PortableSparseCSC) --
# To force the KA generic dispatch, use Int32 array (not Array{Int,2})
for (label, conns, nn) in [("16^3", conns_small, nnodes_small), ("32^3 blob", conns_blob, nnodes_blob)]
    conns_i32 = Int32.(conns)
    b_old = @benchmark create_adjacency_matrix($conns; n=$nn)
    b_new = @benchmark create_adjacency_matrix($conns_i32; n=$nn, weights=1.0f0)
    push!(comp_rows, BenchRow("adjacency_matrix ($label)", median(b_old).time, median(b_new).time))
end

# -- laplacian: SparseMatrixCSC vs PortableSparseCSC --
for (label, am, am_p) in [("16^3", am_small, am_port_small), ("32^3 open", am_medium, am_port_medium)]
    b_old = @benchmark laplacian($am)
    b_new = @benchmark laplacian($am_p)
    push!(comp_rows, BenchRow("laplacian ($label)", median(b_old).time, median(b_new).time))
end

# -- SpMV: SparseMatrixCSC vs PortableSparseCSC --
for (label, L, Lp) in [("16^3", L_small, L_port_small), ("32^3 open", L_medium, L_port_medium)]
    n = size(L, 2)
    x = ones(n)
    y_old = similar(x)
    y_new = similar(x)
    b_old = @benchmark mul!($y_old, $L, $x)
    b_new = @benchmark mul!($y_new, $Lp, $x)
    push!(comp_rows, BenchRow("SpMV mul! ($label)", median(b_old).time, median(b_new).time))
end

print_table("COMPONENT BENCHMARKS: CPU loops (old) vs KA kernels on CPU (new)", comp_rows)

# ---------------------------------------------------------------------------
# 3.  SPARSE OPERATION BENCHMARKS  (PortableSparseCSC)
# ---------------------------------------------------------------------------
println("\n>>> Benchmarking sparse operations on PortableSparseCSC...")

sparse_rows = BenchRow[]

# Build test matrices for sparse ops
L_test = laplacian(am_port_medium)
n_test = size(L_test, 1)

# -- get_diag --
b_old = @benchmark SparseArrays.diag($L_medium)
b_new = @benchmark get_diag($L_test)
push!(sparse_rows, BenchRow("get_diag (32^3)", median(b_old).time, median(b_new).time))

# -- set_diag! --
dv = ones(n_test)
L_copy = sparse_to_portable(L_medium)
b_old = @benchmark begin
    di = SparseArrays.diagind($L_medium)
    $L_medium[di] .= $dv
end
b_new = @benchmark set_diag!($L_copy, $dv)
push!(sparse_rows, BenchRow("set_diag! (32^3)", median(b_old).time, median(b_new).time))

# -- zero_rows_cols! --
bc_nodes = vcat(collect(1:16), collect(n_test-15:n_test))
L_zrc_sparse = copy(L_medium)
L_zrc_port = sparse_to_portable(L_medium)
b_old_zrc = @benchmark begin
    A = copy($L_medium)
    I, J, _ = findnz(A)
    ri = Tortuosity.overlap_indices_fast(I, $bc_nodes)
    ci = Tortuosity.overlap_indices_fast(J, $bc_nodes)
    A.nzval[union(ri, ci)] .= 0.0
end
b_new_zrc = @benchmark begin
    Ap = sparse_to_portable($L_medium)
    zero_rows_cols!(Ap, $bc_nodes)
end
push!(sparse_rows, BenchRow("zero_rows_cols! (32^3)", median(b_old_zrc).time, median(b_new_zrc).time))

# -- dropzeros! --
# Create a matrix with explicit zeros to drop
function make_dirty_portable(L)
    Lp = sparse_to_portable(L)
    nz = nonzeros(Lp)
    # Zero out ~10% of entries
    for i in 1:10:length(nz)
        nz[i] = 0.0
    end
    return Lp
end

function make_dirty_sparse(L)
    Lc = copy(L)
    for i in 1:10:length(Lc.nzval)
        Lc.nzval[i] = 0.0
    end
    return Lc
end

b_old_dz = @benchmark dropzeros!(make_dirty_sparse($L_medium))
b_new_dz = @benchmark dropzeros!(make_dirty_portable($L_medium))
push!(sparse_rows, BenchRow("dropzeros! (32^3)", median(b_old_dz).time, median(b_new_dz).time))

print_table("SPARSE OPS: SparseArrays (old) vs PortableSparseCSC+KA (new)", sparse_rows)

# ---------------------------------------------------------------------------
# 4.  ATOMIC / KERNEL MICRO-BENCHMARKS  (KA on CPU)
# ---------------------------------------------------------------------------
println("\n>>> Benchmarking atomic / kernel primitives on CPU backend...")

atom_rows = BenchRow[]

# -- exclusive_scan! --
for n in [1_000, 100_000]
    inp = ones(Int, n)
    out = similar(inp)
    b_old = @benchmark cumsum!($out, $inp)
    b_new = @benchmark exclusive_scan!($out, $inp)
    push!(atom_rows, BenchRow("exclusive_scan! (n=$n)", median(b_old).time, median(b_new).time))
end

# -- Atomix.@atomic (histogram pattern) --
function histogram_plain!(counts, data)
    fill!(counts, 0)
    for v in data
        counts[v] += 1
    end
end

@kernel function histogram_ka!(counts, @Const(data))
    i = @index(Global)
    @inbounds v = data[i]
    Atomix.@atomic counts[v] += 1
end

n_hist = 50_000
data_hist = rand(1:1000, n_hist)
counts_plain = zeros(Int, 1000)
counts_ka = zeros(Int, 1000)

b_old_hist = @benchmark histogram_plain!($counts_plain, $data_hist)
b_new_hist = @benchmark begin
    fill!($counts_ka, 0)
    histogram_ka!(CPU())($counts_ka, $data_hist; ndrange=$n_hist)
    KernelAbstractions.synchronize(CPU())
end
push!(atom_rows, BenchRow("histogram (Atomix, n=50k)", median(b_old_hist).time, median(b_new_hist).time))

# -- Atomix.modify! (fetch-and-add pattern) --
function scatter_plain!(out, offsets, data, cols)
    for k in eachindex(data)
        j = cols[k]
        pos = offsets[j]
        offsets[j] = pos + 1
        out[pos] = data[k]
    end
end

@kernel function scatter_ka!(out, offsets, @Const(data), @Const(cols))
    k = @index(Global)
    @inbounds j = cols[k]
    ref = Atomix.IndexableRef(offsets, (j,))
    pos = Atomix.modify!(ref, +, 1).first
    @inbounds out[pos] = data[k]
end

n_scat = 50_000
n_cols = 5_000
cols_scat = rand(1:n_cols, n_scat)
# Build offsets from histogram
col_counts = zeros(Int, n_cols)
for c in cols_scat; col_counts[c] += 1; end
offsets_base = cumsum([1; col_counts[1:end-1]])

data_scat = rand(Float64, n_scat)
out_plain = zeros(Float64, n_scat)
out_ka = zeros(Float64, n_scat)

b_old_scat = @benchmark scatter_plain!($out_plain, copy($offsets_base), $data_scat, $cols_scat)
b_new_scat = @benchmark begin
    scatter_ka!(CPU())($out_ka, copy($offsets_base), $data_scat, $cols_scat; ndrange=$n_scat)
    KernelAbstractions.synchronize(CPU())
end
push!(atom_rows, BenchRow("scatter (Atomix.modify!, n=50k)", median(b_old_scat).time, median(b_new_scat).time))

# -- SpMV kernel micro-benchmark --
for n in [4096, 32768]
    A_sp = sprandn(n, n, 6/n)  # ~6 nnz per row like a 3D grid
    A_port = sparse_to_portable(A_sp)
    x = ones(n)
    y1 = similar(x)
    y2 = similar(x)
    b_old = @benchmark mul!($y1, $A_sp, $x)
    b_new = @benchmark mul!($y2, $A_port, $x)
    push!(atom_rows, BenchRow("SpMV kernel (n=$n, ~6nnz/row)", median(b_old).time, median(b_new).time))
end

print_table("ATOMIC / KERNEL PRIMITIVES: Plain loops vs KA+Atomix on CPU", atom_rows)

println("\nDone.")

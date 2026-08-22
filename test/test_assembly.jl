# Structural invariants of the assembled linear system.
#
# The steady solver is three stages — connectivity list → weighted adjacency
# matrix → graph Laplacian → Dirichlet elimination — and each stage carries
# undocumented-by-construction assumptions that the next stage depends on.
# Most notably `build_adjacency_matrix` builds CSC arrays *directly* from the
# connectivity list ("sorted by column, then row"), skipping `sparse()`
# entirely; if a future rewrite of the connectivity builder emits rows in a
# different order the resulting matrix is silently malformed rather than
# throwing. The tests here pin those invariants so that can't happen quietly.

using Test
using LinearAlgebra
using SparseArrays
using Tortuosity
using Tortuosity:
    Imaginator,
    axis_faces,
    build_adjacency_matrix,
    build_connectivity_list,
    build_pore_index,
    build_steady_system,
    effective_diffusivity,
    tortuosity,
    _build_connectivity_list_cpu,
    find_boundary_nodes,
    interpolate_edge_values,
    laplacian,
    reconstruct_field

# Fixtures span the interesting structural cases: a fully-connected box, a
# geometry with solid slabs (so degrees vary), and irregular pore space.
function assembly_fixtures()
    imgs = Tuple{String,Array{Bool,3}}[]
    push!(imgs, ("open 5x6x7", ones(Bool, 5, 6, 7)))
    let img = ones(Bool, 8, 8, 8)
        img[:, :, 1:4] .= false
        push!(imgs, ("half-solid 8^3", img))
    end
    let img = ones(Bool, 6, 6, 6)
        img[2:5, 2:5, 2:5] .= false     # hollow shell: every voxel is a surface voxel
        push!(imgs, ("hollow shell 6^3", img))
    end
    for seed in (3, 11)
        img = Array{Bool}(Imaginator.blobs(; shape=(12, 12, 12), porosity=0.6, blobiness=1, seed=seed))
        img = Array{Bool}(Imaginator.trim_nonpercolating_paths(img; axis=:x))
        count(img) >= 8 || continue
        push!(imgs, ("trimmed blob 12^3 seed=$seed", img))
    end
    return imgs
end

const ASSEMBLY_IMAGES = assembly_fixtures()

# Count face-adjacent pore pairs by brute force, independent of the kernel.
function count_adjacent_pairs(img)
    nx, ny, nz = size(img)
    pairs = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        img[i, j, k] || continue
        i < nx && img[i + 1, j, k] && (pairs += 1)
        j < ny && img[i, j + 1, k] && (pairs += 1)
        k < nz && img[i, j, k + 1] && (pairs += 1)
    end
    return pairs
end

# Assert the CSC arrays describe a well-formed matrix with sorted row indices
# within each column — what SparseArrays and CUSPARSE both assume.
function check_csc_wellformed(A::SparseMatrixCSC)
    colptr = SparseArrays.getcolptr(A)
    rows = rowvals(A)
    @test colptr[1] == 1
    @test colptr[end] == nnz(A) + 1
    @test issorted(colptr)
    @test all(1 .<= rows .<= size(A, 1))
    for j in 1:size(A, 2)
        @test issorted(@view rows[colptr[j]:(colptr[j + 1] - 1)])
    end
end

# --- Connectivity list ---

@testset "build_connectivity_list — $(label)" for (label, img) in ASSEMBLY_IMAGES
    conns = build_connectivity_list(img)
    n = count(img)

    @testset "indices are in range and there are no self-loops" begin
        @test size(conns, 2) == 2
        @test all(1 .<= conns .<= n)
        @test all(conns[k, 1] != conns[k, 2] for k in 1:size(conns, 1))
    end

    @testset "edge count matches a brute-force adjacency count" begin
        # Each adjacent pair appears once in each direction.
        @test size(conns, 1) == 2 * count_adjacent_pairs(img)
    end

    @testset "the edge set is symmetric" begin
        forward = Set((conns[k, 1], conns[k, 2]) for k in 1:size(conns, 1))
        @test length(forward) == size(conns, 1)   # no duplicated edges
        @test forward == Set((j, i) for (i, j) in forward)
    end

    @testset "rows are grouped by column and ascending within a column" begin
        # `build_adjacency_matrix` writes CSC arrays straight from this ordering.
        @test issorted(@view conns[:, 2])
        for j in unique(@view conns[:, 2])
            rows_in_col = conns[findall(==(j), @view conns[:, 2]), 1]
            @test issorted(rows_in_col)
        end
    end

    @testset "node degrees equal the number of pore face-neighbours" begin
        degrees = zeros(Int, n)
        for k in 1:size(conns, 1)
            degrees[conns[k, 2]] += 1
        end
        @test sum(degrees) == size(conns, 1)
        @test all(0 .<= degrees .<= 6)
    end
end

@testset "build_connectivity_list — 2D input is promoted to 3D" begin
    img2d = Bool[1 1 0; 1 1 1; 0 1 1]
    @test build_connectivity_list(img2d) == build_connectivity_list(reshape(img2d, 3, 3, 1))
end

@testset "build_connectivity_list — precomputed inds give the same result" begin
    img = ASSEMBLY_IMAGES[2][2]
    idx = similar(img, Int)
    idx[img] .= 1:count(img)
    @test _build_connectivity_list_cpu(img; inds=idx) == _build_connectivity_list_cpu(img)
end

# --- Edge-weight interpolation ---

@testset "interpolate_edge_values" begin
    conns = [1 2; 2 3; 3 1]
    D = [1.0, 4.0, 9.0]
    g = interpolate_edge_values(D, conns)

    @testset "is the harmonic mean of the two node values" begin
        a, b = D[conns[:, 1]], D[conns[:, 2]]
        @test g ≈ @. 2 * a * b / (a + b)
        # The simplified form is the two-half-cell-resistors-in-series expression
        # the docstring derives it from — algebraically equal, and equal to
        # rounding here. It is deliberately not asserted bit-identical; the
        # Float32 testset below bounds how far the two may drift.
        @test g ≈ @. 1 / (1 / (2a) + 1 / (2b))
    end

    @testset "in Float32 the simplified form tracks the literal one to a few ULP" begin
        # `2ab/(a+b)` and `1/(1/(2a)+1/(2b))` are algebraically equal but round
        # differently: they disagree for roughly half of all Float32 pairs. The
        # bound matters because Float32 is the element type of every GPU solve,
        # and because a reference implementation written the other way must be
        # compared with a tolerance rather than for exact equality.
        ulp_diff(x, y) = abs(Int(reinterpret(Int32, x)) - Int(reinterpret(Int32, y)))

        a32 = Float32[1, 4, 9, 0.5, 1e-3, 7.25, 1e3, 2.5, 1e-6]
        b32 = Float32[4, 9, 1, 2.5, 1e3, 0.125, 1e-3, 2.5, 1e6]
        conns32 = hcat(1:length(a32), (length(a32) + 1):(2 * length(a32)))
        simplified = interpolate_edge_values(vcat(a32, b32), conns32)
        literal = @. 1.0f0 / (1.0f0 / (2 * a32) + 1.0f0 / (2 * b32))

        @test eltype(simplified) === Float32
        @test all(ulp_diff.(simplified, literal) .<= 3)

        # Equal node values are the common case (uniform D) and are exact.
        @test interpolate_edge_values(Float32[2.5, 2.5], [1 2]) == Float32[2.5]
    end

    @testset "reduces to the common value when both nodes agree" begin
        uniform = fill(2.5, 3)
        @test interpolate_edge_values(uniform, conns) ≈ fill(2.5, 3)
    end

    @testset "is bounded by the two node values and symmetric" begin
        a, b = D[conns[:, 1]], D[conns[:, 2]]
        @test all(min.(a, b) .<= g .<= max.(a, b))
        @test interpolate_edge_values(D, conns[:, [2, 1]]) ≈ g
    end

    @testset "a near-zero node value chokes the edge" begin
        # Solid voxels enter as tiny diffusivities; the harmonic mean must
        # collapse toward the small value rather than the average.
        @test interpolate_edge_values([1.0, 1e-8], [1 2]) ≈ [2e-8] rtol = 1e-6
    end
end

# --- Adjacency matrix and Laplacian ---

@testset "build_adjacency_matrix / laplacian — $(label)" for (label, img) in ASSEMBLY_IMAGES
    conns = build_connectivity_list(img)
    n = count(img)
    nedges = size(conns, 1)
    w = collect(range(0.5, 2.0; length=nedges))
    am = build_adjacency_matrix(conns; n=n, weights=w)

    @testset "produces a well-formed CSC matrix" begin
        @test size(am) == (n, n)
        @test nnz(am) == nedges
        check_csc_wellformed(am)
    end

    @testset "matches SparseArrays.sparse on the same triplets" begin
        reference = sparse(conns[:, 1], conns[:, 2], w, n, n)
        @test am == reference
    end

    @testset "scalar weights broadcast to every edge" begin
        am1 = build_adjacency_matrix(conns; n=n)
        @test all(nonzeros(am1) .== 1)
        @test nnz(am1) == nedges
    end

    @testset "Laplacian is symmetric with zero row sums" begin
        # Symmetric weights ⇒ symmetric adjacency ⇒ symmetric Laplacian.
        am_sym = build_adjacency_matrix(conns; n=n, weights=ones(nedges))
        L = laplacian(am_sym)
        @test issymmetric(Array(L))
        # Zero row sum is the discrete conservation statement: a constant field
        # produces no net flux anywhere.
        @test maximum(abs, L * ones(n)) < 1e-10
        @test maximum(abs, vec(sum(Array(L); dims=1))) < 1e-10
    end

    @testset "Laplacian is a symmetric M-matrix" begin
        am_sym = build_adjacency_matrix(conns; n=n, weights=ones(nedges))
        L = Array(laplacian(am_sym))
        d = diag(L)
        @test all(d .>= 0)
        offdiag = L - Diagonal(d)
        @test all(offdiag .<= 0)
        # Diagonal equals the node degree (number of pore neighbours).
        @test d ≈ -vec(sum(offdiag; dims=2))
    end
end

@testset "laplacian — positive semi-definite with the constant null vector" begin
    img = ones(Bool, 4, 4, 4)
    conns = build_connectivity_list(img)
    am = build_adjacency_matrix(conns; n=count(img), weights=ones(size(conns, 1)))
    L = Symmetric(Array(laplacian(am)))
    λ = eigvals(L)
    @test minimum(λ) > -1e-10                 # PSD
    @test count(<(1e-10), λ) == 1             # one connected component
end

@testset "fully-open box has the exact expected edge count" begin
    nx, ny, nz = 5, 6, 7
    conns = build_connectivity_list(ones(Bool, nx, ny, nz))
    expected = 2 * ((nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1))
    @test size(conns, 1) == expected
end

# --- Dirichlet elimination and the assembled SteadyDiffusionProblem ---

@testset "SteadyDiffusionProblem — assembled system, $(label)" for (label, img) in ASSEMBLY_IMAGES
    n = count(img)
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false)
    A = sim.prob.A
    b = sim.prob.b

    inlet_face, outlet_face = axis_faces(:x)
    inlet = find_boundary_nodes(img, inlet_face)
    outlet = find_boundary_nodes(img, outlet_face)

    @testset "shape, element type, and CSC health" begin
        @test size(A) == (n, n)
        @test length(b) == n
        @test eltype(A) === Float64
        @test eltype(b) === Float64
        check_csc_wellformed(A)
    end

    @testset "stays symmetric after Dirichlet elimination" begin
        # Symmetry is what makes CG (KrylovJL_CG) a valid solver here — losing
        # it during an optimisation would degrade convergence silently.
        @test issymmetric(Array(A))
    end

    @testset "boundary rows reduce to diag·x = diag·val" begin
        Ad = Array(A)
        @test all(isfinite, b)
        for node in inlet
            row = Ad[node, :]
            @test count(!iszero, row) == 1
            @test row[node] > 0
            @test b[node] ≈ row[node] * 1.0
        end
        for node in outlet
            row = Ad[node, :]
            @test count(!iszero, row) == 1
            # An outlet value of 0 makes b[node] zero either way, so the
            # surviving diagonal is the only thing distinguishing a correctly
            # eliminated row from an entirely blank (singular) one.
            @test row[node] > 0
            @test b[node] ≈ 0.0
        end
    end

    @testset "the solution satisfies the assembled system it came from" begin
        u = solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u
        @test norm(A * u .- b) <= 1e-8 * max(1.0, norm(b))
    end

    @testset "solution honours the imposed boundary values" begin
        u = solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u
        @test all(isapprox.(u[inlet], 1.0; atol=1e-8))
        @test all(isapprox.(u[outlet], 0.0; atol=1e-8))
    end

    @testset "assembly is deterministic" begin
        sim2 = SteadyDiffusionProblem(img; axis=:x, gpu=false)
        @test sim2.prob.A == A
        @test sim2.prob.b == b
    end
end

@testset "construction and solving leave the caller's arrays untouched" begin
    # `sim.img` aliases the mask the caller handed in — the struct deliberately
    # does not copy it — so any in-place step in assembly would silently rewrite
    # the user's image, and every later `tortuosity(c, img)` would be measured
    # against the wrong geometry. The pore numbering is built with `cumsum!` into
    # a fresh array and masked in place, which is one edit away from being done
    # on `img` itself.
    img = Array{Bool}(Imaginator.blobs(; shape=(12, 12, 12), porosity=0.6, blobiness=1, seed=3))
    D = zeros(size(img))
    D[img] .= 1.5
    img_before, D_before = copy(img), copy(D)

    for matrixfree in (false, true)
        sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, D=D, matrixfree=matrixfree,
                                     warn_nonpercolating=false)
        solve(sim.prob, KrylovJL_CG(); reltol=1e-10)
        @test img == img_before
        @test D == D_before
        @test sim.img === img          # aliased, hence the check above
    end
end

@testset "an isolated boundary voxel still receives its Dirichlet value" begin
    # A zero-degree boundary node has a zero diagonal, so the `diag·x = diag·val`
    # encoding would collapse to `0 = 0`, `dropzeros!` would delete the row, and
    # the prescribed value would never be applied. The voxel would keep c = 0
    # while sitting on a c = 1 face, dragging the inlet-slice mean below the
    # imposed drop and reporting a tortuosity below 1 — impossible, from a solve
    # that reports success. Reachable on any untrimmed image.
    duct = falses(12, 6, 6)
    duct[:, 3:4, 3:4] .= true
    iso = copy(duct)
    iso[1, 6, 6] = true                        # on the inlet face, touching nothing

    sim = SteadyDiffusionProblem(iso; axis=:x, gpu=false, warn_nonpercolating=false)
    node = build_pore_index(BitArray(iso))[1, 6, 6]
    A = Array(sim.prob.A)

    @test A[node, node] > 0                    # the row survives dropzeros!
    @test count(!iszero, A[node, :]) == 1      # …and is diagonal-only
    @test sim.prob.b[node] ≈ A[node, node] * 1.0

    c = reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u, iso)
    @test c[1, 6, 6] ≈ 1.0 atol = 1e-9         # the inlet value is honoured

    # The voxel adds pore volume but carries no flux, so D_eff is untouched and
    # τ rises above the duct's 1.0 rather than dropping below it.
    c_duct = reconstruct_field(
        solve(SteadyDiffusionProblem(duct; axis=:x, gpu=false).prob,
              KrylovJL_CG(); reltol=1e-12).u, duct,
    )
    @test effective_diffusivity(c, iso; axis=:x) ≈
          effective_diffusivity(c_duct, duct; axis=:x) atol = 1e-9
    @test tortuosity(c, iso; axis=:x) > 1
end

@testset "a pore space with no connected pairs still assembles" begin
    # Every pore voxel isolated ⇒ the Laplacian has no stored entries ⇒
    # `apply_dirichlet_bc_fast!` calls `overlap_indices_fast` on an empty index
    # vector. That used to throw `ArgumentError: step cannot be zero` from
    # inside the chunking helper.
    img = falses(4, 1, 1)
    img[1, 1, 1] = true
    img[3, 1, 1] = true
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false)
    @test size(sim.prob.A) == (2, 2)
    @test length(sim.prob.b) == 2
end

@testset "SteadyDiffusionProblem — show" begin
    sim = SteadyDiffusionProblem(ones(Bool, 4, 5, 6); axis=:y, gpu=false)
    io = IOBuffer()
    show(io, sim)
    s = String(take!(io))
    @test occursin("SteadyDiffusionProblem", s)
    @test occursin("shape=(4, 5, 6)", s)
    @test occursin("axis=y", s)
    @test occursin("gpu=false", s)
end

@testset "Dirichlet elimination — exact contract, $(label)" for (label, img) in ASSEMBLY_IMAGES
    # These six identities pin the elimination convention completely, in terms of
    # the pre-elimination Laplacian `L`, and they are checked against what
    # `SteadyDiffusionProblem` actually builds — `build_steady_system`, which
    # never forms `L` at all. They are written as a specification
    # rather than a spot check because the matrix-free plan
    # (docs/plans/2026-08-08-matrix-free-operator.md) replaces the
    # assembled matrix with a stencil operator that has to reproduce exactly this
    # convention: an edge survives only when both endpoints are free, a boundary
    # row keeps its *original* diagonal, and the eliminated coupling is folded
    # into the RHS once at setup.
    n = count(img)
    conns = build_connectivity_list(img)
    L = Array(laplacian(build_adjacency_matrix(conns; n=n, weights=ones(size(conns, 1)))))

    inlet_face, outlet_face = axis_faces(:x)
    inlet = find_boundary_nodes(img, inlet_face)
    outlet = find_boundary_nodes(img, outlet_face)
    bc = vcat(inlet, outlet)
    vals = vcat(ones(length(inlet)), zeros(length(outlet)))
    free = setdiff(1:n, bc)

    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false)
    A = Array(sim.prob.A)
    b = sim.prob.b

    # A zero-degree boundary node has nothing to scale, so its row is given a
    # unit diagonal instead — otherwise `diag·x = diag·val` reads `0 = 0` and the
    # prescribed value is dropped. Identity for every node with a neighbour.
    bc_diag = [iszero(d) ? one(d) : d for d in diag(L)[bc]]

    @test A[free, free] ≈ L[free, free]                 # free–free block untouched
    @test all(iszero, A[free, bc])                      # coupling eliminated…
    @test all(iszero, A[bc, free])                      # …symmetrically
    @test diag(A)[bc] ≈ bc_diag                         # diagonal preserved, or 1
    @test b[bc] ≈ bc_diag .* vals
    @test b[free] ≈ -L[free, bc] * vals                 # folded-in boundary load
end

@testset "with uniform weights the RHS counts inlet neighbours" begin
    # The same folding rule, spelled out concretely: for D = 1 and an inlet value
    # of 1, a free node's RHS entry is simply how many inlet voxels touch it.
    img = ones(Bool, 8, 6, 6)
    img[3:5, 2:4, 2:4] .= false
    n = count(img)
    conns = build_connectivity_list(img)
    inlet = find_boundary_nodes(img, :left)
    outlet = find_boundary_nodes(img, :right)
    bc = Set(vcat(inlet, outlet))
    inlet_set = Set(inlet)

    neighbours_from_inlet = zeros(Int, n)
    for k in 1:size(conns, 1)
        src, dst = conns[k, 1], conns[k, 2]
        dst in bc && continue
        src in inlet_set && (neighbours_from_inlet[dst] += 1)
    end

    b = SteadyDiffusionProblem(img; axis=:x, gpu=false).prob.b
    free = setdiff(1:n, collect(bc))
    @test b[free] ≈ Float64.(neighbours_from_inlet[free])
    @test any(>(0), b[free])                            # the check is not vacuous
end

@testset "Dirichlet elimination matches an independent reduced-system solve" begin
    # Build the Laplacian, partition it into free/boundary blocks, and solve
    # L_ff x_f = -L_fb x_b densely. This is the textbook route and shares no
    # code with build_steady_system, so agreement is real corroboration
    # rather than a tautology.
    checked = 0
    for (label, img) in ASSEMBLY_IMAGES
        n = count(img)
        # The blob fixtures sit just under 1000 nodes; a dense solve at that
        # size costs milliseconds. An earlier 700-node cap silently excluded
        # exactly the two irregular geometries this cross-check exists for.
        n <= 2000 || continue
        conns = build_connectivity_list(img)
        L = Array(laplacian(build_adjacency_matrix(conns; n=n, weights=ones(size(conns, 1)))))

        inlet_face, outlet_face = axis_faces(:x)
        inlet = find_boundary_nodes(img, inlet_face)
        outlet = find_boundary_nodes(img, outlet_face)
        bc = vcat(inlet, outlet)
        length(unique(bc)) == length(bc) || continue
        vals = vcat(ones(length(inlet)), zeros(length(outlet)))
        free = setdiff(1:n, bc)

        x_ref = zeros(n)
        x_ref[bc] = vals
        if !isempty(free)
            x_ref[free] = L[free, free] \ (-L[free, bc] * vals)
        end

        sim = SteadyDiffusionProblem(img; axis=:x, gpu=false)
        u = solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u
        @test u ≈ x_ref atol = 1e-7
        checked += 1
    end
    # Guard against the `continue`s quietly emptying the loop: this is the only
    # check in the file that shares no code with `build_steady_system`, so
    # a silently-skipped run would look identical to a passing one.
    @test checked == length(ASSEMBLY_IMAGES)
end

@testset "index type selection" begin
    # An image past the bound is far larger than any machine on hand, so the
    # rule is tested through the predicates rather than by building one.
    wall = fld(typemax(Int32) - 1, 7)           # 7*nnodes + 1 <= typemax(Int32)

    @test Tortuosity._assembled_index_type(1000) === Int32
    @test Tortuosity._assembled_index_type(wall) === Int32
    @test Tortuosity._assembled_index_type(wall + 1) === Int
    # Both backends widen. The refusal this replaced was a GPU-only branch.
    @test Tortuosity._assembled_index_type(Int64(typemax(Int32))) === Int

    # The ordinal's bound is `nnodes`, the offsets' is `7 * nnodes`, so the two
    # sit a factor of seven apart and the ordinal stays narrow well past the
    # point where the offsets have to widen. That gap is the whole reason `idx`
    # carries its own type.
    @test Tortuosity._ordinal_index_type(1000) === Int32
    @test Tortuosity._ordinal_index_type(wall + 1) === Int32
    @test Tortuosity._ordinal_index_type(typemax(Int32) - 1) === Int32
    @test Tortuosity._ordinal_index_type(Int64(typemax(Int32))) === Int

    # The wrap the bound exists to prevent.
    @test Int32(typemax(Int32)) + Int32(1) == typemin(Int32)
end

# The assertion someone with the machine for it would write, kept here as the
# executable statement of what the predicates above stand in for. It is never
# run: `@test_skip` records it Broken without evaluating it, so it costs nothing
# and cannot turn the suite red.
#
# Why it is skipped rather than sized down: the widening rule keys off
# `7 * nnodes + 1 > typemax(Int32)`, i.e. 306,783,378 pore voxels — about 800³
# at ε = 0.6. Holding one costs 512 MB for the mask, 1.2 GB for the grid index
# array, and 24.4 GB for the `Int64` matrix itself, before the solver allocates
# a single Krylov vector, and roughly 36 GB once it has. There is no smaller
# image that reaches the branch, because the branch is a function of the pore
# count alone.
#
# `Ti=Int64` forced at a tractable size is the nearest thing the suite can run,
# and it does — "Ti=Int64 changes the index width and nothing else" above pins
# that the wide build is the narrow one widened, entry for entry. What that
# cannot reach is the *automatic* widening: an image that selects `Int64` on its
# own, where a mis-stated bound would let `Int32` through and every offset would
# wrap to negative under `@inbounds`.
function _automatic_wide_index_path_holds()
    # ε = 0.65, not the 0.6 the bound sits at: `blobs` lands a few tenths of a
    # percent under its target porosity, and at 0.6 an 800³ image comes out just
    # below 306,783,378 pore voxels — the guard below would fire instead of the
    # branch this exists to exercise.
    img = Array{Bool}(Imaginator.blobs(; shape=(800, 800, 800), porosity=0.65, blobiness=1, seed=1))
    count(img) > fld(typemax(Int32) - 1, 7) || error("fixture does not reach the bound")

    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    # Nobody asked for `Int64`; the image itself is what selects it. Comparing
    # against a `Ti=Int64`-forced build would prove nothing here — past the bound
    # `Int32` is refused outright, so both builds are the same code path, and the
    # second one costs another 24.4 GB to say so.
    eltype(SparseArrays.getcolptr(sim.prob.A)) === Int64 || return false
    # Every offset ascends, which is the property the bound exists to keep: a
    # wrapped `accumulate(+, counts)` goes negative, and `_steady_fill_kernel!`
    # then writes through `@inbounds` off the end of arrays sized from the
    # wrapped count.
    issorted(SparseArrays.getcolptr(sim.prob.A)) || return false

    c = reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1e-10).u, img)
    return tortuosity(c, img; axis=:x) > 1
end

@testset "the automatic 64-bit index path, end to end" begin
    # Needs ~26 GB before the solver allocates, ~36 GB with its Krylov vectors.
    # Skipped unconditionally — see above.
    @test_skip _automatic_wide_index_path_holds()
end

@testset "the Ti keyword is honoured or refused, never ignored" begin
    wall = fld(typemax(Int32) - 1, 7)

    @test Tortuosity._resolve_index_type(nothing, 1000) === Int32
    @test Tortuosity._resolve_index_type(nothing, wall + 1) === Int
    # Asking for more range than the image needs costs memory and nothing else.
    @test Tortuosity._resolve_index_type(Int64, 1000) === Int64
    @test Tortuosity._resolve_index_type(Int32, wall) === Int32
    # Granting Int32 past the bound would reinstate the wrap-around corruption
    # the bound exists to prevent; a keyword does not get to switch that off.
    @test_throws ArgumentError Tortuosity._resolve_index_type(Int32, wall + 1)
    @test_throws ArgumentError Tortuosity._resolve_index_type(Int16, 1000)
    @test_throws ArgumentError Tortuosity._resolve_index_type(Float64, 1000)

    img = ASSEMBLY_IMAGES[2][2]
    @test_throws ArgumentError SteadyDiffusionProblem(img; axis=:x, gpu=false, Ti=Int16)
    # `Ti` describes the assembled matrix; silently ignoring it on the
    # matrix-free path would leave a caller believing they got a wide index.
    @test_throws ArgumentError SteadyDiffusionProblem(
        img; axis=:x, gpu=false, matrixfree=true, Ti=Int64
    )
end

@testset "Ti=Int64 changes the index width and nothing else — $(label)" for
        (label, img) in ASSEMBLY_IMAGES
    n = count(img)
    for (axis, D) in ((:x, nothing), (:y, nothing), (:z, fill(2.0, size(img)) .* img))
        A32, b32 = build_steady_system(img; nnodes=n, axis=axis, D=D)
        A64, b64 = build_steady_system(img; nnodes=n, axis=axis, D=D, Ti=Int64)

        @test A32 isa SparseMatrixCSC{Float64,Int32}
        @test A64 isa SparseMatrixCSC{Float64,Int64}
        # The wide build must be the narrow one widened: same pattern, same
        # values to the last bit. Anything else means an offset was computed
        # differently rather than merely stored differently.
        @test SparseArrays.getcolptr(A32) == SparseArrays.getcolptr(A64)
        @test rowvals(A32) == rowvals(A64)
        @test nonzeros(A32) == nonzeros(A64)
        @test b32 == b64
    end
end

@testset "open box reproduces the exact linear profile" begin
    # For a uniform open grid the discrete solution is the continuous one: the
    # second difference of a linear ramp vanishes, and the lateral no-flux
    # condition is satisfied identically. Any assembly error — a mis-scaled
    # weight, a dropped edge, a wrong boundary row — perturbs this.
    N = 12
    img = ones(Bool, N, N, N)
    for (ax, d) in zip((:x, :y, :z), (1, 2, 3))
        sim = SteadyDiffusionProblem(img; axis=ax, gpu=false)
        c = reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u, img)
        ramp = reshape(1 .- (0:(N - 1)) ./ (N - 1), ntuple(i -> i == d ? N : 1, 3))
        @test maximum(abs, c .- ramp) < 1e-8
    end
end

# --- Scalar diffusivity ---

# A scalar `D` is uniform diffusivity. The kernels already express that case as
# `D === nothing` with the weight carried in `D0`, so a scalar rides that path
# rather than being expanded into a grid-sized array of one repeated value. The
# executable statement of it: assembling with a scalar must produce exactly what
# assembling with an array holding that value on the pore space produces.
@testset "a scalar D assembles the pore-constant array — $(label)" for (label, img) in ASSEMBLY_IMAGES
    # k = 0.1 is in the list deliberately: `2k²/(k+k)` is exactly k for most
    # values and *not* for 0.1, so it is the one that distinguishes the two
    # paths' arithmetic from their answers.
    for k in (2.0, 0.1)
        D = zeros(size(img))
        D[img] .= k                # the array form requires zeros off the pore space

        scalar = SteadyDiffusionProblem(img; axis=:x, gpu=false, D=k, warn_nonpercolating=false)
        array = SteadyDiffusionProblem(img; axis=:x, gpu=false, D=D, warn_nonpercolating=false)
        # Same sparsity pattern exactly; same values to rounding. Not bit-identical,
        # and deliberately not asserted as such: the array path takes the harmonic
        # mean of k with itself at every face, which is a no-op mathematically but
        # not in floating point.
        @test scalar.prob.A.colptr == array.prob.A.colptr
        @test scalar.prob.A.rowval == array.prob.A.rowval
        @test scalar.prob.A.nzval ≈ array.prob.A.nzval
        @test scalar.prob.b ≈ array.prob.b

        # The scalar path is the sharper of the two, which is the point of routing
        # it through `D0` rather than expanding it into an array: `D0` *is* the
        # edge weight, so every off-diagonal entry is exactly -k with no
        # arithmetic in between.
        A = scalar.prob.A
        offdiag = [A.nzval[p] for j in 1:size(A, 2)
                   for p in A.colptr[j]:(A.colptr[j + 1] - 1) if A.rowval[p] != j]
        @test !isempty(offdiag)
        @test all(==(-k), offdiag)

        # The matrix-free operator takes the same route and holds no array for it.
        mf = SteadyDiffusionProblem(img; axis=:x, gpu=false, D=k, matrixfree=true,
                                    warn_nonpercolating=false)
        @test isnothing(mf.prob.A.D)
        @test mf.prob.A.D0 == k
        @test mf.prob.b == scalar.prob.b
    end
end

@testset "a scalar D sets the element type the way an array's does" begin
    # `A` follows `D`'s element type when one is given; a scalar is a `D`, so a
    # Float32 scalar must narrow the matrix exactly as a Float32 array would.
    img = ones(Bool, 6, 5, 4)
    D32 = fill(1.0f0, size(img))
    @test eltype(SteadyDiffusionProblem(img; axis=:x, gpu=false, D=1.0f0).prob.A) === Float32
    @test eltype(SteadyDiffusionProblem(img; axis=:x, gpu=false, D=D32).prob.A) === Float32
    @test eltype(SteadyDiffusionProblem(img; axis=:x, gpu=false, D=1.0).prob.A) === Float64
    # Omitting `D` is still the Float64 host default rather than anything the
    # scalar path leaks into it.
    @test eltype(SteadyDiffusionProblem(img; axis=:x, gpu=false).prob.A) === Float64
end

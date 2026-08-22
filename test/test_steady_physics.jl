# Physics invariants of the steady-state solve, and the closed forms it has to
# reproduce exactly.
#
# Most of these assert properties that must hold for *every* image and every
# implementation: flux conservation across all cross-sections, the discrete
# maximum principle, invariance under relabelling of axes, agreement with 1D
# resistor-network theory for layered diffusivity. They are the safety net for
# rewrites of the assembly or solver — an optimised path that returns a
# slightly-wrong field will break conservation or the maximum principle long
# before it visibly moves τ.
#
# The rest pin τ itself against geometries whose answer is known in closed form:
# a prismatic duct (τ = 1, F = 1/φ), a staircase channel (τ = n(n-1)/(N(N-1)),
# the one exact τ > 1 in the suite that comes from path length rather than from
# a diffusivity contrast), and a pore space that does not connect the two faces
# (no transport at all).

using Test
using Statistics
using Tortuosity
using Tortuosity:
    Imaginator,
    axis_dim,
    effective_diffusivity,
    flux,
    formation_factor,
    reconstruct_field,
    tortuosity

solve_steady(img; axis, D=nothing) = begin
    sim = SteadyDiffusionProblem(img; axis=axis, gpu=false, D=D)
    reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u, img)
end

function physics_fixtures()
    imgs = Tuple{String,Array{Bool,3}}[]
    push!(imgs, ("open 10x9x8", ones(Bool, 10, 9, 8)))
    let img = ones(Bool, 10, 10, 10)
        img[:, 1:5, :] .= false           # straight half-channel
        push!(imgs, ("half channel 10^3", img))
    end
    let img = ones(Bool, 12, 8, 8)
        img[4:8, 3:6, 3:6] .= false       # an interior obstruction the flow must detour around
        push!(imgs, ("obstructed 12x8x8", img))
    end
    for seed in (5, 23)
        img = Array{Bool}(Imaginator.blobs(; shape=(14, 14, 14), porosity=0.65, blobiness=1, seed=seed))
        img = Array{Bool}(Imaginator.trim_nonpercolating_paths(img; axis=:x))
        count(img) >= 50 || continue
        push!(imgs, ("trimmed blob 14^3 seed=$seed", img))
    end
    return imgs
end

const PHYSICS_IMAGES = physics_fixtures()

# Look fixtures up by name — an index would silently point at a different
# geometry if the list is ever reordered, and the two ducts in it are
# indistinguishable by their τ (both exactly 1).
fixture(name) = PHYSICS_IMAGES[findfirst(p -> p[1] == name, PHYSICS_IMAGES)][2]

# A prismatic channel of the given lateral cross-section running the full length
# of `ax`. Written axis-first so the same closed forms can be checked on all
# three transport directions rather than only on `:x`.
function prismatic_duct(ax, lateral; long=16, wide=8)
    d = axis_dim(ax)
    img = falses(ntuple(i -> i == d ? long : wide, 3))
    idx = Any[lateral[1], lateral[2]]
    insert!(idx, d, :)
    img[idx...] .= true
    return img
end

# Number of face-connected pore neighbours of each pore voxel, in column-major
# order. Used to prove a fixture really is the graph a closed form assumes.
function pore_degrees(img)
    nx, ny, nz = size(img)
    degrees = Int[]
    for c in findall(img)
        deg = 0
        for (di, dj, dk) in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
            i, j, k = c[1] + di, c[2] + dj, c[3] + dk
            (1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz) || continue
            img[i, j, k] && (deg += 1)
        end
        push!(degrees, deg)
    end
    return degrees
end

# --- Conservation and boundedness ---

@testset "steady flux is conserved across every cross-section — $(label)" for (label, img) in PHYSICS_IMAGES
    c = solve_steady(img; axis=:x)
    N = size(img, 1)
    fluxes = [flux(c, 1.0, 1.0, img, :x; ind=k) for k in 1:(N - 1)]
    # Divergence-free steady transport: the net flux through any plane normal
    # to the transport axis is the same. This is the single strongest scalar
    # check on the assembled operator plus the solve.
    @test maximum(fluxes) - minimum(fluxes) < 1e-8 * max(1.0, abs(mean(fluxes)))
    # …and therefore D_eff must not depend on where it is measured.
    Deffs = [effective_diffusivity(c, img; axis=:x, ind=k) for k in 1:(N - 1)]
    @test maximum(Deffs) - minimum(Deffs) < 1e-8 * max(1.0, abs(mean(Deffs)))
end

@testset "discrete maximum principle — $(label)" for (label, img) in PHYSICS_IMAGES
    c = solve_steady(img; axis=:x)
    vals = c[img]
    # The system matrix is an irreducibly diagonally dominant M-matrix, so the
    # solution can neither undershoot nor overshoot its boundary data.
    @test minimum(vals) >= -1e-9
    @test maximum(vals) <= 1 + 1e-9
    @test all(isnan, c[.!img])
end

@testset "inlet and outlet slices bracket every interior slice — $(label)" for (label, img) in PHYSICS_IMAGES
    # A corollary of the maximum principle that is cheap to check and would
    # catch a swapped inlet/outlet or a boundary row written to the wrong node.
    c = solve_steady(img; axis=:x)
    means = [mean(filter(!isnan, vec(selectdim(c, 1, k)))) for k in 1:size(img, 1)]
    @test means[1] ≈ 1.0 atol = 1e-9
    @test means[end] ≈ 0.0 atol = 1e-9
    @test all(means[end] - 1e-9 .<= means .<= means[1] + 1e-9)
end

@testset "an unobstructed box drops linearly, slice by slice" begin
    N = 10
    c = solve_steady(ones(Bool, N, 6, 6); axis=:x)
    means = [mean(vec(selectdim(c, 1, k))) for k in 1:N]
    @test means ≈ 1 .- (0:(N - 1)) ./ (N - 1) atol = 1e-9
    @test all(diff(means) .< 0)
end

# --- Symmetry and invariance ---

@testset "τ is invariant under relabelling the axes" begin
    # Permuting the image and the transport axis together must leave every
    # transport property untouched. This catches axis/dimension mix-ups that a
    # cubic test image would hide.
    img = ones(Bool, 11, 9, 7)
    img[3:8, 4:6, 2:5] .= false
    c_x = solve_steady(img; axis=:x)
    ref_tau = tortuosity(c_x, img; axis=:x)
    ref_Deff = effective_diffusivity(c_x, img; axis=:x)

    for perm in ((2, 1, 3), (3, 2, 1), (2, 3, 1))
        img_p = permutedims(img, perm)
        # `permutedims` moves original dimension 1 to wherever it appears in perm
        ax_p = (:x, :y, :z)[findfirst(==(1), perm)]
        c_p = solve_steady(img_p; axis=ax_p)
        @test tortuosity(c_p, img_p; axis=ax_p) ≈ ref_tau atol = 1e-8
        @test effective_diffusivity(c_p, img_p; axis=ax_p) ≈ ref_Deff atol = 1e-8
    end
end

@testset "reversing the image along the axis mirrors the solution" begin
    # c_reversed(x) must equal 1 - c(L - x): the problem is its own mirror
    # image with the boundary values swapped.
    img = ones(Bool, 12, 8, 8)
    img[4:9, 2:5, 3:7] .= false
    c = solve_steady(img; axis=:x)
    img_rev = reverse(img; dims=1)
    c_rev = solve_steady(img_rev; axis=:x)
    @test tortuosity(c_rev, img_rev; axis=:x) ≈ tortuosity(c, img; axis=:x) atol = 1e-8
    mirrored = 1 .- reverse(c; dims=1)
    @test maximum(abs, (c_rev .- mirrored)[img_rev]) < 1e-8
end

@testset "a symmetric geometry gives an antisymmetric field about the midplane" begin
    img = ones(Bool, 13, 8, 8)
    img[5:9, 3:6, 3:6] .= false            # symmetric about x = 7
    @test img == reverse(img; dims=1)
    c = solve_steady(img; axis=:x)
    @test maximum(abs, (c .+ reverse(c; dims=1) .- 1)[img]) < 1e-8
    # …and the midplane sits at exactly half the concentration drop.
    @test mean(filter(!isnan, vec(selectdim(c, 1, 7)))) ≈ 0.5 atol = 1e-8
end

@testset "2D input matches the equivalent 3D image with a singleton dimension" begin
    img2d = ones(Bool, 16, 16)
    img2d[5:12, 4:9] .= false
    img3d = reshape(img2d, 16, 16, 1)
    # Both in-plane axes: `atleast_3d` promotes to (m, n, 1), so :x and :y are
    # the two directions a 2D image can actually be transported along.
    for ax in (:x, :y)
        c2 = solve_steady(img2d; axis=ax)
        c3 = solve_steady(img3d; axis=ax)
        @test isequal(vec(c2), vec(c3))
        @test tortuosity(c2, img2d; axis=ax) ≈ tortuosity(c3, img3d; axis=ax)
    end

    # And the 2D closed form is the 3D one: a prismatic channel of open
    # fraction φ has τ = 1 and F = 1/φ.
    duct2d = falses(16, 8)
    duct2d[:, 3:6] .= true
    φ = count(duct2d) / length(duct2d)
    c = solve_steady(duct2d; axis=:x)
    @test tortuosity(c, duct2d; axis=:x) ≈ 1.0 atol = 1e-9
    @test formation_factor(c, duct2d; axis=:x) ≈ 1 / φ rtol = 1e-9
end

# --- Scaling laws ---

@testset "D_eff scales linearly with the intrinsic diffusivity" begin
    img = fixture("obstructed 12x8x8")
    c = solve_steady(img; axis=:x)
    D_ref = effective_diffusivity(c, img; axis=:x)
    for k in (0.25, 2.0, 100.0)
        @test effective_diffusivity(c, img; axis=:x, D=k) ≈ k * D_ref atol = 1e-8 * k
    end
end

@testset "a uniform D cancels out of the concentration field but scales D_eff" begin
    # ∇·(D∇c) = 0 with constant D reduces to ∇²c = 0, so a scalar `D` handed to
    # the constructor scales the matrix and the right-hand side by the same
    # factor and leaves the solution where D = 1 puts it. It shows up in D_eff,
    # which is what carries the physical units, and divides back out of τ.
    img = fixture("obstructed 12x8x8")
    c_unit = solve_steady(img; axis=:x)
    D_unit = effective_diffusivity(c_unit, img; axis=:x)
    for k in (0.25, 2.0, 100.0)
        sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, D=k)
        c = reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u, img)
        # Same field, to the solver's tolerance rather than to the bit: CG on the
        # scaled system takes a slightly different floating-point path to it.
        @test maximum(abs, (c .- c_unit)[img]) < 1e-8
        @test effective_diffusivity(c, img; axis=:x, D=k) ≈ k * D_unit rtol = 1e-8
        @test tortuosity(c, img; axis=:x, D=k) ≈ tortuosity(c_unit, img; axis=:x) rtol = 1e-8
    end
end

@testset "D_eff is independent of voxel_size (it is a dimensionless ratio here)" begin
    img = fixture("half channel 10^3")
    c = solve_steady(img; axis=:x)
    D_ref = effective_diffusivity(c, img; axis=:x)
    for vs in (0.1, 1.0, 25.0)
        @test effective_diffusivity(c, img; axis=:x, voxel_size=vs) ≈ D_ref atol = 1e-8
    end
end

@testset "τ and F are invariant under a rescaling of the diffusivity" begin
    # D_eff is homogeneous of degree one in D, so τ and F only describe the
    # geometry if the reference diffusivity divides back out. Without that,
    # handing the solver a physical D rather than a normalised one rescales τ
    # and can drive it below 1, which no geometry can do.
    img = fixture("obstructed 12x8x8")
    c = solve_steady(img; axis=:x)
    τ_ref = tortuosity(c, img; axis=:x)
    F_ref = formation_factor(c, img; axis=:x)
    for k in (0.25, 2.0, 100.0)
        @test tortuosity(c, img; axis=:x, D=k) ≈ τ_ref rtol = 1e-8
        @test formation_factor(c, img; axis=:x, D=k) ≈ F_ref rtol = 1e-8
    end
    @test τ_ref >= 1

    # Same invariance for a per-voxel field, where the reference is the largest
    # pore-phase value rather than the scalar itself.
    open_box = ones(Bool, 10, 6, 6)
    D = fill(1.0, size(open_box))
    D[:, 2:4, :] .= 0.3
    c1 = solve_steady(open_box; axis=:x, D=D)
    τ_field = tortuosity(c1, open_box; axis=:x, D=D)
    k = 7.0
    c2 = solve_steady(open_box; axis=:x, D=k .* D)
    @test tortuosity(c2, open_box; axis=:x, D=k .* D) ≈ τ_field rtol = 1e-8

    # An explicit D0 overrides the default reference.
    @test tortuosity(c, img; axis=:x, D=2.0, D0=1.0) ≈ τ_ref / 2 rtol = 1e-8
    @test formation_factor(c, img; axis=:x, D=2.0, D0=1.0) ≈ F_ref / 2 rtol = 1e-8
end

@testset "the reference diffusivity is the largest value in the pore space" begin
    # Every other test of `D0` compares ratios, which any reducer would satisfy.
    # This one pins the value, against a geometry with a closed form.
    #
    # A prismatic duct whose diffusivity varies laterally but not along the
    # transport axis is a set of independent 1D chains in parallel: the
    # concentration profile is the same linear drop in every column, so no
    # lateral gradient exists for the harmonic-mean face weights to act on and
    # D_eff is exactly the mean of D over the whole cross-section, solid counted
    # as zero. There is no discretisation error to hide in.
    img = falses(16, 8, 8)
    img[:, 3:6, 3:6] .= true          # a 4x4 duct, φ = 16/64
    D = zeros(size(img))
    D[:, 3:4, 3:6] .= 2.0             # half the duct fast…
    D[:, 5:6, 3:6] .= 0.6             # …half of it slow
    ε = count(img) / length(img)
    @test ε == 0.25

    c = solve_steady(img; axis=:x, D=D)
    D_eff_exact = (8 * 2.0 + 8 * 0.6) / 64
    @test effective_diffusivity(c, img; axis=:x, D=D) ≈ D_eff_exact rtol = 1e-8
    # τ = D0·ε/D_eff with D0 = max(D over the pore space) = 2.0. Taking the
    # minimum instead gives 0.46 and the mean gives 1.0; dropping D0 altogether
    # gives 0.77. Two of those three are below 1, which no geometry can produce.
    @test tortuosity(c, img; axis=:x, D=D) ≈ 2.0 * ε / D_eff_exact rtol = 1e-8
    @test formation_factor(c, img; axis=:x, D=D) ≈ 2.0 / D_eff_exact rtol = 1e-8
    @test tortuosity(c, img; axis=:x, D=D) > 1

    # The solid phase cannot set the scale. These voxels carry no flux, so a
    # reference taken over the whole array rather than over the pore space would
    # move τ by a factor of 25 while leaving D_eff untouched.
    D_hot_solid = copy(D)
    D_hot_solid[.!img] .= 50.0
    @test maximum(D_hot_solid) == 50.0
    @test tortuosity(c, img; axis=:x, D=D_hot_solid) == tortuosity(c, img; axis=:x, D=D)
    @test formation_factor(c, img; axis=:x, D=D_hot_solid) ==
          formation_factor(c, img; axis=:x, D=D)
end

@testset "scaling the whole diffusivity field scales D_eff by the same factor" begin
    # Exercises the assembly, not just the post-processing: interpolate_edge_values
    # is homogeneous of degree one, so k·D in ⇒ k·D_eff out.
    img = ones(Bool, 10, 6, 6)
    D = fill(1.0, size(img))
    D[:, 2:4, :] .= 0.3
    c1 = solve_steady(img; axis=:x, D=D)
    D_ref = effective_diffusivity(c1, img; axis=:x, D=D)
    k = 7.0
    c2 = solve_steady(img; axis=:x, D=k .* D)
    @test effective_diffusivity(c2, img; axis=:x, D=k .* D) ≈ k * D_ref rtol = 1e-8
    # The concentration field itself is unchanged by a uniform rescaling of D.
    @test maximum(abs, (c1 .- c2)[img]) < 1e-8
end

# --- Exact analytic references ---

@testset "layered diffusivity matches 1D resistors in series" begin
    # A fully-open box whose diffusivity varies only along the transport axis
    # reduces exactly to a chain of conductances g_i = 2·D_i·D_{i+1}/(D_i+D_{i+1})
    # — the same harmonic mean the finite-volume scheme uses at each face. There
    # is no discretisation error to hide behind, so this pins the variable-D
    # assembly path to an analytic answer.
    N = 12
    img = ones(Bool, N, 4, 4)
    layers = [1.0, 5.0, 0.2, 3.0, 0.05, 2.0, 1.5, 8.0, 0.4, 1.0, 6.0, 0.3]
    D = repeat(reshape(layers, N, 1, 1), 1, 4, 4)

    g = [2 * layers[i] * layers[i + 1] / (layers[i] + layers[i + 1]) for i in 1:(N - 1)]
    G = 1 / sum(1 ./ g)
    D_eff_exact = G * (N - 1)

    c = solve_steady(img; axis=:x, D=D)
    for ind in 1:(N - 1)
        @test effective_diffusivity(c, img; axis=:x, ind=ind, D=D) ≈ D_eff_exact rtol = 1e-8
    end

    # The nodal concentrations follow the resistor-chain potential drops.
    I = G
    c_exact = [1.0; 1.0 .- cumsum(I ./ g)]
    @test maximum(abs, c[:, 1, 1] .- c_exact) < 1e-8
end

@testset "a straight duct has τ = 1 and D_eff = φ exactly — axis=$(ax)" for ax in (:x, :y, :z)
    # Independent of duct cross-section and of transport direction: a prismatic
    # channel adds no tortuosity. The last cross-section is the whole domain, so
    # this also pins τ = 1, D_eff = 1 and F = 1 for a fully open box on every
    # axis, in both 3D and (below) 2D.
    for lateral in ((3:6, 3:6), (2:3, 5:8), (1:8, 1:2), (1:8, 1:8))
        img = prismatic_duct(ax, lateral)
        φ = count(img) / length(img)
        c = solve_steady(img; axis=ax)
        @test effective_diffusivity(c, img; axis=ax) ≈ φ atol = 1e-9
        @test tortuosity(c, img; axis=ax) ≈ 1.0 atol = 1e-9
        @test formation_factor(c, img; axis=ax) ≈ 1 / φ rtol = 1e-9
    end
end

# The one closed form in the suite with τ > 1 that comes from the *geometry*
# rather than from a diffusivity contrast. Everything else that pins an exact τ
# does so at τ = 1, so a bug that scaled τ by a path-length factor — a wrong
# domain length L, a flux normalised by the pore count instead of the slice
# area, an ε taken over the wrong denominator — would leave every one of them
# green while moving every reported τ on a real image.
#
# The channel advances monotonically along the transport axis while wandering
# laterally: a full lateral column at even i, a single voxel at odd i, alternating
# which end of the column that voxel sits at. Consecutive groups share exactly one
# face, so the pore space is a simple path; and because i never decreases, each
# plane normal to the axis is crossed by exactly one edge.
function staircase_channel(N, width)
    @assert isodd(N)
    img = falses(N, width, 1)
    for i in 1:N
        if iseven(i)
            img[i, :, 1] .= true
        else
            img[i, isodd((i + 1) ÷ 2) ? 1 : width, 1] = true
        end
    end
    return img
end

@testset "a staircase channel matches the series-resistor τ — $(N)x$(width)" for
        (N, width) in ((5, 3), (5, 5), (9, 3), (9, 5))
    img = staircase_channel(N, width)
    n = count(img)

    # The closed forms below describe a chain of n-1 unit conductances, so they
    # only hold while the pore space *is* that chain. Asserting it here makes a
    # fixture that grew a shortcut fail as itself rather than as a wrong τ.
    degrees = pore_degrees(img)
    @test maximum(degrees) == 2
    @test count(==(1), degrees) == 2      # exactly two ends
    @test n > N                           # the path is genuinely longer than the domain

    c = solve_steady(img; axis=:x)
    # Series chain: the current is I = 1/(n-1) and it crosses each plane exactly
    # once, so J = I/(width·1), L = N-1 and dc = 1. Hence
    # D_eff = (N-1)/((n-1)·width) and τ = ε/D_eff = n(n-1)/(N(N-1)), which for
    # n ≈ path length ℓ is the textbook (ℓ/L)².
    @test effective_diffusivity(c, img; axis=:x) ≈ (N - 1) / ((n - 1) * width) rtol = 1e-9
    @test tortuosity(c, img; axis=:x) ≈ n * (n - 1) / (N * (N - 1)) rtol = 1e-9
    @test formation_factor(c, img; axis=:x) ≈ (n - 1) * width / (N - 1) rtol = 1e-9
    @test tortuosity(c, img; axis=:x) > 1
end

@testset "a pore space that does not connect inlet to outlet carries no flux" begin
    # Two one-voxel ducts that meet only at a corner. Face connectivity — the
    # only kind this package's stencil has — leaves them disconnected; a
    # 26-neighbour connectivity, or an off-by-one that let a stencil reach a
    # diagonal, would join them and report ordinary transport through a geometry
    # that has none.
    blocked = falses(12, 6, 6)
    blocked[1:6, 3, 3] .= true
    blocked[7:12, 4, 4] .= true            # touches (6, 3, 3) at a corner only
    joined = falses(12, 6, 6)
    joined[1:12, 3, 3] .= true             # the same voxel count, sharing a face
    @test count(joined) == count(blocked)

    sim = SteadyDiffusionProblem(blocked; axis=:x, gpu=false, warn_nonpercolating=false)
    c = reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u, blocked)
    # Each cluster is pinned by the single Dirichlet face it touches, so the
    # field is the boundary data exactly and nothing flows anywhere.
    @test all(isapprox.(c[1:6, 3, 3], 1.0; atol=1e-9))
    @test all(isapprox.(c[7:12, 4, 4], 0.0; atol=1e-9))
    @test maximum(abs, [flux(c, 1.0, 1.0, blocked, :x; ind=k) for k in 1:11]) < 1e-12

    c_joined = solve_steady(joined; axis=:x)
    D_joined = effective_diffusivity(c_joined, joined; axis=:x)
    @test D_joined ≈ count(joined) / length(joined) atol = 1e-9   # a straight duct
    # Same porosity, same voxels, one face apart: transport is not merely small
    # but zero to solver precision.
    @test abs(effective_diffusivity(c, blocked; axis=:x)) < 1e-6 * D_joined
    # …so τ diverges rather than coming back as a plausible finite number.
    @test abs(tortuosity(c, blocked; axis=:x)) > 1e6
    @test tortuosity(c_joined, joined; axis=:x) ≈ 1.0 atol = 1e-9
end

@testset "parallel ducts add their conductances" begin
    img = falses(16, 8, 8)
    img[:, 2:3, 2:3] .= true
    img[:, 6:7, 6:7] .= true       # a second, disjoint duct
    φ = count(img) / length(img)
    c = solve_steady(img; axis=:x)
    @test effective_diffusivity(c, img; axis=:x) ≈ φ atol = 1e-9
    @test tortuosity(c, img; axis=:x) ≈ 1.0 atol = 1e-9
end

@testset "a dead-end pocket raises porosity but not D_eff" begin
    # Steady-state transport ignores stagnant volume: adding a blind side
    # branch must leave D_eff bit-for-bit where it was while τ rises by exactly
    # the porosity ratio. A solver that leaks flux into dead ends fails here.
    #
    # The finger touches the duct at exactly one voxel, (8,6,4), so it is a true
    # dead end. A pocket attached along several voxels would also carry no
    # current, but only because those attachment points happen to be
    # iso-potential — a weaker property that stops holding as soon as the pocket
    # spans more than one plane normal to the transport axis.
    base = falses(16, 8, 8)
    base[:, 3:6, 3:6] .= true
    pocket = copy(base)
    pocket[8, 7:8, 4] .= true
    @test count(pocket) == count(base) + 2

    c_base = solve_steady(base; axis=:x)
    c_pocket = solve_steady(pocket; axis=:x)

    D_base = effective_diffusivity(c_base, base; axis=:x)
    D_pocket = effective_diffusivity(c_pocket, pocket; axis=:x)
    @test D_pocket ≈ D_base atol = 1e-9
    @test D_base ≈ count(base) / length(base) atol = 1e-9

    ε_base = count(base) / length(base)
    ε_pocket = count(pocket) / length(pocket)
    @test tortuosity(c_pocket, pocket; axis=:x) ≈ ε_pocket / D_base atol = 1e-9
    @test tortuosity(c_pocket, pocket; axis=:x) > tortuosity(c_base, base; axis=:x)

    # Both finger voxels equilibrate to the single duct voxel they hang off.
    @test abs(c_pocket[8, 7, 4] - c_pocket[8, 6, 4]) < 1e-9
    @test abs(c_pocket[8, 8, 4] - c_pocket[8, 6, 4]) < 1e-9
end

# --- Relations between the reported quantities ---

@testset "τ, D_eff and F are mutually consistent — $(label)" for (label, img) in PHYSICS_IMAGES
    # Fixtures are percolating along :x by construction, so the solve is
    # well-posed for every one of them on this axis.
    c = solve_steady(img; axis=:x)
    ε = Imaginator.phase_fraction(img, true)
    Deff = effective_diffusivity(c, img; axis=:x)
    @test formation_factor(c, img; axis=:x) ≈ 1 / Deff rtol = 1e-12
    @test tortuosity(c, img; axis=:x) ≈ ε / Deff rtol = 1e-12
    # An explicitly supplied porosity overrides the one derived from the image.
    @test tortuosity(c, img; axis=:x, ε=0.5) ≈ 0.5 / Deff rtol = 1e-12
    # Physical bounds: transport can never beat the fully-open limit, so τ ≥ 1.
    @test 0 < Deff <= ε + 1e-9
    @test tortuosity(c, img; axis=:x) >= 1 - 1e-9
end

@testset "every axis of an isotropic geometry gives the same τ" begin
    # A cube with a centred cubic obstruction is symmetric under axis swaps,
    # so all three transport directions must agree exactly.
    img = ones(Bool, 12, 12, 12)
    img[4:9, 4:9, 4:9] .= false
    taus = map((:x, :y, :z)) do ax
        tortuosity(solve_steady(img; axis=ax), img; axis=ax)
    end
    @test maximum(taus) - minimum(taus) < 1e-8
    @test all(taus .> 1)          # an obstruction must cost something
end

# Golden-value regression guard for irregular geometries.
#
# The rest of the suite checks properties (conservation, symmetry, analytic
# limits). Those are strong, but a porous blob has no closed-form τ, and it is
# possible to imagine an assembly bug that preserves every invariant while still
# shifting the answer — a mis-weighted edge that stays symmetric, say, or a
# connectivity list that drops a matched pair of edges.
#
# This file pins the actual numbers on a small set of deterministic fixtures.
# It exists to make "faster but wrong" impossible to miss during optimisation
# work: any change to these values is either a bug or a deliberate improvement
# that needs its constants updated, with a note in the commit saying why.
#
# The values were produced by the implementation at the time of writing, with
# `KrylovJL_CG(reltol=1e-10)`; the comparison tolerance (rtol = 1e-6) is far
# looser than the solver tolerance, so ordinary floating-point and BLAS
# variation across machines will not trip it.

using Test
using Statistics
using Tortuosity
using Tortuosity: Imaginator, effective_diffusivity, reconstruct_field, tortuosity

# Percolating in all three directions, so every axis is a well-posed problem.
function golden_fixture(seed)
    img = BitArray(Imaginator.blobs(; shape=(24, 24, 24), porosity=0.6, blobiness=1, seed=seed))
    for ax in (:x, :y, :z)
        img = BitArray(Imaginator.trim_nonpercolating_paths(img; axis=ax))
    end
    return img
end

# seed => (nnodes, axis => (τ, mean pore concentration))
const GOLDEN_STEADY = Dict(
    1 => (8116, Dict(
        :x => (2.348236854010, 0.530387945271),
        :y => (2.244957574979, 0.499520112524),
        :z => (2.241060706896, 0.495017726086),
    )),
    42 => (8066, Dict(
        :x => (2.434863973495, 0.520306827112),
        :y => (2.336647897669, 0.513123093040),
        :z => (2.350559187560, 0.466861407905),
    )),
    100 => (8113, Dict(
        :x => (2.250899030511, 0.512215192731),
        :y => (2.294031446393, 0.515731327397),
        :z => (2.161104308778, 0.495357719559),
    )),
)

@testset "golden steady-state values — blobs seed=$(seed)" for seed in sort(collect(keys(GOLDEN_STEADY)))
    nnodes_expected, per_axis = GOLDEN_STEADY[seed]
    img = golden_fixture(seed)

    # The fixture itself must be reproducible, or the τ comparison below is
    # meaningless — so this is deliberately a hard equality, not a tolerance.
    #
    # It is also the line most likely to break for an innocent reason: the
    # fixture depends on Julia's `Xoshiro` stream, `ImageFiltering.imfilter`,
    # and `ImageMorphology.label_components`, none of which promise stability
    # across versions. If a Julia or image-package bump trips this, the image
    # changed and ALL the constants below are stale: regenerate the whole table
    # rather than nudging tolerances, and say so in the commit message.
    @test count(img) == nnodes_expected

    for ax in (:x, :y, :z)
        τ_expected, c̄_expected = per_axis[ax]
        sim = SteadyDiffusionProblem(img; axis=ax, gpu=false)
        u = solve(sim.prob, KrylovJL_CG(); reltol=1e-10).u
        c = reconstruct_field(u, img)
        @test tortuosity(c, img; axis=ax) ≈ τ_expected rtol = 1e-6
        # A second, independent scalar summary of the same field: τ depends only
        # on the flux at one face, while the mean sees every voxel.
        @test mean(u) ≈ c̄_expected rtol = 1e-6
    end
end

# The variable-diffusivity path solves on a fully-open domain with the pore
# structure encoded in D, so it exercises `interpolate_edge_values` and the
# edge-weighted assembly rather than the connectivity mask.
const GOLDEN_VARIABLE_D = Dict(:x => 0.242105032300, :z => 0.250603113034)

@testset "golden variable-diffusivity values — $(ax)-axis" for ax in (:x, :z)
    img = golden_fixture(42)
    D = zeros(size(img))
    D[img] .= 1.0
    D[.!img] .= 1e-3
    domain = ones(Bool, size(img))

    sim = SteadyDiffusionProblem(domain; axis=ax, gpu=false, D=D)
    u = solve(sim.prob, KrylovJL_CG(); reltol=1e-10).u
    c = reconstruct_field(u, domain)
    @test effective_diffusivity(c, domain; axis=ax, D=D) ≈ GOLDEN_VARIABLE_D[ax] rtol = 1e-6
end

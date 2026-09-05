# Tests for the cavern (dead-end) detection routines in caverns.jl.
#
# `flux_out` is a pure stencil, so it is pinned against exact hand-computed
# values on a linear ramp — the one field where every edge difference is known
# in closed form. `find_caverns` is then checked on geometries where the answer
# is unambiguous: a prismatic duct has no stagnant volume at all, and a blind
# side branch is stagnant in its entirety.

using Test
using Tortuosity
using Tortuosity: Imaginator, find_caverns, flux_out

# --- flux_out ---

@testset "flux_out" begin
    @testset "every axis contributes |Δc| once per face-connected neighbour" begin
        # Each of the three directions must be summed independently, so the
        # probe field carries a *different* gradient along each axis. A ramp
        # that varies only in x would leave Fy and Fz identically zero and the
        # test would still pass with those two accumulations deleted entirely.
        nx, ny, nz = 6, 4, 5
        a, b, d = 0.75, 0.20, 1.30
        img = trues(nx, ny, nz)
        c = [a * (i - 1) + b * (j - 1) + d * (k - 1) for i in 1:nx, j in 1:ny, k in 1:nz]

        # A voxel picks up its axial gradient twice in the interior and once at
        # each end face, where it has only one neighbour along that axis.
        w(idx, n) = (idx == 1 || idx == n) ? 1 : 2
        expected = [
            a * w(i, nx) + b * w(j, ny) + d * w(k, nz)
            for i in 1:nx, j in 1:ny, k in 1:nz
        ]
        @test flux_out(c, img) ≈ expected
    end

    @testset "a uniform field has no flux anywhere" begin
        img = trues(5, 5, 5)
        @test all(flux_out(fill(3.0, 5, 5, 5), img) .== 0)
    end

    @testset "depends only on differences, not on offset or sign" begin
        img = trues(6, 4, 4)
        c = [0.3 * i + 0.1 * j for i in 1:6, j in 1:4, k in 1:4]
        base = flux_out(c, img)
        @test flux_out(c .+ 17.0, img) ≈ base
        @test flux_out(-c, img) ≈ base
    end

    @testset "solid voxels carry no flux and NaN fills do not leak" begin
        # `reconstruct_field` NaN-fills solids; the mask multiplication relies on
        # `false * NaN === 0.0` to keep those out of the sum.
        img = trues(8, 5, 5)
        img[4, 2:4, 2:4] .= false
        c = fill(NaN, size(img))
        c[img] .= [0.5 * idx for idx in 1:count(img)]
        F = flux_out(c, img)
        @test all(iszero, F[.!img])
        @test all(isfinite, F)
    end

    @testset "an isolated pore voxel has zero flux" begin
        img = falses(5, 5, 5)
        img[3, 3, 3] = true
        c = fill(NaN, size(img))
        c[3, 3, 3] = 1.0
        @test flux_out(c, img)[3, 3, 3] == 0
    end

    @test_throws AssertionError flux_out(zeros(3, 3, 3), trues(4, 4, 4))
end

# --- find_caverns ---

@testset "find_caverns" begin
    @testset "a prismatic duct has no stagnant volume" begin
        img = falses(16, 8, 8)
        img[:, 3:6, 3:6] .= true
        caverns, cavern_fraction = find_caverns(BitArray(img); vmin=-2, iter=1,
                                                axis=:x, reltol=1e-10, gpu=false)
        @test !any(caverns)
        @test cavern_fraction == [0.0, 0.0]
    end

    @testset "a blind side branch is classified as a cavern" begin
        # A genuine dead end: a two-voxel finger touching the duct at exactly
        # one voxel, (8,6,4). It equilibrates to that voxel's concentration, so
        # every edge inside it carries (numerically) zero flux. Attaching along
        # several voxels instead would make it a parallel loop that happens to
        # be current-free only because its endpoints are iso-potential — true
        # here, but for a different and more fragile reason.
        img = falses(16, 8, 8)
        img[:, 3:6, 3:6] .= true
        pocket = falses(size(img))
        pocket[8, 7:8, 4] .= true
        img .|= pocket

        caverns, cavern_fraction = find_caverns(BitArray(img); vmin=-2, iter=1,
                                                axis=:x, reltol=1e-10, gpu=false)
        @test caverns == BitArray(pocket)
        @test cavern_fraction[1] == 0.0
        @test cavern_fraction[2] ≈ count(pocket) / count(img)
    end

    @testset "raising vmin classifies more volume" begin
        img = falses(16, 8, 8)
        img[:, 3:6, 3:6] .= true
        img[8, 7:8, 4] .= true
        strict, _ = find_caverns(BitArray(img); vmin=-2, iter=1, axis=:x,
                                 reltol=1e-10, gpu=false)
        loose, _ = find_caverns(BitArray(img); vmin=1.0, iter=1, axis=:x,
                                reltol=1e-10, gpu=false)
        # A threshold above the duct's own log-flux sweeps up the duct as well.
        @test count(loose) > count(strict)
    end

    @testset "results stay inside the pore space and accumulate over iterations" begin
        img = BitArray(Imaginator.blobs(; shape=(16, 16, 16), porosity=0.6, blobiness=1, seed=4))
        img = BitArray(Imaginator.trim_nonpercolating_paths(img; axis=:z))
        count(img) > 100 || @warn "cavern fixture unexpectedly small"

        iters = 3
        caverns, cavern_fraction = find_caverns(img; vmin=-2, iter=iters, axis=:z,
                                                reltol=1e-8, gpu=false)
        @test size(caverns) == size(img)
        @test all(caverns .<= img)                 # never marks solid voxels
        @test length(cavern_fraction) == iters + 1
        @test cavern_fraction[1] == 0.0
        # Cavern voxels are only ever added, never cleared.
        @test issorted(cavern_fraction)
        @test all(0 .<= cavern_fraction .<= 1)
        @test cavern_fraction[end] ≈ count(caverns) / count(img)
    end
end

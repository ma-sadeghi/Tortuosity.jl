# Tests for the Imaginator image-generation submodule.
#
# Every other test file builds its fixtures from `Imaginator.blobs`, so a silent
# change in its output would move the "ground truth" of the whole suite without
# any test failing on its own terms. Reproducibility from a seed is therefore
# tested first and hardest. The remaining routines are checked against their
# definitions (exact structuring elements, order-preserving normalisation,
# percolation semantics) rather than against snapshots.

using Random
using Test
using Statistics
using Tortuosity
using Tortuosity.Imaginator:
    apply_gaussian_blur,
    ball,
    blobs,
    denoise,
    disk,
    faces,
    norm_to_uniform,
    phase_fraction,
    to_binary,
    trim_nonpercolating_paths,
    _count_percolating

# Count pore/solid face contacts — a proxy for feature fineness.
function interface_area(img)
    a = count(img[1:(end - 1), :, :] .!= img[2:end, :, :])
    b = count(img[:, 1:(end - 1), :] .!= img[:, 2:end, :])
    c = count(img[:, :, 1:(end - 1)] .!= img[:, :, 2:end])
    return a + b + c
end

# --- blobs ---

@testset "blobs" begin
    @testset "a seed makes the image reproducible" begin
        # Fixtures across the whole suite depend on this.
        a = blobs(; shape=(24, 24, 24), porosity=0.6, blobiness=1, seed=42)
        b = blobs(; shape=(24, 24, 24), porosity=0.6, blobiness=1, seed=42)
        @test a == b
        c = blobs(; shape=(24, 24, 24), porosity=0.6, blobiness=1, seed=43)
        @test a != c
    end

    @testset "shape and element type" begin
        img3 = blobs(; shape=(8, 9, 10), porosity=0.5, blobiness=1, seed=1)
        @test size(img3) == (8, 9, 10)
        @test eltype(img3) === Bool
        img2 = blobs(; shape=(16, 12), porosity=0.5, blobiness=1, seed=1)
        @test size(img2) == (16, 12)
    end

    @testset "realised porosity tracks the requested porosity" begin
        # `norm_to_uniform` maps the blurred noise onto a uniform distribution,
        # so thresholding at `porosity` selects that fraction of voxels.
        for ε in (0.3, 0.5, 0.7)
            img = blobs(; shape=(40, 40, 40), porosity=ε, blobiness=1, seed=7)
            @test phase_fraction(img, true) ≈ ε atol = 0.05
        end
    end

    @testset "higher blobiness produces finer features" begin
        coarse = blobs(; shape=(48, 48, 48), porosity=0.5, blobiness=0.5, seed=11)
        fine = blobs(; shape=(48, 48, 48), porosity=0.5, blobiness=2.0, seed=11)
        # Same porosity, but finer structure means far more pore/solid contact.
        @test phase_fraction(fine, true) ≈ phase_fraction(coarse, true) atol = 0.05
        @test interface_area(fine) > interface_area(coarse)
    end
end

# --- Value transforms ---

@testset "norm_to_uniform" begin
    # Explicit RNG: `blobs` seeds the *global* RNG, so a bare `randn` here would
    # be deterministic only as long as this testset runs after the blobs one.
    x = randn(MersenneTwister(31337), 20, 20, 20)

    @testset "spans exactly the requested range" begin
        y = norm_to_uniform(x; scale=(0, 1))
        @test minimum(y) ≈ 0.0 atol = 1e-12
        @test maximum(y) ≈ 1.0 atol = 1e-12
        z = norm_to_uniform(x; scale=(-3.0, 7.0))
        @test minimum(z) ≈ -3.0 atol = 1e-12
        @test maximum(z) ≈ 7.0 atol = 1e-12
    end

    @testset "is order-preserving" begin
        # The transform is a chain of monotone maps (standardise → Gaussian CDF
        # → affine rescale), so voxel ranking must be untouched. This is what
        # makes thresholding at `porosity` select the right fraction.
        y = norm_to_uniform(x; scale=(0, 1))
        @test sortperm(vec(y)) == sortperm(vec(x))
    end

    @testset "the output is roughly uniformly distributed" begin
        y = norm_to_uniform(randn(MersenneTwister(90210), 60, 60, 60); scale=(0, 1))
        # Deciles of a uniform sample land near their nominal positions.
        for q in 0.1:0.1:0.9
            @test quantile(vec(y), q) ≈ q atol = 0.05
        end
    end
end

@testset "to_binary" begin
    x = [0.0 0.25; 0.5 0.75]
    @test to_binary(x, 0.5) == Bool[1 1; 0 0]
    # Strictly less than: a value equal to the threshold is *not* pore.
    @test to_binary([0.5], 0.5) == [false]
    @test to_binary(x) == to_binary(x, 0.5)          # default threshold
    @test to_binary(x, 0.0) == Bool[0 0; 0 0]
    @test to_binary(x, 1.0) == Bool[1 1; 1 1]
end

@testset "apply_gaussian_blur" begin
    x = zeros(21, 21)
    x[11, 11] = 1.0
    y = apply_gaussian_blur(x, 2.0)
    @test size(y) == size(x)
    @test argmax(y) == CartesianIndex(11, 11)        # peak stays put
    @test y[11, 11] < 1.0                            # energy spreads out
    @test all(y .>= -1e-12)
    # Symmetric kernel on a symmetric input gives a symmetric result.
    @test y ≈ reverse(y; dims=1)
    @test y ≈ reverse(y; dims=2)
end

# --- Structuring elements ---

@testset "disk / ball" begin
    @testset "exact small-radius shapes" begin
        @test disk(1) == Bool[0 1 0; 1 1 1; 0 1 0]
        @test count(disk(2)) == 13                   # offsets with i² + j² ≤ 4
        @test count(ball(1)) == 7                    # centre plus six faces
    end

    @testset "size, centring and symmetry" begin
        for r in 1:4
            d = disk(r)
            @test size(d) == (2r + 1, 2r + 1)
            @test d[r + 1, r + 1]                    # centre included
            @test !d[1, 1]                           # corner excluded
            @test d == reverse(d; dims=1) == reverse(d; dims=2)
            @test d == permutedims(d)                # isotropic

            b = ball(r)
            @test size(b) == (2r + 1, 2r + 1, 2r + 1)
            @test b[r + 1, r + 1, r + 1]
            @test !b[1, 1, 1]
            @test b == reverse(b; dims=3)
        end
    end

    @testset "volume approaches the continuous sphere/circle" begin
        @test count(disk(12)) / (π * 12^2) ≈ 1 atol = 0.1
        @test count(ball(8)) / (4 / 3 * π * 8^3) ≈ 1 atol = 0.1
    end
end

@testset "denoise removes specks but keeps bulk features" begin
    img = falses(15, 15, 15)
    img[8, 8, 8] = true                              # isolated single voxel
    img[2:6, 2:6, 2:6] .= true                       # a solid 5³ block
    cleaned = denoise(img, 1)
    # Opening erodes the lone voxel away entirely, leaving its neighbourhood
    # empty; the block is far larger than the structuring element and survives.
    @test !any(cleaned[7:9, 7:9, 7:9])
    @test all(cleaned[3:5, 3:5, 3:5])
end

# --- Face masks ---

@testset "faces" begin
    shape = (4, 5, 6)

    @testset "inlet marks the first slice of the given dimension" begin
        m = faces(shape; inlet=2)
        @test size(m) == shape
        @test all(m[:, 1, :])
        @test count(m) == 4 * 6
    end

    @testset "outlet marks the last slice" begin
        m = faces(shape; outlet=3)
        @test all(m[:, :, 6])
        @test count(m) == 4 * 5
    end

    @testset "both may be given at once" begin
        m = faces(shape; inlet=1, outlet=1)
        @test all(m[1, :, :]) && all(m[4, :, :])
        @test count(m) == 2 * 5 * 6
    end

    @test_throws ErrorException faces(shape)
end

# --- Percolation trimming ---

@testset "trim_nonpercolating_paths" begin
    @testset "keeps a spanning channel and drops an isolated blob" begin
        img = falses(12, 8, 8)
        img[:, 3:5, 3:5] .= true                     # spans x
        img[5:7, 7, 7] .= true                       # isolated island
        trimmed = trim_nonpercolating_paths(img; axis=:x)
        @test all(trimmed[:, 3:5, 3:5])
        @test !any(trimmed[5:7, 7, 7])
        @test count(trimmed) == 12 * 3 * 3
        @test _count_percolating(img; axis=:x) == count(trimmed)
    end

    @testset "the result is a subset of the input and is idempotent" begin
        img = BitArray(blobs(; shape=(20, 20, 20), porosity=0.6, blobiness=1, seed=9))
        for ax in (:x, :y, :z)
            trimmed = trim_nonpercolating_paths(img; axis=ax)
            @test all(trimmed .<= img)               # never adds pore space
            @test trim_nonpercolating_paths(trimmed; axis=ax) == trimmed
            @test _count_percolating(img; axis=ax) == count(trimmed)
        end
    end

    @testset "a fully open image is left alone" begin
        img = trues(6, 6, 6)
        for ax in (:x, :y, :z)
            @test trim_nonpercolating_paths(img; axis=ax) == img
            @test _count_percolating(img; axis=ax) == length(img)
        end
    end

    @testset "a channel that spans only one axis is kept only for that axis" begin
        img = falses(10, 10, 10)
        img[:, 4:6, 4:6] .= true                     # spans x only
        @test count(trim_nonpercolating_paths(img; axis=:x)) == count(img)
        @test count(trim_nonpercolating_paths(img; axis=:y)) == 0
        @test count(trim_nonpercolating_paths(img; axis=:z)) == 0
        @test _count_percolating(img; axis=:x) == count(img)
        @test _count_percolating(img; axis=:y) == 0
        @test _count_percolating(img; axis=:z) == 0
    end
end

# --- Phase fractions ---

@testset "phase_fraction on boolean masks" begin
    img = BitArray(blobs(; shape=(16, 16, 16), porosity=0.55, blobiness=1, seed=3))
    ε = phase_fraction(img, true)
    @test ε + phase_fraction(img, false) ≈ 1.0
    @test ε ≈ count(img) / length(img)
    fracs = phase_fraction(img)
    @test sum(values(fracs)) ≈ 1.0
    @test fracs[true] ≈ ε
end

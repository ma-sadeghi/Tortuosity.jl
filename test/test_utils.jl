using Test
using Tortuosity
using Tortuosity: phase_fraction

# Set up fixtures
img = ones(UInt8, (32, 32, 32))
img[:, :, 1:5] .= 0
img[:, :, 21:32] .= 5
ε0 = 32*32*5 / (32*32*32)
ε1 = 32*32*15 / (32*32*32)
ε5 = 32*32*12 / (32*32*32)

@testset verbose=true "phase_fraction" begin

    @testset "No labels passed" begin
        fracs = phase_fraction(img)
        @test fracs[0] ≈ ε0 atol=1e-4
        @test fracs[1] ≈ ε1 atol=1e-4
        @test fracs[5] ≈ ε5 atol=1e-4
        @test sum(values(fracs)) ≈ 1.0 atol=1e-4
    end

    @testset "Single label passed" begin
        @test phase_fraction(img, 0) ≈ ε0 atol=1e-4
        @test phase_fraction(img, 1) ≈ ε1 atol=1e-4
        @test phase_fraction(img, 5) ≈ ε5 atol=1e-4
    end

    @testset "Multiple labels passed" begin
        @test phase_fraction(img, [0, 1]) ≈ ε0 + ε1 atol=1e-4
        @test phase_fraction(img, [1, 5]) ≈ ε1 + ε5 atol=1e-4
        @test phase_fraction(img, [0, 5]) ≈ ε0 + ε5 atol=1e-4
        @test phase_fraction(img, [0, 1, 5]) ≈ ε0 + ε1 + ε5 atol=1e-4
    end

end

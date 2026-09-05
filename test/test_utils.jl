using HDF5
using Test
using Tortuosity
using Tortuosity.Imaginator: phase_fraction
import Tortuosity: export_to_hdf5

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
        # A one-element list is the scalar method's answer, not a sum over
        # nothing — the degenerate case of the array overload.
        @test phase_fraction(img, [5]) ≈ ε5 atol=1e-4
    end

end

# --- HDF5 export ---

@testset "export_to_hdf5" begin
    path = joinpath(mktempdir(), "export.h5")
    field = rand(4, 5, 6)
    export_to_hdf5(path; concentration=field, tau=1.75, axis="x")

    @test isfile(path)
    h5open(path, "r") do fid
        # Every keyword becomes a dataset named after the keyword.
        @test Set(keys(fid)) == Set(["concentration", "tau", "axis"])
        @test read(fid["concentration"]) == field
        @test read(fid["tau"]) == 1.75
        @test read(fid["axis"]) == "x"
    end

    @testset "writing again truncates rather than appends" begin
        export_to_hdf5(path; only_this=[1.0, 2.0])
        h5open(path, "r") do fid
            @test Set(keys(fid)) == Set(["only_this"])
        end
    end
end

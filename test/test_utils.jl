using HDF5
using Test
using Tortuosity
using Tortuosity.Imaginator: phase_fraction
import Tortuosity: args_to_dict, build_reverse_lookup, export_to_hdf5,
    find_true_indices, format_args_dict

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

@testset "Boolean index helpers" begin
    mat = Bool[
        true  false;
        false true;
        true  true
    ]
    expected = LinearIndices(mat)[findall(mat)]

    @test find_true_indices(mat) == expected

    lookup = build_reverse_lookup(mat)
    for (i, idx) in enumerate(expected)
        @test lookup[idx] == i
    end
end

# --- Command-line argument plumbing ---
#
# Nothing in the package calls these; they exist for the batch-run scripts. Kept
# to one round trip through the pair — the `--key=value` grammar the regex
# implements, and the two ways `format_args_dict` is asked for something it
# cannot produce.

@testset "args_to_dict / format_args_dict" begin
    # Only `--key=value` is a pair: bare flags and positionals are skipped, and
    # a value runs to the next space, so paths and punctuation survive.
    d = args_to_dict(["--verbose", "positional", "--fpath=/tmp/in-1.h5",
                      "--path_export=out.h5", "--gpu_id=3", "--axis=:x"])
    @test d == Dict("fpath" => "/tmp/in-1.h5", "path_export" => "out.h5",
                    "gpu_id" => "3", "axis" => ":x")
    @test isempty(args_to_dict(String[]))

    @test format_args_dict(d) == ("/tmp/in-1.h5", "out.h5", 3)
    @test format_args_dict(d)[3] === 3          # parsed to Int, not left a String

    @test_throws KeyError format_args_dict(Dict("fpath" => "in.h5"))
    bad = Dict("fpath" => "in.h5", "path_export" => "out.h5", "gpu_id" => "first")
    @test_throws ArgumentError format_args_dict(bad)
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

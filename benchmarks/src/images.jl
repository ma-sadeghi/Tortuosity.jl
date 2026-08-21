# The image store: deterministic generation, on-disk cache and integrity index.
#
# Every tool in the campaign must solve the *same* geometry, or a difference in
# the reported tortuosity says nothing about the solvers. Images are therefore
# generated once, cached as one HDF5 file per case, and indexed by a manifest
# that records a SHA-256 of each. Generation is deterministic in the configured
# seed, so a machine that regenerates the store gets byte-identical images — and
# the hash is what turns that from an assumption into something checked.

using Tortuosity: Imaginator
# `Imaginator.blobs` lives in a package extension; without ImageFiltering loaded
# the call is a MethodError rather than a missing-package error.
using ImageFiltering

const MANIFEST_HEADER = "case_id,size,blobiness,porosity_target,porosity,nnodes,sha256"

imagedir(cfg::Config) = joinpath(cfg.root, "data", "images")
manifest_path(cfg::Config) = joinpath(imagedir(cfg), "manifest.csv")
image_path(cfg::Config, case::Case) = joinpath(imagedir(cfg), "$(case.id).h5")

"""One manifest row: what an image is, and what it should hash to."""
struct ImageEntry
    case_id::String
    size::Int
    blobiness::Float64
    porosity_target::Float64
    porosity::Float64
    nnodes::Int
    sha256::String
end

"""The manifest indexed by case id, or an empty index when there is none yet."""
function read_manifest(cfg::Config)
    entries = Dict{String,ImageEntry}()
    path = manifest_path(cfg)
    isfile(path) || return entries
    for line in Iterators.drop(eachline(path), 1)
        isempty(strip(line)) && continue
        f = split(line, ",")
        length(f) == 7 || continue
        entries[f[1]] = ImageEntry(f[1], parse(Int, f[2]), parse(Float64, f[3]),
                                   parse(Float64, f[4]), parse(Float64, f[5]),
                                   parse(Int, f[6]), f[7])
    end
    return entries
end

function append_manifest!(cfg::Config, e::ImageEntry)
    path = manifest_path(cfg)
    mkpath(dirname(path))
    isfile(path) || open(io -> println(io, MANIFEST_HEADER), path, "w")
    open(path, "a") do io
        @printf(io, "%s,%d,%.2f,%.4f,%.6f,%d,%s\n",
                e.case_id, e.size, e.blobiness, e.porosity_target, e.porosity, e.nnodes, e.sha256)
    end
    return e
end

"""Rewrite the manifest from scratch, sorted, dropping duplicate case ids.

Regeneration appends, so a case rebuilt with `--overwrite` would otherwise leave
two rows claiming different hashes for one image. The last row written is the
current one and wins.
"""
function rewrite_manifest!(cfg::Config, entries::AbstractDict{String,ImageEntry})
    path = manifest_path(cfg)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, MANIFEST_HEADER)
        for e in sort(collect(values(entries)); by=e -> (e.size, e.blobiness, e.porosity_target))
            @printf(io, "%s,%d,%.2f,%.4f,%.6f,%d,%s\n",
                    e.case_id, e.size, e.blobiness, e.porosity_target, e.porosity, e.nnodes, e.sha256)
        end
    end
    return path
end

"""Build one image: blobs at the configured seed, then trimmed to the percolating
pore space along the transport axis.

Trimming is not cosmetic. An isolated pore cluster contributes nodes to the
linear system that no boundary condition reaches, which leaves the operator
singular on that subspace; solvers differ in how gracefully they absorb that, so
leaving the clusters in would measure error handling rather than transport.
"""
function build_image(cfg::Config, case::Case)
    axis = Symbol(cfg["campaign"]["axis"])
    seed = Int(cfg["campaign"]["seed"])
    img = Imaginator.blobs(; shape=ntuple(_ -> case.size, 3), porosity=case.porosity,
                           blobiness=case.blobiness, seed=seed)
    return Imaginator.trim_nonpercolating_paths(img; axis=axis)
end

"""Generate `case`'s image unless it is already cached, and index it.

Returns the manifest entry. With `force`, an existing image is rebuilt and its
manifest row replaced.
"""
function ensure_image!(cfg::Config, case::Case; force::Bool=false)
    manifest = read_manifest(cfg)
    path = image_path(cfg, case)
    if !force && isfile(path) && haskey(manifest, case.id)
        return manifest[case.id]
    end

    img = build_image(cfg, case)
    # Hashed and written from one buffer: the hash has to describe the bytes that
    # reach the disk, not a second copy that merely ought to match them.
    data = Array{UInt8}(img)
    entry = ImageEntry(case.id, case.size, case.blobiness, case.porosity,
                       count(img) / length(img), count(img), bytes2hex(sha256(vec(data))))

    mkpath(dirname(path))
    tmp = path * ".partial"
    h5open(tmp, "w") do fid
        # Chunked and deflated: a binary mask compresses by more than an order of
        # magnitude, which is the difference between a store that can be copied to
        # a rented machine and one that cannot.
        dset = create_dataset(fid, "image", datatype(UInt8), dataspace(size(data));
                              chunk=chunk_shape(size(data)), deflate=1, shuffle=true)
        write(dset, data)
        for (k, v) in ("case_id" => case.id, "size" => case.size,
                       "blobiness" => case.blobiness, "porosity_target" => case.porosity,
                       "porosity" => entry.porosity, "nnodes" => entry.nnodes,
                       "sha256" => entry.sha256, "axis" => String(Symbol(cfg["campaign"]["axis"])),
                       "seed" => Int(cfg["campaign"]["seed"]))
            attrs(dset)[k] = v
        end
    end
    # Renamed only once the file is complete, so an interrupted generation leaves
    # a `.partial` to be discarded rather than a truncated image that later reads
    # as a valid case.
    mv(tmp, path; force=true)

    manifest[case.id] = entry
    rewrite_manifest!(cfg, manifest)
    return entry
end

"""Chunk shape for an `N³` mask: whole xy-slabs, capped so a chunk stays small."""
function chunk_shape(dims::NTuple{3,Int})
    nz = max(1, min(dims[3], div(4_000_000, max(1, dims[1] * dims[2]))))
    return (dims[1], dims[2], nz)
end

"""Load a cached image as a `Bool` array, verifying it against the manifest.

Verification is cheap next to any solve and catches the failure that would
otherwise be invisible: a store regenerated under a different package version,
or copied in part, gives images that look right and results that cannot be
compared with anything measured before.
"""
function load_image(cfg::Config, case::Case; verify::Bool=true)
    path = image_path(cfg, case)
    isfile(path) || error("no image for $(case.id) at $path — run generate_images.jl first")
    data = h5open(path, "r") do fid
        read(fid["image"])
    end
    if verify
        manifest = read_manifest(cfg)
        entry = get(manifest, case.id, nothing)
        if entry === nothing
            @warn "$(case.id) is not in the manifest — cannot verify integrity" path
        else
            digest = bytes2hex(sha256(vec(data)))
            digest == entry.sha256 || error(
                "$(case.id) does not match the manifest (got $digest, expected $(entry.sha256)). " *
                "The image store and the manifest describe different geometry; regenerate with " *
                "--overwrite or restore the store.")
        end
    end
    return data .!= 0x00
end

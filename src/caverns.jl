# Cavern detection: finding the pore voxels of a 3D image that carry no
# meaningful steady-state flux, i.e. dead ends or "caverns".
#
# Adapted from Mohammad Mehrnia's caverns.py.

"""
    find_caverns(img::BitArray; vmin=-2, iter=1, axis=:z, reltol=1e-5, gpu=nothing)

Identify the low-flux "cavern" regions of a pore space: voxels that belong to
the connected pore network but carry so little steady-state flux that they are
dead volume as far as transport is concerned.

Porosity counts a blind pocket exactly as it counts a through-going channel,
which is part of why two images of equal porosity can have very different
tortuosity. Separating the stagnant volume from the flux-carrying volume makes
that difference measurable: the returned mask can be subtracted from `img` to
leave the transporting skeleton, or its fraction reported directly as the share
of pore volume that does no work.

The classification is iterative, because filling one shell of stagnant volume
exposes the next behind it. Each pass:

1. Removes the voxels already classified as caverns.
2. Solves a steady diffusion problem on the pore space that remains.
3. Sums `|Δc|` over the face-connected neighbours of every pore voxel
   ([`flux_out`](@ref)).
4. Marks the voxels whose log-flux falls below `vmin` as caverns.
5. Marks whatever that stranded — pore clusters that no longer reach both
   transport faces — as caverns too.

```julia
img = BitArray(Imaginator.blobs(; shape=(64, 64, 64), porosity=0.6, blobiness=1))
caverns, cavern_fraction = Tortuosity.find_caverns(img; iter=3, axis=:z, gpu=false)
```

# Arguments
- `img::BitArray`: 3D pore mask (`true` = pore). A plain `Array{Bool,3}`, which
  is what [`Imaginator.blobs`](@ref Tortuosity.Imaginator.blobs) returns, has to
  be converted with `BitArray` first.

# Keyword Arguments
- `vmin`: log-flux threshold. A voxel is a cavern when `log10(flux) < vmin`.
  The flux is scaled by the number of voxels along `axis` first, so the
  threshold means the same thing at every image size. Default: `-2`.
- `iter`: number of passes. A pass only ever adds caverns, so the reported
  fraction is monotonic and settles after a few passes. Default: `1`.
- `axis`: transport direction (`:x`, `:y`, or `:z`). Default: `:z`.
- `reltol`: relative tolerance of the diffusion solve. The threshold compares
  fluxes orders of magnitude below the mean, so a loose tolerance moves the
  boundary of the cavern set rather than merely blurring it. Default: `1e-5`.
- `gpu`: passed straight to [`SteadyDiffusionProblem`](@ref), and carries that
  constructor's meaning — `true` forces GPU, `false` forces CPU, and `nothing`
  (default) uses a registered backend once the pore space clears
  `GPU_MIN_NODES`. Default: `nothing`.

# Returns
- `caverns::BitArray`: 3D mask of the cavern voxels (`true`).
- `cavern_fraction::Vector{Float64}`: the fraction of `img`'s pore voxels
  classified as caverns after each pass. It has length `iter + 1` and starts
  at `0.0`, the fraction before any pass has run.

See also [`flux_out`](@ref). Pore clusters that never percolated in the first
place are removed instead by
[`trim_nonpercolating_paths`](@ref Tortuosity.Imaginator.trim_nonpercolating_paths).
"""
function find_caverns(
    img::BitArray; vmin=-2, iter=1, axis::Symbol=:z, reltol=1e-5, gpu=nothing
)
    # Voxel count along the transport axis. The solve imposes a unit
    # concentration drop across the whole domain however long it is, so the
    # field is scaled by `N` to recover a per-voxel gradient and make `vmin` a
    # resolution-independent threshold. Imposing that scale through the boundary
    # condition instead would avoid the multiplication.
    N = size(img, axis_dim(axis))

    caverns = falses(size(img))
    # Cavern fraction after each pass, opening at the fraction before any pass.
    cavern_fraction = zeros(iter + 1)
    filled_img = copy(img)
    for i in 1:iter
        filled_img[caverns] .= false

        sim = SteadyDiffusionProblem(filled_img; axis=axis, gpu=gpu)
        sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=reltol)
        c = N .* reconstruct_field(sol.u, filled_img)

        # `reconstruct_field` always returns a host array, so the stencil below
        # runs on the CPU even when the solve did not. It is one pass over the
        # grid against a Krylov solve, so it is not the bottleneck.
        flux = flux_out(c, filled_img)

        caverns[(log10.(flux) .< vmin) .& filled_img] .= true
        # Filling those voxels can strand pore clusters that used to percolate;
        # they carry no flux either, so fold them in. `trim_nonpercolating_paths`
        # lives in the `Imaginator` submodule and is reached by qualification,
        # the same way `dnstools.jl` reaches `Imaginator.phase_fraction`.
        percolating = Imaginator.trim_nonpercolating_paths(img .& .!caverns; axis=axis)
        caverns[img .& .!percolating] .= true

        cavern_fraction[i + 1] = count(caverns) / count(img)
    end

    return caverns, cavern_fraction
end

"""
    flux_out(c, img)

Sum the absolute concentration difference `|Δc|` over the face-connected
neighbours of every voxel, giving a per-voxel measure of how much material
moves through each point of a steady concentration field.

This works in image space: `c` and `img` are full 3D grids of the same shape,
and so is the result. An edge contributes only when both of its voxels are
pore, so solid voxels come out zero and the `NaN` fill that
[`reconstruct_field`](@ref) leaves in them cannot leak into a neighbouring pore
voxel's sum — `false * NaN` is `0.0` in Julia. A voxel on a face of the domain
has one neighbour fewer along that axis and is lower to that extent.

It is a plain edge-difference stencil and carries no diffusivity, so under a
non-uniform `D` it measures the concentration gradient rather than the flux.
[`flux`](@ref) is the diffusivity-weighted measurement, taken through a
cross-section rather than per voxel.

# Arguments
- `c`: steady concentration field over the full grid.
- `img`: 3D boolean pore mask (`true` = pore), the same size as `c`.

# Returns
- An array the size of `c` holding the summed `|Δc|` at each voxel.
"""
function flux_out(c::AbstractArray, img::AbstractArray{Bool})
    @assert size(c) == size(img) "size of c must match size of img"

    # Difference across every face connection, masked by the image. Note that in
    # Julia, false * NaN = 0.0, which is what keeps the solid-voxel fill out.
    Fx = (img[1:end-1, :, :] .& img[2:end, :, :]) .*
        abs.(c[1:end-1, :, :] .- c[2:end, :, :])

    Fy = (img[:, 1:end-1, :] .& img[:, 2:end, :]) .*
        abs.(c[:, 1:end-1, :] .- c[:, 2:end, :])

    Fz = (img[:, :, 1:end-1] .& img[:, :, 2:end]) .*
        abs.(c[:, :, 1:end-1] .- c[:, :, 2:end])

    F = similar(c)
    fill!(F, 0)

    # X-direction edges
    F[1:end-1, :, :] .+= Fx
    F[2:end, :, :] .+= Fx

    # Y-direction edges
    F[:, 1:end-1, :] .+= Fy
    F[:, 2:end, :] .+= Fy

    # Z-direction edges
    F[:, :, 1:end-1] .+= Fz
    F[:, :, 2:end] .+= Fz

    return F
end

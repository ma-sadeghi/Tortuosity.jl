### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ c0de0002-0000-4000-8000-000000000002
begin
    using Tortuosity
    using Tortuosity: Imaginator
    using CUDA             # optional; the CPU path works fine without a device
    using ImageFiltering   # optional dep that `Imaginator.blobs` needs
    import Meshing         # marching cubes, for the 3-D view; kept qualified
    using PlutoPlotly      # because it exports `isosurface` too
    using PlutoUI
    using ProgressLogging
    using Printf
end

# ╔═╡ c0de0001-0000-4000-8000-000000000001
md"""
# How tortuous is this rock?

Diffusion through a porous material is slower than diffusion through open space, because the molecules have to go *around* things. The **tortuosity factor** τ is the number that says how much slower.

[`Tortuosity.jl`](https://github.com/ma-sadeghi/Tortuosity.jl) computes it directly from a binary 3-D image: it builds a finite-volume Laplacian over the pore voxels, solves it with a preconditioned conjugate gradient, and reads τ off the resulting flux. The same code runs on the CPU or on a GPU — there is a switch further down, and the two agree to about seven digits.

Two buttons drive the notebook: one grows a microstructure, one solves it. Everything in between — slicing, rotating, recolouring — is free and updates as you touch it.
"""

# ╔═╡ c0de0003-0000-4000-8000-000000000003
md"""
## 1. Grow a microstructure

`Imaginator.blobs` makes a random porous image the way PoreSpy does: blur white noise, flatten the histogram, threshold at the target porosity. Then `trim_nonpercolating_paths` throws away the pockets that don't connect the inlet face to the outlet face, since they hold fluid but carry no steady flux.

Set the dials, then press the button — nothing regenerates until you do.
"""

# ╔═╡ c0de0004-0000-4000-8000-000000000004
@bind geom confirm(
    PlutoUI.combine() do Child
        md"""
        |                    |                                                                                  |
        | -----------------: | :------------------------------------------------------------------------------- |
        |     **image size** | $(Child("n", Slider(128:64:512; default=128, show_value=true))) voxels per side   |
        |      **porosity**  | $(Child("porosity", Slider(0.40:0.01:0.85; default=0.65, show_value=true)))       |
        |     **blobiness**  | $(Child("blobiness", Slider(0.4:0.1:3.0; default=1.0, show_value=true)))          |
        |          **seed**  | $(Child("seed", Slider(1:99; default=7, show_value=true)))                        |
        | **transport axis** | $(Child("axis", Select([:x => "x", :y => "y", :z => "z"])))                       |
        """
    end;
    label="Grow it",
)

# ╔═╡ c0de0007-0000-4000-8000-000000000007
md"""
## 2. Look inside

Light is pore, dark is solid.
"""

# ╔═╡ c0de000a-0000-4000-8000-00000000000a
md"""
### ...and from the outside

Marching cubes on a downsampled copy of the image. Drag to rotate.
"""

# ╔═╡ c0de000b-0000-4000-8000-00000000000b
@bind vol PlutoUI.combine() do Child
    md"""
    |             |                                                                    |
    | ----------: | :----------------------------------------------------------------- |
    | **render**  | $(Child("show", CheckBox(default=true)))                           |
    |  **phase**  | $(Child("phase", Select([:solid => "solid", :pore => "pore space"]))) |
    | **detail**  | $(Child("detail", Slider(16:8:72; default=40, show_value=true)))   |
    """
end

# ╔═╡ c0de000d-0000-4000-8000-00000000000d
md"""
## 3. Solve it

Three steps: assemble the sparse Laplacian over the pore voxels, solve `A c = b` with a two-level preconditioned conjugate gradient, then scatter the pore-ordered solution back into the image. The progress bar under the cell tracks them.

The first press on a fresh notebook also pays Julia's one-time compilation of the solver, so press it twice on the same image if you want the steady-state timing.
"""

# ╔═╡ c0de0011-0000-4000-8000-000000000011
md"""
## 4. The concentration field

The inlet face is held at `c = 1`, the outlet at `c = 0`, and every other face is no-flux. The colour is what the solver found in between; the blank gaps are solid.
"""

# ╔═╡ c0de0013-0000-4000-8000-000000000013
md"""
Averaged over each slice the profile is close to a straight line, which is what a homogeneous medium should give. The interesting part is the spread around it: the shaded band is ±1σ across the slice, and it widens wherever the pore space is most constricted.
"""

# ╔═╡ c0de0015-0000-4000-8000-000000000015
md"""
## Notes

**CPU or GPU.** The switch above is the only difference — the same `SteadyDiffusionProblem` and `solve` calls run either way, because the CUDA kernels live in a package extension that loads with `using CUDA`. Measured here on a laptop RTX PRO 5000 against ten CPU threads, a 128³ image took 0.25 s on the device and 4.8 s on the host, for the same τ to four decimal places. Your numbers will differ; the ratio is the point. The GPU system is assembled in `Float32` and then refined against a `Float64` residual, which is why τ matches the CPU answer to roughly seven digits rather than exactly.

**Threads.** This notebook is running on $(Threads.nthreads()) CPU thread(s). The solver is memory-bound and scales well with cores, so if that says one, start Pluto from a Julia session launched as `julia --threads=auto` — Pluto hands its own thread count down to the notebook process.

**The big end of the slider.** At 512³ (134 million voxels) the GPU path takes about 10 s and 10 GiB of device memory, and growing the image takes longer than solving it, because the blob generator blurs a full `Float64` noise field on the CPU. The same size on the CPU path is a much longer wait and tens of gigabytes of RAM — worth trying, but start it knowing that.

**Bigger still.** 512³ is where this notebook stops, not where the package does. It is built for hundreds of millions of voxels, and the benchmarks in the repository go up to 1000³ using a matrix-free operator that never stores the Laplacian at all.
"""

# ╔═╡ c0de0016-0000-4000-8000-000000000016
md"""
---
### Appendix: helpers

Plotting and formatting only; nothing here is part of Tortuosity.jl.
"""

# ╔═╡ c0de0017-0000-4000-8000-000000000017
TableOfContents(; title="Contents", depth=2)

# ╔═╡ c0de0018-0000-4000-8000-000000000018
"""
Where the image cell leaves its result for the solve cell to pick up.

Pluto re-runs a cell whenever a variable it reads is *reassigned*. Writing into
this box is not a reassignment, so the solve stays put until its own button is
pressed instead of chasing every new image.
"""
const staged = Ref{Any}(nothing)

# ╔═╡ c0de0005-0000-4000-8000-000000000005
img, t_image = let
    t = @elapsed begin
        raw = Imaginator.blobs(;
            shape=ntuple(_ -> geom.n, 3),
            porosity=geom.porosity,
            blobiness=geom.blobiness,
            seed=geom.seed,
        )
        connected = Imaginator.trim_nonpercolating_paths(raw; axis=geom.axis)
    end
    # Hand the image to the solver through the box rather than directly, so that
    # growing a new one doesn't drag the expensive solve along with it.
    staged[] = (; img=connected, axis=geom.axis)
    connected, t
end;

# ╔═╡ c0de0008-0000-4000-8000-000000000008
@bind view PlutoUI.combine() do Child
    md"""
    |                      |                                                                             |
    | -------------------: | :-------------------------------------------------------------------------- |
    | **slice normal to**  | $(Child("axis", Select([:z => "z", :x => "x", :y => "y"])))                  |
    |    **slice index**   | $(Child("i", Slider(1:size(img, 1); default=size(img, 1) ÷ 2, show_value=true))) |
    """
end

# ╔═╡ c0de0019-0000-4000-8000-000000000019
function device_options()
    cpu = "CPU — $(Threads.nthreads()) threads"
    CUDA.functional() || return [:cpu => cpu * " (no CUDA device found)"]
    return [:cpu => cpu, :gpu => "GPU — $(CUDA.name(CUDA.device()))"]
end

# ╔═╡ c0de000e-0000-4000-8000-00000000000e
md"""
Run on $(@bind device Select(device_options())) $(@bind go CounterButton("Compute the tortuosity"))
"""

# ╔═╡ c0de000f-0000-4000-8000-00000000000f
result = let
    go  # press the button to re-run this cell
    input = staged[]
    on_gpu = device === :gpu
    if isnothing(input) || count(input.img) == 0
        nothing
    else
        @withprogress name = "tortuosity" begin
            @logprogress "assembling the linear system" 0.05
            t_build = @elapsed begin
                sim = SteadyDiffusionProblem(input.img; axis=input.axis, gpu=on_gpu)
                on_gpu && CUDA.synchronize()
            end
            @logprogress "conjugate-gradient solve" 0.20
            t_solve = @elapsed begin
                sol = solve(sim; verbose=false)
                # The device runs asynchronously, so without this the clock would
                # stop before the solve does.
                on_gpu && CUDA.synchronize()
            end
            @logprogress "reconstructing the field" 0.90
            c = reconstruct_field(sol.u, input.img)
            τ = tortuosity(c, input.img; axis=sim.axis)
            (; img=input.img, device, sim, sol, c, τ, t_build, t_solve)
        end
    end
end;

# ╔═╡ c0de001a-0000-4000-8000-00000000001a
axis_dim(axis::Symbol) = axis === :x ? 1 : axis === :y ? 2 : 3

# ╔═╡ c0de001b-0000-4000-8000-00000000001b
cross_section(A, dim, i) = Array(selectdim(A, dim, clamp(i, 1, size(A, dim))))

# ╔═╡ c0de001c-0000-4000-8000-00000000001c
const PORE_SCALE = [[0.0, "#28323c"], [1.0, "#eef1f4"]]

# ╔═╡ c0de001d-0000-4000-8000-00000000001d
function image_plot(S; colorscale, zmin, zmax, title="", showscale=false)
    trace = heatmap(;
        # `float.` because Plotly's JSON writer cannot serialise a `BitMatrix`:
        # a Bool slice would reach the browser as an empty plot.
        z=permutedims(float.(S)), colorscale, zmin, zmax, showscale,
        colorbar=attr(; thickness=12, len=0.85),
        hovertemplate="(%{x}, %{y}) &nbsp; %{z:.3f}<extra></extra>",
    )
    layout = Layout(;
        title=attr(; text=title, font=attr(; size=14)),
        height=420, margin=attr(; l=10, r=10, t=45, b=10),
        xaxis=attr(; constrain="domain", showgrid=false, zeroline=false),
        yaxis=attr(; scaleanchor="x", scaleratio=1, showgrid=false, zeroline=false),
    )
    return plot(trace, layout)
end

# ╔═╡ c0de0009-0000-4000-8000-000000000009
image_plot(
    cross_section(img, axis_dim(view.axis), view.i);
    colorscale=PORE_SCALE, zmin=0, zmax=1,
    title="pore space — slice $(view.i) normal to $(view.axis)",
)

# ╔═╡ c0de0012-0000-4000-8000-000000000012
isnothing(result) ? md"" : image_plot(
    cross_section(result.c, axis_dim(view.axis), view.i);
    colorscale="Viridis", zmin=0, zmax=1, showscale=true,
    title="concentration — slice $(view.i) normal to $(view.axis)",
)

# ╔═╡ c0de001e-0000-4000-8000-00000000001e
"""
Average `A` over `f`-sized blocks, giving a smooth Float32 field to contour.

Threaded over the output's third index, so each thread owns a disjoint set of
destination cells. At 512³ the scalar version is several seconds of the 3-D
view's budget on its own.
"""
function block_mean(A::AbstractArray{Bool,3}, f::Int)
    f <= 1 && return Float32.(A)
    out = zeros(Float32, cld.(size(A), f))
    cnt = zeros(Float32, size(out))
    Threads.@threads for K in axes(out, 3)
        @inbounds for k in ((K - 1) * f + 1):min(K * f, size(A, 3)),
            j in axes(A, 2), i in axes(A, 1)

            I, J = cld(i, f), cld(j, f)
            out[I, J, K] += A[i, j, k]
            cnt[I, J, K] += 1
        end
    end
    return out ./ cnt
end

# ╔═╡ c0de001f-0000-4000-8000-00000000001f
"""Weld a marching-cubes triangle soup into shared vertices plus 0-based indices."""
function weld(verts, tris)
    seen = Dict{eltype(verts),Int}()
    pts = similar(verts, 0)
    remap = Vector{Int}(undef, length(verts))
    for (n, v) in pairs(verts)
        remap[n] = get!(seen, v) do
            push!(pts, v)
            length(pts)
        end
    end
    idx = [(remap[t[1]] - 1, remap[t[2]] - 1, remap[t[3]] - 1) for t in tris]
    return pts, idx
end

# ╔═╡ c0de0021-0000-4000-8000-000000000021
function profile_plot(c, dim)
    N = size(c, dim)
    stats = map(1:N) do i
        v = filter(!isnan, vec(selectdim(c, dim, i)))
        isempty(v) && return (NaN, NaN)
        μ = sum(v) / length(v)
        return (μ, sqrt(sum(abs2, v .- μ) / length(v)))
    end
    μ = first.(stats)
    σ = last.(stats)
    band = scatter(;
        x=[1:N; N:-1:1], y=[μ .+ σ; reverse(μ .- σ)],
        fill="toself", fillcolor="rgba(45,125,210,0.20)", line=attr(; width=0),
        name="±1σ across the slice", hoverinfo="skip",
    )
    mean_line = scatter(;
        x=1:N, y=μ, mode="lines", name="slice-averaged c",
        line=attr(; width=3, color="#2d7dd2"),
    )
    layout = Layout(;
        height=320, margin=attr(; l=55, r=20, t=20, b=45),
        xaxis=attr(; title="voxel along the transport axis", zeroline=false),
        yaxis=attr(; title="concentration", range=[-0.05, 1.05], zeroline=false),
        legend=attr(; orientation="h", y=-0.25),
    )
    return plot([band, mean_line], layout)
end

# ╔═╡ c0de0014-0000-4000-8000-000000000014
isnothing(result) ? md"" : profile_plot(result.c, axis_dim(result.sim.axis))

# ╔═╡ c0de0022-0000-4000-8000-000000000022
fmt_count(n) = replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+$)" => ",")

# ╔═╡ c0de0020-0000-4000-8000-000000000020
function volume_plot(phase::AbstractArray{Bool,3}; detail=40, smoothing=0.8)
    f = max(1, cld(maximum(size(phase)), detail))
    field = block_mean(phase, f)
    field = Float32.(imfilter(field, Kernel.gaussian(ntuple(_ -> smoothing, 3)), "replicate"))
    # A solid rind of zeros closes the surface where the structure runs off the
    # edge of the image; without it the box looks hollow from the outside.
    padded = zeros(Float32, size(field) .+ 2)
    padded[2:(end - 1), 2:(end - 1), 2:(end - 1)] .= field
    # Padded index p covers original voxels centred on (p - 1.5) * f + 0.5.
    axis_range(n) = range(0.5 - 0.5 * f, 0.5 + (n + 0.5) * f; length=n + 2)
    verts, tris = Meshing.isosurface(
        padded, Meshing.MarchingCubes(; iso=0.5f0), axis_range.(size(field))...
    )
    pts, idx = weld(verts, tris)

    trace = mesh3d(;
        x=[p[1] for p in pts], y=[p[2] for p in pts], z=[p[3] for p in pts],
        i=[t[1] for t in idx], j=[t[2] for t in idx], k=[t[3] for t in idx],
        color="#8aa6bd", flatshading=false, hoverinfo="skip",
        lighting=attr(; ambient=0.45, diffuse=0.85, specular=0.25, roughness=0.6),
        lightposition=attr(; x=120, y=200, z=300),
    )
    blank = attr(; showbackground=false, showticklabels=false, title="", showgrid=false)
    layout = Layout(;
        title=attr(;
            text="$(fmt_count(length(idx))) triangles at 1/$(f) resolution",
            font=attr(; size=13),
        ),
        height=520, margin=attr(; l=0, r=0, t=45, b=0),
        scene=attr(;
            aspectmode="data", xaxis=blank, yaxis=blank, zaxis=blank,
            camera=attr(; eye=attr(; x=1.5, y=1.5, z=1.1)),
        ),
    )
    return plot(trace, layout)
end

# ╔═╡ c0de000c-0000-4000-8000-00000000000c
if vol.show
    volume_plot(vol.phase === :solid ? .!img : img; detail=vol.detail)
else
    md"*(3-D view off)*"
end

# ╔═╡ c0de0023-0000-4000-8000-000000000023
"""A row of small captioned facts, used under the image and the solve."""
function readout(items...)
    cells = join(
        ("<span style=\"padding:0 0.9rem\">$(it)</span>" for it in items),
        "<span style=\"opacity:0.35\">|</span>",
    )
    return HTML("""<div style="font-size:0.95rem; opacity:0.8; text-align:center;
                                padding:0.4rem 0">$(cells)</div>""")
end

# ╔═╡ c0de0006-0000-4000-8000-000000000006
readout(
    "$(join(size(img), " × ")) = $(fmt_count(length(img))) voxels",
    "connected porosity $(@sprintf("%.1f", 100 * count(img) / length(img))) %",
    "grown in $(@sprintf("%.2f", t_image)) s",
)

# ╔═╡ c0de0024-0000-4000-8000-000000000024
function score_card(r)
    total = r.t_build + r.t_solve
    device_name = r.device === :gpu ? "on the GPU" : "on the CPU"
    HTML("""
    <div style="border:1px solid rgba(127,158,189,0.45); border-radius:10px;
                background:rgba(127,158,189,0.10); padding:1.2rem 1rem; text-align:center">
      <div style="text-transform:uppercase; letter-spacing:0.12em; font-size:0.75rem;
                  opacity:0.65">tortuosity factor</div>
      <div style="font-size:3.2rem; font-weight:700; line-height:1.2">
        &tau; = $(@sprintf("%.4f", r.τ))</div>
      <div style="font-size:0.95rem; opacity:0.8">
        D<sub>eff</sub>/D<sub>0</sub> = $(@sprintf("%.4f", effective_diffusivity(r.c, r.img; axis=r.sim.axis)))
        &nbsp;·&nbsp; formation factor = $(@sprintf("%.3f", formation_factor(r.c, r.img; axis=r.sim.axis)))
      </div>
      <hr style="border:none; border-top:1px solid rgba(127,158,189,0.35); margin:0.9rem 0">
      <div style="font-size:0.95rem; opacity:0.85">
        $(fmt_count(length(r.sol.u))) unknowns
        &nbsp;·&nbsp; $(r.sol.iters) CG iterations
        &nbsp;·&nbsp; assembled in $(@sprintf("%.2f", r.t_build)) s
        &nbsp;·&nbsp; solved in $(@sprintf("%.2f", r.t_solve)) s $(device_name)
      </div>
      <div style="font-size:1.05rem; font-weight:600; padding-top:0.4rem">
        $(@sprintf("%.1f", length(r.img) / 1e6 / total)) million voxels per second
      </div>
    </div>
    """)
end

# ╔═╡ c0de0010-0000-4000-8000-000000000010
if isnothing(result)
    count(img) == 0 ?
    md"""
    !!! danger "Nothing gets through"
        No pore cluster connects the inlet face to the outlet face along `$(geom.axis)`.
        Raise the porosity, or lower the blobiness, and grow it again.
    """ :
    md"""
    !!! tip "Ready"
        Press **Compute the tortuosity** above.
    """
elseif result.img !== img
    md"""
    !!! warning "Out of date"
        The numbers below belong to the previous image. Press **Compute the tortuosity**
        again to catch up.
    """
else
    score_card(result)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
Meshing = "e6723b4c-ebff-59f1-b4b7-d97aa5274f73"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Tortuosity = "80bb5c7f-3561-4bbc-a7c3-5e29a4cab6cc"

[compat]
CUDA = "~5.11.3"
ImageFiltering = "~0.7.12"
Meshing = "~0.7.0"
PlutoPlotly = "~0.6.6"
PlutoUI = "~0.7.83"
ProgressLogging = "~0.1.6"
Tortuosity = "~0.1.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "e1059cf8890d7802f9cbe0e8c55c7adc8f35e0cc"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.24.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AMD]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "45a1272e3f809d36431e57ab22703c6896b8908f"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.3"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "daa72978cd7a624246e894a4f4f067706d4e17e2"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "daf5b2aab5b1c1fdcb65b05883cdb4b18abac1b9"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.30.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceFillArraysExt = "FillArrays"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceGPUArraysCoreTrackerExt = ["GPUArraysCore", "Tracker"]
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b8651b2eb5796a386b0398a20b519a6a6150f75c"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.3"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "e386db8b4753b42caac75ac81d0a4fe161a68a97"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.BracketingNonlinearSolve]]
deps = ["CommonSolve", "ConcreteStructs", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "7ad7171d693ae5552ac43862e7f6b61df4471c2b"
uuid = "70df07ce-3d50-431d-a3e7-ca6ddb60ac1e"
version = "1.12.1"

    [deps.BracketingNonlinearSolve.extensions]
    BracketingNonlinearSolveChainRulesCoreExt = ["ChainRulesCore", "ForwardDiff"]
    BracketingNonlinearSolveForwardDiffExt = "ForwardDiff"

    [deps.BracketingNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "54751d09f9acf05ea7b7ee6baa6a99677c788880"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.11.3"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c32d22f2f563ce192c88a44b09c2b569f1e7a980"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.4+1"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "2bbaa78dd79a27e354ac97c17dca290069f5c56f"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.3.4+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "159b1c1f03e6355bb9c6d8954e5f51019788dd26"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "2.1.1"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c0314d9fb0ebd00e404feba4c3fbc04c9975abc1"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.21.0+1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonWorldInvalidations]]
deps = ["PrecompileTools"]
git-tree-sha1 = "bc209b1a67dd03551fe305d72e88128764b89cf5"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.2.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.1+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcreteStructs]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a72c3b5ce6d2a477f55b5c9b8756e91e695c67f6"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.8"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "BracketingNonlinearSolve", "ConcreteStructs", "DocStringExtensions", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "Static", "StaticArraysCore", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "9d333db14895e8c7d4857ed228eb1e72d3b302ec"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.218.0"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDynamicQuantitiesExt = "DynamicQuantities"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseFlexUnitsExt = "FlexUnits"
    DiffEqBaseForwardDiffExt = ["ForwardDiff"]
    DiffEqBaseGTPSAExt = "GTPSA"
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseMooncakeExt = "Mooncake"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseSparseArraysExt = "SparseArrays"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FlexUnits = "76e01b6b-c995-4ce6-8559-91e72a3d4e95"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffEqCallbacks]]
deps = ["ConcreteStructs", "DataStructures", "DiffEqBase", "DifferentiationInterface", "LinearAlgebra", "Markdown", "PrecompileTools", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "88cdec45374d53393bf88268102a2b018c897178"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "4.19.2"

    [deps.DiffEqCallbacks.extensions]
    DiffEqCallbacksFunctorsExt = "Functors"

    [deps.DiffEqCallbacks.weakdeps]
    Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "0693d8b0a4608ff289d228ab4c598df5894845cd"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.21"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = ["GPUArraysCore", "Adapt"]
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceHyperHessiansExt = "HyperHessians"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    HyperHessians = "06b494a0-c8e0-40cc-ad32-d99506a00a6c"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EnzymeCore]]
git-tree-sha1 = "971d7831cc85f43bc9f51d615a3f7f21270c2f1d"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.21"

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

    [deps.EnzymeCore.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "c8f8eefadaa330d982bbf5e395c34e53a13ab668"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "1.4.0"
weakdeps = ["Polyester", "Static"]

    [deps.FastBroadcast.extensions]
    FastBroadcastPolyesterExt = "Polyester"
    FastBroadcastStaticExt = "Static"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastPower]]
deps = ["PrecompileTools"]
git-tree-sha1 = "3c7269c236978d434a16ebe99f9743b9d706270a"
uuid = "a4df4552-cc26-4903-aec0-212e50a0e84b"
version = "1.5.0"

    [deps.FastPower.extensions]
    FastPowerEnzymeExt = "Enzyme"
    FastPowerForwardDiffExt = "ForwardDiff"
    FastPowerMeasurementsExt = "Measurements"
    FastPowerMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    FastPowerMooncakeExt = "Mooncake"
    FastPowerReverseDiffExt = "ReverseDiff"
    FastPowerTrackerExt = "Tracker"

    [deps.FastPower.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "2bcce3ad6f6977d617928d7707fdc86ac83cce03"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "1.13.0"

    [deps.FunctionWrappersWrappers.extensions]
    FunctionWrappersWrappersEnzymeExt = ["Enzyme", "EnzymeCore"]
    FunctionWrappersWrappersMooncakeExt = "Mooncake"

    [deps.FunctionWrappersWrappers.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "0811627284eba0d19bf64d0d247c63fa1e5c26b6"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.5.14"

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

    [deps.GPUArrays.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "b863a2e71f89328e2af69069490aec2448106e59"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.17.1"

    [deps.GPUCompiler.weakdeps]
    LLVMDowngrader_jll = "f52de702-fb25-5922-94ba-81dd59b07444"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "a589b6c1a0eff953571f5d8b0474f5020831114d"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.1.1"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Preferences", "Static"]
git-tree-sha1 = "af9ab7d1f70739a47f03be78771ebda38c3c71bf"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.18"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "52116260a234af5f69969c5286e6a5f8dc3feab8"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.12"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "895205d762ae24a01689f8cc7ad584b55f1fd005"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.7"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c7345ab1a7ca4dc8a02c9f6510da0d9857bbe513"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.7.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "a5b87110fa95d711355af44832497745aa93fb52"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.42"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "71e740d00d71cdb15145d7fe0d6000ec70534598"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.9"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d4bfee24427f4f441bd9212a107e375c39663aab"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.13.1"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "d77aea19c9a71059a021acd99b0a4343e9661d94"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.47+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LinearSolve]]
deps = ["AMD", "ArrayInterface", "ConcreteStructs", "DocStringExtensions", "EnumX", "GPUArraysCore", "InteractiveUtils", "Krylov", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "OpenBLAS_jll", "PrecompileTools", "Preferences", "PureKLU", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "Setfield", "SparseArrays", "SparseColumnPivotedQR", "StaticArraysCore"]
git-tree-sha1 = "ec49ed72f6024be2f9833fe630fbd72f6048f8ad"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "3.87.0"

    [deps.LinearSolve.extensions]
    LinearSolveAMDGPUExt = "AMDGPU"
    LinearSolveAlgebraicMultigridExt = "AlgebraicMultigrid"
    LinearSolveBLISExt = ["blis_jll", "LAPACK_jll"]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = ["cuSOLVER"]
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveCUSOLVERRFExt = ["CUSOLVERRF", "SparseArrays"]
    LinearSolveChainRulesCoreExt = "ChainRulesCore"
    LinearSolveCliqueTreesExt = ["CliqueTrees", "SparseArrays"]
    LinearSolveElementalExt = "Elemental"
    LinearSolveEnzymeExt = ["EnzymeCore", "SparseArrays"]
    LinearSolveFastAlmostBandedMatricesExt = "FastAlmostBandedMatrices"
    LinearSolveFastLapackInterfaceExt = "FastLapackInterface"
    LinearSolveForwardDiffExt = "ForwardDiff"
    LinearSolveGinkgoExt = ["Ginkgo", "SparseArrays"]
    LinearSolveHSLExt = ["HSL", "SparseArrays"]
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMUMPSExt = ["MUMPS", "SparseArrays"]
    LinearSolveMetalExt = "Metal"
    LinearSolveMooncakeExt = "Mooncake"
    LinearSolvePETScExt = ["PETSc", "SparseArrays", "SparseMatricesCSR"]
    LinearSolvePETScMPIExt = ["PETSc", "PartitionedArrays", "SparseArrays", "SparseMatricesCSR"]
    LinearSolveParUExt = ["ParU_jll", "SparseArrays"]
    LinearSolvePardisoExt = ["Pardiso", "SparseArrays"]
    LinearSolvePartitionedSolversExt = ["PartitionedArrays", "PartitionedSolvers"]
    LinearSolvePureUMFPACKExt = ["PureUMFPACK", "SparseArrays"]
    LinearSolveRecursiveFactorizationExt = "RecursiveFactorization"
    LinearSolveSTRUMPACKExt = ["SparseArrays", "STRUMPACK_jll"]
    LinearSolveSparseArraysExt = "SparseArrays"
    LinearSolveSparspakExt = ["SparseArrays", "Sparspak"]
    LinearSolveSpecializingFactorizationsExt = "SpecializingFactorizations"
    LinearSolveSuperLUDISTExt = ["SparseArrays", "SuperLUDIST"]

    [deps.LinearSolve.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    CUSOLVERRF = "a8cc9031-bad2-4722-94f5-40deabb4245c"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    Elemental = "902c3f28-d1ec-5e7e-8399-a24c3845ee38"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    FastLapackInterface = "29a986be-02c6-4525-aec4-84b980013641"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Ginkgo = "4c8bd3c9-ead9-4b5e-a625-08f1338ba0ec"
    HSL = "34c5aeac-e683-54a6-a0e9-6e0fdc586c50"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    LAPACK_jll = "51474c39-65e3-53ba-86ba-03b1b862ec14"
    MUMPS = "55d2b088-9f4e-11e9-26c0-150b02ea6a46"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PETSc = "ace2c81b-2b5f-4b1e-a30d-d662738edfe0"
    ParU_jll = "9e0b026c-e8ce-559c-a2c4-6a3d5c955bc9"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    PartitionedArrays = "5a9dfac6-5c52-46f7-8278-5e2210713be9"
    PartitionedSolvers = "11b65f7f-80ac-401b-9ef2-3db765482d62"
    PureUMFPACK = "b7e1f0a2-3c4d-4e5f-9a0b-1c2d3e4f5a6b"
    RecursiveFactorization = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
    STRUMPACK_jll = "86fbd0b9-476f-557c-b766-62c724b42d8c"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    Sparspak = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
    SpecializingFactorizations = "fa08b7a1-13d3-4faf-875d-5cbc1520e3f3"
    SuperLUDIST = "4cd002a6-0da4-410d-a012-232df062f478"
    blis_jll = "6136c539-28a5-5bf0-87cc-b183200dce32"
    cuSOLVER = "887afef0-6a32-4de5-add4-7827692ba8fc"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "514e8475e33c6faf3155efee5f3c10d9e65a11ab"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.174"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    ForwardDiffNNlibExt = ["ForwardDiff", "NNlib"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "PrecompileTools"]
git-tree-sha1 = "f2cde0ae772162f20287b803e623966d0d94dea9"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.8"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.Meshing]]
git-tree-sha1 = "5705d4f32545f7bcecdff75759a20d17812e5caf"
uuid = "e6723b4c-ebff-59f1-b4b7-d97aa5274f73"
version = "0.7.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "60beb0717782a3bbe0f7df56decad0ef89048c23"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.12"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.NVTX]]
deps = ["JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "a9083c3e469e63cca454d1fc3b19472d9d92c14a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.3"
weakdeps = ["Colors"]

    [deps.NVTX.extensions]
    NVTXColorsExt = "Colors"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NaNStatistics]]
deps = ["PrecompileTools"]
git-tree-sha1 = "22f75e63359030bcac76b145d7cf97bcb3f84d37"
uuid = "b946abbf-3ea7-4610-9019-9858bfdeaf2d"
version = "0.6.58"

    [deps.NaNStatistics.extensions]
    NaNStatisticsDimensionalDataExt = "DimensionalData"
    NaNStatisticsHwlocExt = "Hwloc"
    NaNStatisticsUnitfulExt = "Unitful"

    [deps.NaNStatistics.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"
    Hwloc = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.NonlinearSolveBase]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "CommonSolve", "Compat", "ConcreteStructs", "DifferentiationInterface", "EnzymeCore", "FastClosures", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "LogExpFunctions", "Markdown", "MaybeInplace", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "cb824030a0d5769555704db95ee3cbe0f76ed53d"
uuid = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
version = "2.30.3"

    [deps.NonlinearSolveBase.extensions]
    NonlinearSolveBaseBandedMatricesExt = "BandedMatrices"
    NonlinearSolveBaseChainRulesCoreExt = "ChainRulesCore"
    NonlinearSolveBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    NonlinearSolveBaseForwardDiffExt = "ForwardDiff"
    NonlinearSolveBaseLineSearchExt = "LineSearch"
    NonlinearSolveBaseLinearSolveExt = "LinearSolve"
    NonlinearSolveBaseMooncakeExt = "Mooncake"
    NonlinearSolveBaseReverseDiffExt = "ReverseDiff"
    NonlinearSolveBaseSparseArraysExt = "SparseArrays"
    NonlinearSolveBaseSparseMatrixColoringsExt = "SparseMatrixColorings"
    NonlinearSolveBaseTrackerExt = "Tracker"

    [deps.NonlinearSolveBase.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LineSearch = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
    LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.OrdinaryDiffEqCore]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "ConcreteStructs", "DataStructures", "DiffEqBase", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappersWrappers", "InteractiveUtils", "LinearAlgebra", "Logging", "MacroTools", "MuladdMacro", "Polyester", "PrecompileTools", "Preferences", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Static", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "c0a22f012d3e98fd78afafe8e5152ace7a2276bc"
uuid = "bbf590c4-e513-4bbe-9b18-05decba2e5d8"
version = "3.33.1"

    [deps.OrdinaryDiffEqCore.extensions]
    OrdinaryDiffEqCoreMooncakeExt = "Mooncake"
    OrdinaryDiffEqCoreSparseArraysExt = "SparseArrays"

    [deps.OrdinaryDiffEqCore.weakdeps]
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.OrdinaryDiffEqStabilizedRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "RecursiveArrayTools", "Reexport", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "764e11fb6a26ee8f2d2d10778d531fd708f35d28"
uuid = "358294b1-0aab-51c3-aafe-ad5ab194a2ad"
version = "1.11.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "3de8f5e6e90ebfa8d6d1f86997d6cdcd6a912ff3"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.7"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "2b9e3d771adfe535a4fdda855f4741fdaacd3f7f"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.6"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "16bbc30b5ebea91e9ce1671adc03de2832cff552"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.19"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "cd00e28071a75c98664663f87c2ae447d25c0503"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.7.1"

    [deps.PreallocationTools.extensions]
    PreallocationToolsEnzymeCoreExt = "EnzymeCore"
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1b8aa19f229b1cea7fc93874a52e49db6a854450"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.4.8"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.PureKLU]]
deps = ["LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "ef341b8e734ffa12c0464a58ca1c8a214d7a4235"
uuid = "0c0d3e7f-3a8b-4f7e-b6f1-9a4d2e7c1f01"
version = "1.4.1"

    [deps.PureKLU.extensions]
    PureKLUForwardDiffExt = "ForwardDiff"

    [deps.PureKLU.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "d0282d612f22dcad7b81cf487b746e63aa2a6709"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.54.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsFastBroadcastPolyesterExt = ["FastBroadcast", "Polyester"]
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStatisticsExt = "Statistics"
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "PrecompileTools", "SHA", "Serialization"]
git-tree-sha1 = "6cb9b83354089fcb33c643773acfd3235f2f1bf5"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.25"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "72312aa278823c0e99ce31186e22d917d2d11f99"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.46"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLLogging", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "d56cb3b924cdf9297c171113e6a6ae9694e370eb"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.155.2"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDifferentiationInterfaceExt = "DifferentiationInterface"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReverseDiffExt = "ReverseDiff"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLJacobianOperators]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DifferentiationInterface", "FastClosures", "LinearAlgebra", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "ad167d716fc134c0873c76ba0469ede56a678732"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.17"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "Preferences"]
git-tree-sha1 = "4e1e21f14a284f892eb62923a356c70a2a0c68e1"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "1.10.1"
weakdeps = ["Tracy"]

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

[[deps.SciMLOperators]]
deps = ["Accessors", "Adapt", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "0b5b895913f1269a8c80b59435c61d5a7d79970c"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.30.0"
weakdeps = ["LoopVectorization", "SparseArrays"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsLoopVectorizationExt = "LoopVectorization"
    SciMLOperatorsSparseArraysExt = "SparseArrays"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "5c2f9dbf6f07eea6bc9e93f117b00b7939a79f9e"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.5"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseColumnPivotedQR]]
deps = ["LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "01ac65d0bf2f42a9491b80c4076f4e02b462a9e6"
uuid = "a57abbd0-fea5-4d57-96be-5e525945e8e4"
version = "2.1.7"
weakdeps = ["AMD"]

    [deps.SparseColumnPivotedQR.extensions]
    SparseColumnPivotedQRAMDExt = "AMD"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "474a5283ad435618090122872eea6a8165ea6bcf"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.4.6"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "SciMLPublic", "Static"]
git-tree-sha1 = "2a635e15d5035c53b345077c947f31ff91744078"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.10.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e206cf4850fd7ac4255ffd2b98922f563e18ac53"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.20"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "389592b8592e5fb48f498ff60dc6acb4a0e62953"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.4"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "5316097111523c9a970596a5b33cfea5f92e8581"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.9"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "773065c6e0e903924a9d838259be74338422aef2"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.5.0"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrecompileTools", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "7eb6da9656581ac9fbffaef3ef7950a090a5002a"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.55"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "7c73336785b21f723f5b143f6e99cf6c43b37dc1"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.6"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3748bd928e68c7c346b52125cf41fff0de6937d0"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.29"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.Tortuosity]]
deps = ["Atomix", "CommonSolve", "DiffEqCallbacks", "ImageMorphology", "KernelAbstractions", "LinearAlgebra", "LinearSolve", "NaNStatistics", "OrdinaryDiffEqStabilizedRK", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "6552afd89173e56fac559fa5f2e2a1eccf5427ca"
uuid = "80bb5c7f-3561-4bbc-a7c3-5e29a4cab6cc"
version = "0.1.0"

    [deps.Tortuosity.extensions]
    TortuosityAMDGPUExt = "AMDGPU"
    TortuosityCUDAExt = "CUDA"
    TortuosityHDF5Ext = "HDF5"
    TortuosityImageFilteringExt = "ImageFiltering"
    TortuosityLsqFitExt = "LsqFit"
    TortuosityMetalExt = "Metal"

    [deps.Tortuosity.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
    ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
    LsqFit = "2fda8390-95c7-5789-9bda-21331edee243"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "21b39bfb1fab6156b61fbcba4c86c57b6216d2c3"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.2"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "807a234dc5e6132dd6cf4c9317ca0917c4001ab3"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.74"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "da8c1f6eee04831f14edcfa5dae611d309807e57"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.3.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─c0de0001-0000-4000-8000-000000000001
# ╠═c0de0002-0000-4000-8000-000000000002
# ╟─c0de0003-0000-4000-8000-000000000003
# ╟─c0de0004-0000-4000-8000-000000000004
# ╠═c0de0005-0000-4000-8000-000000000005
# ╟─c0de0006-0000-4000-8000-000000000006
# ╟─c0de0007-0000-4000-8000-000000000007
# ╟─c0de0008-0000-4000-8000-000000000008
# ╠═c0de0009-0000-4000-8000-000000000009
# ╟─c0de000a-0000-4000-8000-00000000000a
# ╟─c0de000b-0000-4000-8000-00000000000b
# ╟─c0de000c-0000-4000-8000-00000000000c
# ╟─c0de000d-0000-4000-8000-00000000000d
# ╟─c0de000e-0000-4000-8000-00000000000e
# ╠═c0de000f-0000-4000-8000-00000000000f
# ╟─c0de0010-0000-4000-8000-000000000010
# ╟─c0de0011-0000-4000-8000-000000000011
# ╠═c0de0012-0000-4000-8000-000000000012
# ╟─c0de0013-0000-4000-8000-000000000013
# ╟─c0de0014-0000-4000-8000-000000000014
# ╟─c0de0015-0000-4000-8000-000000000015
# ╟─c0de0016-0000-4000-8000-000000000016
# ╟─c0de0017-0000-4000-8000-000000000017
# ╟─c0de0018-0000-4000-8000-000000000018
# ╟─c0de0019-0000-4000-8000-000000000019
# ╟─c0de001a-0000-4000-8000-00000000001a
# ╟─c0de001b-0000-4000-8000-00000000001b
# ╟─c0de001c-0000-4000-8000-00000000001c
# ╟─c0de001d-0000-4000-8000-00000000001d
# ╟─c0de001e-0000-4000-8000-00000000001e
# ╟─c0de001f-0000-4000-8000-00000000001f
# ╟─c0de0020-0000-4000-8000-000000000020
# ╟─c0de0021-0000-4000-8000-000000000021
# ╟─c0de0022-0000-4000-8000-000000000022
# ╟─c0de0023-0000-4000-8000-000000000023
# ╟─c0de0024-0000-4000-8000-000000000024
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

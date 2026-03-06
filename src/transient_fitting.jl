# Analytical solutions to homogeneous transient diffusion and curve-fitting utilities

"""
    effective_diffusivity(sim::TransientState, prob::TransientProblem, method::Symbol; depth=0.5, t_fit=(0, sim.t[end]), terms=100, D0=1.0)

Fit an analytical transient‑diffusion solution to simulation data and return the
effective diffusivity. Supports `bc_outlet=0` and `bc_outlet=nothing` (insulated).

Supported observables:
- `:conc` — concentration at a normalized depth.
- `:mass` — normalized mass uptake over full volume. independent of 'depth' kwarg
- `:flux` — not implemented.

Boundary modes must be `bc_inlet=1, bc_outlet=0` or `bc_inlet=1, bc_outlet=nothing`.

# Keyword arguments
- `depth` — depth for `:conc` or `:flux` fits, redefined to depth of closest index for fit,
         TransientProblem defaults to depths from 0 to 1
- `t_fit` — time window `(tmin, tmax)` for fitting, defaults all timesteps in TransientState.
- `terms` — number of series terms in the analytical solution (infinite series).
- `D0` — initial diffusivity guess scalar.

# Returns
`D_eff, φ_eff, xdata, ydata, fit, model`
"""
function effective_diffusivity(
    t,
    C,
    prob::TransientProblem,
    method::Symbol;
    depth=0.5,
    t_fit=(0, t[end]),
    terms=100,
    D0=1.0,
)
    D0 = Float64.(D0)
    param = [D0, porosity(prob)]

    idx_min = argmin(abs.(t .- t_fit[1]))
    idx_max = argmin(abs.(t .- t_fit[2]))
    N = size(prob.img, axis_dim(prob.axis))

    xdata = t[idx_min:idx_max]
    ydata = nothing
    model = nothing
    @assert !isnothing(prob.bc_inlet) "The inlet boundary being insulated is not supported for effective diffusivity fitting"
    C1 = prob.bc_inlet
    C2 = isnothing(prob.bc_outlet) ? C1 : prob.bc_outlet
    # Insulated outlet is modelled as a symmetric slab of double length
    L = (N - 1) * prob.dx * (isnothing(prob.bc_outlet) ? 2 : 1)

    if method == :conc
        depth_idx = round(Int, 1 + depth * (N - 1))
        depth = (depth_idx - 1) * prob.dx # snap to nearest grid point

        ydata = get_slice_conc(C[idx_min:idx_max], prob, depth_idx)
        depth += prob.dx / 2 # flux sits between slice idx and idx+1
        model =
            (t, p) -> p[2] * analytic_conc(p[1], depth, t; C1=C1, C2=C2, L=L, terms=terms)

    elseif method == :mass
        ydata = (mass_intake(C[1:idx_max], prob))[idx_min:end]
        model =
            (t, p) ->
                p[2] * (C1 + C2) / 2 .*
                analytic_mass(p[1], t; C1=C1, C2=C2, L=L, terms=terms)

    elseif method == :flux
        depth_idx = round(Int, 0.5 + (depth) * (N - 1)) # flux lives between slices
        depth_idx == N && (depth_idx = N - 1) # clamp: no flux past last slice
        depth = (depth_idx - 0.5) * prob.dx # snap to nearest inter-slice position

        ydata = get_flux(C[idx_min:idx_max], prob; ind=depth_idx)
        model =
            (t, p) -> p[2] * analytic_flux(p[1], depth, t; C1=C1, C2=C2, L=L, terms=terms)

    else
        error(
            "Built-in diffusivity fitting only supports method ':conc', ':mass', and ':flux'.",
        )
    end

    fit = curve_fit(model, xdata, ydata, param)
    D_eff = fit.param[1]
    φ_eff = fit.param[2]

    return D_eff, φ_eff, xdata, ydata, fit, model
end
function effective_diffusivity(
    sim::TransientState,
    prob::TransientProblem,
    method::Symbol;
    depth=0.5,
    t_fit=(0, sim.t[end]),
    terms=100,
    D0=1.0,
)
    return effective_diffusivity(
        sim.t,
        sim.C,
        prob::TransientProblem,
        method::Symbol;
        depth=depth,
        t_fit=t_fit,
        terms=terms,
        D0=D0,
    )
end

function voxel_tortuosity(
    sim::TransientState,
    prob::TransientProblem;
    depth=0.5,
    n_samples=200,
    t_fit=(0, sim.t[end]),
    terms=100,
)

    # --- determine slice index ---
    N = size(prob.img, axis_dim(prob.axis))
    depth_idx = round(Int, 1 + depth * (N - 1))

    # Sample voxels uniformly by pore-voxel index (not spatial position)
    slice_coords = slice_vec_indices(prob, depth_idx)
    slice_voxels = length(slice_coords)
    @assert slice_voxels >= n_samples "the number of samples to fit cannot exceed pore voxels at that index"

    voxels = slice_coords[round.(Int, LinRange(1, slice_voxels, n_samples))]

    # --- prepare outputs ---
    D_list = Float64[]
    x_list = Float64[]
    SE_D_list = Float64[]
    SE_x_list = Float64[]

    # --- time window ---
    idx_min = argmin(abs.(sim.t .- t_fit[1]))
    idx_max = argmin(abs.(sim.t .- t_fit[2]))
    @assert idx_min < idx_max "t_fit window must contain at least two time points"
    xdata = sim.t[idx_min:idx_max]

    # --- boundary conditions ---
    C1 = prob.bc_inlet
    C2 = isnothing(prob.bc_outlet) ? C1 : prob.bc_outlet
    L = (N - 1) * prob.dx * (isnothing(prob.bc_outlet) ? 2 : 1) # symmetric slab for insulated outlet

    model = (t, p) -> analytic_conc(p[1], p[2], t; C1=C1, C2=C2, L=L, terms=terms)

    D_init = if prob.D isa Number
        Float64(prob.D)
    else
        Float64(sum(prob.D[prob.img]) / count(prob.img))
    end
    p0 = [D_init, depth]

    # --- fit each voxel ---
    for i in voxels
        ydata = map(A -> A[i], sim.C[idx_min:idx_max])

        fit = curve_fit(model, xdata, ydata, p0)

        push!(D_list, fit.param[1])
        push!(x_list, fit.param[2])

        sigma = stderror(fit)
        push!(SE_D_list, sigma[1])
        push!(SE_x_list, sigma[2])
    end

    return D_list, x_list, SE_D_list, SE_x_list, voxels
end

function tortuosity()
    return nothing
end

"""
    analytic_conc(D, x, t; terms=100, C1=1, C2=0, C0=0, L=1)

Analytical concentration in a 1D slab with constant diffusivity (Crank, *The
Mathematics of Diffusion*, 2nd ed., p. 50).

# Arguments
- `D`: diffusion coefficient.
- `x`: position along slab length (0 to `L`).
- `t`: time or array of times.

# Keyword Arguments
- `terms`: number of Fourier series terms (default: 100).
- `C1`: concentration at `x = 0` boundary (default: 1).
- `C2`: concentration at `x = L` boundary (default: 0).
- `C0`: initial concentration throughout slab (default: 0).
- `L`: slab thickness (default: 1).
"""
function analytic_conc(D, x, t; terms=100, C1=1, C2=0, C0=0, L=1)
    # Series indices reshaped for broadcasting: time along dim 1, terms along dim 2
    n = reshape(1:terms, 1, terms)
    m = 2 .* n .- 1

    t_is_scalar = t isa Number
    t_vec = t_is_scalar ? [t] : collect(t)
    t = reshape(t_vec, :, 1)

    out = dropdims(
        C1 .+ (C2 - C1) .* x ./ L .+
        2 ./ π .* sum(
            (C2 .* cos.(π .* n) .- C1) ./ (n) .* sin.(n .* π .* x ./ L) .*
            exp.(-D .* (n) .^ 2 .* π^2 .* t / L^2);
            dims=2,
        ) .+
        4 .* C0 ./ π .* sum(
            1 ./ m .* sin.((2 .* n .- 1) .* (π .* x ./ L)) .*
            exp.(-D .* m .^ 2 .* π^2 .* t ./ L^2);
            dims=2,
        );
        dims=2,
    )

    return t_is_scalar ? out[1] : out
end

"""
    analytic_mass(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

Normalized mass uptake `M_t/M_inf` in a 1D slab with constant diffusivity,
where `M_inf = L * ((C1 + C2)/2 - C0)` (Crank, p. 50).

# Arguments
- `D`: diffusion coefficient.
- `t`: time or array of times.

# Keyword Arguments
- `terms`: number of Fourier series terms (default: 100).
- `C1`: concentration at `x = 0` boundary (default: 1).
- `C2`: concentration at `x = L` boundary (default: 0).
- `C0`: initial concentration throughout slab (default: 0).
- `L`: slab thickness (default: 1).
"""
function analytic_mass(D, t; terms=100, C1=1, C2=0, C0=0, L=1)
    n = (2 .* reshape(1:terms, 1, terms) .- 1) .^ 2
    t_is_scalar = t isa Number
    t_vec = t_is_scalar ? [t] : collect(t)
    t = reshape(t_vec, :, 1)

    out = dropdims(
        1 .- 8 ./ (π^2) .* sum((1 ./ n) .* exp.(-D .* n .* (π ./ L) .^ 2 .* t); dims=2);
        dims=2,
    )
    return t_is_scalar ? out[1] : out
end

"""
    analytic_flux(D, x, t; terms=100, C1=1, C2=0, C0=0, L=1)

Analytical diffusive flux in a 1D slab, computed as

    J(x,t) = -D * ∂C(x,t)/∂x

where `C(x,t)` is the Fourier–series solution of the diffusion equation
with boundary concentrations `C1` at `x = 0` and `C2` at `x = L`, and
initial concentration `C0` throughout the slab.

This function evaluates the *local instantaneous flux* at position `x`
and time(s) `t` using the exact analytical derivative of the series
solution (Crank, *The Mathematics of Diffusion*, 2nd ed., Ch. 3).

The series is truncated after `terms` modes. Larger `terms` improve
accuracy at early times and near boundaries.

# Arguments
- `D`: diffusion coefficient.
- `x`: spatial position at which the flux is evaluated (scalar).
- `t`: time or array of times.

# Keyword Arguments
- `terms`: number of Fourier terms to include (default: 100).
- `C1`: concentration at the `x = 0` boundary (default: 1).
- `C2`: concentration at the `x = L` boundary (default: 0).
- `C0`: initial concentration throughout the slab (default: 0).
- `L`: slab thickness (default: 1).

# Returns
Flux `J(x,t)` as a scalar or array matching the shape of `t`.

"""
function analytic_flux(D, x, t; terms=100, C1=1, C2=0, C0=0, L=1)
    # n and m indices
    n = reshape(1:terms, 1, terms)
    m = 2 .* n .- 1

    # reshape t if needed
    t_is_scalar = t isa Number
    t_vec = t_is_scalar ? [t] : collect(t)
    t = reshape(t_vec, :, 1)

    # wave numbers
    kn = n .* π ./ L
    km = m .* π ./ L

    # derivative of linear term
    dCdx_linear = (C2 - C1) ./ L

    # derivative of first sine series
    dCdx_series1 =
        (2 / π) .* sum(
            ((C2 .* cos.(π .* n) .- C1) ./ n) .* (kn .* cos.(kn .* x)) .*
            exp.(-D .* (n .^ 2) .* π^2 .* t ./ L^2);
            dims=2,
        )

    # derivative of second sine series
    dCdx_series2 =
        (4 * C0 / π) .* sum(
            (1 ./ m) .* (km .* cos.(km .* x)) .* exp.(-D .* (m .^ 2) .* π^2 .* t ./ L^2);
            dims=2,
        )

    # total derivative
    dCdx = dCdx_linear .+ dropdims(dCdx_series1 .+ dCdx_series2; dims=2)

    # flux = -D * dC/dx
    return -D .* (t_is_scalar ? dCdx[1] : dCdx)
end

"""
    analytic_∑flux(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

Cumulative amount of substance that has diffused through the slab in the
interval `(0, t)` (Crank, p. 51).

# Arguments
- `D`: diffusion coefficient.
- `t`: time or array of times.

# Keyword Arguments
- `terms`: number of Fourier series terms (default: 100).
- `C1`: concentration at `x = 0` boundary (default: 1).
- `C2`: concentration at `x = L` boundary (default: 0).
- `C0`: initial concentration throughout slab (default: 0).
- `L`: slab thickness (default: 1).
"""
function analytic_∑flux(D, t; terms=100, C1=1, C2=0, C0=0, L=1)
    n = reshape(1:terms, 1, terms)
    m = (2 .* n .- 1) .^ 2

    t_is_scalar = t isa Number
    t_vec = t_is_scalar ? [t] : collect(t)
    t = reshape(t_vec, :, 1)

    out = dropdims(
        D * (C1 - C2) .* t ./ L .+
        (2 * L / π^2) .* sum(
            (C1 .* cos.(π .* n) .- C2) ./ (n .^ 2) .*
            (1 .- exp.(-D .* (n) .^ 2 .* π^2 .* t / L^2));
            dims=2,
        ) .+
        (4 * C0 * L / π^2) .*
        sum(1 ./ m .* (1 .- exp.(-D .* m .* π^2 .* t ./ L^2)); dims=2);
        dims=2,
    )
    return t_is_scalar ? out[1] : out
end

# Analytical solutions to homogeneous transient diffusion and curve-fitting utilities

"""
    fit_effective_diffusivity(t, C, prob, method; depth=0.5, t_fit=(0, t[end]), terms=100)
    fit_effective_diffusivity(sim, prob, method; ...)

Fit a 1D analytical transient-diffusion solution to simulation data and extract
the effective transport coefficient. The fit uses a single degree of freedom:
the inverse tortuosity `τ⁻¹ = D_eff / φ`, where `φ` is the geometric porosity.

Boundary modes must be `bc_inlet=<number>, bc_outlet=<number>` or
`bc_inlet=<number>, bc_outlet=nothing` (insulated outlet). Time-dependent
(Function) boundaries are not supported.

# Supported observables
- `:conc` — concentration at a specified normalized depth.
- `:mass` — normalized mass uptake over the pore volume (independent of `depth`).
- `:flux` — flux between voxel slices at a specified depth.

# Keyword Arguments
- `depth` — fractional depth in `[0,1]` for `:conc` and `:flux`.
             The value is snapped to the nearest voxel index.
- `t_fit` — time window `(tmin, tmax)` over which the fit is performed.
- `terms` — number of eigenfunction terms in the analytic series solution.

# Returns
`τ, D_eff, xdata, ydata, fit, model`

Where `τ` is the fitted tortuosity (`D_pore * φ / D_eff`) and `D_eff` is
the effective diffusivity. If `prob.D` is a scalar field, `D_pore` is set
to `mean(prob.D[img])`.
"""
function fit_effective_diffusivity(
    t, C, prob::TransientDiffusionProblem, method::Symbol;
    depth=0.5, t_fit=(0, t[end]), terms=100,
)
    @assert !(prob.bc_inlet isa Function) "fit_effective_diffusivity does not support f(t) inlet boundary conditions."
    @assert !(prob.bc_outlet isa Function) "fit_effective_diffusivity does not support f(t) outlet boundary conditions."
    @assert !isnothing(prob.bc_inlet) "The inlet boundary being insulated is not supported for fitting"

    φ = Imaginator.phase_fraction(prob.img, true)

    D_pore = prob.D isa Number ? Float64(prob.D) : Float64(sum(prob.D[prob.img]) / count(prob.img))
    param = [D_pore]

    idx_min = argmin(abs.(t .- t_fit[1]))
    idx_max = argmin(abs.(t .- t_fit[2]))
    N = size(prob.img, axis_dim(prob.axis))

    xdata = t[idx_min:idx_max]
    ydata = nothing
    model = nothing

    # Insulated outlet is modelled as a symmetric slab of double length
    insulated = isnothing(prob.bc_outlet)
    C1 = prob.bc_inlet
    C2 = insulated ? prob.bc_inlet : prob.bc_outlet
    L = (N - 1) * prob.dx * (insulated ? 2 : 1)

    if method == :conc
        # Cell-center convention: concentration lives at the center of each voxel,
        # so voxel index i maps to physical position (i - 0.5) * dx.
        depth_idx = round(Int, 1 + depth * (N - 1))
        depth_actual = (depth_idx - 0.5) * prob.dx

        ydata = slice_concentration(
            C[idx_min:idx_max], prob.img, prob.axis, depth_idx;
            grid_to_vec=prob.grid_to_vec, pore_only=true,
        )
        model = (t, p) -> slab_concentration(p[1], depth_actual, t; C1=C1, C2=C2, L=L, terms=terms)

    elseif method == :mass
        ydata = (mass_uptake(C[1:idx_max], prob.img))[idx_min:end]
        model = (t, p) -> φ * (C1 + C2) / 2 .* slab_mass_uptake(p[1], t; C1=C1, C2=C2, L=L, terms=terms)

    elseif method == :flux
        # Cell-face convention: flux is computed at the face between voxels
        # depth_idx and depth_idx+1, so the physical position is depth_idx * dx.
        depth_idx = round(Int, 0.5 + depth * (N - 1))
        if depth_idx == N
            depth_idx = N - 1
        end
        depth_actual = depth_idx * prob.dx

        ydata = flux(
            C[idx_min:idx_max], prob.D, prob.dx, prob.img, prob.axis;
            ind=depth_idx, grid_to_vec=prob.grid_to_vec,
        )
        model = (t, p) -> φ .* slab_flux(p[1], depth_actual, t; C1=C1, C2=C2, L=L, terms=terms)

    else
        error("Built-in diffusivity fitting only supports method ':conc', ':mass', and ':flux'.")
    end

    fit = curve_fit(model, xdata, ydata, param)
    D_app = fit.param[1]  # apparent diffusivity: D_pore / τ
    τ = D_pore / D_app
    D_eff = φ * D_app

    return τ, D_eff, xdata, ydata, fit, model
end
function fit_effective_diffusivity(
    sim::TransientState, prob::TransientDiffusionProblem, method::Symbol;
    depth=0.5, t_fit=(0, sim.t[end]), terms=100,
)
    return fit_effective_diffusivity(
        sim.t, sim.C, prob, method; depth=depth, t_fit=t_fit, terms=terms,
    )
end

"""
    fit_voxel_diffusivity(sim, prob; depth=0.5, n_samples=200,
                          t_fit=(0, sim.t[end]), terms=100, fit_depth=false)

Estimate voxel-wise tortuosity by fitting the transient concentration response
at a fixed slice depth to the analytic 1D diffusion solution. Samples
`n_samples` pore voxels uniformly from the slice at `depth` and fits each
independently.

# Keyword Arguments
- `depth`: fractional position along the main axis (`0` = inlet, `1` = outlet).
- `n_samples`: number of pore voxels to fit.
- `t_fit`: time window `(t_min, t_max)` for fitting.
- `terms`: number of eigenfunction terms in the analytic series.
- `fit_depth`: if `true`, also fit the effective depth as a free parameter.

# Returns
If `fit_depth == false`: `(tau_list, SE_tau_list, voxels)`
If `fit_depth == true`: `(tau_list, x_list, SE_tau_list, SE_x_list, voxels)`
"""
function fit_voxel_diffusivity(
    sim::TransientState, prob::TransientDiffusionProblem;
    depth=0.5, n_samples=200,
    t_fit=(0, sim.t[end]), terms=100,
    fit_depth::Bool=false,
)
    @assert !(prob.bc_inlet isa Function) "fit_voxel_diffusivity does not support f(t) inlet boundary conditions."
    @assert !(prob.bc_outlet isa Function) "fit_voxel_diffusivity does not support f(t) outlet boundary conditions."
    @assert !isnothing(prob.bc_inlet) "fit_voxel_diffusivity requires a numeric inlet boundary value."

    N = size(prob.img, axis_dim(prob.axis))
    depth_idx = round(Int, 1 + depth * (N - 1))

    slice_coords = slice_vec_indices(prob, depth_idx)
    slice_voxels = length(slice_coords)
    @assert slice_voxels >= n_samples "the number of samples to fit cannot exceed pore voxels at that index"

    voxels = slice_coords[round.(Int, LinRange(1, slice_voxels, n_samples))]

    tau_list = Float64[]
    x_list = Float64[]
    SE_tau_list = Float64[]
    SE_x_list = Float64[]

    idx_min = argmin(abs.(sim.t .- t_fit[1]))
    idx_max = argmin(abs.(sim.t .- t_fit[2]))
    @assert idx_min < idx_max "t_fit window must contain at least two time points"
    xdata = sim.t[idx_min:idx_max]

    insulated = isnothing(prob.bc_outlet)
    C1 = prob.bc_inlet
    C2 = insulated ? C1 : prob.bc_outlet
    L = (N - 1) * prob.dx * (insulated ? 2 : 1)

    # Cell-center convention: voxel index i maps to physical position (i - 0.5) * dx
    depth_actual = (depth_idx - 0.5) * prob.dx
    if fit_depth
        model = (t, p) -> slab_concentration(1 / p[1], p[2], t; C1=C1, C2=C2, L=L, terms=terms)
    else
        model = (t, p) -> slab_concentration(1 / p[1], depth_actual, t; C1=C1, C2=C2, L=L, terms=terms)
    end

    D_pore = prob.D isa Number ? Float64(prob.D) : Float64(sum(prob.D[prob.img]) / count(prob.img))
    p0 = fit_depth ? [1 / D_pore, depth] : [1 / D_pore]

    for i in voxels
        ydata = map(A -> A[i], sim.C[idx_min:idx_max])

        fit = curve_fit(model, xdata, ydata, p0)
        sigma = stderror(fit)

        push!(tau_list, fit.param[1] * D_pore)
        push!(SE_tau_list, sigma[1])

        if fit_depth
            push!(x_list, fit.param[2])
            push!(SE_x_list, sigma[2])
        end
    end

    if fit_depth
        return tau_list, x_list, SE_tau_list, SE_x_list, voxels
    else
        return tau_list, SE_tau_list, voxels
    end
end

# --- Analytical solutions for 1D slab diffusion ---

"""
    slab_concentration(D, x, t; terms=100, C1=1, C2=0, C0=0, L=1)

Analytical concentration in a 1D slab with constant diffusivity (Crank, *The
Mathematics of Diffusion*, 2nd ed., p. 50).
"""
function slab_concentration(D, x, t; terms=100, C1=1, C2=0, C0=0, L=1)
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
    slab_mass_uptake(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

Normalized mass uptake `M_t/M_inf` in a 1D slab with constant diffusivity
(Crank, p. 50).
"""
function slab_mass_uptake(D, t; terms=100, C1=1, C2=0, C0=0, L=1)
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
    slab_flux(D, x, t; terms=100, C1=1, C2=0, C0=0, L=1)

Analytical diffusive flux `J(x,t) = -D * ∂C/∂x` in a 1D slab (Crank, Ch. 3).
Returns 0 at `t=0` to avoid Fourier series truncation artifacts.
"""
function slab_flux(D, x, t; terms=100, C1=1, C2=0, C0=0, L=1)
    n = reshape(1:terms, 1, terms)
    m = 2 .* n .- 1

    t_is_scalar = t isa Number
    t_vec = t_is_scalar ? [t] : collect(t)
    t_mat = reshape(t_vec, :, 1)

    kn = n .* π ./ L
    km = m .* π ./ L

    dCdx_linear = (C2 - C1) / L

    dCdx_series1 =
        (2 / π) .* sum(
            ((C2 .* cos.(π .* n) .- C1) ./ n) .* (kn .* cos.(kn .* x)) .*
            exp.(-D .* (n .^ 2) .* π^2 .* t_mat ./ L^2);
            dims=2,
        )

    dCdx_series2 =
        (4 * C0 / π) .* sum(
            (1 ./ m) .* (km .* cos.(km .* x)) .* exp.(-D .* (m .^ 2) .* π^2 .* t_mat ./ L^2);
            dims=2,
        )

    dCdx = dCdx_linear .+ dropdims(dCdx_series1 .+ dCdx_series2; dims=2)

    # Enforce flux=0 at t=0 to avoid trig series truncation error
    if t_is_scalar
        return t == 0 ? 0.0 : -D * dCdx[1]
    else
        flux = -D .* dCdx
        flux[t_vec .== 0] .= 0
        return flux
    end
end

"""
    slab_cumulative_flux(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

Cumulative amount of substance that has diffused through the slab in the
interval `(0, t)` (Crank, p. 51).
"""
function slab_cumulative_flux(D, t; terms=100, C1=1, C2=0, C0=0, L=1)
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

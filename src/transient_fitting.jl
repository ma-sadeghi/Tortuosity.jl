# Analytical solutions to homogeneous transient diffusion and curve-fitting utilities

"""
    fit_effective_diffusivity(t, c_hist, prob, method; depth=0.5, t_fit=(0, t[end]), terms=100)
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
    t, c_hist, prob::TransientDiffusionProblem, method::Symbol;
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
    c1 = prob.bc_inlet
    c2 = insulated ? prob.bc_inlet : prob.bc_outlet
    L = (N - 1) * prob.voxel_size * (insulated ? 2 : 1)

    if method == :conc
        # Cell-centered FV with Dirichlet clamped at the centers of voxels 1
        # and N, so in the analytical coordinate those live at x=0 and x=L
        # respectively. Voxel i therefore sits at (i - 1) * voxel_size.
        depth_idx = round(Int, 1 + depth * (N - 1))
        depth_actual = (depth_idx - 1) * prob.voxel_size

        ydata = slice_concentration(
            c_hist[idx_min:idx_max], prob.img, prob.axis, depth_idx;
            pore_index=prob.pore_index, pore_only=true,
        )
        model = (t, p) -> slab_concentration(p[1], depth_actual, t; c1=c1, c2=c2, L=L, terms=terms)

    elseif method == :mass
        ydata = (mass_uptake(c_hist[1:idx_max], prob.img))[idx_min:end]
        model = (t, p) -> φ * (c1 + c2) / 2 .* slab_mass_uptake(p[1], t; c1=c1, c2=c2, L=L, terms=terms)

    elseif method == :flux
        # Flux is evaluated between voxels depth_idx and depth_idx+1, at the
        # face midpoint between their centers. In the analytical coordinate
        # that face sits at (depth_idx - 0.5) * voxel_size — half a voxel
        # downstream of voxel depth_idx's center, not at voxel depth_idx+1.
        depth_idx = round(Int, 0.5 + depth * (N - 1))
        if depth_idx == N
            depth_idx = N - 1
        end
        depth_actual = (depth_idx - 0.5) * prob.voxel_size

        ydata = flux(
            c_hist[idx_min:idx_max], prob.D, prob.voxel_size, prob.img, prob.axis;
            ind=depth_idx, pore_index=prob.pore_index,
        )
        model = (t, p) -> φ .* slab_flux(p[1], depth_actual, t; c1=c1, c2=c2, L=L, terms=terms)

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
    sol::TransientSolution, prob::TransientDiffusionProblem, method::Symbol;
    depth=0.5, t_fit=(0, sol.t[end]), terms=100,
)
    return fit_effective_diffusivity(
        sol.t, sol.u, prob, method; depth=depth, t_fit=t_fit, terms=terms,
    )
end

"""
    fit_voxel_diffusivity(sol, prob; depth=0.5, n_samples=200,
                          t_fit=(0, sol.t[end]), terms=100, fit_depth=false)

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
If `fit_depth == false`: `(taus, SE_taus, voxels)`
If `fit_depth == true`: `(taus, xs, SE_taus, SE_xs, voxels)`
"""
function fit_voxel_diffusivity(
    sol::TransientSolution, prob::TransientDiffusionProblem;
    depth=0.5, n_samples=200,
    t_fit=(0, sol.t[end]), terms=100,
    fit_depth::Bool=false,
)
    @assert !(prob.bc_inlet isa Function) "fit_voxel_diffusivity does not support f(t) inlet boundary conditions."
    @assert !(prob.bc_outlet isa Function) "fit_voxel_diffusivity does not support f(t) outlet boundary conditions."
    @assert !isnothing(prob.bc_inlet) "fit_voxel_diffusivity requires a numeric inlet boundary value."

    N = size(prob.img, axis_dim(prob.axis))
    depth_idx = round(Int, 1 + depth * (N - 1))

    slice_coords = slice_indices(prob, depth_idx)
    slice_voxels = length(slice_coords)
    @assert slice_voxels >= n_samples "the number of samples to fit cannot exceed pore voxels at that index"

    voxels = slice_coords[round.(Int, LinRange(1, slice_voxels, n_samples))]

    taus = Float64[]
    xs = Float64[]
    SE_taus = Float64[]
    SE_xs = Float64[]

    idx_min = argmin(abs.(sol.t .- t_fit[1]))
    idx_max = argmin(abs.(sol.t .- t_fit[2]))
    @assert idx_min < idx_max "t_fit window must contain at least two time points"
    xdata = sol.t[idx_min:idx_max]

    insulated = isnothing(prob.bc_outlet)
    c1 = prob.bc_inlet
    c2 = insulated ? c1 : prob.bc_outlet
    L = (N - 1) * prob.voxel_size * (insulated ? 2 : 1)

    # Cell-centered FV with Dirichlet clamped at voxel 1 and voxel N centers;
    # in the analytical coordinate voxel i lives at (i - 1) * voxel_size.
    depth_actual = (depth_idx - 1) * prob.voxel_size
    if fit_depth
        model = (t, p) -> slab_concentration(1 / p[1], p[2], t; c1=c1, c2=c2, L=L, terms=terms)
    else
        model = (t, p) -> slab_concentration(1 / p[1], depth_actual, t; c1=c1, c2=c2, L=L, terms=terms)
    end

    D_pore = prob.D isa Number ? Float64(prob.D) : Float64(sum(prob.D[prob.img]) / count(prob.img))
    p0 = fit_depth ? [1 / D_pore, depth] : [1 / D_pore]

    for i in voxels
        ydata = map(A -> A[i], sol.u[idx_min:idx_max])

        fit = curve_fit(model, xdata, ydata, p0)
        sigma = stderror(fit)

        push!(taus, fit.param[1] * D_pore)
        push!(SE_taus, sigma[1])

        if fit_depth
            push!(xs, fit.param[2])
            push!(SE_xs, sigma[2])
        end
    end

    if fit_depth
        return taus, xs, SE_taus, SE_xs, voxels
    else
        return taus, SE_taus, voxels
    end
end

# --- Analytical solutions for 1D slab diffusion ---

"""
    slab_concentration(D, x, t; terms=100, c1=1, c2=0, c0=0, L=1)

Analytical concentration in a 1D slab with constant diffusivity (Crank, *The
Mathematics of Diffusion*, 2nd ed., p. 50).
"""
function slab_concentration(D, x, t; terms=100, c1=1, c2=0, c0=0, L=1)
    n = reshape(1:terms, 1, terms)
    m = 2 .* n .- 1

    t_is_scalar = t isa Number
    t_vec = t_is_scalar ? [t] : collect(t)
    t = reshape(t_vec, :, 1)

    out = dropdims(
        c1 .+ (c2 - c1) .* x ./ L .+
        2 ./ π .* sum(
            (c2 .* cos.(π .* n) .- c1) ./ (n) .* sin.(n .* π .* x ./ L) .*
            exp.(-D .* (n) .^ 2 .* π^2 .* t / L^2);
            dims=2,
        ) .+
        4 .* c0 ./ π .* sum(
            1 ./ m .* sin.((2 .* n .- 1) .* (π .* x ./ L)) .*
            exp.(-D .* m .^ 2 .* π^2 .* t ./ L^2);
            dims=2,
        );
        dims=2,
    )

    return t_is_scalar ? out[1] : out
end

"""
    slab_mass_uptake(D, t; terms=100, c1=1, c2=0, c0=0, L=1)

Normalized mass uptake `M_t/M_inf` in a 1D slab with constant diffusivity
(Crank, p. 50).
"""
function slab_mass_uptake(D, t; terms=100, c1=1, c2=0, c0=0, L=1)
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
    slab_flux(D, x, t; terms=100, c1=1, c2=0, c0=0, L=1)

Analytical diffusive flux `J(x,t) = -D * ∂c/∂x` in a 1D slab (Crank, Ch. 3).
Returns 0 at `t=0` to avoid Fourier series truncation artifacts.
"""
function slab_flux(D, x, t; terms=100, c1=1, c2=0, c0=0, L=1)
    n = reshape(1:terms, 1, terms)
    m = 2 .* n .- 1

    t_is_scalar = t isa Number
    t_vec = t_is_scalar ? [t] : collect(t)
    t_mat = reshape(t_vec, :, 1)

    kn = n .* π ./ L
    km = m .* π ./ L

    dcdx_linear = (c2 - c1) / L

    dcdx_series1 =
        (2 / π) .* sum(
            ((c2 .* cos.(π .* n) .- c1) ./ n) .* (kn .* cos.(kn .* x)) .*
            exp.(-D .* (n .^ 2) .* π^2 .* t_mat ./ L^2);
            dims=2,
        )

    dcdx_series2 =
        (4 * c0 / π) .* sum(
            (1 ./ m) .* (km .* cos.(km .* x)) .* exp.(-D .* (m .^ 2) .* π^2 .* t_mat ./ L^2);
            dims=2,
        )

    dcdx = dcdx_linear .+ dropdims(dcdx_series1 .+ dcdx_series2; dims=2)

    # Enforce flux=0 at t=0 to avoid trig series truncation error
    if t_is_scalar
        return t == 0 ? 0.0 : -D * dcdx[1]
    else
        flux = -D .* dcdx
        flux[t_vec .== 0] .= 0
        return flux
    end
end

"""
    slab_cumulative_flux(D, t; terms=100, c1=1, c2=0, c0=0, L=1)

Cumulative amount of substance that has diffused through the slab in the
interval `(0, t)` (Crank, p. 51).
"""
function slab_cumulative_flux(D, t; terms=100, c1=1, c2=0, c0=0, L=1)
    n = reshape(1:terms, 1, terms)
    m = (2 .* n .- 1) .^ 2

    t_is_scalar = t isa Number
    t_vec = t_is_scalar ? [t] : collect(t)
    t = reshape(t_vec, :, 1)

    out = dropdims(
        D * (c1 - c2) .* t ./ L .+
        (2 * L / π^2) .* sum(
            (c1 .* cos.(π .* n) .- c2) ./ (n .^ 2) .*
            (1 .- exp.(-D .* (n) .^ 2 .* π^2 .* t / L^2));
            dims=2,
        ) .+
        (4 * c0 * L / π^2) .*
        sum(1 ./ m .* (1 .- exp.(-D .* m .* π^2 .* t ./ L^2)); dims=2);
        dims=2,
    )
    return t_is_scalar ? out[1] : out
end

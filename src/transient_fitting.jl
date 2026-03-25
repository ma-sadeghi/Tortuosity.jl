#file to accompany data generated with transient.jl
#contains analytical solutions to homogenous transient_diffusion
#and related utilities


##---------- Curve Fitting Utility ---------

"""
    effective_diffusivity(sim::TransientState, prob::TransientProblem, method::Symbol;
                          depth=0.5, t_fit=(0, sim.t[end]), terms=100)

Fit a 1D analytical transient‑diffusion solution to simulation data and extract the
effective transport coefficient. The fit uses a single physical degree of freedom:
the inverse tortuosity `τ⁻¹ = D_eff / φ`, where `φ` is the geometric porosity.

The result may not always match the tortuosity obtained from the steady state solution

Supported boundary modes are `(1, 0)` (Dirichlet–Neumann) and `(1, NaN)` (Dirichlet–insulated).
The inlet boundary must be numeric; insulated inlets are not supported.

# Supported observables
- `:conc` — concentration at a specified normalized depth.
- `:mass` — normalized mass uptake over the pore volume (independent of `depth`).
- `:flux` — flux between voxel slices at a specified depth.
note: for small domain images, 

# Keyword Arguments
- `depth` — fractional depth in `[0,1]` for `:conc` and `:flux`.  
             The value is snapped to the nearest voxel index.
- `t_fit` — time window `(tmin, tmax)` over which the fit is performed.
- `terms` — number of eigenfunction terms in the analytic series solution.

# Returns
A tuple:
τ, D_eff, xdata, ydata, fit, mode

Where:
- `τ` — fitted tortuosity. D_pore *φ/(D_eff)
    note: if prob.D is a scalar field, D_pore is set to mean(prob.D[img])
- `D_eff` — effective diffusivity.
- `xdata` — time points used in the fit.
- `ydata` — observable values used in the fit.
- `fit` — the `LsqFit` result object.
- `model` — the analytic model function used for fitting.

# Notes
- For `(1, NaN)` boundary conditions, the homogenous function domain is
    mirrored to enforce an effective insulated outlet.

This routine is useful for extracting homogenized transport coefficients,
validating transient behavior, and comparing different observables against
the same analytic model.
"""
function effective_diffusivity(t, C, prob::TransientProblem, method::Symbol; depth = 0.5, t_fit = (0, t[end]), terms = 100)
    
    # --- validate boundary type -----
    @assert !(prob.bound_mode[1] isa Function) "effective_diffusivity does not support f(t) inlet boundary conditions."
    @assert !(prob.bound_mode[2] isa Function) "effective_diffusivity does not support f(t) outlet boundary conditions."
    @assert !isnan(prob.bound_mode[1]) "effective_diffusivity does not support the inlet boundary being insulated"
    
    #porosity
    φ = porosity(prob)

    # inital guess for parameter vector
    D_pore = prob.D isa Number ? Float64(prob.D) : Float64(sum(prob.D[prob.img]) / count(prob.img)) 
    param = [D_pore]

    # get indexes for fitting window
    idx_min = argmin(abs.(t .- t_fit[1]))
    idx_max = argmin(abs.(t .- t_fit[2]))

    #index corresponding to normalized depth
    N = prob.dims[AXIS_DEFINITION[prob.axis]]

    #initialize fitting data
    xdata = t[idx_min:idx_max]
    ydata = nothing #depends on method

    model = nothing #depends on method
    
    #find effective boundary conditions of problem
    mirrored = isnan(prob.bound_mode[2]) #one insulated bound equivalent to mirroring domain
    C1 = prob.bound_mode[1]
    C2 = mirrored ? prob.bound_mode[1] : prob.bound_mode[2]
    L = N*prob.dx * (1+mirrored)

    #make assignments based on boundary mode and which observable is being fit to
    if method == :conc
        depth_idx = round(Int, 1 + depth*(N-1)) #voxel index 1 -> depth dx/2
        depth = (depth_idx-0.5)*prob.dx #redefine depth to closest value that matches an index

        ydata = get_slice_conc(C[idx_min:idx_max], prob, depth_idx; pore_only = true)
        model = (t, p) -> analytic_conc(p[1], depth, t; C1=C1, C2=C2, L=L, terms = terms)

    elseif method == :mass
        ydata = (mass_intake(C[1:idx_max], prob))[idx_min:end]
        model = (t, p) -> φ*(C1 + C2)/2 .*analytic_mass(p[1], t; C1=C1, C2=C2, L=L, terms = terms)


    elseif method == :flux
        depth_idx = round(Int, 0.5 + (depth)*(N-1)) #flux is between slices, offset by half index
        depth_idx == N && (depth_idx = N-1) #cannot calculate flux between index N and N+1
        depth = depth_idx*prob.dx #redefine depth to closest value that matches an index

        ydata = get_flux(C[idx_min:idx_max], prob;ind= depth_idx)
        model = (t, p) -> φ*analytic_flux(p[1], depth, t; C1=C1, C2=C2, L=L, terms = terms)

    else error("Built-in diffusivity fitting only supports method ':conc', ':mass', and ':flux'.") end
            

    #preform fit
    fit = curve_fit(model, xdata, ydata, param)
    D_rel = fit.param[1] #this is 1/tortuosity or D_eff/porosity
    τ = D_pore / D_rel
    D_eff = φ * D_pore * D_rel

    #optional extra info in returns for plotting or getting quality of fit
    return τ, D_eff, xdata, ydata, fit, model
end
effective_diffusivity(sim::TransientState, prob::TransientProblem, method::Symbol; depth = 0.5, t_fit = (0, sim.t[end]), terms = 100) = effective_diffusivity(sim.t, sim.C, prob::TransientProblem, method::Symbol; depth = depth, t_fit = t_fit, terms = terms)


"""
    voxel_tortuosity(sim, prob; depth=0.5, n_samples=200,
                     t_fit=(0, sim.t[end]), terms=100,
                     fit_depth=false)

Estimate voxel‑wise tortuosity by fitting the transient concentration
response at a fixed slice depth to the analytic 1D diffusion solution.

This routine extracts `n_samples` pore voxels from a transverse slice at
the specified fractional depth `depth ∈ [0,1]` along the transport axis,
and performs an independent nonlinear least‑squares fit on to the 
homogenous solution for diffusion in a 1D slab for each voxel.
The fitted parameter is `τ⁻¹` (the inverse tortuosity), and optionally
the effective depth `x` if `fit_depth=true`, which can better fit
    in cases where the steady state concentration and or curve shape
    is very different from the homogenous case at that depth.

# Arguments
- `sim::TransientState`: time‑dependent simulation state containing
  concentration fields `sim.C` and time vector `sim.t`.
- `prob::TransientProblem`: problem definition containing geometry,
  diffusivity, boundary conditions, and voxel spacing.

# Keyword Arguments
- `depth::Float64 = 0.5`  
  Fractional position along the main axis at which voxels are sampled.
  `depth=0` corresponds to the inlet boundary, `depth=1` to the outlet.
- `n_samples::Int = 200`  
  Number of pore voxels to fit. Voxels are sampled uniformly in index
  space from the set of pore voxels at the chosen slice.
- `t_fit::Tuple = (0, sim.t[end])`  
  Time window `(t_min, t_max)` over which the fit is performed.
- `terms::Int = 100`  
  Number of eigenfunction terms used in the analytic series solution.
- `fit_depth::Bool = false`  
  If `true`, the effective depth `x` is treated as a free parameter and
  fitted jointly with `τ⁻¹`. If `false`, the depth is fixed to the
  supplied `depth` value.

# Returns
If `fit_depth == false`:
(tau_list, SE_tau_list, voxels)
- `tau_list`: fitted values of `τ⁻¹` for each voxel  
- `SE_tau_list`: standard errors of the fitted `τ⁻¹`  
- `voxels`: linear indices of the sampled voxels

If `fit_depth == true`:
(tau_list, x_list, SE_tau_list, SE_x_list, voxels)
- `x_list`: fitted effective depths  
- `SE_x_list`: standard errors of the fitted depths  

# Notes
- The analytic model uses the appropriate Dirichlet–Neumann or
  Dirichlet–Dirichlet equivalent length depending on the boundary
  conditions in `prob.bound_mode`.
- If the diffusivity field is spatially varying, the initial guess for
  `τ⁻¹` uses the mean diffusivity over pore voxels.
- The function asserts that the slice contains at least `n_samples`
  pore voxels.

This routine is useful for probing spatial variability in transient
response, validating homogenization assumptions, and diagnosing
depth‑dependent deviations from 1D analytic behavior.
"""
function voxel_tortuosity(sim::TransientState, prob::TransientProblem;
                          depth=0.5, n_samples=200,
                          t_fit=(0, sim.t[end]), terms=100,
                          fit_depth::Bool = false
    )

    # --- determine slice index ---
    N = prob.dims[AXIS_DEFINITION[prob.axis]]
    depth_idx = round(Int, 1 + depth*(N-1))

    # get references to voxels to be fit, uniform w.r.t voxel density as opposed to spatially in image
    slice_coords = slice_vec_indices(prob, depth_idx)
    slice_voxels = length(slice_coords)
    @assert slice_voxels >= n_samples "the number of samples to fit cannot exceed pore voxels at that index"

    voxels = slice_coords[round.(Int, LinRange(1, slice_voxels, n_samples))]

    # --- prepare outputs ---
    tau_list = Float64[]
    x_list = Float64[] #effective depth or x_eff is not a good name for what this represents...
    # also get an idea of if the fit is decent
    SE_tau_list = Float64[]
    SE_x_list = Float64[]

    # --- time window ---
    idx_min = argmin(abs.(sim.t .- t_fit[1]))
    idx_max = argmin(abs.(sim.t .- t_fit[2]))
    @assert idx_min < idx_max "t_fit window must contain at least two time points"
    xdata = sim.t[idx_min:idx_max]

    # --- boundary conditions ---
    @assert !(prob.bound_mode[1] isa Function) "voxel_tortuosity does not support f(t) inlet boundary conditions."
    @assert !(prob.bound_mode[2] isa Function) "voxel_tortuosity does not support f(t) outlet boundary conditions."
    @assert !isnan(prob.bound_mode[1]) "voxel_tortuosity requires a numeric inlet boundary value; NaN is not allowed."
    C1 = prob.bound_mode[1]
    C2 = isnan(prob.bound_mode[2]) ? C1 : prob.bound_mode[2] #reflect solution for insulated bound equivalent
    L = (N-1)*prob.dx * (isnan(prob.bound_mode[2]) ? 2 : 1) #double length for insulated bound equivalent

    # --- homogenous solution ---
    model = fit_depth ? 
        (t, p) -> analytic_conc(1/p[1], p[2], t; C1=C1, C2=C2, L=L, terms=terms) :
        (t, p) -> analytic_conc(1/p[1], depth, t; C1=C1, C2=C2, L=L, terms=terms)

    #account for scalar field diffusivity option
    D_pore = prob.D isa Number ? Float64(prob.D) : Float64(sum(prob.D[prob.img]) / count(prob.img)) 
    p0 = fit_depth ? [1/D_pore, depth] : [1/D_pore]  # initial guess

    # --- fit each voxel ---
    for i in voxels
        ydata = map(A -> A[i], sim.C[idx_min:idx_max])

        fit = curve_fit(model, xdata, ydata, p0)
        sigma = stderror(fit)

        push!(tau_list, fit.param[1]*D_pore)
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

##---------- Analytical Solutions ----------------
# for homogenous versions of transient diffusion problems of interest
# solutions for two dirichlet bounds and a initial concentration that is the same at all x


"""
analytic_conc(D,x,t; terms=100, C1=1, C2=0, C0=0, L=1)

The Mathematics of Diffusion, Second Edition, Crank, pg. 50
analytical solution for diffusion in a slab of length L with constant diffusivity
returns Array of concentrations at positions x and times t
# Arguments
D: diffusion coefficient
x: position along slab length (0 to L)
t: time or array of times
# Keyword Arguments
terms: number of terms to include of the infinite series, default 50
C1: concentration at x=0 boundary, default 1
C2: concentration at x=L boundary, default 0
C0: initial concentration throughout slab, default 0
L: length of the slab, default 1
"""
function analytic_conc(D,x,t; terms=100, C1=1, C2=0, C0=0, L=1)
    
    #every array is along a different dimension, only sum along the series
    n = reshape(1:terms, 1, terms)
    m = 2 .*n .-1


    t_is_scalar = t isa Number  
    t_vec = t_is_scalar ? [t] : collect(t)  
    t = reshape(t_vec, :, 1)  

    out = dropdims(
        C1 .+ (C2-C1).*x./L .+
        2 ./π.*sum( (C2.*cos.(π .* n).-C1)./(n).* sin.(n.*π.*x./L ).*exp.(-D.*(n).^2 .*π^2 .*t/L^2)  ,dims = 2) .+ 
        4 .*C0./π .* sum(1 ./m.*sin.((2 .*n.-1).*(π.*x./L)).*exp.(-D.*m .^2 .*π^2 .*t ./L^2)   ,dims =2),
        dims = 2
    )

    return t_is_scalar ? out[1] : out

end

"""
analytic_mass(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

The Mathematics of Diffusion, Second Edition, Crank, pg. 50
analytical solution for normalized mass uptake in period (0,t) in a slab of length L with constant diffusivity
value is M_t/M_inf where M_inf = L*((C1+C2)/2 - C0)

returns Array of mass ratios at times in t
# Arguments
D: diffusion coefficient
t: time or array of times
# Keyword Arguments
terms: number of terms to include of the infinite series, default 50
C1: concentration at x=0 boundary, default 1
C2: concentration at x=L boundary, default 0
C0: initial concentration throughout slab, default 0
L: length of the slab, default 1
"""
function analytic_mass(D, t; terms=100, C1=1, C2=0, C0=0, L=1) #the concentration values are irrelevant if it's normalized
    
    #t array is along a different dimension, only sum along the series
    n = (2 .*reshape(1:terms, 1, terms).-1).^2
    t_is_scalar = t isa Number  
    t_vec = t_is_scalar ? [t] : collect(t)  
    t = reshape(t_vec, :, 1)  

    #also get rid of the summation term dimension
    out =  dropdims(1 .- 8 ./ (π^2) .* sum( (1 ./ n)  .* exp.(-D .* n .* ( π ./ L ).^2 .* t), dims=2), dims=2)
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
    dCdx_series1 = (2/π) .* sum(
            ((C2 .* cos.(π .* n) .- C1) ./ n) .* (kn .* cos.(kn .* x)) .*
             exp.(-D .* (n.^2) .* π^2 .* t ./ L^2),
            dims=2
    )

    # derivative of second sine series
    dCdx_series2 =(4*C0/π) .* sum(
            (1 ./ m) .* (km .* cos.(km .* x)) .* 
            exp.(-D .* (m.^2) .* π^2 .* t ./ L^2),
            dims=2
    )

    # total derivative
    dCdx = dCdx_linear .+ dropdims(dCdx_series1 .+ dCdx_series2, dims=2)

    # flux = -D * dC/dx
    # enforce flux=0 at t=0 to avoid trig series error
    if t_is_scalar
        return t == 0 ? -D*(C2 - C1)/L : 0
    else
        flux = -D .* dCdx
        flux[t .== 0] .= 0
        return flux
    end
end

"""
analytic_∑flux(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

The Mathematics of Diffusion, Second Edition, Crank, pg. 51
total amount of diffusing substance which has passed through the membrane in time (0,t)

returns Array of amounts at times t
# Arguments
D: diffusion coefficient
t: time or array of times
# Keyword Arguments
terms: number of terms to include of the infinite series, default 50
C1: concentration at x=0 boundary, default 1
C2: concentration at x=L boundary, default 0
C0: initial concentration throughout slab, default 0
L: length of the slab, default 1
"""
function analytic_∑flux(D, t; terms=100, C1=1, C2=0, C0=0, L=1)
    
    #every array is along a different dimension, only sum along the series
    n = reshape(1:terms, 1, terms)
    m = (2 .*n .-1).^2

    t_is_scalar = t isa Number  
    t_vec = t_is_scalar ? [t] : collect(t)  
    t = reshape(t_vec, :, 1)  

    out = dropdims(
        D*(C1-C2).*t./L .+
        (2*L/π^2).*sum( (C1.*cos.(π .* n).-C2)./(n.^2).* (1 .-exp.(-D.*(n).^2 .*π^2 .*t/L^2))  ,dims = 2) .+ 
        (4*C0*L/π^2) .* sum(1 ./m.* (1 .-exp.(-D.*m .*π^2 .*t ./L^2))   ,dims =2),
        dims = 2
    )
    return t_is_scalar ? out[1] : out

end

#file to accompany data generated with transient.jl
#contains analytical solutions to homogenous transient_diffusion
#and related utilities


##---------- Curve Fitting Utility ---------

"""
    effective_diffusivity(sim, prob, method; depth=0.5, t_fit=(0, sim.t[end]), terms=100, D0=1.0)

Fit an analytical transient‑diffusion solution to simulation data and return the
effective diffusivity. For bound_mode's (1,0) and (1,insulated)

Supported observables:
- `:conc` — concentration at a normalized depth.
- `:mass` — normalized mass uptake over full volume. independent of 'depth' kwarg
- `:flux` — not implemented.

Boundary modes must be `(1, 0)` or `(1, NaN)`.

# Keyword arguments
- `depth` — depth for `:conc` or `:flux` fits, redefined to depth of closest index for fit,
         TransientProblem defaults to depths from 0 to 1
- `t_fit` — time window `(tmin, tmax)` for fitting, defaults all timesteps in TransientState.
- `terms` — number of series terms in the analytical solution (infinite series).
- `D0` — initial diffusivity guess scalar.

# Returns
`D_eff, σ, fit, xdata, ydata`
"""
function effective_diffusivity(sim::TransientState, prob::TransientProblem, method::Symbol; depth = 0.5, t_fit = (0, sim.t[end]), terms = 100, D0 =1.0)
    
    D0 = Float64.(D0) #avoid issue if passing in int 
    param = [D0, porosity(prob)] # inital guess for parameter vector

    # get indexes for fitting window
    idx_min = argmin(abs.(sim.t .- t_fit[1]))
    idx_max = argmin(abs.(sim.t .- t_fit[2]))

    #index corresponding to normalized depth
    N = prob.dims[AXIS_DEFINITION[prob.axis]]

    #initialize fitting data
    xdata = sim.t[idx_min:idx_max]
    ydata = nothing #depends on method

    model = nothing #depends on method
    
    #find effective boundary conditions of problem
    C1 = prob.bound_mode[1]
    C2 = prob.bound_mode[2]
    L = (N-1)*prob.dx
    @assert !isnan(C1) "The depth = 0 boundary being insulated is not supported for effective diffusivity fitting"
    if isnan(C2)
        C2 = C1; L*=2 #effectively insulated bound on depth = (0,1), mirrored on (1,2)
    end

    #make assignments based on boundary mode and which observable is being fit to
    if method == :conc
        depth_idx = round(Int, 1 + depth*(N-1))
        depth = (depth_idx-1)*prob.dx #redefine depth to closest value that matches an index

        observable = A -> get_slice_conc(A, prob, depth_idx) #conc over time at that depth
        ydata = map(observable, sim.C[idx_min:idx_max])
        depth += prob.dx/2 # match reality that the flux is between two slices at idx, idx+1
        model = (t, p) -> p[2]*analytic_conc(p[1], depth, t; C1=C1, C2=C2, L=L, terms = terms)

    elseif method == :mass
        ydata = (mass_intake(sim.C[1:idx_max], prob))[idx_min:end]
        model = (t, p) -> p[2]*(C1 + C2)/2 .*analytic_mass(p[1], t; C1=C1, C2=C2, L=L, terms = terms)


    elseif method == :flux
        depth_idx = round(Int, 0.5 + (depth)*(N-1)) #flux is between slices, offset by half index
        depth_idx == N && (depth_idx = N-1) #cannot calculate flux between index N and N+1
        depth = (depth_idx-0.5)*prob.dx #redefine depth to closest value that matches an index

        observable = A -> get_flux(A, prob;ind= depth_idx) #conc over time at that depth
        ydata = map(observable, sim.C[idx_min:idx_max])
        model = (t, p) -> p[2]*analytic_flux(p[1], depth, t; C1=C1, C2=C2, L=L, terms = terms)

    else throw("Built-in diffusivity fitting only supports method ':conc', ':mass', and ':flux'.") end
            

    #preform fit
    fit = curve_fit(model, xdata, ydata, param)
    D_eff = fit.param[1] #need to consider whether this is D_eff of full volume or of pores and update accordingly, depends on method
    φ_eff = fit.param[2]


    #optional extra info in returns for plotting
    return D_eff, φ_eff, fit, xdata, ydata
end


function tortuosity()

nothing
end
##---------- Analytical Solutions ----------------
# for homogenous versions of transient diffusion problems of interest
# firstly for two opposite dirichlet bounds

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

    if length(t)>1
        t = reshape(t, length(t))
    end


    return dropdims(
        C1 .+ (C2-C1).*x./L .+
        2 ./π.*sum( (C2.*cos.(π .* n).-C1)./(n).* sin.(n.*π.*x./L ).*exp.(-D.*(n).^2 .*π^2 .*t/L^2)  ,dims = 2) .+ 
        4 .*C0./π .* sum(1 ./m.*sin.((2 .*n.-1).*(π.*x./L)).*exp.(-D.*m .^2 .*π^2 .*t ./L^2)   ,dims =2),
        dims = 2
    )

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
    if length(t)>1
        t = reshape(t, length(t))
    end

    #also get rid of the summation term dimension
    return  dropdims(1 .- 8 ./ (π^2) .* sum( (1 ./ n)  .* exp.(-D .* n .* ( π ./ L ).^2 .* t), dims=2), dims=2)
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
    if length(t) > 1
        t = reshape(t, length(t))
    end

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
    return -D .* dCdx
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

    if length(t)>1
        t = reshape(t, length(t))
    end

    return dropdims(
        D*(C1-C2).*t./L .+
        (2*L/π^2).*sum( (C1.*cos.(π .* n).-C2)./(n.^2).* (1 .-exp.(-D.*(n).^2 .*π^2 .*t/L^2))  ,dims = 2) .+ 
        (4*C0*L/π^2) .* sum(1 ./m.* (1 .-exp.(-D.*m .*π^2 .*t ./L^2))   ,dims =2),
        dims = 2
    )

end

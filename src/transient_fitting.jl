#file to accompany data generated with transient.jl
#contains analytical solutions to homogenous transient_diffusion
#and related utilities


##---------- Curve Fitting Utility ---------

"""
    effective_diffusivity(sim, prob, method; depth=0.5, t_fit=(0, sim.t[end]), terms=100, D0=1.0)

Fit an analytical transient‑diffusion solution to simulation data and return the
effective diffusivity.

Supported observables:
- `:conc` — concentration at a normalized depth.
- `:mass` — normalized mass uptake over full volume. independent of 'depth' kwarg
- `:flux` — not implemented.

Boundary modes must be `(1, 0)` or `(1, NaN)`.

# Keyword arguments
- `depth` — normalized depth for `:conc` fits.
- `t_fit` — time window `(tmin, tmax)` for fitting.
- `terms` — number of series terms in the analytical solution.
- `D0` — initial diffusivity guess (scalar or 1‑element vector).

# Returns
`D_eff, σ, fit, xdata, ydata`
"""
function effective_diffusivity(sim::TransientState, prob::TransientProblem, method::Symbol; depth = 0.5, t_fit = (0, sim.t[end]), terms = 100, D0 =1.0)
    
    #D0 (initial guess) will get passed to model as parameter vector
    D0 = Float64.(D0) #avoid annoying issue if passing in int 
    size(D0) == () && (D0 = [D0]) 

    # get indexes for fitting window
    idx_min = argmin(abs.(sim.t .- t_fit[1]))
    idx_max = argmin(abs.(sim.t .- t_fit[2]))

    #index corresponding to normalized depth
    N = prob.dims[AXIS_DEFINITION[prob.axis]]
    depth_idx = round(Int, N*depth)

    #initialize fitting data
    xdata = sim.t[idx_min:idx_max]
    ydata = nothing

    model = nothing #depends on method
    
    #insulated bound distribution is equivalent to half of symmetrical 2 dirichlet bounds distribution
    C1 = 1; C2 = 0; L=1
    if prob.bound_mode[1] == 1 && prob.bound_mode[2] == 0
        nothing #this fits with assignments outside of if statement
    elseif prob.bound_mode[1] == 1 && isnan(prob.bound_mode[2])
        C2 = 1; L=2 #effectively insulated bound
    else throw("Built-in diffusivity fitting only supports bound modes (1,0) and (1,NaN).") end

    #make assignments based on boundary mode and which observable is being fit to
    if method == :conc
        observable = A -> get_slice_conc(A, prob, depth_idx) #conc over time at that depth
        ydata = map(observable, sim.C[idx_min:idx_max])
        model = (t, p) -> analytic_conc(p[1], depth, t; C1=C1, C2=C2, L=L, terms = terms)

    elseif method == :mass
        ydata = (normalized_mass_intake(sim))[idx_min:idx_max] #this needs work
        model = (t, p) -> analytic_mass(p[1], t; C1=C1, C2=C2, L=L, terms = terms)

    elseif method == :flux
        throw(":flux not implemented")

    else throw("Built-in diffusivity fitting only supports method ':conc', ':mass', and ':flux'.") end
            

    #preform fit
    fit = curve_fit(model, xdata, ydata, D0)
    σ = stderror(fit)[1]
    D_eff = fit.param[1] #need to consider whether this is D_eff of full volume or of pores and update accordingly, depends on method

    #optional extra info in returns
    return D_eff, σ, fit, xdata, ydata
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
analytic_flux(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

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
function analytic_flux(D, t; terms=100, C1=1, C2=0, C0=0, L=1)
    
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

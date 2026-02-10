#file to accompany data generated with transient.jl
#contains analytical solutions to homogenous transient_diffusion
#and related utilities


##---------- Curve Fitting Abstraction ---------

using LsqFit

"""
Fits an analytical diffusion model to simulation data.

Arguments:
 model: function (t, p) → y
 t: vector of simulation times
 p: vector of parameters i.e. [D_eff]
 y: vector of observable values (same length as t)

Keyword arguments:
 t_span: tuple (tmin, tmax) time window to fit over
 p0: initial parameter guess

Returns:
 D_eff: fitted diffusion coefficient
 σ: standard error of the fit
 fit: the full LsqFit result
"""
function D_eff_from_fit(model, t, y; t_span= (0,maximum(t)), D0=[1.0])
    size(D0) == () && (D0 = [D0]) #parameters passed as vector
    
    # restrict to fitting window
    idx_min = argmin(abs.(t .- t_span[1]))
    idx_max = argmin(abs.(t .- t_span[2]))

    xdata = t[idx_min:idx_max]
    ydata = y[idx_min:idx_max]


    fit = curve_fit(model, xdata, ydata, D0)
    σ = stderror(fit)[1]
    D_eff = fit.param[1]

    return D_eff, σ, fit
end

##---------- Analytical Solutions ----------------
# for homogenous versions of transient diffusion problems of interest
# firstly for two opposite dirichlet bounds

"""
two_bounds_homog_C(D,x,t; terms=100, C1=1, C2=0, C0=0, L=1)

The Mathematics of Diffusion, Second Edition, Crank, pg. 50
analytical solution for diffusion in a slab of length L with constant diffusivity
returns Array of concentrations at positions x and times t
# Arguments
D: diffusion coefficient
x: position or array of positions along slab length (0 to L)
t: time or array of times
# Keyword Arguments
terms: number of terms to include of the infinite series, default 50
C1: concentration at x=0 boundary, default 1
C2: concentration at x=L boundary, default 0
C0: initial concentration throughout slab, default 0
L: length of the slab, default 1
"""
function two_bounds_homog_C(D,x,t; terms=100, C1=1, C2=0, C0=0, L=1)
    
    #every array is along a different dimension, only sum along the series
    n = reshape(1:terms, 1,1, terms)
    m = 2 .*n .-1

    if length(t)>1
        t = reshape(t, length(t))
    end
    if length(x)>1
        x = reshape(x, 1, length(x))
    end

    return dropdims(
        C1 .+ (C2-C1).*x./L .+
        2 ./π.*sum( (C2.*cos.(π .* n).-C1)./(n).* sin.(n.*π.*x./L ).*exp.(-D.*(n).^2 .*π^2 .*t/L^2)  ,dims = 3) .+ 
        4 .*C0./π .* sum(1 ./m.*sin.((2 .*n.-1).*(π.*x./L)).*exp.(-D.*m .^2 .*π^2 .*t ./L^2)   ,dims =3),
        dims = 3
    )

end

"""
two_bounds_homog_M(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

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
function two_bounds_homog_M(D, t; terms=100, C1=1, C2=0, C0=0, L=1) #the concentration values are irrelevant if it's normalized
    
    #t array is along a different dimension, only sum along the series
    n = (2 .*reshape(1:terms, 1, terms).-1).^2
    if length(t)>1
        t = reshape(t, length(t))
    end

    #also get rid of the summation term dimension
    return  dropdims(1 .- 8 ./ (π^2) .* sum( (1 ./ n)  .* exp.(-D .* n .* ( π ./ L ).^2 .* t), dims=2), dims=2)
end


"""
two_bounds_homog_Q(D, t; terms=100, C1=1, C2=0, C0=0, L=1)

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
function two_bounds_homog_Q(D, t; terms=100, C1=1, C2=0, C0=0, L=1)
    
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

# one dirichlet bounds, others insulated analytical homogenous solutions

#insulated bound distribution is equivalent to half of symmetrical 2 dirichlet bounds distribution
function one_bound_homog_C(D,x,t; terms=100, C1=1, C0=0, L=1)

    two_bounds_homog_C(D,x,t; terms = terms, C1=C1, C2=C1, C0 = C0, L= 2*L ) 
end


#insulated bound distribution is equivalent to half of symmetrical 2 dirichlet bounds distribution
function one_bound_homog_M(D, t; terms=100, C1=1, C0=0, L=1) #the concentration values are irrelevant if it's normalized
    two_bounds_homog_M(D, t; terms = terms, C1=1, C2=C1, L=2*L) 
end

##--- abstraction of curve fitting --- TO DO

function D_eff_equilibrium()
    nothing
end

function D_eff_from_fit(model, xdata, ydata, t_span)

end
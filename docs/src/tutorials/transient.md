# Transient Diffusion

The steady-state solver tells you the *equilibrium* transport properties of a porous medium. But porous features like dead-end channels and bottlenecks can produce transient behavior that deviates from homogeneous predictions. The transient solver captures these effects by solving the time-dependent diffusion equation.

## Boundary conditions

`TransientDiffusionProblem` accepts three types of boundary conditions for `bc_inlet` and `bc_outlet`:

| Type | Meaning | Example |
|------|---------|---------|
| `Number` | Dirichlet (fixed concentration) | `bc_inlet=1.0` |
| `nothing` | Insulated (zero-flux Neumann) | `bc_outlet=nothing` |
| `Function` | Time-dependent Dirichlet | `bc_inlet=t -> sin(2π*t)` |

## Basic workflow

The transient API is a thin adapter on top of [SciML's ODE machinery](https://docs.sciml.ai/OrdinaryDiffEq/stable/). You construct a `TransientDiffusionProblem` (the PDE discretisation), then call `solve(prob, alg; kwargs...)` exactly as you would with any other SciML problem:

```@example transient
using Plots
using Tortuosity

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
φ = Imaginator.phase_fraction(img, true)

# Steady-state tortuosity for comparison
sim_ss = SteadyDiffusionProblem(img; axis=:x, gpu=false)
sol_ss = solve(sim_ss.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)
c_ss = reconstruct_field(sol_ss.u, img)
τ_ss = tortuosity(c_ss, img; axis=:x)
```

Now set up and run the transient simulation:

```@example transient
prob = TransientDiffusionProblem(img; bc_inlet=1, bc_outlet=0, axis=:x, gpu=false)

# Run to near steady state, saving a snapshot every 0.05 time units.
sol = solve(prob, ROCK4();
    saveat   = 0.05,
    callback = StopAtFluxBalance(prob; abstol=0.005),
    tspan    = (0.0, 20.0))
```

Key parameters of `TransientDiffusionProblem`:
- **`bc_inlet`, `bc_outlet`** — boundary conditions (see table above).
- **`axis`** — transport direction.
- **`D`** — diffusivity (scalar or array, default `1.0`).
- **`voxel_size`** — physical voxel spacing. If `nothing`, set to `1/(N_axis - 1)` so the domain spans `[0, 1]`.
- **`gpu`** — same auto-detection as steady-state; see [GPU backends](@ref) for how to activate CUDA, Metal, or AMDGPU.

Key keyword arguments to `solve`:
- **`saveat`** — snapshot interval in diffusion time units (`length²/D`). **Not** the internal ODE timestep, which is adaptive.
- **`callback`** — a stop condition (see below) or a `CallbackSet` combining multiple callbacks.
- **`tspan`** — integration interval; defaults to `(0.0, Inf)` (terminate via callback). Pass a finite upper bound to cap the run.
- **`reltol`, `abstol`** — ODE tolerances. The Laplacian is mildly stiff, so the defaults are slightly tighter than OrdinaryDiffEq's.

The returned `sol` is a `TransientSolution` with:
- `sol.t::Vector{Float64}` — snapshot times.
- `sol.u::Vector{Vector{T}}` — CPU-resident snapshot arrays, one per saved time.
- `sol.retcode::Symbol` — `:Success` if the solver reached `tspan[2]`, `:Terminated` if a stop condition fired first.
- `sol.prob`, `sol.alg` — the originating problem and algorithm.
- `sol.ode_sol` — the raw `ODESolution` for power users who need solver diagnostics like `sol.ode_sol.destats`.

Snapshots live on CPU even when the solver runs on GPU — the wrapper uses `DiffEqCallbacks.SavingCallback` internally to materialise each saved state to CPU, so `sol.u` is safe for long-running simulations that would otherwise exhaust VRAM.

## Comparing to the homogeneous solution

Let's compare the transient outlet flux from the porous image to the analytical solution for a homogeneous slab with $D_\text{eff} = 1/\tau_\text{ss}$:

```@example transient
# Outlet flux from the transient simulation at each snapshot
flux_out = map(c -> flux(c, prob.D, prob.voxel_size, prob.img, prob.axis; ind=:end, pore_index=prob.pore_index), sol.u)

# Analytical outlet flux for a homogeneous slab (x=1 is the outlet face)
t_ana = range(0, 1.5*sol.t[end], 200)[2:end]
J_ana = φ .* slab_flux(1/τ_ss, 1, t_ana)

plot(sol.t, flux_out,
    title = "Outlet Flux over Time", xlabel = "time", ylabel = "flux",
    seriestype = :scatter, label = "transient (porous image)",
    legend = :bottomright
)
plot!(t_ana, J_ana,
    label = "homogeneous (D = 1/τ, τ = $(round(τ_ss, digits=2)))"
)
savefig("outlet_flux.svg"); nothing # hide
```

```@example transient
HTML("""<figure><img src=$(joinpath(Main.buildpath,"outlet_flux.svg"))><figcaption>Porous material transient flux vs. analytical homogeneous solution</figcaption></figure>""") # hide
```

The porous medium approaches steady state differently than a homogeneous slab — this is the transient "fingerprint" of the microstructure.

!!! note
    The discrepancy may be exaggerated for low-resolution images or images with a low domain-to-pore size ratio.

## Stop conditions

The transient solver composes with SciML's callback ecosystem. Pass any `DiscreteCallback`, `ContinuousCallback`, or `CallbackSet` as the `callback=` kwarg and the solver terminates when the callback fires.

Tortuosity ships four stop-condition factories tuned for diffusion problems:

| Factory | What it checks | Typical use case |
|---|---|---|
| `StopAtSteadyState(; abstol, reltol)` | `|dc[i]/dt| ≤ max(abstol, reltol·|c[i]|)` for every pore voxel | Strict near-steady-state detection — reads `du/dt` from the integrator cache, no geometry needed |
| `StopAtFluxBalance(prob; abstol, reltol)` | `|flux_inlet − flux_outlet| ≤ max(abstol, reltol·max(|f_in|, |f_out|))` | Porous-media-native: tolerance expressed in the same flux units used to derive effective diffusivity |
| `StopAtSaturation(c; abstol, reltol)` | `mean(u) ≥ c − max(abstol, reltol·|c|)` via `ContinuousCallback` rootfinding | Fill-until-saturation scenarios (e.g. insulated outlet) |
| `StopAtPeriodicState(freq, prob; reltol, …)` | Consecutive periods of an oscillating BC agree within `reltol·amplitude` | Sine-wave inlet: detect periodic steady state |

### Tolerance conventions

Each factory expresses its tolerance in whatever units are native to the observable it measures, which means the *interpretation* of `abstol` and `reltol` varies:

- **`StopAtSteadyState`** — threshold on `|dc/dt|` per voxel. Same units as OrdinaryDiffEq's adaptive step controller. Default `abstol=1e-8, reltol=1e-6`.
- **`StopAtFluxBalance`** — threshold on the inlet-outlet flux differential. Units of `D·Δc/Δx`. Default `abstol=1e-4, reltol=1e-3`.
- **`StopAtSaturation`** — offset on the target mean concentration. Terminates at the exact crossing of `mean(u) = c − tol` via rootfinding. Default `abstol=1e-4, reltol=1e-3` — the nonzero defaults protect against the asymptotic-approach footgun (picking `c` equal to the asymptotic limit would otherwise never fire).
- **`StopAtPeriodicState`** — relative tolerance only, scaled by the observed oscillation amplitude. Default `reltol=1e-2`.

### `tspan` as a hard cap

Every `solve` call also accepts a `tspan` kwarg. Pass `tspan=(0.0, t_max)` to enforce a maximum run duration; the solver will terminate at `t_max` (with `sol.retcode == :Success`) even if no callback fired. The combination of `tspan` + `callback` gives you both a soft convergence criterion and a hard timeout:

```julia
sol = solve(prob, ROCK4();
    saveat   = 0.05,
    callback = StopAtSteadyState(abstol=1e-5, reltol=1e-3),
    tspan    = (0.0, 100.0))  # safety cap
```

### Custom callbacks

Anything that works for a regular `ODEProblem` works here too. Compose with `CallbackSet`, write your own `DiscreteCallback`, or use other callbacks from `DiffEqCallbacks` (e.g., `PositiveDomain`). See the [DiffEqCallbacks docs](https://docs.sciml.ai/DiffEqCallbacks/stable/) for the full menu.

## Next steps

- [Advanced Transient](advanced_transient.md) — voxel-wise tortuosity fitting, periodic boundaries, and worked examples of each stop condition.

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

The transient workflow has three steps: create a `TransientDiffusionProblem`, initialize the state with `init_state`, and advance with `solve!`.

```@example transient
using Plots
using Tortuosity

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
φ = Imaginator.phase_fraction(img, true)

# Steady-state tortuosity for comparison
sim_ss = SteadyDiffusionProblem(img; axis=:x, gpu=false)
sol_ss = solve(sim_ss.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)
C_ss = reconstruct_field(sol_ss.u, img)
τ_ss = tortuosity(C_ss, img; axis=:x)
```

Now set up and run the transient simulation:

```@example transient
# dt is the snapshot interval — concentration is saved every dt time units.
# This is NOT the internal ODE timestep (which is adaptive).
dt = 0.05

prob = TransientDiffusionProblem(img, dt; bc_inlet=1, bc_outlet=0, axis=:x, gpu=false)
sim = init_state(prob)

# Stop when inlet and outlet fluxes converge (near steady state)
solve!(sim, prob, stop_at_flux_balance(0.005, prob))
```

Key parameters of `TransientDiffusionProblem`:
- **`dt`** — snapshot interval in diffusion time units ($\text{length}^2 / D$). Not the internal solver timestep.
- **`bc_inlet`, `bc_outlet`** — boundary conditions (see table above).
- **`axis`** — transport direction.
- **`D`** — diffusivity (scalar or array, default `1.0`).
- **`dx`** — voxel spacing (default `1.0`).
- **`gpu`** — same auto-detection as steady-state; see [GPU backends](@ref) for how to activate CUDA, Metal, or AMDGPU.

## Comparing to the homogeneous solution

Let's compare the transient outlet flux from the porous image to the analytical solution for a homogeneous slab with $D_\text{eff} = 1/\tau_\text{ss}$:

```@example transient
# Outlet flux from the transient simulation at each snapshot
flux_out = map(C -> flux(C, prob.D, prob.dx, prob.img, prob.axis; ind=:end, grid_to_vec=prob.grid_to_vec), sim.C)

# Analytical outlet flux for a homogeneous slab (x=1 is the outlet face)
t_ana = range(0, 1.5*sim.t[end], 200)[2:end]
J_ana = φ .* slab_flux(1/τ_ss, 1, t_ana)

plot(sim.t, flux_out,
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

`solve!` takes a stop condition function with signature `f(t_hist, C_hist) -> Bool`. When it returns `true`, the solver stops. Two commonly used built-in conditions:

- **`stop_at_time(t)`** — stops at time `t`.
- **`stop_at_flux_balance(delta, prob)`** — stops when inlet and outlet fluxes converge within `delta`.

You can also write your own:

```julia
my_stop = (t_hist, C_hist) -> t_hist[end] > 5.0
```

See [Advanced Transient](advanced_transient.md) for the full set of stop conditions and advanced techniques.

## Next steps

- [Advanced Transient](advanced_transient.md) — voxel-wise tortuosity fitting, periodic boundaries, and more stop conditions

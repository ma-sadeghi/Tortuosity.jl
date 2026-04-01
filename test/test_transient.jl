using Statistics
using Test
using Tortuosity
using Tortuosity: vec_to_grid, vec_to_slice, get_flux, analytic_flux

@testset "Transient diffusion" begin

    N = 32
    img = trues((N, N, N))
    img[:,1:N÷2,:] .= false #block off half the image, tortuosity of 1 and porosity of ~0.5
    porosity = Tortuosity.porosity(img)
    axis = :x
    dt = 0.05

    # --- steady state reference ---
    sim_ss = TortuositySimulation(img; axis=axis, gpu=false)
    sol_ss = solve(sim_ss.prob, KrylovJL_CG(); reltol=1e-6)

    # --- transient simulation ---
    prob = TransientProblem(img, dt; bound_mode=(1,0), axis=axis, gpu=false)
    sim = init_state(prob)

    stop_condition = Tortuosity.stop_at_delta_flux(1e-3, prob)
    solve!(sim, prob, stop_condition)

    @test length(sim.t) > 2
    @test length(sim.C[1]) == count(img)

    # --- boundary conditions ---
    inlet = vec_to_slice(sim.C[end],prob,1)
    outlet = vec_to_slice(sim.C[end],prob,N)

    @test all(isnan.(inlet) .|| inlet .≈ 1.0)      # inlet Dirichlet
    @test all(isnan.(outlet) .|| outlet .≈ 0.0)    # outlet Dirichlet

    # --- steady state consistency ---
    C_final = sim.C[end]
    @test all(isapprox.(C_final, Array(sol_ss.u); atol=1e-2))

    # --- flux monotonicity ---
    flux = get_flux(sim.C, prob)
    @test flux[1] ≈ 0.0 atol=1e-6
    @test flux[end] > flux[2]

    # --- analytic homogeneous comparison ---
    t_ana = sim.t
    J_ana = porosity * analytic_flux(1.0, 1.0, t_ana) # homogenous flux sol at outlet D=1.0

    @test J_ana[end] ≈ flux[end] atol=1e-2
end
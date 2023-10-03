using LinearSolve
using NPZ
using Test
using Tortuosity

# Load test fixtures
test_data = npzread("test/testdata.npz")
img = test_data["im"]
tau_gt = test_data["tau"]
ff_gt = test_data["ff"]

@testset "Blobs 3D, $(ax)-axis" for (i, ax) in enumerate([:x, :y, :z])
    # Compute τ and ℱ for each axis
    sim = TortuositySimulation(img, axis=ax)
    sol = solve(sim.prob, KrylovJL_CG(), reltol=1e-6)
    c = vec_to_field(sol.u, img)
    tau = tortuosity(c, ax)
    @test tau ≈ tau_gt[i] atol=1e-4
    ff = formation_factor(c, ax)
    @test ff ≈ ff_gt[i] atol=1e-4
end

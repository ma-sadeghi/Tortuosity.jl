using Tortuosity
using Tortuosity: Imaginator, TortuositySimulation, tortuosity, vec_to_grid

shape = (32, 32, 32)
img = Imaginator.blobs(; shape=shape, porosity=0.5, blobiness=1, seed=2);
img = Imaginator.trim_nonpercolating_paths(img; axis=:x);
sim = TortuositySimulation(img; axis=:x, gpu=true);
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);
c = vec_to_grid(sol.u, img);
τ = tortuosity(c; axis=:x);

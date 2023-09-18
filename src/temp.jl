using AlgebraicMultigrid
using LinearSolve
using Plots
using Random

includet("dnstools.jl")
includet("imgen.jl")
includet("numpytools.jl")
includet("pdetools.jl")
includet("plottools.jl")
includet("simulations.jl")
includet("topotools.jl")


Random.seed!(2)

img = blobs(shape=(64, 64, 64), porosity=0.65, blobiness=1)
img = denoise(img, 2)
display(imshow(img, slice=1))

prob = tortuosity_fdm(img, axis=:x)
sol_cg = solve(prob, KrylovJL_CG(), verbose=true, reltol=1e-5)
sol_amg = solve(prob.A, prob.b, RugeStubenAMG(), maxiter=1000, reltol=1e-5)

c_cg = vec_to_field(sol_cg.u, img)
c_amg = vec_to_field(sol_amg, img)

display(imshow(c_cg, slice=1))
display(imshow(c_amg, slice=1))

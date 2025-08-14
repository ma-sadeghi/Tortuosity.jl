
using DifferentialEquations

function f!(du, u, p, t)
    D = p
    dx = 1.0 / (length(u) - 1)
    for i in 2:length(u)-1
        du[i] = D * (u[i+1] - 2u[i] + u[i-1]) / dx^2
    end
    du[1] = du[end] = 0.0  # boundary conditions
end

N = 100
u0 = zeros(N)
tspan = (0.0, 10.0)
D = 0.01

prob = ODEProblem(f!, u0, tspan, D)
sol = solve(prob, Tsit5())

using Plots
plot(sol, xlabel="x", ylabel="C", label="Concentration over time")
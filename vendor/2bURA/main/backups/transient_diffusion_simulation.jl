
using DifferentialEquations
using Plots

# Parameters
pore_num = 40
L = 0.01
spacing = L / pore_num
pore_diam = spacing * 0.75
throat_diam_factor = 0.5
throat_diam = pore_diam * throat_diam_factor

# Create a 2D grid of pores
N = pore_num^2
function index(i, j)
    return (j - 1) * pore_num + i
end

# Initial concentration
initial_concentration = zeros(N)
left_value = 1.0
right_value = 0.0

# Apply boundary conditions
function apply_bc!(u)
    for j in 1:pore_num
        u[index(1, j)] = left_value
        u[index(pore_num, j)] = right_value
    end
end

# Diffusion model
function diffusion!(du, u, p, t)
    D = 2.095e-5  # Diffusivity
    for i in 1:pore_num
        for j in 1:pore_num
            idx = index(i, j)
            du[idx] = 0.0
            neighbors = []
            if i > 1
                push!(neighbors, index(i - 1, j))
            end
            if i < pore_num
                push!(neighbors, index(i + 1, j))
            end
            if j > 1
                push!(neighbors, index(i, j - 1))
            end
            if j < pore_num
                push!(neighbors, index(i, j + 1))
            end
            for n in neighbors
                du[idx] += D * (u[n] - u[idx]) / spacing^2
            end
        end
    end
    apply_bc!(du)
end

# Time span
tspan = (0.0, 10.0)
prob = ODEProblem(diffusion!, initial_concentration, tspan)
sol = solve(prob, Tsit5(), saveat=0.5)

# Visualization
final_conc = reshape(sol[end], (pore_num, pore_num))
heatmap(final_conc, title="Concentration Map (at t = end)", color=:rainbow)

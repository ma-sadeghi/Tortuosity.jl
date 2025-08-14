using DifferentialEquations, SparseArrays, Plots, LinearAlgebra
using ColorSchemes

# Grid setup
pore_num = 40
L = 0.01
spacing = L / pore_num
pore_diam = spacing * 0.75
throat_diam_factor = 0.5
throat_diam = pore_diam * throat_diam_factor
throat_len = spacing - pore_diam / 2 - pore_diam / 2
A_throat = π * (throat_diam / 2)^2

# Physical parameters
D_true = 2.09488e-5  # m^2/s, oxygen in air
g_throat = D_true * A_throat / throat_len

function generate_conductance_matrix(N, g)
    num_pores = N * N
    A = spzeros(num_pores, num_pores)

    for i in 1:N, j in 1:N
        p = (j - 1) * N + i
        if i < N
            q = (j - 1) * N + i + 1
            A[p, q] = g
            A[q, p] = g
        end
        if j < N
            q = j * N + i
            A[p, q] = g
            A[q, p] = g
        end
    end

    return A
end

G = generate_conductance_matrix(pore_num, g_throat)

row_sums = vec(sum(G, dims=2))
L = spdiagm(0 => row_sums) - G


C_left = 1.0
C_right = 0.0
num_pores = pore_num^2

function apply_BC!(L, b, N; left=C_left, right=C_right)
    for j in 1:N
        left_idx = (j - 1) * N + 1
        right_idx = (j - 1) * N + N

        L[left_idx, :] .= 0
        L[left_idx, left_idx] = 1
        b[left_idx] = left

        L[right_idx, :] .= 0
        L[right_idx, right_idx] = 1
        b[right_idx] = right
    end
end

function diffusion_ode!(du, u, p, t)
    du[:] = -L * u
end

u0 = fill(0.0, num_pores)  # Initial concentration
tspan = (0.0, 5.0)
prob = ODEProblem(diffusion_ode!, u0, tspan)
sol = solve(prob, saveat=0.1)

C_final = sol[end]
C_mat = reshape(C_final, (pore_num, pore_num)) |> reverse  # Rotate to match orientation

reversed_rainbow = reverse(ColorSchemes.rainbow.colors)

heatmap(C_mat, c=reversed_rainbow, aspect_ratio=:equal,
    title="Concentration Map (t = $(tspan[2]))",
    xlabel="x", ylabel="y", colorbar_title="Concentration")

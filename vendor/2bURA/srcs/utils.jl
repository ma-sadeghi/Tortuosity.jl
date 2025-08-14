# utils.jl

# Tastefully written by Harry Kim
# Date: June 1st, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 


## File: Contains  helper functrions for visualization/geometry

function generate_mask(N, sphere_radius, num_spheres)
    mask = ones(Float32, N, N)
    rng = MersenneTwister(1234) # for reproducibility
    for _ in 1:num_spheres
        x_c = rand(rng, sphere_radius+1:N-sphere_radius)
        y_c = rand(rng, sphere_radius+1:N-sphere_radius)
        for i in 1:N, j in 1:N
            if (i - x_c)^2 + (j - y_c)^2 ≤ sphere_radius^2
                mask[i, j] = 0.0
            end
        end
    end
    return mask
end

function visualize_final_concentration(sol, C_left, C_right)
    final_C_cpu = reshape(Array(sol.u[end]), N, N) # Copy to CPU
    final_C_norm = (final_C_cpu .- C_right) ./ (C_left - C_right)
    p = heatmap(final_C_norm',
        title="Final Concentration ($(selected_backend) backend)",
        yflip=true, colorbar=true, c=reverse(ColorSchemes.rainbow.colors))
    display(p)
end
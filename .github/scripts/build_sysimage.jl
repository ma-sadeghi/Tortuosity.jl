using PackageCompiler
using Libdl # To get the correct file extension

@info "Julia build script started."

# --- Configuration from Command Line Arguments ---
output_name = "TortuositySysimage-$(Sys.MACHINE)"
cpu_target = "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)"
packages_to_compile = ["Tortuosity"] # Ensure this matches your package name/module
precompile_script = ".github/scripts/precompile.jl" # Assumed to be in repo root relative to workflow execution dir

# --- Print Configuration ---
@info "--- Sysimage Build Configuration ---"
@info "Packages: $packages_to_compile"
@info "Output: $output_name"
@info "Precompile Script: $precompile_script"
@info "CPU Target: $cpu_target"
@info "------------------------------------"

# --- Validation ---
if !isfile(precompile_script)
    @warn "Precompile script '$precompile_script' not found in current directory. Continuing without precompilation file."
    precompile_args = NamedTuple() # Don't pass the argument if file not found
else
    precompile_args = (precompile_execution_file=precompile_script,)
end

# --- Build ---
expected_extension = ".$(Libdl.dlext)"
output_path = output_name * expected_extension
@info "Calling create_sysimage..."

create_sysimage(
    packages_to_compile;
    sysimage_path=output_path,
    cpu_target=cpu_target,
    include_transitive_dependencies=true,
    precompile_args..., # Splat the NamedTuple containing precompile arg if applicable
)

@info "create_sysimage call finished."

# --- Verify and Prepare Output for GitHub Actions ---

if !isfile(output_path)
    error("Sysimage build failed, output file '$output_path' not found.")
else
    @info "Successfully built sysimage: $output_path"
    # Set environment variables for subsequent GitHub Actions steps
    # This is the modern way using environment files
    if "GITHUB_ENV" in keys(ENV)
        github_env_file = ENV["GITHUB_ENV"]
        open(github_env_file, "a") do io
            println(io, "SYSIMAGE_ARTIFACT_PATH=$output_path")
            println(io, "SYSIMAGE_ARTIFACT_NAME=$output_path")
        end
        @info "Wrote artifact path/name to GITHUB_ENV"
    else
        @warn "GITHUB_ENV not found in environment variables. Cannot set artifact path/name for subsequent steps."
    end
end

@info "Julia build script finished successfully."

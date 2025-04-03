using PackageCompiler
using Libdl # To get the correct file extension

@info "Julia build script started."
@info "Number of args: ", length(ARGS)
for (i, arg) in enumerate(ARGS)
    @info "Arg $i: $arg"
end

if length(ARGS) != 2
    error("Usage: build_sysimage_script.jl <output_base_name> <cpu_target>")
end

# --- Configuration from Command Line Arguments ---
output_base_name = ARGS[1]
cpu_target_to_use = ARGS[2]
packages_to_compile = ["Tortuosity"] # Ensure this matches your package name/module
precompile_script = ".github/scripts/precompile.jl" # Assumed to be in repo root relative to workflow execution dir

# --- Print Configuration ---
@info "--- Sysimage Build Configuration ---"
@info "Packages: ", packages_to_compile
@info "Output Base: ", output_base_name
@info "Precompile Script: ", precompile_script
@info "CPU Target: ", cpu_target_to_use
@info "---------------------------------"

# --- Validation ---
if !isfile(precompile_script)
    @warn "Precompile script '$precompile_script' not found in current directory. Continuing without precompilation file."
    precompile_args = NamedTuple() # Don't pass the argument if file not found
else
    precompile_args = (precompile_execution_file=precompile_script,)
end

# --- Build ---
expected_extension = ".$(Libdl.dlext)"
output_path = output_base_name * expected_extension
@info "Calling create_sysimage..."

create_sysimage(
    packages_to_compile;
    sysimage_path=output_path,
    cpu_target=cpu_target_to_use,
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
    github_env_file = ENV["GITHUB_ENV"]
    open(github_env_file, "a") do io
        println(io, "SYSIMAGE_ARTIFACT_PATH=$output_path")
        println(io, "SYSIMAGE_ARTIFACT_NAME=$output_path")
    end
    @info "Wrote artifact path/name to GITHUB_ENV"
end

@info "Julia build script finished successfully."

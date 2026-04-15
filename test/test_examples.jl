# Smoke-test every script in examples/ by evaluating it end-to-end.
# Catches regressions like missing imports, undefined symbols, or
# `gpu=true` without a backend loaded. Only checks that each script
# runs without raising — outputs and accuracy are covered elsewhere.

examples_dir = joinpath(@__DIR__, "..", "examples")
scripts = ("demo_basic.jl", "demo_variable_diffusivity.jl", "demo_transient.jl")

# Sanitize each script for headless CI:
#   - strip `using Plots` (Plots is not a test dependency; all plotting
#     calls in the examples are guarded by `PLOT && ...`)
#   - force `PLOT = false` so the guarded calls don't fire
#   - force `USE_GPU = false` so we stay on CPU regardless of what the
#     script's default happens to be
function sanitize_for_ci(src)
    src = replace(src, r"^using Plots\s*$"m => "")
    src = replace(src, r"^PLOT\s*=\s*true\s*$"m => "PLOT = false")
    src = replace(src, r"^USE_GPU\s*=\s*true\s*$"m => "USE_GPU = false")
    return src
end

@testset "$script" for script in scripts
    path = joinpath(examples_dir, script)
    src = sanitize_for_ci(read(path, String))
    sandbox = Module(:ExampleSmokeTest)
    @test (Base.include_string(sandbox, src, script); true)
end

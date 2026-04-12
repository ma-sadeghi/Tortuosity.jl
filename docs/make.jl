using Documenter
using Tortuosity

ENV["GKSwstype"] = "100"
const buildpath = haskey(ENV, "CI") ? ".." : ""

# Run docstring-level jldoctests as a separate step. We don't pass `modules`
# to `makedocs` because that would also turn on the "docstring not referenced"
# check, which would flood CI with warnings — api.md is hand-curated markdown
# rather than `@docs` blocks, so most docstrings are technically "unreferenced"
# from Documenter's point of view. Calling `doctest(Tortuosity)` separately
# runs the in-source jldoctests and nothing else.
doctest(Tortuosity; manual=false)

format = Documenter.HTML(;
    edit_link="main",
    prettyurls=get(ENV, "CI", nothing) == "true",
    assets=[joinpath("assets", "style.css")],
)

makedocs(;
    sitename="Tortuosity.jl",
    format=format,
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Steady-State Tortuosity" => "tutorials/steady_state.md",
            "Variable Diffusivity" => "tutorials/variable_diffusivity.md",
            "Transient Diffusion" => "tutorials/transient.md",
            "Advanced Transient" => "tutorials/advanced_transient.md",
        ],
        "Imaginator" => "imaginator.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/ma-sadeghi/Tortuosity.jl.git",
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
    forcepush=true,
    push_preview=true,
    devbranch="main",
)

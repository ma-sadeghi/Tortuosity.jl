using Documenter
using Tortuosity

ENV["GKSwstype"] = "100"
const buildpath = haskey(ENV, "CI") ? ".." : ""

format = Documenter.HTML(;
    edit_link="main",
    prettyurls=get(ENV, "CI", nothing) == "true",
    assets=[joinpath("assets", "style.css")],
)

makedocs(;
    sitename="Tortuosity.jl",
    format=format,
    # `modules` is what teaches Documenter which docstrings the `@docs` blocks
    # in api.md should resolve against, and it's also what enables jldoctests
    # embedded in those docstrings to run. `checkdocs=:exports` tells
    # Documenter to only warn about unreferenced *exported* symbols (internal
    # helpers are fine). `warnonly=[:missing_docs]` keeps the build green if
    # an export is added without a corresponding `@docs` block — the CI log
    # still shows the warning so we don't forget to wire it up.
    modules=[Tortuosity, Tortuosity.Imaginator],
    checkdocs=:exports,
    warnonly=[:missing_docs],
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

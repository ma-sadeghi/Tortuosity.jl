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

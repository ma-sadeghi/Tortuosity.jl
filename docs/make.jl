using Documenter
using Tortuosity

ENV["GKSwstype"] = "100"

format = Documenter.HTML(;
    edit_link="main",
    prettyurls=get(ENV, "CI", nothing) == "true",
    assets=[joinpath("assets", "style.css")],
)

makedocs(;
    sitename="Tortuosity.jl",
    format=format,
    pages=[
        "Getting started" => "index.md",
        "Imaginator" => "imaginator.md",
        "Variable diffusivity" => "variable_diffusivity.md",
        "Benchmark" => "benchmark.md",
    ],
)

deploydocs(;
    repo="github.com/ma-sadeghi/Tortuosity.jl.git",
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
    forcepush=true,
    push_preview=true,
    devbranch="main",
)

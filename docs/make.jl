using Documenter
using Tortuosity

format = Documenter.HTML(;
    edit_link="main",
    prettyurls=get(ENV, "CI", nothing) == "true",
    assets=[joinpath("assets", "style.css")],
)

makedocs(; sitename="Tortuosity.jl", format=format)

deploydocs(;
    repo="github.com/ma-sadeghi/Tortuosity.jl.git",
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
    forcepush=true,
    push_preview=true,
    devbranch="main",
)

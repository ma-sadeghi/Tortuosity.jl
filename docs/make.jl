using Documenter, Tortuosity

format = Documenter.HTML(;
    edit_link="main",
    # prettyurls=get(ENV, "CI", nothing) == "true",
    assets=[joinpath("assets", "style.css")],
)

makedocs(; sitename="Tortuosity.jl", format=format)

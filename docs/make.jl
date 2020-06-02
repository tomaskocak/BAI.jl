using BAI
using Documenter

makedocs(;
    modules=[BAI],
    authors="Tomas Kocak <tomas.kocak@gmail.com> and contributors",
    repo="https://github.com/tomaskocak/BAI.jl/blob/{commit}{path}#L{line}",
    sitename="BAI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tomaskocak.github.io/BAI.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tomaskocak/BAI.jl",
)

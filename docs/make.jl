using SubspaceTools
using Documenter

DocMeta.setdocmeta!(SubspaceTools, :DocTestSetup, :(using SubspaceTools); recursive=true)

makedocs(;
    modules=[SubspaceTools],
    authors="Simon Neville",
    sitename="SubspaceTools.jl",
    format=Documenter.HTML(;
        canonical="https://spneville.github.io/SubspaceTools.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/spneville/SubspaceTools.jl",
    devbranch="main",
)

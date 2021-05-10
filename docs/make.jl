using Documenter, SuiteSparse

makedocs(
    modules = [SuiteSparse],
    sitename = "SuiteSparse",
    pages = Any[
        "SuiteSparse" => "index.md"
        ]
    )

deploydocs(repo = "github.com/JuliaLang/SuiteSparse.jl.git")

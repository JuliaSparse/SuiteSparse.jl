using Pkg
using Test

@testset "SuiteSparse UUID" begin
    project_filename = joinpath(dirname(@__DIR__), "Project.toml")
    project = Pkg.TOML.parsefile(project_filename)
    uuid = project["uuid"]
    correct_uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"
    @test uuid == correct_uuid
end

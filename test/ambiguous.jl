using SuiteSparse
using Test

@testset "detect_ambiguities" begin
    @test_nowarn detect_ambiguities(SuiteSparse; recursive=true, ambiguous_bottom=false)
end

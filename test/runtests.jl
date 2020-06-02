using BAI
using Test


@testset "All tests" begin
    @testset "general" begin
        include("general_test.jl")
    end

    @testset "reducers" begin
        include("reducers_test.jl")
    end

    @testset "response oracle" begin
        include("response_oracle_test.jl")
    end
end
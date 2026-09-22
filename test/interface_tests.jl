using StateSpaceSets
using Test

@testset "AbstractVector indexing" begin
    X = StateSpaceSet(rand(10, 3))
    @test size(X) == (10,)
    @test axes(X) == (Base.OneTo(10),)
    @test eltype(X) == typeof(X[1])
    @test X[1:end, 2] == X[:, 2]
    @test X[2:end, 3] == X[2:10, 3]
end

@testset "slicing preserves the point representation" begin
    data = rand(12, 3)
    names = [:x, :y, :z]

    for C in (SVector, MVector, Vector)
        X = StateSpaceSet(data; container = C, names)

        @test eltype(copy(X)) == eltype(X)
        @test eltype(X[2:7]) == eltype(X)
        @test X[2:7, :x] == X[2:7, 1]

        mask = isodd.(eachindex(X))
        @test Matrix(X[mask]) == Matrix(X)[mask, :]

        Y = X[:, 2:3]
        @test Y[1] isa C
        @test Matrix(Y) == Matrix(X)[:, 2:3]
        @test Y[:, :y] == X[:, :y]

        empty = X[1:0]
        @test isempty(empty)
        @test dimension(empty) == dimension(X)
        @test eltype(empty) == eltype(X)

        V = view(X, 2:5)
        @test eltype(V) == eltype(X)
        @test V[:, :x] == X[2:5, :x]
    end
end

using Test, StateSpaceSets
using Statistics

@testset "statistics representation parity" begin
    data = rand(128, 4)
    names = [:x, :y, :z, :w]
    covariance = Statistics.cov(data)
    correlation = Statistics.cor(data)
    means = vec(mean(data; dims = 1))

    @testset "container = $(container)" for container in (SVector, MVector, Vector)
        d = StateSpaceSet(data; container, names)

        @test StateSpaceSets.cov(d) ≈ covariance
        @test StateSpaceSets.cor(d) ≈ correlation

        μ, Σ = mean_and_cov(d)
        @test μ ≈ means
        @test Σ ≈ covariance

        standardized = standardize(d)
        @test standardized[1] isa container
        @test standardized.names == names
        for column in columns(standardized)
            @test abs(mean(column)) < 1e-12
            @test std(column) ≈ 1
        end

        # Views should expose the same statistics interface as concrete sets.
        plain = StateSpaceSet(data; container)
        sub = view(plain, 1:64)
        @test StateSpaceSets.cov(sub) ≈ Statistics.cov(data[1:64, :])
        @test StateSpaceSets.cor(sub) ≈ Statistics.cor(data[1:64, :])
        μsub, Σsub = mean_and_cov(sub)
        @test μsub ≈ vec(mean(data[1:64, :]; dims = 1))
        @test Σsub ≈ Statistics.cov(data[1:64, :])
    end

    # Non-floating static data should use the generic Statistics fallback rather
    # than failing in the optimized SVector-only kernel.
    ints = StateSpaceSet(rand(1:10, 64, 3))
    @test StateSpaceSets.cov(ints) ≈ Statistics.cov(Matrix(ints))
    @test StateSpaceSets.cor(ints) ≈ Statistics.cor(Matrix(ints))
end

using StateSpaceSets
using LinearAlgebra, Random, Statistics, Test

@testset "rectangular box" begin
    @testset "D=$(D)" for D in (2, 31)
        seed = 1234  # Seed random number generator for reproducibility

        # Define rectangular box region
        min_bounds = rand(D) .- 5
        max_bounds = rand(D) .+ 5

        # Generate sampler and isinside functions
        region = HRectangle(min_bounds, max_bounds)

        gen, isinside = statespace_sampler(region, seed)

        # Test generated points are inside the box region
        for i in 1:250
            x = gen()
            @test all(min_bounds .<= x .<= max_bounds)
            @test isinside(x)
        end
    end
    rect = HRectangle(SVector(0,0), SVector(1,1))
    gen, isinside = statespace_sampler(rect, 1)
    @test gen() isa Vector
    gen, isinside = statespace_sampler(rect, UniformSampler(), 1)
    @test isinside(gen())
end

@testset "sphere" begin
    @testset "r = $(r)" for r in (0.1, 4.0)
        @testset "D=$(D)" for D in (2, 31)
            @testset "inside=$(inside)" for inside in (true, false)
                seed = 1234
                center = fill(rand(), D)

                R = inside ? HSphere : HSphereSurface
                region = R(r, center)

                # Generate sampler and isinside functions
                gen, isinside = statespace_sampler(region, seed)

                # Test generated points are inside the sphere region
                for i in 1:50
                    x = gen()
                    if inside
                        @test norm(x - center) < r
                    else
                        @test norm(x - center) ≈ r
                    end
                    @test isinside(x)
                end
            end
        end
    end
    rect = HSphere(0.1, SVector(0.1, 0.1))
    gen, isinside = statespace_sampler(rect, 1)
    @test gen() isa Vector
end

@testset "Latin hypercube #17" begin
    D = 4
    n = 32
    mins = [-2.0, 1.0, 4.0, -7.0]
    maxs = [3.0, 5.0, 9.0, -1.0]
    region = HRectangle(mins, maxs)
    gen, isinside = statespace_sampler(region, LatinHypercubeSampler(n), 1234)

    # Copy because the sampler deliberately reuses thread-local scratch storage.
    points = [copy(gen()) for _ in 1:n]
    @test all(isinside, points)

    # Every coordinate occupies every stratum exactly once in each complete batch.
    for d in 1:D
        strata = sort([
            floor(Int, n * (p[d] - mins[d]) / (maxs[d] - mins[d]))
            for p in points
        ])
        @test strata == collect(0:n-1)
    end

    points2 = [copy(gen()) for _ in 1:n]
    @test all(isinside, points2)
    for d in 1:D
        strata = sort([
            floor(Int, n * (p[d] - mins[d]) / (maxs[d] - mins[d]))
            for p in points2
        ])
        @test strata == collect(0:n-1)
    end

    g1, _ = statespace_sampler(region, LatinHypercubeSampler(8), 42)
    g2, _ = statespace_sampler(region, LatinHypercubeSampler(8), 42)
    @test [copy(g1()) for _ in 1:8] == [copy(g2()) for _ in 1:8]
    @test_throws ArgumentError LatinHypercubeSampler(0)
    @test_throws ArgumentError statespace_sampler(HRectangle([0.0], [0.0]), 1)
end

@testset "grid strategy dispatch" begin
    grid = (range(-1.0, 1.0; length = 10), range(2.0, 4.0; length = 10))
    gen, inside = statespace_sampler(grid, LatinHypercubeSampler(16), 7)
    @test all(inside(copy(gen())) for _ in 1:16)
end

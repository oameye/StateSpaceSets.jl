# Always benchmark the checkout containing this file rather than a registered StateSpaceSets.
pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using BenchmarkTools
using StateSpaceSets

function bench_hcat(N)
    x = rand(N)
    y = StateSpaceSet(rand(N, 2))
    z = StateSpaceSet(rand(N))
    ym = Matrix(y)
    zm = Matrix(z)

    println("\nN = $N")
    println("StateSpaceSet multi-input hcat")
    @btime hcat($x, $y, $z)

    println("Dense Matrix hcat lower-bound/reference")
    @btime hcat($x, $ym, $zm)

    println("Matrix-conversion workaround from issue #38")
    @btime StateSpaceSet(hcat($x, $ym, $zm))

    println("Mixed point-container promotion")
    ys = StateSpaceSet(rand(N, 2); container = SVector)
    ymv = StateSpaceSet(rand(N, 2); container = MVector)
    yv = StateSpaceSet(rand(N, 2); container = Vector)
    @btime hcat($ys, $ymv)
    @btime hcat($ys, $yv)
end

# N=10 reproduces the scale used in upstream #38; N=10_000 exposes throughput behavior.
bench_hcat(10)
bench_hcat(10_000)

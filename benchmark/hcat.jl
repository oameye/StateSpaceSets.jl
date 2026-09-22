pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using BenchmarkTools
using StateSpaceSets

x = rand(10)
y = StateSpaceSet(rand(10, 2))
z = StateSpaceSet(rand(10))

@btime hcat($x, $y, $z)
@btime StateSpaceSet(hcat($x, $(Matrix(y)), $(Matrix(z))))

x = rand(10_000)
y = StateSpaceSet(rand(10_000, 2))
z = StateSpaceSet(rand(10_000))

@btime hcat($x, $y, $z)

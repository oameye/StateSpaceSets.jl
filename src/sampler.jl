export statespace_sampler
export HSphere, HRectangle, HSphereSurface
export SamplingMethod, UniformSampler, LatinHypercubeSampler

using Random
using LinearAlgebra: norm

abstract type Region end
abstract type SamplingMethod end

"""
    UniformSampler()

Ordinary independent uniform sampling inside a region. This is the default strategy used by
`statespace_sampler(region)` and preserves the historical StateSpaceSets sampler behavior.
"""
struct UniformSampler <: SamplingMethod end

"""
    LatinHypercubeSampler(batchsize)

Random Latin-hypercube sampling for [`HRectangle`](@ref). Every consecutive block of
`batchsize` samples contains exactly one sample in each one-dimensional stratum along every
state-space dimension. A new randomized Latin hypercube is generated when a block is exhausted.

The implementation is dependency-free and reuses its plan and scratch storage between batches.
"""
struct LatinHypercubeSampler <: SamplingMethod
    batchsize::Int
    function LatinHypercubeSampler(batchsize::Integer)
        batchsize > 0 || throw(ArgumentError("Latin-hypercube batch size must be positive."))
        new(Int(batchsize))
    end
end

"""
    statespace_sampler(region [, method = UniformSampler()] [, seed]) → sampler, isinside

Create a sampling function and a predicate for a state-space `region`.

`sampler()` returns a point in the region using the selected `SamplingMethod`. The returned
point is a thread-local mutable `Vector`; copy it if it needs to be stored across subsequent
sampler calls. The default method is [`UniformSampler`](@ref).

Supported regions are:
- `HSphere(radius, center)` for the interior of a hypersphere;
- `HSphereSurface(radius, center)` for its surface;
- `HRectangle(mins, maxs)` for a hyperrectangle.

`LatinHypercubeSampler(n)` is defined for `HRectangle` and produces randomized Latin-hypercube
blocks of `n` points. The historical calls `statespace_sampler(region)` and
`statespace_sampler(region, seed)` remain equivalent to `UniformSampler()`.
"""
function statespace_sampler end

statespace_sampler(region::Region) =
    statespace_sampler(region, UniformSampler(), abs(rand(Int)))
statespace_sampler(region::Region, seed::Integer) =
    statespace_sampler(region, UniformSampler(), seed)
statespace_sampler(region::Region, method::SamplingMethod) =
    statespace_sampler(region, method, abs(rand(Int)))

"""
    HSphere(r::Real, center::AbstractVector)
    HSphere(r::Real, D::Int)

A state space region denoting all points _within_ a hypersphere.
"""
struct HSphere{T, V<:AbstractVector{T}} <: Region
    radius::T
    center::V
end
HSphere(r::Real, D::Int) = HSphere(r, zeros(eltype(r), D))

"""
    HSphereSurface(r::Real, center::AbstractVector)
    HSphereSurface(r::Real, D::Int)

A state space region denoting all points _on the surface_ (boundary)
of a hypersphere.
"""
struct HSphereSurface{T, V<:AbstractVector{T}} <: Region
    radius::T
    center::V
end
HSphereSurface(r::Real, D::Int) = HSphereSurface(r, zeros(eltype(r), D))

"""
    HRectangle(mins::AbstractVector, maxs::AbstractVector)

A state space region denoting all points _within_ the hyperrectangle.
"""
struct HRectangle{T, V<:AbstractVector{T}} <: Region
    mins::V
    maxs::V
end
HRectangle(mins::Tuple, maxs::Tuple) = HRectangle(SVector(mins), SVector(maxs))

function statespace_sampler(region::HSphere, ::UniformSampler, seed::Integer)
    return sphereregion(region.radius, region.center, Xoshiro(seed), true)
end

function statespace_sampler(region::HSphereSurface, ::UniformSampler, seed::Integer)
    return sphereregion(region.radius, region.center, Xoshiro(seed), false)
end

function sphereregion(r, center, rng, inside)
    @assert r ≥ 0
    dim = length(center)
    dummies = [zeros(typeof(r), dim) for _ in 1:Threads.nthreads()]
    generator = SphereGenerator(r, center, dummies, inside, rng, length(center))
    if inside
        isinside = (x) -> norm(x .- center) < r
    else
        isinside = (x) -> norm(x .- center) ≈ r
    end
    return generator, isinside
end

struct SphereGenerator{T, V<:AbstractVector{T}, R} <: Function
    radius::T
    center::V
    dummies::Vector{Vector{T}}
    inside::Bool
    rng::R
    D::Int
end

function (s::SphereGenerator)()
    dummy = s.dummies[Threads.threadid()]
    r = s.radius
    randn!(s.rng, dummy)
    n = LinearAlgebra.norm(dummy)
    ρ = s.inside ? (rand(s.rng)^(1/s.D))*r : r
    dummy .*= ρ/n
    dummy .+= s.center
    return dummy
end

function _rectangle_data(region::HRectangle)
    as = region.mins
    bs = region.maxs
    length(as) == length(bs) > 0 || throw(ArgumentError(
        "Rectangle bounds must have the same positive dimension.",
    ))
    all(i -> as[i] < bs[i], eachindex(as)) || throw(ArgumentError(
        "Every lower rectangle bound must be strictly smaller than its upper bound.",
    ))
    T = as[1] isa AbstractFloat ? eltype(as) : Float64
    return T.(as), T.(bs .- as)
end

function _rectangle_isinside(region::HRectangle)
    as = region.mins
    bs = region.maxs
    return x -> length(x) == length(as) && all(i -> as[i] ≤ x[i] < bs[i], eachindex(x))
end

function statespace_sampler(region::HRectangle, ::UniformSampler, seed::Integer)
    mins, difs = _rectangle_data(region)
    dummies = [zeros(eltype(mins), length(mins)) for _ in 1:Threads.nthreads()]
    gen = RectangleGenerator(mins, difs, dummies, Xoshiro(seed))
    return gen, _rectangle_isinside(region)
end

struct RectangleGenerator{T, V<:AbstractVector{T}, R} <: Function
    mins::V
    difs::V
    dummies::Vector{Vector{T}}
    rng::R
end

function (s::RectangleGenerator)()
    dummy = s.dummies[Threads.threadid()]
    rand!(s.rng, dummy)
    dummy .*= s.difs
    dummy .+= s.mins
    return dummy
end

mutable struct LatinHypercubeGenerator{T, V<:AbstractVector{T}, R} <: Function
    mins::V
    difs::V
    dummies::Vector{Vector{T}}
    plan::Matrix{T}
    permutation::Vector{Int}
    rng::R
    next::Int
    lock::ReentrantLock
end

function _refill_latin_hypercube!(s::LatinHypercubeGenerator)
    D, n = size(s.plan)
    @inbounds for d in 1:D
        for k in 1:n
            s.permutation[k] = k
        end
        shuffle!(s.rng, s.permutation)
        for k in 1:n
            # One random point inside every stratum; rand ∈ [0, 1), so upper bounds
            # remain excluded exactly as for the historical rectangle sampler.
            u = (s.permutation[k] - 1 + rand(s.rng)) / n
            s.plan[d, k] = s.mins[d] + u*s.difs[d]
        end
    end
    s.next = 1
    return s
end

function statespace_sampler(
        region::HRectangle, method::LatinHypercubeSampler, seed::Integer,
    )
    mins, difs = _rectangle_data(region)
    D = length(mins)
    n = method.batchsize
    T = eltype(mins)
    dummies = [zeros(T, D) for _ in 1:Threads.nthreads()]
    gen = LatinHypercubeGenerator(
        mins, difs, dummies, Matrix{T}(undef, D, n), collect(1:n),
        Xoshiro(seed), n + 1, ReentrantLock(),
    )
    _refill_latin_hypercube!(gen)
    return gen, _rectangle_isinside(region)
end

function (s::LatinHypercubeGenerator)()
    dummy = s.dummies[Threads.threadid()]
    lock(s.lock)
    try
        s.next > size(s.plan, 2) && _refill_latin_hypercube!(s)
        k = s.next
        s.next += 1
        @inbounds for d in axes(s.plan, 1)
            dummy[d] = s.plan[d, k]
        end
    finally
        unlock(s.lock)
    end
    return dummy
end

"""
    statespace_sampler(grid::NTuple{N, AbstractRange} [, method] [, seed])

For a tuple of ranges, use their extrema to construct an `HRectangle` and dispatch to the
selected sampling strategy.
"""
function statespace_sampler(grid::NTuple{N, AbstractRange}) where {N}
    region = HRectangle(minimum.(grid), maximum.(grid))
    return statespace_sampler(region)
end

function statespace_sampler(grid::NTuple{N, AbstractRange}, seed::Integer) where {N}
    region = HRectangle(minimum.(grid), maximum.(grid))
    return statespace_sampler(region, seed)
end

function statespace_sampler(
        grid::NTuple{N, AbstractRange}, method::SamplingMethod,
    ) where {N}
    region = HRectangle(minimum.(grid), maximum.(grid))
    return statespace_sampler(region, method)
end

function statespace_sampler(
        grid::NTuple{N, AbstractRange}, method::SamplingMethod, seed::Integer,
    ) where {N}
    region = HRectangle(minimum.(grid), maximum.(grid))
    return statespace_sampler(region, method, seed)
end

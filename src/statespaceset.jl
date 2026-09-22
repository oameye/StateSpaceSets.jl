using StaticArraysCore, LinearAlgebra
using Base.Iterators: flatten
using Statistics

export AbstractStateSpaceSet, minima, maxima
export SVector, SMatrix, MVector
export minmaxima, columns, standardize, dimension
export cov, cor, mean_and_cov

# D = dimension, T = element type, V = container type, N = names
# note that the container type is given as keyword `container` to
# all functions that somehow end up making a state space set.
abstract type AbstractStateSpaceSet{D, T, V, N} <: AbstractVector{V} end

# Core extensions and functions:
"""
    dimension(thing) -> D

Return the dimension of the `thing`, in the sense of state space dimensionality.
"""
dimension(::AbstractStateSpaceSet{D}) where {D} = D
Base.vec(X::AbstractStateSpaceSet) = X.data
containertype(::AbstractStateSpaceSet{D,T,V}) where {D,T,V} = V

# Rebuild a concrete set without silently changing point representation or metadata.
# Keeping D/T/V from the source also makes empty row selections well-defined.
function _statespaceset_like(
        d::AbstractStateSpaceSet{D,T,V}, data; names = d.names,
    ) where {D,T,V}
    return StateSpaceSet{D,T,V}(collect(data); names)
end

@inline function _selected_names(names, indices)
    isnothing(names) && return nothing
    return collect(names)[indices]
end

###########################################################################################
# Base extensions
###########################################################################################
for f in (
        :length, :sort!, :iterate, :firstindex, :size,
    )
    @eval Base.$(f)(d::AbstractStateSpaceSet, args...; kwargs...) = $(f)(vec(d), args...; kwargs...)
end

Base.:(==)(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = vec(d1) == vec(d2)
Base.copy(d::AbstractStateSpaceSet) = _statespaceset_like(d, copy(vec(d)))
Base.sort(d::AbstractStateSpaceSet) = sort!(copy(d))
@inline Base.eltype(::Type{<:AbstractStateSpaceSet{D, T, V}}) where {D, T, V} = V
@inline Base.IteratorSize(::Type{<:AbstractStateSpaceSet}) = Base.HasLength()
@inline Base.IndexStyle(::Type{<:AbstractStateSpaceSet}) = IndexLinear()
Base.eachcol(ds::AbstractStateSpaceSet) = (ds[:, i] for i in 1:dimension(ds))

# StateSpaceSet is formally a one-dimensional AbstractVector. Base lowers `end` in a
# multi-index expression such as X[1:end, j] through lastindex(X, 1), so define that
# method explicitly while retaining standard AbstractVector semantics in higher dimensions.
@inline Base.lastindex(d::AbstractStateSpaceSet) = length(d)
@inline Base.lastindex(d::AbstractStateSpaceSet, dim::Integer) = dim == 1 ? length(d) : 1

"""
    columns(ssset) -> x, y, z, ...

Return the individual columns of the state space set allocated as `Vector`s.
Equivalent with `collect(eachcol(ssset))`.
"""
function columns end
@generated function columns(data::AbstractStateSpaceSet{D}) where {D}
    gens = [:(data[:, $k]) for k=1:D]
    quote tuple($(gens...)) end
end

###########################################################################################
# Indexing
###########################################################################################
# 1D indexing over the container elements:
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int) = vec(d)[i]
@inline Base.getindex(d::AbstractStateSpaceSet, i) = _statespaceset_like(d, vec(d)[i])

# 2D indexing with second index being column (reduces indexing to 1D indexing)
@inline Base.getindex(d::AbstractStateSpaceSet, i, ::Colon) = d[i]

# 2D convenience indexing where the dataset behaves as a matrix with each column a
# dynamic-variable timeseries. This does not change the formal one-dimensional Array shape.
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int, j::Int) = vec(d)[i][j]
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, j::Int) =
    [vec(d)[k][j] for k in eachindex(d)]
@inline Base.getindex(d::AbstractStateSpaceSet, i::AbstractVector, j::Int) =
    [vec(d)[k][j] for k in i]
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int, j::AbstractVector) = d[i][j]
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, ::Colon) = d
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, v::AbstractVector) =
    StateSpaceSet(
        [d[i][v] for i in eachindex(d)];
        container = containertype(d), names = _selected_names(d.names, v),
    )
@inline Base.getindex(d::AbstractStateSpaceSet, v1::AbstractVector, v::AbstractVector) =
    StateSpaceSet(
        [d[i][v] for i in v1];
        container = containertype(d), names = _selected_names(d.names, v),
    )

# Set index stuff
@inline Base.setindex!(d::AbstractStateSpaceSet, v, i::Int) = (vec(d)[i] = v)

function Base.dotview(d::AbstractStateSpaceSet, ::Colon, ::Int)
    error("`setindex!` is not defined for StateSpaceSets and the given arguments. "*
          "Best to create a new dataset or `Vector{SVector}` instead of in-place operations.")
end

###########################################################################
# Named dimensions
###########################################################################
Base.getindex(d::AbstractStateSpaceSet, i, s::Symbol) = d[i, name2idx(d.names, s)]
function name2idx(names, s)
    if isnothing(names)
        error("The state space set does not have names.")
    end
    j = findfirst(isequal(s), names)
    if isnothing(j)
        error("Name $(s) does not exist in names $(names)")
    end
    return j
end

###########################################################################
# Appending/concatenating
###########################################################################
Base.append!(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = (append!(vec(d1), vec(d2)); d1)
Base.push!(d::AbstractStateSpaceSet, new_item) = (push!(vec(d), new_item); d)
Base.vcat(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = append!(copy(d1), d2)

const _HCatInput = Union{AbstractVector{<:Real}, AbstractStateSpaceSet}
Base.hcat(x::AbstractStateSpaceSet, y::_HCatInput, zs::_HCatInput...) = _hcat(x, y, zs...)
Base.hcat(x::AbstractVector{<:Real}, y::AbstractStateSpaceSet, zs::_HCatInput...) = _hcat(x, y, zs...)

@inline _hcat_dimension(::AbstractVector{<:Real}) = 1
@inline _hcat_dimension(x::AbstractStateSpaceSet) = dimension(x)

# Container promotion is deliberately local to StateSpaceSet concatenation. It follows a
# least-restrictive policy: dynamic Vector dominates mutable-static MVector, which dominates
# immutable-static SVector. This makes the result independent of argument order and avoids
# silently discarding mutability.
@inline _container_rank(::Type{<:SVector}) = 1
@inline _container_rank(::Type{<:MVector}) = 2
@inline _container_rank(::Type{<:Vector}) = 3
@inline _container_rank(::Type{<:AbstractVector}) = 3

_hcat_typeinfo(::Type{<:AbstractVector{T}}) where {T<:Real} = (1, T, 0)
_hcat_typeinfo(::Type{<:AbstractStateSpaceSet{D,T,V}}) where {D,T,V} =
    (D, T, _container_rank(V))

# Dimensions and point-container families are encoded in the argument types. Compute the
# output metadata once at specialization time instead of rebuilding a runtime type from D on
# every call. This keeps the hot hcat path inference-friendly for downstream algorithms.
@generated function _hcat_output_metadata(::Type{TT}) where {TT<:Tuple}
    infos = map(_hcat_typeinfo, TT.parameters)
    D = sum(first(info) for info in infos)
    T = promote_type((info[2] for info in infos)...)
    rank = maximum(info[3] for info in infos)
    V = rank == 1 ? SVector{D,T} : rank == 2 ? MVector{D,T} : Vector{T}
    return :(Val{$D}(), $V)
end

@inline _val(::Val{D}) where {D} = D

@inline function _fill_hcat_point!(dest, xs, i)
    k = 1
    @inbounds for x in xs
        if x isa AbstractStateSpaceSet
            p = x[i]
            for j in eachindex(p)
                dest[k] = p[j]
                k += 1
            end
        else
            dest[k] = x[i]
            k += 1
        end
    end
    return dest
end

function _hcat_data(::Type{SVector{D,T}}, xs, L, ::Int) where {D,T}
    data = Vector{SVector{D,T}}(undef, L)
    scratch = Vector{T}(undef, D)
    @inbounds for i in 1:L
        _fill_hcat_point!(scratch, xs, i)
        data[i] = SVector{D,T}(scratch)
    end
    return data
end

function _hcat_data(::Type{MVector{D,T}}, xs, L, ::Int) where {D,T}
    data = Vector{MVector{D,T}}(undef, L)
    scratch = Vector{T}(undef, D)
    @inbounds for i in 1:L
        _fill_hcat_point!(scratch, xs, i)
        data[i] = MVector{D,T}(scratch)
    end
    return data
end

function _hcat_data(::Type{Vector{T}}, xs, L, D::Int) where {T}
    data = Vector{Vector{T}}(undef, L)
    @inbounds for i in 1:L
        point = Vector{T}(undef, D)
        _fill_hcat_point!(point, xs, i)
        data[i] = point
    end
    return data
end

function _hcat_names(xs)
    names = Symbol[]
    for x in xs
        x isa AbstractStateSpaceSet || return nothing
        isnothing(x.names) && return nothing
        append!(names, x.names)
    end
    return names
end

function _hcat(xs::_HCatInput...)
    L = length(first(xs))
    @inbounds for x in xs
        length(x) == L || error("StateSpaceSets must be of same length")
    end
    Dval, V = _hcat_output_metadata(typeof(xs))
    D = _val(Dval)
    T = eltype(V)
    data = _hcat_data(V, xs, L, D)
    return StateSpaceSet{D,T,V}(data; names = _hcat_names(xs))
end

#####################################################################################
#                                   Pretty Printing                                 #
#####################################################################################
function Base.summary(d::AbstractStateSpaceSet{D, T}) where {D, T}
    N = length(d)
    s = "$D-dimensional $(nameof(typeof(d))){$(T)} with $N points"
    return s
end

function matstring(d::AbstractStateSpaceSet{D, T}) where {D, T}
    N = length(d)
    if N > 50
        mat = zeros(eltype(eltype(d)), 50, D)
        for (i, a) in enumerate(flatten((1:25, N-24:N)))
            mat[i, :] .= d[a]
        end
    else
        mat = Matrix(d)
    end
    s = sprint(io -> show(IOContext(io, :limit=>true), MIME"text/plain"(), mat))
    s = join(split(s, '\n')[2:end], '\n')
    if !isnothing(d.names)
        e = "\nand named dimensions: $(join(d.names, ", "))"
    else
        e = ""
    end
    tos = summary(d)*e*"\n"*s
    return tos
end

Base.show(io::IO, ::MIME"text/plain", d::AbstractStateSpaceSet) = print(io, matstring(d))
Base.show(io::IO, d::AbstractStateSpaceSet) = print(io, summary(d))

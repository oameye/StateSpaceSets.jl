
#####################################################################################
#                                 Minima and Maxima                                 #
#####################################################################################
"""
    minima(dataset)

Return a point container of the same type as `dataset` containing the minimum element
of each timeseries.
"""
function minima(data::AbstractStateSpaceSet{D, T, V}) where {D, T<:Real, V}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] < m[i]
                m[i] = point[i]
            end
        end
    end
    return V(m)
end

"""
    maxima(dataset)

Return a point container of the same type as `dataset` containing the maximum element
of each timeseries.
"""
function maxima(data::AbstractStateSpaceSet{D, T, V}) where {D, T<:Real, V}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > m[i]
                m[i] = point[i]
            end
        end
    end
    return V(m)
end

"""
    minmaxima(dataset)
Return `minima(dataset), maxima(dataset)` without doing the computation twice.
"""
function minmaxima(data::AbstractStateSpaceSet{D, T, V}) where {D, T<:Real, V}
    mi = Vector(data[1])
    ma = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > ma[i]
                ma[i] = point[i]
            elseif point[i] < mi[i]
                mi[i] = point[i]
            end
        end
    end
    return V(mi), V(ma)
end

#####################################################################################
#                                     SVD                                           #
#####################################################################################
using LinearAlgebra
# SVD of Base seems to be much faster when the "long" dimension of the matrix
# is the first one, probably due to Julia's column major structure.
# This does not depend on using `svd` or `svdfact`, both give same timings.
# In fact it is so much faster, that it is *much* more worth it to
# use `Matrix(data)` instead of `reinterpret` in order to preserve the
# long dimension being the first.
"""
    svd(d::AbstractStateSpaceSet) -> U, S, Vtr
Perform singular value decomposition on the dataset.
"""
function LinearAlgebra.svd(d::AbstractStateSpaceSet)
    F = svd(Matrix(d))
    return F[:U], F[:S], F[:Vt]
end

#####################################################################################
#                                standardize                                        #
#####################################################################################
using Statistics: mean, std

"""
    standardize(d::StateSpaceSet) → r

Create a standardized version of the input set where each column is transformed to
have mean 0 and standard deviation 1. The point-container type and dimension names of
`d` are preserved.
"""
function standardize(d::AbstractStateSpaceSet)
    xs, _, _ = standardized_timeseries(d)
    return StateSpaceSet(hcat(xs...); container = containertype(d), names = d.names)
end

function standardized_timeseries(d::AbstractStateSpaceSet)
    xs = columns(d)
    means = mean.(xs)
    stds = std.(xs)
    for i in eachindex(xs)
        xs[i] .= (xs[i] .- means[i]) ./ stds[i]
    end
    return xs, means, stds
end


#####################################################################################
#                          covariance/correlation matrix                            #
#####################################################################################
import Statistics: cov, cor
using Statistics: mean, std
using StaticArraysCore: MMatrix, MVector, SMatrix, SVector

"""
    cov(d::StateSpaceSet)

Compute the covariance matrix from the columns of `d`, where `m[i, j]` is the covariance
between `d[:, i]` and `d[:, j]`. The default `SVector` representation uses the optimized
static implementation; other point-container types use the dense matrix implementation.
"""
cov(x::AbstractStateSpaceSet{D,T,V}) where {D,T<:AbstractFloat,V<:SVector} = fastcov(vec(x))
cov(x::AbstractStateSpaceSet) = Statistics.cov(Matrix(x))

"""
    mean_and_cov(d::StateSpaceSet) → μ, m

Return the column means `μ` and covariance matrix `m`. The default `SVector`
representation uses the optimized static implementation; other point-container types
use the dense matrix implementation.
"""
mean_and_cov(x::AbstractStateSpaceSet{D,T,V}) where {D,T<:AbstractFloat,V<:SVector} = fastmean_and_cov(vec(x))
function mean_and_cov(x::AbstractStateSpaceSet)
    m = Matrix(x)
    μ = vec(mean(m; dims = 1))
    return μ, Statistics.cov(m)
end

"""
    cor(d::StateSpaceSet)

Compute the correlation matrix from the columns of `d`, where `m[i, j]` is the
correlation between `d[:, i]` and `d[:, j]`. The default `SVector` representation uses
the optimized static implementation; other point-container types use the dense matrix
implementation.
"""
cor(x::AbstractStateSpaceSet{D,T,V}) where {D,T<:AbstractFloat,V<:SVector} = fastcor(vec(x))
cor(x::AbstractStateSpaceSet) = Statistics.cor(Matrix(x))

function fastcov(x::AbstractVector{SVector{D, T}}) where {D, T<:AbstractFloat}
    μ = mean(x)
    return fastcov(μ, x)
end

function fastcov(μ, x::AbstractVector{SVector{D, T}}) where {D, T<:AbstractFloat}
    N = length(x) - 1
    C = MMatrix{D, D}(zeros(D, D))
    Δx = MVector{D}(zeros(D))
    @inbounds for xᵢ in x
        Δx .= xᵢ - μ
        C .+= Δx * transpose(Δx)
    end
    C ./= N
    return SMatrix{D, D}(C)
end

function fastmean_and_cov(x::AbstractVector{SVector{D, T}}) where {D, T<:AbstractFloat}
    μ = mean(x)
    Σ = fastcov(μ, x)
    return μ, Σ
end

# Non-allocating and faster than writing a wrapper.
function fastcor(x::AbstractVector{SVector{D, T}}) where {D, T<:AbstractFloat}
    μ, Σ = fastmean_and_cov(x)
    σ = std(x)
    C = MMatrix{D, D}(zeros(D, D))
    for j in 1:D
        for i in 1:D
            C[i, j] = Σ[i, j] / (σ[i] * σ[j])
        end
    end
    return SMatrix{D, D}(C)
end
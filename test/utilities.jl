#####
##### setup and utilities for tests
#####

using Random, StaticArrays, FiniteDifferences, ForwardDiff, DiffResults, LogExpFunctions

Random.seed!(0x3b1ac7d1ef4ad7c3eab4e377ca14b76c) # consistent test runs

"""
Random parameters with `N` sectors. `L` is added to positive parameters, to bound away from
`0`. For unit testing.
"""
function random_parameters(::Val{N}, L = 0.1) where N
    σ = rand() * 2.0 .+ L
    Ω̂s = randn(SVector{N})
    ϵs = rand(SVector{N}) .* 2.0 .+ L
    p̂s = randn(SVector{N})
    Ĉ = rand() * 3.0 + L
    Ê = logsumexp((Ĉ .* ϵs .+ p̂s) .* (1-σ) .+ Ω̂s) / (1-σ)
    (; Ê, σ, Ω̂s, ϵs, p̂s, Ĉ)
end

"""
Relative residual from the Newton solver, should be `< tol`. For correctness checks.
"""
function newton_relative_residual(; Ê, Ω̂s, σ, p̂s, ϵs, Ĉ)
    lhs = logsumexp(@. Ω̂s + (1 - σ) * (p̂s + ϵs * Ĉ))
    res = (lhs - Ê * (1 - σ)) / (1 + abs(Ê))
end

"Derivative of (univariate) `f` at `x` (= 0 by default)."
∂(f; x = 0.0, kwargs...) = central_fdm(5, 1; factor = 1e6, kwargs...)(f, x)

"Add `x` at `i`."
function add_at(v::SVector, i::Int, x)
    m = MVector(v)
    m[i] += x
    SVector(m)
end

"Value and derivative using ForwardDiff."
function fwd_d(f::F, x::Real) where F
    r = ForwardDiff.derivative!(DiffResults.DiffResult(x, x), f, x)
    DiffResults.value(r), DiffResults.derivative(r)
end

"Value and gradient using ForwardDiff."
function fwd_∇(f, x::AbstractVector)
    r = DiffResults.DiffResult(x[1], x)
    r = ForwardDiff.gradient!(r, f, x)
    DiffResults.value(r), DiffResults.gradient(r)
end

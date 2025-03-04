"""
$(DocStringExtensions.README)
"""
module NonhomotheticCES

export NonhomotheticCESUtility, log_consumption_aggregator, log_sectoral_consumptions

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, DocStringExtensions
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, xexpy
using StaticArrays: SVector

####
#### problem setup for user-facing API
####

"""
$(SIGNATURES)

Check parameters, throw an error message for invalid ones.
"""
function check_σ_ϵ(σ, ϵ)
    @argcheck σ > 0 DomainError
    @argcheck σ ≠ 1 DomainError
    @argcheck all(ϵ .> 0) DomainError
    nothing
end

struct NonhomotheticCESUtility{N,Tσ<:Real,TΩ̂<:Real,Tϵ<:Real}
    σ::Tσ
    Ω̂::SVector{N,TΩ̂}
    ϵ::SVector{N,Tϵ}
    function NonhomotheticCESUtility(σ::Tσ, Ω̂::SVector{N,TΩ̂},
                   ϵ::SVector{N,Tϵ}) where {N,Tσ<:Real,TΩ̂<:Real,Tϵ<:Real}
        check_σ_ϵ(σ, ϵ)
        new{N,Tσ,TΩ̂,Tϵ}(σ, Ω̂, ϵ)
    end
end

"""
$(SIGNATURES)

Non-homothetic CES preferences, as defined in

*Comin, D., Lashkari, D., & Mestieri, Martí (2021). Structural change with long-run
 income and price effects. Econometrica, 89(1), 311–374.*

### Definition

Let ``E = ∑ᵢ pᵢ Cᵢ`` denote that *total expenditure*. The consumption aggregator `C` is
defined implicitly by

```math
E^{1 - \\sigma} = \\sum_i \\Omega_i (C^{\\epsilon_i} p_i)^{1 - \\sigma}
```

In the actual calculation and parametrization, we use logs (``Ê = log(E)`` etc) for improved
floating point accuracy.

### Arguments

- `σ`: elasticity of substitution between goods of different sectors, `> 0`, `≠ 1`.
- `Ω̂s`: **log** of sector weights
- `ϵs`: sectoral non-homotheticity parameters.

### Suggestions

- Use `SVector`s for vector arguments whenever possible.

- You may want to normalize some normalize some way, eg to `ϵ = 1` and `Ω̂ = 0` for a
  *base good* as described in equation (10) of the paper.
"""
function NonhomotheticCESUtility(σ::Real, Ω̂s::AbstractVector, ϵs::AbstractVector)
    N = length(Ω̂s)
    @argcheck N == length(ϵs)
    NonhomotheticCESUtility(σ, SVector{N}(Ω̂s), SVector{N}(ϵs))
end

# FIXME: SizedVector constructors

####
#### scaled problem
####

"""
Given `x`, solve
```math
x = \\log(∑ᵢ \\exp(aᵢ + ϵᵢ y(x)))
```
for `y`.

It is assumed that caller does all the sanity checks for parameters.

# Important

The original problem needs to be mapped into this one with

| original problem             | scaled problem |
|------------------------------|----------------|
| ``(1 - σ) \\log(E)``         | ``x``          |
| ``(1 - σ) \\log(C)``         | ``Ĉ``          |
| ``\\log(Ωᵢ Pᵢ^{1-\\sigma})`` | ``zᵢ``         |
| ``ϵᵢ``                       | ``ϵᵢ`` (same)  |
"""
struct ScaledProblem{N,TA<:Real,TE<:Real}
    a::SVector{N,TA}
    ϵ::SVector{N,TE}
end

"""
$(SIGNATURES)

Calculate the right hand side and implicit derivative ``dy/dx``of
```math
x = \\log(\\sum_i \\exp(a_i + ϵ_i y(x)))
```
"""
function rhs_and_dydx(a::SVector{2}, ϵ::SVector{2}, y)
    # NOTE simple algorithm for 2-dimensional problem
    ϵ1, ϵ2 = ϵ
    b1 = a[1] + ϵ1 * y
    b2 = a[2] + ϵ2 * y
    if b1 > b2                  # code below assumes b1 ≤ b2
        b1, b2 = b2, b1
        ϵ1, ϵ2 = ϵ2, ϵ1
    end
    Δ = b2 - b1                 # always ≥ 0
    rhs = log1pexp(Δ) + b1
    dydx = (1 + exp(-Δ)) / (xexpy(ϵ1, -Δ) + ϵ2)
    rhs, dydx
end

function rhs_and_dydx(a::SVector{N}, ϵ::SVector{N}, y) where N
    z = @. a + ϵ * y
    m = maximum(z)
    Z = @. exp(z - m)
    ∑Z = sum(Z)
    rhs = log(∑Z) + m
    dydx = ∑Z / sum(Z .* ϵ)
    rhs, dydx
end

"""
$(SIGNATURES)

Newton solver that dispatches on type. Solves
```math
x = \\log(\\sum_i \\exp(a_i + ϵ_i y(x)))
```
starting with `y = y0`.

Dispatch on type allows escaping to `ForwardDiff.Dual`. Should only be called by
[`newton_solver`](@ref).
"""
function typed_newton_solver(::Type{T}, a, ϵ, x, newton_steps, y0) where {T<:Real}
    y = y0
    for _ in 1:newton_steps
        rhs, dydx = rhs_and_dydx(a, ϵ, y)
        y -= (rhs - x) * dydx
    end
    y
end

"""
$(SIGNATURES)

Solve the problem `P` using `newton_steps`, starting from `y0`.
"""
function newton_solver(P::ScaledProblem{N,TA,TE}, x::TX, newton_steps::Integer,
                       y0) where {N,TA,TE,TX}
    T = promote_type(TA,TE,TX)
    typed_newton_solver(T, P.a, P.ϵ, x, newton_steps, y0)
end

"""
$(SIGNATURES)

Calculate and return

1. `Z`, where each element is ``Zᵢ ∝ exp(zᵢ)``, scaled to protect from under- and overflow,
where ``zᵢ = aᵢ + ϵ_i y``,

2. `sum(Z .* ϵ)`.

These are used for partial derivatives (see below).
"""
function calculate_Z_∑Zϵ(a, ϵ, y)
    z = @. a + ϵ * y
    Z = exp.(z .- maximum(z))
    Z, dot(Z, ϵ)
end

calculate_∂x(Z, ∑Zϵ) = sum(Z) / ∑Zϵ

"""
$(SIGNATURES)

Note that ``∂y/∂ϵ = y ⋅ ∂y/∂a``, there is no separate function for this.
"""
calculate_∂a(Z, ∑Zϵ) = Z ./ (-∑Zϵ)


####
#### tangents (for the initial guess)
####

"""
$(SIGNATURES)

A callable that implements the line `y = (x - location) * slope`.
"""
struct Tangent{T}
    location::T
    slope::T
end

(t::Tangent)(x) = (x - t.location) * t.slope

"""
$(SIGNATURES)

Calculate the tangent at `y` and the `x` obtained from the right hand side.
"""
function calculate_tangent(P::ScaledProblem, y)
    x, dydx = rhs_and_dydx(P.a, P.ϵ, y)
    Tangent(x - y / dydx, dydx)
end

"""
$(SIGNATURES)

Calculate the `x` coordinate of the intersection of two tangents.
"""
function calculate_intersection_x(a::Tangent, b::Tangent)
    denom = a.slope - b.slope
    @argcheck denom ≠ 0
    (a.location * a.slope - b.location * b.slope) / denom
end

####
#### precalculated problem
####

"""
$(SIGNATURES)

A [`ScaledProblem`](@ref) with some precalculated quantities to speed up the solution.
"""
Base.@kwdef struct PrecalculatedProblem{N,M,TA<:Real,TE<:Real,TT<:NTuple{M}}
    P::ScaledProblem{N,TA,TE}
    tangents::TT
end

"""
$(SIGNATURES)

Calculate the two asymptotes at ``±∞`` and return them as a tuple of [`Tangent`](@ref)s.
"""
function calculate_asymptotes(P::ScaledProblem)
    (; a, ϵ) = P
    ϵ_min, i_min = findmin(ϵ)
    ϵ_max, i_max = findmax(ϵ)
    # tangents to ±∞
    t1 = Tangent(a[i_min], 1 / ϵ_min)
    t2 = Tangent(a[i_max], 1 / ϵ_max)
    (t1, t2)
end

"""
$(SIGNATURES)

Envelope (minimum) of tangents evaluated at `x`.
"""
tangents_envelope(tangents, x) = mapreduce(t -> t(x), min, tangents)

"""
$(SIGNATURES)

Calculate the envelope of intersecting the two asymptotes, fitting a tangent at the `x`
coordinate, then repeating it with each pair. Yields a total of 5 tangents and provides
a good initial guess.
"""
function calculate_envelope5(P::ScaledProblem; newton_steps = 5)
    t1, t2 = calculate_asymptotes(P)
    function _tangent_at_intersection(tA, tB, tangents)
        xc = calculate_intersection_x(t1, t2)
        yc = newton_solver(P, xc, newton_steps, tangents_envelope(tangents, xc))
        calculate_tangent(P, yc)
    end
    # tangent at intersection of t1 and t2
    tc = _tangent_at_intersection(t1, t2, (t1, t2))
    # tangents at subsequent intersections
    tc1 = _tangent_at_intersection(t1, tc, (t1, t2, tc))
    tc2 = _tangent_at_intersection(t2, tc, (t1, t2, tc, tc1))
    (t1, t2, tc, tc1, tc2)
end

function newton_solver(PP::PrecalculatedProblem, x::Real, newton_steps,
                       y::Real = tangents_envelope(PP.tangents, x))
    newton_solver(PP.P, x, newton_steps, y)
end

const NEWTON_STEPS = 6

"""
$(SIGNATURES)

Calculate the **log** consumption aggregator for the given utility, with **log** sector
prices `p̂` and **log** expenditure `Ê`; within (absolute) tolerance `tol`.
"""
function log_consumption_aggregator(U::NonhomotheticCESUtility{N}, p̂::SVector{N}, Ê::Real;
                                    newton_steps = NEWTON_STEPS) where {N}
    @argcheck all(isfinite, p̂) DomainError
    @argcheck isfinite(Ê) DomainError
    (; σ, Ω̂, ϵ) = U
    a = @. Ω̂ + (1 - σ) * p̂
    x = (1 - σ) * Ê
    P = ScaledProblem(a, ϵ)
    y = newton_solver(P, x, newton_steps,
                      tangents_envelope(calculate_asymptotes(P), x))
    y / (1 - σ)
end

struct LogConsumptionAggregatorFix{TU,TP,TPP}
    U::TU
    p̂::TP
    PP::TPP
    newton_steps::Int
end

####
#### partial application
####

"""
$(SIGNATURES)

Partial application version. Uses more precalculated steps and fewer Newton steps,
customize as needed.
"""
function log_consumption_aggregator(U::NonhomotheticCESUtility{N}, p̂::SVector{N};
                                    precalculation_newton_steps = NEWTON_STEPS,
                                    newton_steps = ceil(Int, precalculation_newton_steps / 2)) where {N}
    @argcheck all(isfinite, p̂) DomainError
    (; σ, Ω̂, ϵ) = U
    a = @. Ω̂ + (1 - σ) * p̂
    P = ScaledProblem(a, ϵ)
    tangents = calculate_envelope5(P; newton_steps)
    PP = PrecalculatedProblem(P, tangents)
    LogConsumptionAggregatorFix(U, p̂, PP, newton_steps)
end

function (F::LogConsumptionAggregatorFix)(Ê::Real)
    @argcheck isfinite(Ê)
    (; U, PP, newton_steps) = F
    (; σ) = U
    x = (1 - σ) * Ê
    y = newton_solver(PP, x, newton_steps)
    y / (1 - σ)
end


"""
$(SIGNATURES)

Calculate **log** sectoral consumptions, return as a vector. Arguments are in logs; see
[`log_consumption_aggregator`](@ref) which whould also be used to obtain `Ĉ`.

The budget constraint holds, ie
```julia
logsumexp(log_sectoral_consumptions(U, p̂s, Ê, Ĉ) .+ p̂s) ≈ Ê
```
"""
function log_sectoral_consumptions(U::NonhomotheticCESUtility, p̂, Ê, Ĉ)
    (; σ, Ω̂, ϵ) = U
    Ω̂ .- σ .* (p̂ .- Ê) + ((1 - σ) * Ĉ) .* ϵ
end

end # module

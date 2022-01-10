####
#### internal methods, not part of the API
####

####
#### scaled problem
####

"""
Given `Ê`, solves
```math
Ê = \\log(∑ᵢ \\exp(zᵢ + ϵᵢ Ĉ))
```
for `Ĉ`. It is assumed that called does all the sanity checks for parameters.

# Important

The original problem needs to be mapped into this one with

| original problem    | scaled problem |
|---------------------|----------------|
| ``(1 - σ) Ê``       | ``Ê``          |
| ``(1 - σ) Ĉ``       | ``Ĉ``          |
| ``Ω̂ᵢ + (1 - σ) p̂ᵢ`` | ``zᵢ``         |
| ``ϵᵢ``              | ``ϵᵢ`` (same)  |
"""
struct ScaledProblem{N,Tz<:Real,Tϵ<:Real}
    zs::SVector{N,Tz}
    ϵs::SVector{N,Tϵ}
end

Broadcast.broadcastable(problem::ScaledProblem) = Ref(problem)


"""
$(SIGNATURES)

Solve the scaled `problem` using Newton's method. `atol` is the absolute tolerance (should
be positive) between the left and right hand sides, and `M` is the maximum number of
iterations.
"""
function calculate_Ĉ_newton(problem::ScaledProblem, Ê::Real, atol, M = 100)
    @unpack zs, ϵs = problem
    Ĉ = (Ê - logsumexp(zs)) / mean(ϵs)
    for _ in 1:M
        x = zs .+ ϵs .* Ĉ
        f = logsumexp(x) - Ê
        abs(f) ≤ atol && return Ĉ
        X = exp.(x .- maximum(x)) # numerical stability
        f′ = dot(X, ϵs) / sum(X)
        Δ = f / f′
        Ĉ′ = Ĉ - Δ
        Δ ≠ 0 && Ĉ′ == Ĉ && return Ĉ # no change because abs(Δ) < eps(Ĉ)
        Ĉ = Ĉ′
    end
    error("no solution after $M iterations")
end

###
### consumption aggregator
###

"""
$(SIGNATURES)

Check parameters, throw an error message for invalid ones.
"""
function check_σ_ϵs(σ, ϵs)
    @argcheck σ > 0 DomainError
    @argcheck σ ≠ 1 DomainError
    @argcheck all(ϵs .> 0) DomainError
    nothing
end

"""
Default tolerance for the residual in `Ê = ...` for calculating `Ĉ`.
"""
const DEFAULT_TOL = 1e-10

# `T` is for a separate, conditional dispatch path to catch `ForwardDiff.Dual`.
function calculate_Ĉ(::Type{T}, Ê, σ, Ω̂s, ϵs, p̂s;
                     tol = DEFAULT_TOL,
                     skipcheck::Bool = false) where {T <: Real}
    skipcheck || check_σ_ϵs(σ, ϵs)
    scaled_problem = ScaledProblem(Ω̂s .+ (1 - σ) .* p̂s, ϵs)
    atol = (1 + abs(Ê)) * abs(tol)
    calculate_Ĉ_newton(scaled_problem, (1 - σ) * Ê, atol) / (1 - σ)
end

"""
$(SIGNATURES)

Calculate `Ĉ` from parameters.

# Arguments

- `skipcheck`: skip sanity checks

- `tol`: (relative) tolerance, translated to absolute tolerance `(1 + abs(Ê)) * abs(tol)`.
"""
function calculate_Ĉ(Ê::TÊ, σ::Tσ, Ω̂s::AbstractVector{TΩ̂}, ϵs::AbstractVector{Tϵ},
                     p̂s::AbstractVector{Tp̂};
                     tol = DEFAULT_TOL,
                     skipcheck::Bool = false) where {Tσ,TΩ̂,Tϵ,TÊ,Tp̂}
    T = promote_type(Tσ,TΩ̂,Tϵ,TÊ,Tp̂)
    calculate_Ĉ(T, Ê, σ, Ω̂s, ϵs, p̂s; tol, skipcheck)
end


###
### partial derivatives
###
### Coded here internally so that they can be reused for various AD frameworks.

"""
$(SIGNATURES)

Calculate and return

1. `Zs`, where each element is ``Zᵢ ∝ exp(zᵢ)``, scaled to protect from under- and overflow,
where ``zᵢ = (Ĉ ϵᵢ + pᵢ) (1 - σ) + Ωᵢ``,

2. `sum(Zs .* ϵs)`.

These are useful for partial derivatives.
"""
function calculate_Zs_∑Zϵ(Ĉ, Ê, σ, Ω̂s, ϵs, p̂s)
    zs = (Ĉ .* ϵs .+ p̂s) .* (1 - σ) .+ Ω̂s
    Zs = exp.(zs .- maximum(zs))
    Zs, dot(Zs, ϵs)
end

"""
$(SIGNATURES)

Calculate `∂Ĉ/∂Ê`.
"""
@inline calculate_∂Ê(Zs, ∑Zϵ) = sum(Zs) / ∑Zϵ

"""
$(SIGNATURES)

Calculate `∂Ĉ/∂p̂ᵢ` for each price in `p̂s`.

Also note that the result can be reused, as ``∂Ĉ/∂ϵᵢ=Ĉ⋅∂Ĉ/∂p̂ᵢ`` and
``∂Ĉ/∂Ω̂ᵢ=∂p̂ᵢ/(1-σ)``.
"""
@inline calculate_∂p̂s(Zs, ∑Zϵ) = Zs ./ -∑Zϵ

"""
$(SIGNATURES)

Calculate `∂Ĉ/∂σ`.
"""
@inline function calculate_∂σ(Zs, ∑Zϵ, Ĉ, Ê, σ, ϵs, p̂s)
    (Ĉ + sum(Zs .* (p̂s .- Ê)) / ∑Zϵ) / (1 - σ)
end

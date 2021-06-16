####
#### internal methods, not part of the API
####

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
Default tolerance for calculating `Ĉ`.
"""
const DEFAULT_CTOL = 1e-8

"""
$(SIGNATURES)

An initial guess for `Ĉ`, to start the bracketing.
"""
function guess_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s)
    (Ê - mean(Ω̂s) / (1 - σ) - mean(p̂s)) / mean(ϵs)
end

"""
$(SIGNATURES)

Calculate `Ĉ` from parameters. `T` is for a separate, conditional dispatch path to catch
`ForwardDiff.Dual`.
"""
function calculate_Ĉ(::Type{T}, Ê, σ, Ω̂s, ϵs, p̂s;
                     Ĉtol = DEFAULT_CTOL,
                     skipcheck::Bool = false) where {T <: Real}
    skipcheck || check_σ_ϵs(σ, ϵs)
    f(Ĉ) = logsumexp((Ĉ .* ϵs .+ p̂s) .* (1 - σ) .+ Ω̂s) / (1 - σ) - Ê
    Ĉ0 = guess_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s)
    @argcheck isfinite(Ĉ0) "infinite initial guess"
    Ĉ1, fĈ1, Ĉ2, fĈ2 = bracket_increasing(f, Ĉ0, 1.0, 2.0, 100)
    Ĉ, r = bisection(f, Ĉ1, fĈ1, Ĉ2, fĈ2, Ĉtol, 100)
    # FIXME check residual r?
    Ĉ
end

function calculate_Ĉ(Ê::TÊ, σ::Tσ, Ω̂s::AbstractVector{TΩ̂}, ϵs::AbstractVector{Tϵ},
                     p̂s::AbstractVector{Tp̂};
                     Ĉtol = DEFAULT_CTOL,
                     skipcheck::Bool = false) where {Tσ,TΩ̂,Tϵ,TÊ,Tp̂}
    T = promote_type(Tσ,TΩ̂,Tϵ,TÊ,Tp̂)
    calculate_Ĉ(T, Ê, σ, Ω̂s, ϵs, p̂s; Ĉtol, skipcheck)
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

"""
Placeholder for a short summary about NonhomotheticCES.
"""
module NonhomotheticCES

export NonhomotheticCESUtility, log_consumption_aggregator, log_sectoral_consumptions

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using ForwardDiff: Dual, value, partials
using LinearAlgebra: dot
using LogExpFunctions: logsumexp
using StaticArrays: SVector
using Statistics: mean
using UnPack: @unpack

include("utilities.jl")
include("internals.jl")

struct NonhomotheticCESUtility{N,Tσ<:Real,TΩ̂<:Real,Tϵ<:Real}
    σ::Tσ
    Ω̂s::SVector{N,TΩ̂}
    ϵs::SVector{N,Tϵ}
    function NonhomotheticCESUtility(σ::Tσ, Ω̂s::SVector{N,TΩ̂},
                   ϵs::SVector{N,Tϵ}) where {N,Tσ<:Real,TΩ̂<:Real,Tϵ<:Real}
        check_σ_ϵs(σ, ϵs)
        new{N,Tσ,TΩ̂,Tϵ}(σ, Ω̂s, ϵs)
    end
end

"""
$(SIGNATURES)

Non-homothetic CES preferences, as defined in

*Comin, D., Lashkari, D., & Mestieri, Martí (2021). Structural change with long-run
 income and price effects. Econometrica, 89(1), 311–374.*

### Arguments

- `σ`: elasticity of substitution between goods of different sectors, `> 0`, `≠ 1`.
- `Ω̂s`: **log** of sector weights
- `ϵs`: sectoral non-homotheticity parameters.

### Suggestions

- Use `SVector`s for vector arguments whenever possible.

- Normalize to `ϵ = 1` and `Ω̂ = 0` for a *base good* as described in equation (10) of the
  paper. This is not mandatory, but improves numerical performance.
"""
function NonhomotheticCESUtility(σ::Real, Ω̂s::AbstractVector, ϵs::AbstractVector)
    N = length(Ω̂s)
    @argcheck N == length(ϵs)
    NonhomotheticCESUtility(σ, SVector{N}(Ω̂s), SVector{N}(ϵs))
end
# FIXME: SizedVector constructors

partials_product(x, y, α = 1) = mapreduce((x, y) -> partials(x) * y * α, +, x, y)

"""
$(SIGNATURES)

Calculate the **log** consumption aggregator for the given utility, with **log** expenditure
`Ê`, and **log** sector prices `p̂s`, within (absolute) tolerance `tol`.
"""
function log_consumption_aggregator(NHCES::NonhomotheticCESUtility{N,Tσ,TΩ̂,Tϵ}, Ê::TÊ,
                                    p̂s::SVector{N,Tp̂};
                                    tol = 1e-20) where {N,Tσ,TΩ̂,Tϵ,TÊ,Tp̂}
    @unpack σ, Ω̂s, ϵs = NHCES
    T = promote_type(Tσ,TΩ̂,Tϵ,TÊ,Tp̂)
    if T <: Dual
        vÊ, vσ, vΩ̂s, vϵs, vp̂s = value(Ê), value(σ), value.(Ω̂s), value.(ϵs), value.(p̂s)
        Ĉ = calculate_Ĉ(vÊ, vσ, vΩ̂s, vϵs, vp̂s; Ĉtol = tol, skipcheck = true)
        Zs, ∑Zϵ = calculate_Zs_∑Zϵ(Ĉ, vÊ, vσ, vΩ̂s, vϵs, vp̂s)
        # NOTE: we could be clever here and only calculate what is actually needed
        ∂p̂s = calculate_∂p̂s(Zs, ∑Zϵ)
        dĈ = (partials(Ê) * calculate_∂Ê(Zs, ∑Zϵ) +                      # ∂Ê
              partials(σ) * calculate_∂σ(Zs, ∑Zϵ, Ĉ, vÊ, vσ, vϵs, vp̂s) + # ∂σ
              partials_product(Ω̂s, ∂p̂s, inv(1 - vσ)) +                   # ∂Ω̂
              partials_product(ϵs, ∂p̂s, Ĉ) +                             # ∂ϵs
              partials_product(p̂s, ∂p̂s))                                 # ∂p̂s
        T(Ĉ, dĈ)
    else
        calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s; Ĉtol = tol, skipcheck = true)
    end
end

"""
$(SIGNATURES)

Calculate **log** sectoral consumptions, return as a vector. Arguments are in logs; see
[`log_consumption_aggregator`](@ref) which whould also be used to obtain `Ĉ`.
"""
function log_sectoral_consumptions(NHCES::NonhomotheticCESUtility, Ê, p̂s, Ĉ)
    @unpack σ, Ω̂s, ϵs = NHCES
    Ω̂s .- σ .* (p̂s .- Ê) + ((1 - σ) * Ĉ) .* ϵs
end

end # module

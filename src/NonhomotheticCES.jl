"""
Placeholder for a short summary about NonhomotheticCES.
"""
module NonhomotheticCES

export NonhomotheticCESUtility, log_consumption_aggregator, log_sectoral_consumptions

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using LinearAlgebra: dot
using LogExpFunctions: logsumexp
using StaticArrays: SVector
using Statistics: mean

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

Broadcast.broadcastable(U::NonhomotheticCESUtility) = Ref(U)

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

- Normalize to `ϵ = 1` and `Ω̂ = 0` for a *base good* as described in equation (10) of the
  paper. This is not mandatory, but improves numerical performance.
"""
function NonhomotheticCESUtility(σ::Real, Ω̂s::AbstractVector, ϵs::AbstractVector)
    N = length(Ω̂s)
    @argcheck N == length(ϵs)
    NonhomotheticCESUtility(σ, SVector{N}(Ω̂s), SVector{N}(ϵs))
end
# FIXME: SizedVector constructors

"""
$(SIGNATURES)

Calculate the **log** consumption aggregator for the given utility, with **log** sector
prices `p̂s` and **log** expenditure `Ê`; within (absolute) tolerance `tol`.
"""
function log_consumption_aggregator(U::NonhomotheticCESUtility{N,Tσ,TΩ̂,Tϵ},
                                    p̂s::SVector{N,Tp̂}, Ê::TÊ;
                                    tol = DEFAULT_TOL) where {N,Tσ,TΩ̂,Tϵ,TÊ,Tp̂}
    (; σ, Ω̂s, ϵs) = U
    @argcheck isfinite(Ê) && all(isfinite, p̂s) DomainError
    calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s; tol = tol, skipcheck = true)
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
function log_sectoral_consumptions(U::NonhomotheticCESUtility, p̂s, Ê, Ĉ)
    (; σ, Ω̂s, ϵs) = U
    Ω̂s .- σ .* (p̂s .- Ê) + ((1 - σ) * Ĉ) .* ϵs
end

end # module

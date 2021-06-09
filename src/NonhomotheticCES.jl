"""
Placeholder for a short summary about NonhomotheticCES.
"""
module NonhomotheticCES

export NHCES, log_consumption_aggregator

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

struct NHCES{N,Tσ<:Real,TΩ̂<:Real,Tϵ<:Real}
    σ::Tσ
    Ω̂s::SVector{N,TΩ̂}
    ϵs::SVector{N,Tϵ}
    function NHCES(σ::Tσ, Ω̂s::SVector{N,TΩ̂},
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

Arguments:

- `σ`: elasticity of substitution between goods of different sectors, `> 0`, `≠ 1`.
- `Ω̂s`: **log** of sector weights
- `ϵs`: sectoral non-homotheticity parameters.

Use `SVector`s for vector arguments whenever possible.
"""
function NHCES(σ::Real, Ω̂s::AbstractVector, ϵs::AbstractVector)
    N = length(Ω̂s)
    @argcheck N == length(ϵs)
    NHCES(σ, SVector{N}(Ω̂s), SVector{N}(ϵs))
end
# FIXME: SizedVector constructors

function log_consumption_aggregator(preferences::NHCES{N,Tσ,TΩ̂,Tϵ}, Ê::TÊ,
                                    p̂s::SVector{N,Tp̂};
                                    tol = 1e-20) where {N,Tσ,TΩ̂,Tϵ,TÊ,Tp̂}
    @unpack σ, Ω̂s, ϵs = preferences
    T = promote_type(Tσ,TΩ̂,Tϵ,TÊ,Tp̂)
    if T <: Dual
        vÊ, vσ, vΩ̂s, vϵs, vp̂s = value(Ê), value(σ), value.(Ω̂s), value.(ϵs), value.(p̂s)
        Ĉ = calculate_Ĉ(vÊ, vσ, vΩ̂s, vϵs, vp̂s; Ĉtol = tol, skipcheck = true)
        Zs, ∑Zϵ = calculate_Zs_∑Zϵ(Ĉ, vÊ, vσ, vΩ̂s, vϵs, vp̂s)
        dĈ = (partials(Ê) * calculate_∂Ê(Zs, ∑Zϵ) +
              partials(σ) * calculate_∂σ(Zs, ∑Zϵ, Ĉ, vÊ, vσ, vϵs, vp̂s)
              )
        T(Ĉ, dĈ)
    else
        calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s; Ĉtol = tol, skipcheck = true)
    end
end

end # module

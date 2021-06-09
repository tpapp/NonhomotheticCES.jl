####
#### internal methods, not part of the API
####

function calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s; Ĉtol = 0x1p-10, skipcheck::Bool = false)
    if !skipcheck
        @argcheck σ > 0 && σ ≠ 1 && all(ϵs .> 0)
    end
    f(Ĉ) = logsumexp((Ĉ .* ϵs .+ p̂s) .* (1 - σ) .+ Ω̂s) / (1 - σ) - Ê
    Ĉ0 = (Ê - mean(Ω̂s) / (1 - σ) - mean(p̂s)) / mean(ϵs)
    Ĉ1, fĈ1, Ĉ2, fĈ2 = bracket_increasing(f, Ĉ0, 1.0, 2.0, 50)
    Ĉ, r = bisection(f, Ĉ1, fĈ1, Ĉ2, fĈ2, Ĉtol, 50)
    # FIXME check residual r?
    Ĉ
end

"""
$(SIGNATURES)

Calculate and return

1. `Zs`, where each element is ``Zᵢ ∝ exp(zᵢ)``, scaled to protect from under- and overflow,
where ``zᵢ = (Ĉ ϵᵢ + pᵢ) (1 - σ) + Ωᵢ``,

2. `sum(Zs .* ϵs)`.
"""
function calculate_Zs_∑Zϵ(Ĉ, Ê, σ, Ω̂s, ϵs, p̂s)
    zs = (Ĉ .* ϵs .+ p̂s) .* (1 - σ) .+ Ω̂s
    Zs = exp.(zs .- maximum(zs))
    Zs, sum(Zs .* ϵs)
end

calculate_∂Ê(Zs, ∑Zϵ) = sum(Zs) / ∑Zϵ

calculate_∂p̂s(Zs, ∑Zϵ) = Zs ./ -∑Zϵ

calculate_∂ϵs(∂p̂s, Ĉ) = Ĉ .* ∂p̂s

calculate_∂Ω̂s(∂p̂s, σ) = ∂p̂s ./ (1 - σ)

calculate_∂σ(Zs, ∑Zϵ, Ĉ, Ê, σ, ϵs, p̂s) = (Ĉ + sum(Zs .* (p̂s .- Ê)) / ∑Zϵ) / (1 - σ)

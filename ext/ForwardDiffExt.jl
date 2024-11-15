module ForwardDiffExt

using ForwardDiff: Dual, value, partials, valtype
import NonhomotheticCES

partials_product(x, y, α = 1) = mapreduce((x, y) -> partials(x) * y * α, +, x, y)

function NonhomotheticCES.calculate_Ĉ(::Type{T}, Ê, σ, Ω̂s, ϵs, p̂s;
                     tol = DEFAULT_TOL, skipcheck = false) where {T<:Dual}
    vÊ, vσ, vΩ̂s, vϵs, vp̂s = value(Ê), value(σ), value.(Ω̂s), value.(ϵs), value.(p̂s)
    Ĉ = NonhomotheticCES.calculate_Ĉ(valtype(T), vÊ, vσ, vΩ̂s, vϵs, vp̂s;
                                     tol = tol, skipcheck = skipcheck)
    Zs, ∑Zϵ = NonhomotheticCES.calculate_Zs_∑Zϵ(Ĉ, vÊ, vσ, vΩ̂s, vϵs, vp̂s)
    # NOTE: we could be clever here and only calculate what is actually needed
    ∂p̂s = NonhomotheticCES.calculate_∂p̂s(Zs, ∑Zϵ)
    dĈ = (partials(Ê) * NonhomotheticCES.calculate_∂Ê(Zs, ∑Zϵ) +                    # ∂Ê
        partials(σ) * NonhomotheticCES.calculate_∂σ(Zs, ∑Zϵ, Ĉ, vÊ, vσ, vϵs, vp̂s) + # ∂σ
        partials_product(Ω̂s, ∂p̂s, inv(1 - vσ)) +                                    # ∂Ω̂
        partials_product(ϵs, ∂p̂s, Ĉ) + # ∂ϵs
        partials_product(p̂s, ∂p̂s))     # ∂p̂s
    T(Ĉ, dĈ)
end

end

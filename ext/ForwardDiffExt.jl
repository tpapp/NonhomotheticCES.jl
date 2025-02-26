module ForwardDiffExt

using ForwardDiff: Dual, value, partials, valtype
import NonhomotheticCES

"""
Helper function for type stable accumulation without explicit initialization.
"""
__accum(::Nothing, y) = y
__accum(x, y) = x + y

function NonhomotheticCES.typed_newton_solver(::Type{T}, a, ϵ, x, N, y0) where {T<:Dual}
    vx, vy0, va, vϵ = value(x), value(y0), value.(a), value.(ϵ)
    vy = NonhomotheticCES.typed_newton_solver(valtype(T), va, vϵ, vx, N, vy0)
    dy = nothing
    Z, ∑Zϵ = NonhomotheticCES.calculate_Z_∑Zϵ(va, vϵ, vy)
    if x isa Dual
        dy = __accum(dy, partials(x) * NonhomotheticCES.calculate_∂x(Z, ∑Zϵ))
    end
    if eltype(a) <: Dual || eltype(ϵ) <: Dual
        D = NonhomotheticCES.calculate_∂a(Z, ∑Zϵ)
        if eltype(a) <: Dual
            dy = __accum(dy, mapreduce((a, d) -> partials(a) * d, +, a, D))
        end
        if eltype(ϵ) <: Dual
            dy = __accum(dy, mapreduce((ϵ, d) -> partials(ϵ) * vy * d, +, ϵ, D))
        end
    end
    T(vy, dy)
end

end

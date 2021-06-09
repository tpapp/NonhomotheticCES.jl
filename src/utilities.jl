####
#### utilities for numerical methods
####

"""
$(SIGNATURES)

Find `a` and `b` so that `f(a) * f(b) ≤ 0`, bracketing a root. Return `a, f(a), b, f(b)`.

`f` has to be increasing. At each iteration, `x` is adjusted by a stepsize in the
appropriate direction. `Δ` is the initial value of the stepsize, and is increased by a
factor of `G` after each step. At most `max_iterations` iterations are performed, after
which the function errors.
"""
function bracket_increasing(f, x, Δ, G, max_iterations)
    @argcheck Δ > 0 && G > 1
    fx = f(x)
    fx == 0 && return x, fx, f, fx
    if fx > 0
        Δ = -Δ
    end
    for _ in 1:max_iterations
        x2 = x + Δ
        fx2 = f(x2)
        fx2 * fx ≤ 0 && return x, fx, x2, fx2
        x, fx = x2, fx2
        Δ *= G
    end
    error("too many iterations")
end

"""
$(SIGNATURES)

Bisection method for finding and `x₁` near `x₀` such that `f(x₀) = 0` and
`abs(x₁ - x₀) ≤ xtol`. Returns `x₁, f(x₁)`.
"""
function bisection(f, a::T, fa::T, b::T, fb::T, xtol, max_iterations) where {T}
    fa == 0 && return a, fa
    fb == 0 && return b, fb
    fa * fb < 0 || error("not bracketed")
    for _ in 1:max_iterations
        m = (a + b) / 2
        fm = f(m)
        abs(a - b) ≤ xtol && return m, fm
        if fm * fa > 0
            a, fa = m, fm
        else
            b, fb = m, fm
        end
    end
    error("too many iterations")
end

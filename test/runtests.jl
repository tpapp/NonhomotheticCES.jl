using NonhomotheticCES
using NonhomotheticCES:
    calculate_Ĉ, calculate_Zs, calculate_∂Ê, calculate_∂p̂s, calculate_∂ϵs,
    calculate_∂Ω̂s, calculate_∂σ

using FiniteDifferences, LogExpFunctions, Test, StaticArrays, UnPack

"""
Random parameters with `N` sectors. `L` is added to positive parameters, to bound away from
`0`.
"""
function random_parameters(::Val{N}, L = 0.1) where N
    (Ê = rand() * 5.0,
     σ = rand() * 2.0 .+ L,
     Ω̂s = randn(SVector{N}),
     ϵs = rand(SVector{N}) .* 2.0 .+ L,
     p̂s = randn(SVector{N}))
end

∂(f; x = 0.0, kwargs...) = central_fdm(5, 1; factor = 1e6, kwargs...)(f, x)

"Add `x` at `i`."
function add_at(v::SVector, i::Int, x)
    m = MVector(v)
    m[i] += x
    SVector(m)
end

@testset "finding Ĉ" begin
    for _ in 1:100
        @unpack Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(4))
        Ĉ = calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s)
        @test logsumexp((Ĉ .* ϵs .+ p̂s) .* (1-σ) .+ Ω̂s)/(1-σ) ≈ Ê atol = 1e-2
    end
end

@testset "partials building blocks" begin
    Ĉtol = 0x1p-20
    atol = 1e-2
    rtol = 1e-2
    for _ in 1:100
        N = Val{2}()
        @unpack Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(N)
        Ĉ = calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s; Ĉtol = Ĉtol)
        Zs = calculate_Zs(Ĉ, Ê, σ, Ω̂s, ϵs, p̂s)
        ∂p̂s = calculate_∂p̂s(Zs, ϵs)
        ∂ϵs = calculate_∂ϵs(∂p̂s, Ĉ)
        ∂Ω̂s = calculate_∂Ω̂s(∂p̂s, σ)
        @test calculate_∂Ê(Zs, ϵs) ≈
            ∂(h -> calculate_Ĉ(Ê + h, σ, Ω̂s, ϵs, p̂s; Ĉtol = Ĉtol)) atol = atol rtol = rtol
        for (j, ∂p̂) in pairs(∂p̂s)
            ∂p̂_fd = ∂(h -> calculate_Ĉ(Ê, σ, Ω̂s, ϵs, add_at(p̂s, j, h); Ĉtol = Ĉtol))
            @test ∂p̂ ≈ ∂p̂_fd atol = atol rtol = rtol
        end
        for (j, ∂ϵ) in pairs(∂ϵs)
            ∂ϵ_fd = ∂(h -> calculate_Ĉ(Ê, σ, Ω̂s, add_at(ϵs, j, h), p̂s; Ĉtol = Ĉtol);
                      max_range = ϵs[j] * 0.99)
            @test ∂ϵ ≈ ∂ϵ_fd atol = atol rtol = rtol
        end
        for (j, ∂Ω̂) in pairs(∂Ω̂s)
            ∂Ω̂_fd = ∂(h -> calculate_Ĉ(Ê, σ, add_at(Ω̂s, j, h), ϵs, p̂s; Ĉtol = Ĉtol))
            @test ∂Ω̂ ≈ ∂Ω̂_fd atol = atol rtol = rtol
        end
        @test calculate_∂σ(Zs, Ĉ, Ê, σ, ϵs, p̂s) ≈
            ∂(h -> calculate_Ĉ(Ê, σ + h, Ω̂s, ϵs, p̂s; Ĉtol = Ĉtol);
              max_range = σ * 0.99) atol = atol rtol = rtol
    end
end

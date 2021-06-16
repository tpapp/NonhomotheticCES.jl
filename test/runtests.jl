using NonhomotheticCES
using NonhomotheticCES:         # internals
    calculate_Ĉ, calculate_Zs_∑Zϵ, calculate_∂Ê, calculate_∂p̂s, calculate_∂σ

using FiniteDifferences, LogExpFunctions, Test, StaticArrays, UnPack, Random, LinearAlgebra
import ForwardDiff, DiffResults

####
#### setup and utilities
####

Random.seed!(0x3b1ac7d1ef4ad7c3eab4e377ca14b76c) # consistent test runs

"""
Random parameters with `N` sectors. `L` is added to positive parameters, to bound away from
`0`. For unit testing.
"""
function random_parameters(::Val{N}, L = 0.1) where N
    σ = rand() * 2.0 .+ L
    Ω̂s = randn(SVector{N})
    ϵs = rand(SVector{N}) .* 2.0 .+ L
    p̂s = randn(SVector{N})
    Ĉ = rand() * 3.0 + L
    Ê = logsumexp((Ĉ .* ϵs .+ p̂s) .* (1-σ) .+ Ω̂s)/(1-σ)
    (; Ê, σ, Ω̂s, ϵs, p̂s, Ĉ)
end

"Derivative of (univariate) `f` at `x` (= 0 by default)."
∂(f; x = 0.0, kwargs...) = central_fdm(5, 1; factor = 1e6, kwargs...)(f, x)

"Add `x` at `i`."
function add_at(v::SVector, i::Int, x)
    m = MVector(v)
    m[i] += x
    SVector(m)
end

"Value and derivative using ForwardDiff."
function fwd_d(f::F, x::Real) where F
    r = ForwardDiff.derivative!(DiffResults.DiffResult(x, x), f, x)
    DiffResults.value(r), DiffResults.derivative(r)
end

"Value and gradient using ForwardDiff."
function fwd_∇(f, x::AbstractVector)
    r = DiffResults.DiffResult(x[1], x)
    r = ForwardDiff.gradient!(r, f, x)
    DiffResults.value(r), DiffResults.gradient(r)
end

####
#### internals
####

@testset "finding Ĉ" begin
    tol = 1e-10
    for _ in 1:100
        @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(4))
        @test @inferred(calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s; Ĉtol = tol)) ≈ Ĉ atol = tol
    end
end

@testset "partials building blocks" begin
    Ĉtol = 1e-10
    atol = 1e-2
    rtol = 1e-2
    for _ in 1:100
        N = Val{2}()
        @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(N)
        Zs, ∑Zϵ = @inferred calculate_Zs_∑Zϵ(Ĉ, Ê, σ, Ω̂s, ϵs, p̂s)
        ∂p̂s = @inferred calculate_∂p̂s(Zs, ∑Zϵ)
        @test @inferred(calculate_∂Ê(Zs, ∑Zϵ)) ≈
            ∂(h -> calculate_Ĉ(Ê + h, σ, Ω̂s, ϵs, p̂s; Ĉtol = Ĉtol)) atol = atol rtol = rtol
        for (j, ∂p̂) in pairs(∂p̂s)
            ∂p̂_fd = ∂(h -> calculate_Ĉ(Ê, σ, Ω̂s, ϵs, add_at(p̂s, j, h); Ĉtol = Ĉtol))
            @test ∂p̂ ≈ ∂p̂_fd atol = atol rtol = rtol
        end
        for (j, ∂ϵ) in pairs(∂p̂s .* Ĉ)
            ∂ϵ_fd = ∂(h -> calculate_Ĉ(Ê, σ, Ω̂s, add_at(ϵs, j, h), p̂s; Ĉtol = Ĉtol);
                      max_range = ϵs[j] * 0.9)
            @test ∂ϵ ≈ ∂ϵ_fd atol = atol rtol = rtol
        end
        for (j, ∂Ω̂) in pairs(∂p̂s ./ (1 - σ))
            ∂Ω̂_fd = ∂(h -> calculate_Ĉ(Ê, σ, add_at(Ω̂s, j, h), ϵs, p̂s; Ĉtol = Ĉtol))
            @test ∂Ω̂ ≈ ∂Ω̂_fd atol = atol rtol = rtol
        end
        @test calculate_∂σ(Zs, ∑Zϵ, Ĉ, Ê, σ, ϵs, p̂s) ≈
            ∂(h -> calculate_Ĉ(Ê, σ + h, Ω̂s, ϵs, p̂s; Ĉtol = Ĉtol);
              max_range = σ * 0.9) atol = atol rtol = rtol
    end
end

####
#### API and Forwarddiff
####

@testset "argument checks" begin
    @test_throws DomainError NonhomotheticCESUtility(-0.1, SVector(1.0, 2.0), SVector(1.0, 2.0))
    @test_throws DomainError NonhomotheticCESUtility(1, SVector(1.0, 2.0), SVector(1.0, 2.0))
    @test_throws DomainError NonhomotheticCESUtility(0.1, SVector(1.0, 2.0), SVector(-1.0, 2.0))
end

@testset "API and AD checks" begin
    @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(2))
    max_range = min(σ, minimum(ϵs) / 5) * 0.9
    tol = 1e-10
    pref = NonhomotheticCESUtility(σ, Ω̂s, ϵs)
    @test @inferred(log_consumption_aggregator(pref, p̂s, Ê; tol = tol)) ≈ Ĉ atol = tol
    @testset "directional derivative" begin
        f = h -> log_consumption_aggregator(NonhomotheticCESUtility(σ + h,
                                                                    Ω̂s .+ SVector(2h, 3h),
                                                                    ϵs .+ SVector(4h, 5h)),
                                            p̂s .+ SVector(8h, 9h), Ê + 7h)
        v, d = @inferred fwd_d(f, 0.0)
        @test v ≈ Ĉ atol = tol
        @test d ≈ ∂(f; max_range) rtol = 1e-4
    end
    @testset "gradient" begin
        f = h -> log_consumption_aggregator(NonhomotheticCESUtility(σ + h[1],
                                                                    Ω̂s .+ SVector(h[2], h[3]),
                                                                    ϵs .+ SVector(h[4], h[5])),
                                            p̂s .+ SVector(h[7], h[8]),
                                            Ê + h[6])
        v, ∇ = @inferred fwd_∇(f, zeros(8))
        @test v ≈ Ĉ atol = tol
        ∇_fd = [∂(h -> (z = zeros(8); z[i] = h; f(z)); max_range) for i in 1:8]
        @test norm(∇ .- ∇_fd, Inf) ≤ 1e-7
    end
    @test @inferred(log_sectoral_consumptions(pref, p̂s, Ê, Ĉ)) ≈
        @. Ω̂s - σ * (p̂s - Ê) + (1 - σ) * ϵs * Ĉ
    @testset "non-finite inputs" begin
        @test_throws DomainError log_consumption_aggregator(pref, p̂s, -Inf)
    end
end

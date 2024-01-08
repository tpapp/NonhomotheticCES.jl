using NonhomotheticCES
using NonhomotheticCES:         # internals
    ScaledProblem, calculate_Ĉ_newton, calculate_Ĉ, calculate_Zs_∑Zϵ, calculate_∂Ê,
    calculate_∂p̂s, calculate_∂σ

using Logging

using Test
using UnPack: @unpack
using LinearAlgebra: norm

include("utilities.jl")

####
#### internals
####

@testset "finding Ĉ" begin
    tol = 1e-5
    for _ in 1:100
        @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(4))
        Ĉ2 = @inferred(calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s; tol = tol))
        @test newton_relative_residual(; Ê,  σ, Ω̂s, ϵs, p̂s, Ĉ = Ĉ2) ≤ tol
    end
end

@testset "partials building blocks" begin
    tol = 1e-10
    atol = 1e-2
    rtol = 1e-2
    for _ in 1:100
        N = Val{2}()
        @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(N)
        Zs, ∑Zϵ = @inferred calculate_Zs_∑Zϵ(Ĉ, Ê, σ, Ω̂s, ϵs, p̂s)
        ∂p̂s = @inferred calculate_∂p̂s(Zs, ∑Zϵ)
        @test @inferred(calculate_∂Ê(Zs, ∑Zϵ)) ≈
            ∂(h -> calculate_Ĉ(Ê + h, σ, Ω̂s, ϵs, p̂s; tol = tol)) atol = atol rtol = rtol
        for (j, ∂p̂) in pairs(∂p̂s)
            ∂p̂_fd = ∂(h -> calculate_Ĉ(Ê, σ, Ω̂s, ϵs, add_at(p̂s, j, h); tol = tol))
            @test ∂p̂ ≈ ∂p̂_fd atol = atol rtol = rtol
        end
        for (j, ∂ϵ) in pairs(∂p̂s .* Ĉ)
            ∂ϵ_fd = ∂(h -> calculate_Ĉ(Ê, σ, Ω̂s, add_at(ϵs, j, h), p̂s; tol = tol);
                      max_range = ϵs[j] * 0.9)
            @test ∂ϵ ≈ ∂ϵ_fd atol = atol rtol = rtol
        end
        for (j, ∂Ω̂) in pairs(∂p̂s ./ (1 - σ))
            ∂Ω̂_fd = ∂(h -> calculate_Ĉ(Ê, σ, add_at(Ω̂s, j, h), ϵs, p̂s; tol = tol))
            @test ∂Ω̂ ≈ ∂Ω̂_fd atol = atol rtol = rtol
        end
        @test calculate_∂σ(Zs, ∑Zϵ, Ĉ, Ê, σ, ϵs, p̂s) ≈
            ∂(h -> calculate_Ĉ(Ê, σ + h, Ω̂s, ϵs, p̂s; tol = tol);
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

@testset "API checks" begin
    @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(2))
    tol = 1e-10
    U = NonhomotheticCESUtility(σ, Ω̂s, ϵs)
    Ĉ2 = @inferred(log_consumption_aggregator(U, p̂s, Ê; tol = tol)) # we compare to this below
    @test newton_relative_residual(; Ĉ = Ĉ2, σ, Ω̂s, ϵs, p̂s, Ê) ≤ tol

    @test @inferred(log_sectoral_consumptions(U, p̂s, Ê, Ĉ)) ≈
        @. Ω̂s - σ * (p̂s - Ê) + (1 - σ) * ϵs * Ĉ

    @testset "non-finite inputs" begin
        @test_throws DomainError log_consumption_aggregator(U, p̂s, -Inf)
    end

    @testset "homotheticity" begin
        for _ in 1:100
            @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(rand(2:5)))
            U = NonhomotheticCESUtility(σ, Ω̂s, ϵs)
            ν = 1.4
            Ĉ1 = log_consumption_aggregator(U, p̂s, Ê)
            Ĉ2 = log_consumption_aggregator(U, p̂s .+ ν, Ê + ν)
            @test Ĉ1 ≈ Ĉ2 atol = 1e-5
            @test Ĉ1 ≈ Ĉ atol = 1e-5
            ĉ1 = log_sectoral_consumptions(U, p̂s, Ê, Ĉ)
            ĉ2 = log_sectoral_consumptions(U, p̂s .+ ν, Ê + ν, Ĉ)
            @test ĉ1 ≈ ĉ2 atol = 1e-5
        end
    end

    @testset "budget constraint" begin
        for _ in 1:100
            @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(rand(2:5)))
            U = NonhomotheticCESUtility(σ, Ω̂s, ϵs)
            Ĉ2 = log_consumption_aggregator(U, p̂s, Ê; tol = tol)
            @test newton_relative_residual(; Ĉ = Ĉ2, σ, Ω̂s, ϵs, p̂s, Ê) ≤ tol
            ĉs = log_sectoral_consumptions(U, p̂s, Ê, Ĉ)
            @test logsumexp(ĉs .+ p̂s) ≈ Ê
        end
    end
end

@testset "directional derivative" begin
    @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(2))
    max_range = min(σ, minimum(ϵs) / 5) * 0.9
    function fh(h)
        U = NonhomotheticCESUtility(σ + h, Ω̂s .+ SVector(2h, 3h), ϵs .+ SVector(4h, 5h))
        log_consumption_aggregator(U, p̂s .+ SVector(8h, 9h), Ê + 7h)
    end
    v, d = @inferred fwd_d(fh, 0.0)
    @test v == log_consumption_aggregator(NonhomotheticCESUtility(σ, Ω̂s, ϵs), p̂s, Ê)
    @test d ≈ ∂(fh; max_range) rtol = 1e-4 atol = 1e-4
end

@testset "gradient" begin
    @unpack Ĉ, Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(2))
    max_range = min(σ, minimum(ϵs) / 5) * 0.9
    f = h -> log_consumption_aggregator(NonhomotheticCESUtility(σ + h[1],
                                                                Ω̂s .+ SVector(h[2], h[3]),
                                                                ϵs .+ SVector(h[4], h[5])),
                                        p̂s .+ SVector(h[7], h[8]),
                                        Ê + h[6])
    v, ∇ = @inferred fwd_∇(f, zeros(8))
    @test v == log_consumption_aggregator(NonhomotheticCESUtility(σ, Ω̂s, ϵs), p̂s, Ê)
    ∇_fd = [∂(h -> (z = zeros(8); z[i] = h; f(z)); max_range) for i in 1:8]
    @test norm(∇ .- ∇_fd, Inf) ≤ 1e-5
end

@testset "collected test cases from previous failures" begin
    @testset "abs(Δ) < eps(Ĉ) nonconvergence" begin
        zs = SVector(1.0358525676185755, 1.2562213779011235)
        ϵs = SVector(0.41559183742572103, 0.4499184434235435)
        Ê = 0.30482937178846753
        tol = 8.861127361373689e-21
        Ĉ = @inferred calculate_Ĉ_newton(ScaledProblem(zs, ϵs), Ê, tol)
        @test isfinite(Ĉ)
    end
end

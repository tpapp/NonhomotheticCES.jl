using NonhomotheticCES, Test, LogExpFunctions, StaticArrays, FiniteDifferences
using NonhomotheticCES:         # internals
    calculate_Z_∑Zϵ, calculate_∂x, calculate_∂a, newton_solver, ScaledProblem,
    PrecalculatedProblem, tangents_envelope, calculate_envelope5, rhs_and_dydx
import Random, ForwardDiff
Random.seed!(0x3b1ac7d1ef4ad7c3eab4e377ca14b76c) # consistent test runs

####
#### utilities for tests
####

"""
Random parameters with `N` sectors. `L` is added to positive parameters, to bound away from
`0`. For unit testing.
"""
function random_parameters(::Val{N}, L = 0.1) where N
    σ = rand() * 2.0 .+ L
    Ω̂ = randn(SVector{N})
    ϵ = rand(SVector{N}) .* 2.0 .+ L
    p̂ = randn(SVector{N})
    Ĉ = rand() * 3.0 + L
    Ê = logsumexp((Ĉ .* ϵ .+ p̂) .* (1-σ) .+ Ω̂) / (1-σ)
    (; Ê, σ, Ω̂, ϵ, p̂, Ĉ)
end

"""
Relative residual from the Newton solver, should be `< tol`. For correctness checks.
"""
function newton_relative_residual(; Ê, Ω̂, σ, p̂, ϵ, Ĉ)
    lhs = logsumexp(@. Ω̂ + (1 - σ) * (p̂ + ϵ * Ĉ))
    res = (lhs - Ê * (1 - σ)) / (1 + abs(Ê))
end

"Derivative of (univariate) `f` at `x` (= 0 by default)."
const DD = central_fdm(5, 1; factor = 1e6)

####
#### internals tests
####

_residual(P::ScaledProblem, x, y) = logsumexp(P.a .+ P.ϵ .* y) - x

@testset "rhs_and_dydx" begin
    for _ in 1:1000
        # create a ScaledProblem with random size and parameters
        N = rand(2:5)
        a = randn(SVector{N})
        ϵ = max.(rand(SVector{N}) .* 2, 1e-7)
        y = randn() * 10
        rhs, dydx = rhs_and_dydx(a, ϵ, y)
        Ba = BigFloat.(a)
        Bϵ = BigFloat.(ϵ)
        By = BigFloat(y)
        Brhs = logsumexp(@. Ba + Bϵ * By)
        @test rhs ≈ Brhs
        Bdydx = 1 / DD(By -> logsumexp(@. Ba + Bϵ * By), By)
        @test dydx ≈ Bdydx
    end
end

@testset "scaled problem building" begin
    for _ in 1:100
        # create a ScaledProblem with random size and parameters
        N = rand(2:5)
        a = randn(SVector{N})
        ϵ = max.(rand(SVector{N}) .* 2, 1e-7)
        P = ScaledProblem(a, ϵ)
        PP = PrecalculatedProblem(P, calculate_envelope5(P))
        # test asymptotics
        xL = 1e5
        @test _residual(P, xL, tangents_envelope(PP.tangents, xL)) ≤ 1e-8
        @test _residual(P, -xL, tangents_envelope(PP.tangents, -xL)) ≤ 1e-8
        # test solver at random points
        for _ in 1:100
            x = randn() * 10
            # test Newton solver residuals
            y = newton_solver(PP, x, 5)
            @test _residual(P, x, y) ≤ 1e-4
            # if _residual(P, x, y) > 1e-4
            #     @show P x y
            # end
        end
    end
end

@testset "partials building blocks and AD of scaled problem" begin
    tol = 1e-10
    atol = 1e-2
    rtol = 1e-2
    for _ in 1:100
        N = 2
        a = randn(SVector{N})
        ϵ = rand(SVector{N}) .+ 0.1
        y = randn()
        x = logsumexp(@. a + ϵ * y)
        Z, ∑Zϵ = @inferred calculate_Z_∑Zϵ(a, ϵ, y)
        ∂x = @inferred calculate_∂x(Z, ∑Zϵ)
        ∂a = @inferred calculate_∂a(Z, ∑Zϵ)

        let P = ScaledProblem(a, ϵ), f = x -> newton_solver(P, x, 10, 0.0)
            @test ∂x ≈ DD(f, x)
            @test ∂x ≈ @inferred ForwardDiff.derivative(f, x)
        end
        let x = x, ϵ = ϵ, f = a -> newton_solver(ScaledProblem(a, ϵ), x, 10, 0.0)
            @test ∂a ≈ grad(DD, f, a)[1]
            @test ∂a ≈ @inferred ForwardDiff.gradient(f, a)
        end
        let x = x, a = a, f = ϵ -> newton_solver(ScaledProblem(a, ϵ), x, 10, 0.0)
            ∂ϵ = ∂a .* y
            @test ∂ϵ ≈ grad(DD, f, ϵ)[1]
            @test ∂ϵ ≈ @inferred ForwardDiff.gradient(f, ϵ)
        end
    end
end

####
#### API and AD tests
####

@testset "argument checks" begin
    @test_throws DomainError NonhomotheticCESUtility(-0.1, SVector(1.0, 2.0),
                                                     SVector(1.0, 2.0))
    @test_throws DomainError NonhomotheticCESUtility(1, SVector(1.0, 2.0),
                                                     SVector(1.0, 2.0))
    @test_throws DomainError NonhomotheticCESUtility(0.1, SVector(1.0, 2.0),
                                                     SVector(-1.0, 2.0))
end


@testset "finding Ĉ" begin
    tol = 1e-5
    for _ in 1:100
        (; Ĉ, Ê, σ, Ω̂, ϵ, p̂) = random_parameters(Val(4))
        U = NonhomotheticCESUtility(σ, Ω̂, ϵ)
        Ĉ2 = @inferred log_consumption_aggregator(U, p̂, Ê)
        @test newton_relative_residual(; Ê,  σ, Ω̂, ϵ, p̂, Ĉ = Ĉ2) ≤ tol
    end
end

@testset "homotheticity" begin
    for _ in 1:100
        (; Ĉ, Ê, σ, Ω̂, ϵ, p̂) = random_parameters(Val(rand(2:5)))
        U = NonhomotheticCESUtility(σ, Ω̂, ϵ)
        ν = 1.4
        Ĉ1 = log_consumption_aggregator(U, p̂, Ê)
        Ĉ2 = log_consumption_aggregator(U, p̂ .+ ν, Ê + ν)
        @test Ĉ1 ≈ Ĉ2 atol = 1e-5
        @test Ĉ1 ≈ Ĉ atol = 1e-5
        ĉ1 = log_sectoral_consumptions(U, p̂, Ê, Ĉ)
        ĉ2 = log_sectoral_consumptions(U, p̂ .+ ν, Ê + ν, Ĉ)
        @test ĉ1 ≈ ĉ2 atol = 1e-5
    end
end

@testset "API checks" begin
    (; Ĉ, Ê, σ, Ω̂, ϵ, p̂) = random_parameters(Val(2))
    tol = 1e-10
    U = NonhomotheticCESUtility(σ, Ω̂, ϵ)
    Ĉ2 = @inferred(log_consumption_aggregator(U, p̂, Ê)) # we compare to this below
    @test newton_relative_residual(; Ĉ = Ĉ2, σ, Ω̂, ϵ, p̂, Ê) ≤ tol
    @test @inferred(log_sectoral_consumptions(U, p̂, Ê, Ĉ)) ≈
        @. Ω̂ - σ * (p̂ - Ê) + (1 - σ) * ϵ * Ĉ

    @test_throws DomainError log_consumption_aggregator(U, p̂, -Inf)
    @test_throws DomainError log_consumption_aggregator(U, SVector(-Inf, 1.0), 1.0)

    @testset "budget constraint" begin
        for _ in 1:100
            (; Ĉ, Ê, σ, Ω̂, ϵ, p̂) = random_parameters(Val(rand(2:5)))
            U = NonhomotheticCESUtility(σ, Ω̂, ϵ)
            Ĉ2 = log_consumption_aggregator(U, p̂, Ê)
            @test newton_relative_residual(; Ĉ = Ĉ2, σ, Ω̂, ϵ, p̂, Ê) ≤ tol
            ĉ = log_sectoral_consumptions(U, p̂, Ê, Ĉ)
            @test logsumexp(ĉ .+ p̂) ≈ Ê
        end
    end
end

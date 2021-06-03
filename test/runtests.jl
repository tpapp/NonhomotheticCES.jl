using NonhomotheticCES
using NonhomotheticCES: calculate_Ĉ
using LogExpFunctions, Test, StaticArrays, UnPack

function random_parameters(::Val{N}) where N
    (Ê = rand() * 5.0,
     σ = rand() * 2.0,
     Ω̂s = randn(SVector{N}),
     ϵs = rand(SVector{N}) .* 3.0,
     p̂s = randn(SVector{N}))
end

@testset "finding Ĉ" begin
    for _ in 1:100
        @unpack Ê, σ, Ω̂s, ϵs, p̂s = random_parameters(Val(4))
        Ĉ = calculate_Ĉ(Ê, σ, Ω̂s, ϵs, p̂s)
        @test logsumexp((Ĉ .* ϵs .+ p̂s) .* (1-σ) .+ Ω̂s)/(1-σ) ≈ Ê atol = 1e-2
    end
end

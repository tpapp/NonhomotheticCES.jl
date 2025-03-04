# NonhomotheticCES.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/NonhomotheticCES.jl/workflows/CI/badge.svg)](https://github.com/tpapp/NonhomotheticCES.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/NonhomotheticCES.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/NonhomotheticCES.jl?branch=master)

A small package to solve for the consumption aggregator for non-homothetic CES preferences as described in

*Comin, D., Lashkari, D., & Mestieri, Martí (2021). Structural change with long-run income and price effects. Econometrica, 89(1), 311–374.*

## API

```julia
using NonhomotheticCES, StaticArrays

U = NonhomotheticCESUtility(σ,  # σ
                            Ω̂s, # LOG sectoral Ωs
                            ϵs) # sectoral ϵs

Ĉ = log_consumption_aggregator(U,
                               p̂s, # LOG prices
                               Ê)  # LOG expenditure

ĉs = log_sectoral_consumptions(U, p̂s, Ê, Ĉ)
```

The API uses a Newton solver with a good initial guess and a **fixed number of steps**, which helps with smoothness of the result (at the cost of accuracy). If you really, really need accuracy to the last [ulp](https://en.wikipedia.org/wiki/Unit_in_the_last_place), just use more steps.

There is a partial application version that calculates an even better initial guess, using fewer Newton steps later on by default:

```julia
A = log_consumption_aggregator(U, p̂s)
Ĉ = A(Ê)
```

See the docstrings for the details.

## Integrations

Partial derivatives are implemented for AD frameworks:

1. [X] [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
2. [ ] [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) **WIP**

## Example

σ = 0.5, Ω̂₁ = 0.0, Ω̂₂ = 0.0, ϵ₁ = 1.0, ϵ₂ = 2.0, p̂₁ = 0.1, p̂₂ = 0.0 (see [script/plot.jl](./script/plot.jl)).

<img src="script/example.png" width="50%">

#####
##### generate plot for readme
#####
##### - its directory should be the current one
##### - PGFPlotsX should be available in the environment

using NonhomotheticCES, PGFPlotsX, StaticArrays

U = NonhomotheticCESUtility(0.5, SVector(0.0, 0.0), SVector(1.0, 2.0))

E = range(0.001, 0.5; length = 100)
p̂ = SVector(0.1, 0.0)
Ĉ = [log_consumption_aggregator(U, p̂, log(E)) for E in E]
ĉs = [log_sectoral_consumptions(U, p̂, log(E), Ĉ) for (E, Ĉ) in zip(E, Ĉ)]

pgfsave(joinpath(@__DIR__, "example.png"),
        @pgf Axis({ xlabel = "expenditure", legend_pos = "north west",
                    yticklabel_style={ "/pgf/number format/fixed",
                                       "/pgf/number format/precision" = 2 },
                    scaled_y_ticks = false,
                    },
                  Plot({ no_marks, black }, Table(E, exp.(Ĉ))),
                  LegendEntry(raw"$C$"),
                  Plot({ no_marks, red }, Table(E, exp.(first.(ĉs)))),
                  LegendEntry(raw"$c_1$"),
                  Plot({ no_marks, blue }, Table(E, exp.(last.(ĉs)))),
                  LegendEntry(raw"$c_2$"));
        dpi = 600)

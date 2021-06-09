"""
Placeholder for a short summary about NonhomotheticCES.
"""
module NonhomotheticCES

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using LogExpFunctions: logsumexp
using Statistics: mean
using UnPack: @unpack

include("utilities.jl")
include("internals.jl")

end # module

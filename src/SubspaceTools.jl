module SubspaceTools

using LinearAlgebra
using UnPack
using Printf
using InteractiveUtils

"""
    AllowedFloat = Union{Float32, Float64}

The allowed real number types `R` that are compatible with the allowed
matrix types `T`
"""
AllowedFloat = Union{Float32, Float64}

AllowedComplex = Union{ComplexF32, ComplexF64}

"""
    AllowedTypes = Union{Float32, Float64, ComplexF32, ComplexF64}

The allowed types `T` of the matrix whose eigenpairs are to be computed.
"""
AllowedTypes = Union{AllowedFloat, AllowedComplex}

Allowed64 = Union{Float64, ComplexF64}

abstract type Cache end
abstract type Results end

export AllowedTypes
export AllowedFloat

# Davidson
export solver, solver!
export DavidsonCache
export EigenPairs
export ConvInfo
export workarrays

# SIL
export sil!
export SILCache

include("wrapper.jl")
include("davidson/cache.jl")
include("davidson/eigenpairs.jl")
include("davidson/convinfo.jl")
include("davidson/solver.jl")
include("sil/cache.jl")
include("sil/sil.jl")

end

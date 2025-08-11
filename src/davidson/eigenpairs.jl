mutable struct EigenPairs{T, R} <: Results where {T<:AllowedTypes,
                                                  R<:AllowedFloat}

    # Number of roots
    nroots::Int64

    # Matrix dimension
    matdim::Int64
    
    # Eigenvalues
    values::Vector{R}

    # Eigenvectors
    vectors::Matrix{T}

    # Residual norms
    residuals::Vector{R}

    # Convergence flags
    converged::Vector{Bool}

    # Inner constructor
    function EigenPairs{T, R}(nroots::Int64,
                              matdim::Int64) where {T<:AllowedTypes,
                                                    R<:AllowedFloat}

        # Make sure that the real type R is consistent with the matrix
        # type T
        @assert R == (T <: Allowed64 ? Float64 : Float32)

        # Eigenpairs, residual norms, and convergence flags
        values = Vector{R}(undef, nroots)
        vectors = Matrix{T}(undef, matdim, nroots)
        residuals = Vector{R}(undef, nroots)
        converged = Vector{Bool}(undef, nroots)

        new{T, R}(nroots, matdim, values, vectors, residuals, converged)
        
    end
    
end

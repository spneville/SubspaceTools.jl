mutable struct ConvInfo{R} <: Results where {R<:AllowedFloat}

    # Number of roots
    nroots::Int64

    # Residual norms
    residuals::Vector{R}

    # Convergence flags
    converged::Vector{Bool}

    # Inner constructor
    function ConvInfo{R}(nroots::Int64) where {R<:AllowedFloat}

        # Residual norms, and convergence flags
        residuals = Vector{R}(undef, nroots)
        converged = Vector{Bool}(undef, nroots)

        new{R}(nroots, residuals, converged)
        
    end
    
end

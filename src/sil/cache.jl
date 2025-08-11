mutable struct SILCache{T, R} <: Cache where {T<:AllowedTypes,
                                              R<:AllowedFloat}

    # Type parameter
    T::Type
    
    # Matrix-vector multiplication function
    f::Function

    # Matrix dimension
    matdim::Int64

    # Inner constructor
    function DavidsonCache{T, R}(f::Function,
                                 matdim::Int64
                                 ) where {T<:AllowedTypes,
                                          R<:AllowedFloat}

        println("Here")
        exit()
        
    end
        
end

mutable struct SILCache{T, R} <: Cache where {T<:AllowedTypes,
                                              R<:AllowedFloat}

    # Type parameter
    T::Type
    
    # Matrix-vector multiplication function
    f::Function

    # Matrix dimension
    matdim::Int64

    # Maximum Lanczos subspace dimension
    maxvec::Int64

    # Error threshold
    ϵ::R

    # Counters, etc.
    Kdim::Int64

    # LAPACK ?syev and ?heev work arrays
    lwork::Int64
    info::Int64
    
    # Work arrays
    Twork::Vector{T}
    Rwork::Vector{R}

    # Twork array offsets
    qvec_start::Int64
    rvec_start::Int64
    alpha_start::Int64
    beta_start::Int64
    Gmat_start::Int64
    eigvec_start::Int64
    evwork_start::Int64
    Fvcoeff_start::Int64

    # Rwork array offsets
    eigval_start::Int64
    revwork_start::Int64
    
    # Inner constructor
    function SILCache{T, R}(f::Function,
                            matdim::Int64,
                            maxvec::Int64,
                            ϵ::R,
                            Twork::Vector{T},
                            Rwork::Vector{R}
                            ) where {T<:AllowedTypes,
                                     R<:AllowedFloat}

        # Make sure that the real type R is consistent with the matrix
        # type T
        @assert R == (T <: Allowed64 ? Float64 : Float32)

        # Twork array offsets
        qvec_start, rvec_start, alpha_start,
        beta_start, Gmat_start, eigvec_start,
        evwork_start, Fvcoeff_start = sil_Twork_offsets(matdim, maxvec)

        # Rwork array offsets
        eigval_start, revwork_start = sil_Rwork_offsets(maxvec)

        # Counters, etc.
        Kdim = 0

        # LAPACK ?syev and ?heev work arrays
        # We will use the same dimension for both the work and rwork
        # arrays
        lwork = 3 * maxvec
        info = 0
        
        new{T, R}(T, f, matdim, maxvec, ϵ, Kdim, lwork, info, 
                  Twork, Rwork, qvec_start, rvec_start,
                  alpha_start, beta_start, Gmat_start,
                  eigvec_start, evwork_start, Fvcoeff_start,
                  eigval_start, revwork_start)
        
    end
        
end

function sil_workarrays(T::DataType, matdim::Int64,
                        maxvec::Int64)

    @assert T <: AllowedTypes
    
    if T <: Union{Float32, ComplexF32}
        R = Float32
    else
        R = Float64
    end

    Tdim = sil_Tworksize(matdim, maxvec)
    Twork = Vector{T}(undef, Tdim)
    fill!(Twork, 0.0)
    
    Rdim = sil_Rworksize(maxvec)
    Rwork = Vector{R}(undef, Rdim)
    fill!(Rwork, 0.0)
    
    return Twork, Rwork
    
end

function sil_Tworksize(matdim::Int64, maxvec::Int64)

    dim = 0

    # Lanczos subspace vectors
    dim += matdim * maxvec

    # r-vector
    dim += matdim

    # On-diagonal elements of the Lanczos subspace matrix (α)
    dim += maxvec

    # Off-diagonal elements of the Lanczos subspace matrix (β)
    dim += maxvec

    # Lanczos subspace matrix (G)
    dim += maxvec^2

    # Eigenvectors of the subspace matrix
    dim += maxvec^2

    # LAPACK ?syev and ?heev work arrays
    dim += 3 * maxvec

    # Coefficients of F(A)*v in the Lanczos vector basis
    dim += maxvec
    
    return dim
    
end

function sil_Rworksize(maxvec::Int64)

    dim = 0

    # Eigenvalues of the subspace matrix
    dim += maxvec

    # LAPACK ?syev and ?heev work arrays
    dim += 3 * maxvec
    
    return dim
    
end

function sil_Twork_offsets(matdim::Int64, maxvec::Int64)

    lwork = 3 * maxvec

    qvec_start = 1
    qvec_end = qvec_start + matdim * maxvec - 1

    rvec_start = qvec_end + 1
    rvec_end = rvec_start + matdim - 1

    alpha_start = rvec_end + 1
    alpha_end = alpha_start + maxvec - 1

    beta_start = alpha_end + 1
    beta_end = beta_start + maxvec - 1
    
    Gmat_start = beta_end + 1
    Gmat_end = Gmat_start + maxvec^2 - 1

    eigvec_start = Gmat_end + 1
    eigvec_end = eigvec_start + maxvec^2 - 1

    evwork_start = eigvec_end + 1
    evwork_end = evwork_start + lwork - 1

    Fvcoeff_start = evwork_end + 1
    Fvcoeff_end = Fvcoeff_start + maxvec - 1
    
    return qvec_start, rvec_start, alpha_start, beta_start,
    Gmat_start, eigvec_start, evwork_start, Fvcoeff_start
    
end

function sil_Rwork_offsets(maxvec::Int64)

    lwork = 3 * maxvec
    
    eigval_start = 1
    eigval_end = eigval_start + maxvec - 1

    revwork_start = eigval_end + 1
    revwork_end = revwork_start + lwork - 1
    
    return eigval_start, revwork_start
    
end

function qvec(cache::SILCache, range1::UnitRange{Int64},
              range2::UnitRange{Int64})

    @assert range1 == 1:cache.matdim

    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.qvec_start + (range2.start - 1) * dim1
    iend = istart + len - 1

    q = reshape(view(cache.Twork, istart:iend),
                (dim1, dim2))

    return q
    
end

function rvec(cache::SILCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.rvec_start + range.start - 1
    iend = istart + len - 1
    
    r = view(cache.Twork, istart:iend)
    
    return r
    
end

function alpha(cache::SILCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.alpha_start + range.start - 1
    iend = istart + len - 1
    
    α = view(cache.Twork, istart:iend)
    
    return α
    
end

function beta(cache::SILCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.beta_start + range.start - 1
    iend = istart + len - 1
    
    β = view(cache.Twork, istart:iend)
    
    return β
    
end

function Gmat(cache::SILCache, range1::UnitRange{Int64},
              range2::UnitRange{Int64})

    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.Gmat_start + (range2.start - 1) * dim1
    iend = istart + len - 1

    G = reshape(view(cache.Twork, istart:iend),
                (dim1, dim2))
    
    return G
    
end

function eigvec(cache::SILCache, range1::UnitRange{Int64},
                range2::UnitRange{Int64})

    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.eigvec_start + (range2.start - 1) * dim1
    iend = istart + len - 1

    eigvec = reshape(view(cache.Twork, istart:iend),
                     (dim1, dim2))
    
    return eigvec
    
end

function eigval(cache::SILCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.eigval_start + range.start - 1
    iend = istart + len - 1
    
    eigval = view(cache.Rwork, istart:iend)
    
    return eigval
    
end

function evwork(cache::SILCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.evwork_start + range.start - 1
    iend = istart + len - 1
    
    work = view(cache.Twork, istart:iend)

    return work
    
end

function Fvcoeff(cache::SILCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.Fvcoeff_start + range.start - 1
    iend = istart + len - 1
    
    C = view(cache.Twork, istart:iend)

    return C
    
end

function revwork(cache::SILCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.revwork_start + range.start - 1
    iend = istart + len - 1

    rwork = view(cache.Rwork, istart:iend)

    return rwork
    
end


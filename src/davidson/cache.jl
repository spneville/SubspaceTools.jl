mutable struct DavidsonCache{T, R} <: Cache where {T<:AllowedTypes,
                                                   R<:AllowedFloat}
    
    # Type parameter
    T::Type
    
    # Matrix-vector multiplication function
    f::Function

    # Diagonal of the matrix whose eigenpairs are sought
    diag::Vector{T}
    
    # Number of roots to compute
    nroots::Int64

    # Dimension of the matrix whose eigenpairs are sought
    matdim::Int64

    # Block size
    blocksize::Int64

    # Maximum subspace dimension
    maxvec::Int64

    # Residual norm convergence threshold
    tol::Float64

    # Maximum number of iterations
    niter::Int64

    # Counters, etc.
    currdim::Int64
    nconv::Int64
    nnew::Int64
    nsigma::Int64
    iconv::Vector{Int64}

    # LAPACK ?syev and ?heev work arrays
    lwork::Int64
    info::Int64

    # -1.0, 0.0, and +1.0
    minus_one::T
    zero::T
    one::T

    # Work arrays
    Twork::Vector{T}
    Rwork::Vector{R}

    # TWork array offsets
    bvec_start::Int64
    sigvec_start::Int64
    Gmat_start::Int64
    alpha_start::Int64
    rnorm_start::Int64
    work2_start::Int64
    work3_start::Int64
    work4_start::Int64
    evwork_start::Int64

    # Rwork array offsets
    rho_start::Int64
    rho1_start::Int64
    work1_start::Int64
    revwork_start::Int64
    
    # Inner constructor
    function DavidsonCache{T, R}(f::Function,
                                 diag::Vector{T},
                                 nroots::Int64,
                                 matdim::Int64,
                                 blocksize::Int64,
                                 maxvec::Int64,
                                 tol::Float64,
                                 niter::Int64,
                                 Twork::Vector{T},
                                 Rwork::Vector{R}
                                 ) where {T<:AllowedTypes,
                                          R<:AllowedFloat}

        # Make sure that the real type R is consistent with the matrix
        # type T
        @assert R == (T <: Allowed64 ? Float64 : Float32)
        
        # Counters, etc.
        currdim = 0
        nconv = 0
        nnew = 0
        nsigma = 0
        iconv = Vector{Int64}(undef, blocksize)

        # LAPACK ?syev and ?heev work arrays
        # We will use the same dimension for both the work and rwork
        # arrays
        lwork = 3 * maxvec
        info = 0
        
        # -1.0, 0.0, and +1.0
        minus_one::T = -1.0
        zero::T = 0.0
        one::T = 1.0

        # Twork array offsets
        bvec_start, sigvec_start, Gmat_start, alpha_start,
        work2_start, work3_start, work4_start,
        evwork_start = davidson_Twork_offsets(matdim, blocksize,
                                              maxvec)
        
        # Rwork array offsets
        rho_start, rho1_start, rnorm_start, work1_start,
        revwork_start = davidson_Rwork_offsets(matdim, blocksize,
                                               maxvec)
        
        new{T, R}(T, f, diag, nroots, matdim, blocksize, maxvec, tol,
                  niter, currdim, nconv, nnew,
                  nsigma, iconv, lwork, info, minus_one,
                  zero, one, Twork, Rwork, bvec_start, sigvec_start,
                  Gmat_start, alpha_start, rnorm_start, work2_start,
                  work3_start, work4_start, evwork_start, rho_start,
                  rho1_start, work1_start, revwork_start)
        
    end

end

"""
    davidson_workarrays(T, matdim, blocksize, maxvec)

Constructs the `Twork` and `Rwork` work arrays required to make the
in-place `solver!` function allocation-free.

# Arguments

* `T<:AllowedTypes`: Matrix type
* `matdim::Int64`: Dimension of the matrix
* `blocksize::Int64`: Block size
* `maxvec::Int64`: Maximum subspace dimension

# Return values

The return value is of the form `Twork, Rwork = davidson_workarrays(…)`, where

* `Twork::Vector{T<:AllowedTypes}`
* `Rwork::Vector{R<:AllowedFloat}`, where `R` is compatible with `T`

"""
function davidson_workarrays(T::DataType, matdim::Int64,
                             blocksize::Int64, maxvec::Int64)

    @assert T <: AllowedTypes
    
    if T <: Union{Float32, ComplexF32}
        R = Float32
    else
        R = Float64
    end
    
    Tdim = davidson_Tworksize(matdim, blocksize, maxvec)
    Twork = Vector{T}(undef, Tdim)
    
    Rdim = davidson_Rworksize(matdim, blocksize, maxvec)
    Rwork = Vector{R}(undef, Rdim)
    
    return Twork, Rwork
    
end

function davidson_Tworksize(matdim::Int64, blocksize::Int64,
                            maxvec::Int64)

    dim = 0
    
    # Subspace vectors
    dim += matdim * maxvec
        
    # Sigma vectors
    dim += matdim * maxvec
    
    # Subspace matrix
    dim += maxvec * maxvec

    # Subspace eigenvectors
    dim += maxvec * maxvec

    # Residual norms
    dim += blocksize
    
    # Work2 array
    dim += matdim * blocksize

    # Work3 array
    dim += maxvec * blocksize

    # Work4 array
    dim += blocksize * blocksize
    
    # LAPACK ?syev and ?heev work arrays
    dim += 3 * maxvec

    return dim
    
end

function davidson_Rworksize(matdim::Int64, blocksize::Int64,
                            maxvec::Int64)

    dim = 0
    
    # Subspace eigenvalues plus a working copy
    dim += maxvec
    dim += maxvec

    # Residual norms
    dim += blocksize
    
    # Work1 array
    dim += matdim

    # LAPACK ?syev and ?heev work arrays
    dim += 3 * maxvec

    return dim
    
end

function davidson_Twork_offsets(matdim::Int64, blocksize::Int64,
                                maxvec::Int64)

    lwork = 3 * maxvec
    
    bvec_start = 1
    bvec_end = bvec_start + matdim*maxvec - 1
    
    sigvec_start = bvec_end + 1
    sigvec_end = sigvec_start + matdim*maxvec - 1
    
    Gmat_start = sigvec_end + 1
    Gmat_end = Gmat_start + maxvec*maxvec - 1
    
    alpha_start = Gmat_end + 1
    alpha_end = alpha_start + maxvec*maxvec - 1
    
    work2_start = alpha_end + 1
    work2_end = work2_start + matdim*blocksize - 1
    
    work3_start = work2_end + 1
    work3_end = work3_start + maxvec*blocksize - 1
    
    work4_start = work3_end + 1
    work4_end = work4_start + blocksize*blocksize - 1
    
    evwork_start = work4_end + 1
    evwork_end = evwork_start + lwork - 1

    return bvec_start, sigvec_start, Gmat_start, alpha_start,
    work2_start, work3_start, work4_start, evwork_start
    
end

function davidson_Rwork_offsets(matdim::Int64, blocksize::Int64,
                                maxvec::Int64)

    lwork = 3 * maxvec

    rho_start = 1
    rho_end = rho_start + maxvec - 1

    rho1_start = rho_end + 1
    rho1_end = rho1_start + maxvec - 1

    rnorm_start =rho1_end + 1
    rnorm_end = rnorm_start + blocksize - 1
    
    work1_start = rnorm_end + 1
    work1_end = work1_start + matdim - 1
    
    revwork_start = work1_end + 1
    revwork_end = revwork_start + lwork - 1

    return rho_start, rho1_start, rnorm_start, work1_start,
    revwork_start
    
end

function bvec(cache::DavidsonCache, range1::UnitRange{Int64},
              range2::UnitRange{Int64})

    @assert range1 == 1:cache.matdim
    
    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.bvec_start + (range2.start - 1) * dim1
    iend = istart + len - 1
    
    b = reshape(view(cache.Twork, istart:iend),
                (dim1, dim2))
    
    return b
    
end

function sigvec(cache::DavidsonCache, range1::UnitRange{Int64},
                range2::UnitRange{Int64})

    @assert range1 == 1:cache.matdim

    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.sigvec_start + (range2.start - 1) * dim1
    iend = istart + len - 1
    
    σ = reshape(view(cache.Twork, istart:iend),
                (dim1, dim2))
    
    return σ
    
end

function Gmat(cache::DavidsonCache, range1::UnitRange{Int64},
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

function alpha(cache::DavidsonCache, range1::UnitRange{Int64},
              range2::UnitRange{Int64})

    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.alpha_start + (range2.start - 1) * dim1
    iend = istart + len - 1
    
    α = reshape(view(cache.Twork, istart:iend),
                (dim1, dim2))

    return α
    
end

function rho(cache::DavidsonCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.rho_start + range.start - 1
    iend = istart + len - 1
    
    ρ = view(cache.Rwork, istart:iend)

    return ρ
    
end

function rho1(cache::DavidsonCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.rho1_start + range.start - 1
    iend = istart + len - 1
    
    ρ1 = view(cache.Rwork, istart:iend)

    return ρ1
    
end

function rnorm(cache::DavidsonCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.rnorm_start + range.start - 1
    iend = istart + len - 1
    
    resnorm = view(cache.Rwork, istart:iend)

    return resnorm
    
end

function work1(cache::DavidsonCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.work1_start + range.start - 1
    iend = istart + len - 1
    
    w1 = view(cache.Rwork, istart:iend)

    return w1
    
end

function work2(cache::DavidsonCache, range1::UnitRange{Int64},
               range2::UnitRange{Int64})

    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.work2_start + (range2.start - 1) * dim1
    iend = istart + len - 1
    
    w2 = reshape(view(cache.Twork, istart:iend),
                 (dim1, dim2))

    return w2
    
end

function work3(cache::DavidsonCache, range1::UnitRange{Int64},
               range2::UnitRange{Int64})

    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.work3_start + (range2.start - 1) * dim1
    iend = istart + len - 1
    
    w3 = reshape(view(cache.Twork, istart:iend),
                 (dim1, dim2))

    return w3
    
end

function work4(cache::DavidsonCache, range1::UnitRange{Int64},
               range2::UnitRange{Int64})

    dim1 = range1.stop - range1.start + 1
    dim2 = range2.stop - range2.start + 1

    len = length(range1) * length(range2)

    istart = cache.work4_start + (range2.start - 1) * dim1
    iend = istart + len - 1
    
    w4 = reshape(view(cache.Twork, istart:iend),
                 (dim1, dim2))

    return w4
    
end

function evwork(cache::DavidsonCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.evwork_start + range.start - 1
    iend = istart + len - 1
    
    work = view(cache.Twork, istart:iend)

    return work
    
end

function revwork(cache::DavidsonCache, range::UnitRange{Int64})

    len = length(range)

    istart = cache.revwork_start + range.start - 1
    iend = istart + len - 1

    rwork = view(cache.Rwork, istart:iend)

    return rwork
    
end

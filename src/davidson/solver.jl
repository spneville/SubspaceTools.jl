"""
    solver(f, diag, nroots, matdim; tol=1e-4, blocksize=nroots+5,
           maxvec=4*blocksize, niter=100, verbose=false)

# Arguments

* `f::Function`: In-place matrix-vector multiplication function
* `diag::Vector{T}`: Diagonal of the matrix whose eigenpairs are sought, where [`T<:AllowedTypes`](@ref AllowedTypes)
* `nroots::Int64`: Number of eigenpairs to compute
* `matdim::Int64`: Dimension of the matrix

# Optional keyword arguments

* `tol::Float64`: Residual norm convergence threshold
* `blocksize::Int64`: Block size
* `maxvec::Int64`: Maximum subspace dimension
* `niter::Int64`: Maximum number of iterations
* `verbose::Bool`: Verbose output flag. If true, then a summary is printed
                   at the end of each iteration

# Return values

The return value is of the form `result = solver(…)`, where `result` is an
`EigenPairs` object with the following fields:

* `vectors::Matrix{T}`: Eigenvectors
* `values::Vector{R}`: Eigenvalues
* `residuals::Vector{R}`: Residual norms of the eigenvectors
* `converged::Vector{Bool}`: Convergence flags (`converged[i] == true` if the `i` root converged)

!!! warning "Check for convergence"

    If `verbose==false`, no warning is printed if one or more roots fail
    to converge. The contents of `result.converged` should therefore
    be checked.

# A note on the matrix-vector multiplication function

Let the matrix whose eigenpairs are sought be denoted by `A`. The in-place
`Function` `f` recieves takes as arguments an `AbstractMatrix{T}` of vectors `v`
and an `AbstractMatrix{T}` of matrix-vector products `Av` that is to be computed,
where [`T<:AllowedTypes`](@ref AllowedTypes). The required form of this function
is detailed in [Matrix-vector multiplication function](@ref matvec) section.

"""
function solver(f::Function,
                diag::AbstractVector{T},
                nroots::Int64,
                matdim::Int64;
                tol=1e-4,
                blocksize=nroots+5,
                maxvec=4*blocksize,
                niter=100,
                verbose=false,
                kwargs...) where {T<:AllowedTypes}

    # Real type
    T <: Allowed64 ? R = Float64 : R = Float32
    
    # Work arrays
    Twork, Rwork = workarrays(T, matdim, blocksize, maxvec)
    
    # EigenPair result
    eigenpairs = EigenPairs{T, R}(nroots, matdim)

    # Call the in-place solver
    convinfo = solver!(eigenpairs.vectors, eigenpairs.values, f, diag,
                       nroots, matdim, Twork, Rwork;
                       tol=tol, blocksize=blocksize, maxvec=maxvec,
                       niter=niter, verbose=verbose, guess=false,
                       kwargs...)

    # Fill in the convergence information
    for i in 1:nroots
        eigenpairs.residuals[i] = convinfo.residuals[i]
        eigenpairs.converged[i] = convinfo.converged[i]
    end
    
    return eigenpairs
    
end

function solver!(vectors::AbstractMatrix{T},
                 values::AbstractVector{R},
                 f::Function,
                 diag::AbstractVector{T},
                 nroots::Int64,
                 matdim::Int64;
                 tol=1e-4,
                 blocksize=nroots+5,
                 maxvec=4*blocksize,
                 niter=100,
                 verbose=false,
                 guess=false,
                 kwargs...) where {T<:AllowedTypes, R<:AllowedFloat}
    
    Twork, Rwork = workarrays(T, matdim, blocksize, maxvec)
    
    convinfo = solver!(vectors, values, f, diag, nroots, matdim, Twork,
                       Rwork; tol=tol, blocksize=blocksize, maxvec=maxvec,
                       niter=niter, verbose=verbose, guess=guess,
                       kwargs...)

    return convinfo
    
end

"""
    solver!(vectors, values, f, diag, nroots, matdim, [Twork, Rwork];
            tol=1e-4, blocksize=nroots+5, maxvec=4*blocksize, niter=100,
            verbose=false, guess=false)

# Arguments

* `vectors::Matrix{T}`: Eigenvectors, where [`T<:AllowedTypes`](@ref AllowedTypes)
* `values::Vector{R}`: Eigenvalues, where [`R<:AllowedFloat`](@ref AllowedFloat)
* `f::Function`: In-place matrix-vector multiplication function
* `diag::Matrix{T}`: Diagonal of the matrix whose eigenpairs are sought, where [`T<:AllowedTypes`](@ref AllowedTypes)
* `nroots::Int64`: Number of eigenpairs to compute
* `matdim::Int64`: Dimension of the matrix

# Optional arguments

The following two pre-allocated work arrays may be supplied:

* `Twork::Vector{T}`: Type [`T<:AllowedTypes`](@ref AllowedTypes) work array
* `Rwork::Vector{R}`: Type [`R<:AllowedFloat`](@ref AllowedFloat) work array

See [Work arrays](@ref WorkArrays) for the procedure for constructing these.

# Optional keyword arguments

* `tol::Float64`: Residual norm convergence threshold
* `blocksize::Int64`: Block size
* `maxvec::Int64`: Maximum subspace dimension
* `niter::Int64`: Maximum number of iterations
* `verbose::Bool`: Verbose output flag. If true, then a summary is printed
                   at the end of each iteration
* `guess::Bool`: If `true`, then on input, the `vectors` array is taken to
                 contain the guess vectors

# Return values

The return value is of the form `result = solver!(…)`, where `result` is an
`ConvInfo` object with the following fields:

* `residuals::Vector{R}`: Residual norms of the eigenvectors
* `converged::Vector{Bool}`: Convergence flags (`converged[i] == true` if the `i` root converged)

!!! warning "Check for convergence"

    If `verbose==false`, no warning is printed if one or more roots fail
    to converge. The contents of `result.converged` should therefore
    be checked.

"""
function solver!(vectors::AbstractMatrix{T},
                 values::AbstractVector{R},
                 f::Function,
                 diag::AbstractVector{T},
                 nroots::Int64,
                 matdim::Int64,
                 Twork::Vector{T},
                 Rwork::Vector{R};
                 tol=1e-4,
                 blocksize=nroots+5,
                 maxvec=4*blocksize,
                 niter=100,
                 verbose=false,
                 guess=false,
                 kwargs...) where {T<:AllowedTypes, R<:AllowedFloat}

    # Check on the input
    checkinp(nroots, blocksize, maxvec, matdim, Twork, Rwork)
    
    # Davidson cache
    cache = DavidsonCache{T, R}(f, diag, nroots, matdim, blocksize,
                                maxvec, tol, niter, Twork, Rwork)

    # ConvInfo result
    convinfo = ConvInfo{R}(nroots)
    
    # Construct the guess vectors
    if guess
        guessvec_user(cache, vectors)
    else
        guessvec(cache)
    end
        
    # Run the generalised Davidson algorithm
    run_gendav(cache, verbose; kwargs...)

    # Eigenvalues
    ρ = rho(cache, 1:nroots)
    copy!(values, ρ)
    
    # Get the eigenvectors
    eigenvectors!(vectors, cache)

    # Fill in the convergence information
    res = rnorm(cache, 1:nroots)
    for i in 1:nroots
        convinfo.residuals[i] = res[i]
        cache.iconv[i] == 1 ? convinfo.converged[i] = true :
            convinfo.converged[i] = false
    end
    
    return convinfo
    
end

function checkinp(nroots::Int64, blocksize::Int64, maxvec::Int64,
                  matdim::Int64, Twork::Vector{<:AllowedTypes},
                  Rwork::Vector{<:AllowedFloat})

    # The block size must be greater than or equal to the number
    # of roots
    if blocksize < nroots
        @error "Block size is less than the no. roots" blocksize nroots
        exit()
    end

    # The maximum subspace dimension must be a multiple of the
    # block size
    if mod(maxvec, blocksize) != 0
        @error "The maximum subspace dimension must be a multiple" *
            " of the block size"
        exit()
    end

    # The maximum subspace dimension must be greater than the blocksize
    if maxvec <= blocksize
        @error "The maximum subspace dimension must be greater" *
            " than the blocksize" maxvec blocksize
    end

    # The maximum subspace dimension cannot be greater than the matrix
    # dimension
    if maxvec > matdim
        @error "The maximum subspace dimension cannot be greater" *
            " than the matrix dimension" maxvec matdim
        exit()
    end

    # Work array dimensions
    if size(Rwork)[1] < Rworksize(matdim, blocksize, maxvec)
        @error "Rwork is not large enough"
        exit() 
    end

    if size(Twork)[1] < Tworksize(matdim, blocksize, maxvec)
        @error "Twork is not large enough"
        exit()
    end
    
end

function guessvec(cache::Cache)

    #
    # For now, we will take the unit vectors (0,...,0,1,0,..,0)
    # as our guess vectors with the non-zero elements corresponding
    # to the lowest value diagonal matrix elements
    #

    @unpack matdim, blocksize = cache
    
    # Sort the diagonal matrix elements
    # Note that these are constrained to be real for a symmetric or
    # Hermitian matrix
    hii = work1(cache, 1:matdim)
    for i in 1:matdim
        @inbounds hii[i] = real(cache.diag[i])
    end
    ix = sortperm(hii)

    # Construct the guess vectors
    b = bvec(cache, 1:matdim, 1:blocksize)
    fill!(b, 0.0)
    for i in 1:cache.blocksize
        @inbounds b[ix[i],i] = 1.0
    end
    
end

function guessvec_user(cache::DavidsonCache{T},
                       vectors::AbstractMatrix{T}) where T<:AllowedTypes

    @unpack nroots, blocksize, matdim, one, zero = cache

    # Number of user-supplied guess vectors
    nuser = size(vectors)[2]
    
    # Check on the number of user-supplied vectors
    if nuser < nroots
        @error "The no. of user-supplied guess vectors is less" *
            " than the no. roots" nuser nroots
        exit()
    end
    if nuser > blocksize
        @error "The no. of user-supplied guess vectors is greater" *
            " than the block size" nuser blocksize
        exit()
    end

    # If there is only a single vector, then simply load that
    # and return
    if nuser == blocksize
        b = bvec(cache, 1:matdim, 1:blocksize)
        @views b[:,1] = vectors[:,1]
        return
    end
        
    # The user supplies nuser guess vectors, and we need
    # blocksize guess vectors.
    # So, we need to generate nextra = blocksize - nuser
    # extra guess vectors
    nextra = blocksize - nuser
    
    # Input vectors
    v = vectors
    
    # Combined basis {vᵢ} ⋃ {xᵢ}, where the xᵢ are random extra normalised
    # vectors
    g = work2(cache, 1:matdim, 1:blocksize)
    for i in 1:nuser
        @inbounds @views g[:,i] = v[:,i]
    end
    for i in 1:nextra
        for j in 1:matdim
            @inbounds g[j,nuser+i] = rand()
        end
        @views norm = sqrt(dot(g[:,nuser+i], g[:,nuser+i]))
        for j in 1:matdim
            @inbounds g[j,nuser+i] /= norm
        end
        
    end
        
    # Overlaps of the combined basis vectors
    S = work3(cache, 1:blocksize, 1:blocksize)
    BLAS.gemm!('C', 'N', one, g, g, zero, S)

    # Inverse square root of the overlap matrix
    # Note that invsqrt_matrix will overwrite the S matrix
    Sinvsq = work4(cache, 1:blocksize, 1:blocksize)
    invsqrt_matrix!(Sinvsq, S, cache)

    # Symmetric orthogonalisation of the combined basis vectors
    b = bvec(cache, 1:matdim, 1:blocksize)
    BLAS.gemm!('N', 'N', one, g, Sinvsq, zero, b)
        
end

function run_gendav(cache::Cache, verbose::Bool; kwargs...)

    #
    # Initialisation
    #
    # currdim: the current dimension of the subspace
    # nnew:    the no. new subspace vectors added in a given iteration
    # nconv:   the no. of converged roots
    # nsigma:  the total no. sigma vectors calculated
    cache.currdim = cache.blocksize
    cache.nnew = cache.blocksize
    cache.nconv = 0
    cache.nsigma = 0

    # Loop over iterations
    for k in 1:cache.niter

        # Compute the σ-vectors
        sigma_vectors(cache; kwargs...)

        # Compute the new elements in the subspace matrix
        subspace_matrix(cache)
        
        # Compute the eigenpairs of the subspace matrix
        subspace_diag(cache)

        # Compute the residual vectors. Note that these will be stored
        # in the bvec array and subsequently transformed in place
        # to obtain the correction vectors, and then the new
        # subspace vectors
        residual_vectors(cache)

        # Print the report for this iteration
        if verbose print_report(k, cache) end

        # Stop here if all the roots are converged
        if all(i -> i == 1, cache.iconv[1:cache.nroots]) break end

        # Compute the correction vectors
        correction_vectors(cache)

        # Compute the new subspace vectors
        subspace_vectors(cache)

        # Subspace collapse?
        if cache.currdim + cache.nnew * 2 > cache.maxvec
            subspace_collapse(cache)
        else
            cache.currdim = cache.currdim + cache.nnew
        end

        # If we are here and this is the last iteration, then
        # we failed to converge all roots
        if k == cache.niter @warn "Not all roots converged" end
        
    end

end

function sigma_vectors(cache::Cache; kwargs...)

    @unpack nnew, currdim, matdim = cache
    
    # Update the total no. σ-vector calculations
    cache.nsigma += nnew
    
    # Indices of the first and last subspace vectors for which
    # σ-vectors are required
    ki = currdim - nnew + 1
    kf = currdim

    # Compute the σ-vectors
    b = bvec(cache, 1:matdim, ki:kf)
    σ = sigvec(cache, 1:matdim, ki:kf)

    if haskey(kwargs, :data)
        cache.f(b, σ, kwargs[:data])
    else
        cache.f(b, σ)
    end

end

function subspace_matrix(cache::DavidsonCache{T}) where T<:AllowedTypes

    # Dimensions
    @unpack matdim, currdim, nnew, zero, one = cache

    # Work array
    bσ = work3(cache, 1:currdim, 1:nnew)
    fill!(bσ, 0.0)

    # b^T sigma matrix product
    i1 = currdim - nnew + 1
    i2 = currdim
    b = bvec(cache, 1:matdim, 1:i2)
    σ = sigvec(cache, 1:matdim, i1:i2)
            
    BLAS.gemm!('C', 'N', one, b, σ, zero, bσ)
    
    # Re-order the Gmat working vector to be consistent
    # with the new subspace dimension
    olddim = currdim - nnew
    if olddim > 0
        
        # Make a copy of the subspace matrix from the last iteration
        G = Gmat(cache, 1:olddim, 1:olddim)
        tmp = alpha(cache, 1:olddim, 1:olddim)
        copy!(tmp, G)

        # Fill in the old subspace block in the new ordering
        G = Gmat(cache, 1:currdim, 1:currdim)

        for j in 1:olddim
            for i in 1:olddim
                @inbounds G[i,j] = tmp[i,j]
            end
        end

    end

    # Fill in the Gmat array
    G = Gmat(cache, 1:currdim, 1:currdim)

    for i in 1:currdim
        for j in 1:nnew
            j1 = currdim - nnew + j
            @inbounds G[i,j1] = bσ[i,j]
            @inbounds G[j1,i] = conj(G[i,j1])
        end
    end
    
end

function subspace_diag(cache::DavidsonCache{T}) where T<:AllowedFloat

    currdim = cache.currdim
    
    jobz = "V"
    uplo = "L"
    n = currdim
    a = alpha(cache, 1:currdim, 1:currdim)
    lda = currdim
    w = rho(cache, 1:currdim)
    lwork = cache.lwork
    work = evwork(cache, 1:lwork)
    info = cache.info

    # Fill in the subspace matrix
    G = Gmat(cache, 1:currdim, 1:currdim)
    
    for j in 1:currdim
        for i in 1:currdim
            @inbounds a[i,j] = G[i,j]
        end
    end

    # Call to ?syev
    syev!(jobz, uplo, n, a, lda, w, work, lwork, info)
    
end

function subspace_diag(cache::DavidsonCache{T}) where T<:AllowedComplex

    currdim = cache.currdim
    
    jobz = "V"
    uplo = "L"
    n = currdim
    a = alpha(cache, 1:currdim, 1:currdim)    
    lda = currdim
    w = rho(cache, 1:currdim)
    lwork = cache.lwork
    work = evwork(cache, 1:lwork)
    rwork = revwork(cache, 1:lwork)
    info = cache.info

    # Fill in the subspace matrix
    G = Gmat(cache, 1:currdim, 1:currdim)
    for j in 1:currdim
        for i in 1:currdim
            @inbounds a[i,j] = G[i,j]
        end
    end

    # Call to ?heev
    heev!(jobz, uplo, n, a, lda, w, work, lwork, rwork, info)
    
end

function residual_vectors(cache::Cache)

    @unpack matdim, maxvec, currdim, blocksize, nnew, tol,
    minus_one, zero, one = cache

    # Subspace eigenvectors    
    α = alpha(cache, 1:currdim, 1:currdim)
    
    # Subspace eigenvalues
    #rho = view(cache.rho, 1:currdim)

    ρ = rho(cache, 1:currdim)
    
    # Working arrays
    α_bar = work3(cache, 1:currdim, 1:blocksize)    
    
    #
    # Compute the residual vectors
    # r_k = Sum_i α_ik * (σ_i - ρ_k b_i),
    #
    # (α_bar)_ik = α_ik * rho_K
    for k in 1:blocksize
        for i in 1:currdim
            @inbounds α_bar[i,k] = α[i,k] * ρ[k]
        end
    end
    
    # σ α
    σ = sigvec(cache, 1:matdim, 1:currdim)
    α = alpha(cache, 1:currdim, 1:blocksize)
    σα = work2(cache, 1:matdim, 1:blocksize)
    
    BLAS.gemm!('N', 'N', one, σ, α, zero, σα)
    
    # σ α - b α_bar
    b = bvec(cache, 1:matdim, 1:currdim)
    BLAS.gemm!('N', 'N', minus_one, b, α_bar, one, σα)
        
    #
    # Save the residual vectors for the unconverged roots
    #
    # Initialisation
    ki=currdim + 1
    kf=currdim + nnew
    resnorm = rnorm(cache, 1:blocksize)
    iconv = cache.iconv
    ρ1 = rho1(cache, 1:maxvec)
    
    # Loop over roots
    nnew = 0
    iconv .= 0

    for k in 1:blocksize

        # Residual norm
        r = reshape(work2(cache, 1:matdim, k:k), (matdim))
        @inbounds resnorm[k] = real(sqrt(dot(r, r)))
        
        # Update the convergence information
        if resnorm[k] < tol iconv[k] = 1 end
    
        # Save the residual vector and corresponding eigenvalue if it
        # corresponds to an unconverged root
        if iconv[k] == 0

            nnew = nnew + 1

            k1 = ki-1+nnew
            
            b = reshape(bvec(cache, 1:matdim, k1:k1), (matdim))
            
            r = reshape(work2(cache, 1:matdim, k:k), (matdim))
            
            b = copy!(b, r) 
            
            @inbounds ρ1[nnew] = ρ[k]

        end

    end
    
end

function print_report(k::Int64, cache::Cache)

    # Table header
    if k == 1
        println("\n", "*"^36)
        println(" Iteration  Nvec  Max rnorm   Nconv")
        println("*"^36)
    end

    # Information for the current iteration
    @unpack blocksize, currdim, iconv, nroots = cache

    resnorm = rnorm(cache, 1:blocksize)

    maxres = maximum(resnorm[1:nroots])

    nconv = sum(iconv)

    @printf("%6d    %6d  %.4e %6d \n", k, currdim, maxres, nconv)
    
end

function correction_vectors(cache::Cache)

    @unpack matdim, maxvec, currdim, nnew, diag = cache

    ρ1 = rho1(cache, 1:maxvec)
    
    b = bvec(cache, 1:matdim, 1:maxvec)
    
    #
    # Diagonal preconditioned residue correction vectors
    #

    # Indices of the positions in the bvec array in which the
    # correction vectors will be stored
    ki = currdim + 1
    kf = currdim + nnew
    
    # Loop over correction vectors
    k1 = 0
    for k in ki:kf

        k1 += 1
        
        # Loop over elements of the correction vector
        for i in 1:matdim
            @inbounds b[i,k] = -b[i,k] / (diag[i]-ρ1[k1])
        end
            
    end

end

function subspace_vectors(cache::Cache)

    @unpack T, matdim, maxvec, currdim, nnew, zero, one = cache

    b = bvec(cache, 1:matdim, 1:maxvec)

    # Indices of the positions in the bvec array in which the new
    # subspace vectors will be stored
    ki = currdim + 1
    kf = currdim + nnew
    
    #
    # New orthonormalisation of the correction vectors
    #
    # Performed in two steps:
    #
    # (1) Gram-Schmidt orthogonalisation against the previous subspace
    #     vectors
    #
    # (2) Symmetric orthogonalisation within the space spanned by the
    #     intermediately orthogonalised correction vectors from (1)
    #

    # Overlaps between the previous subspace vectors and the correction
    # vectors
    Smat = work3(cache, 1:currdim, 1:nnew)
    
    bprev = bvec(cache, 1:matdim, 1:currdim)
    
    bnew = bvec(cache, 1:matdim, ki:kf)
    
    BLAS.gemm!('C', 'N', one, bprev, bnew, zero, Smat)

    # GS orthogonalisation of the correction vectors against the previous
    # subspace vectors
    k1 = 0
    for k in ki:kf
        k1 += 1

        for i in 1:currdim
            for j in 1:matdim
                @inbounds b[j,k] -= Smat[i,k1] * b[j,i]
            end
            
        end

        bk = reshape(bvec(cache, 1:matdim, k:k), (matdim))
        
        len_bk = sqrt(dot(bk, bk))

        for j in 1:matdim
            @inbounds bk[j] /= len_bk
        end
        
    end

    # Overlaps between the intermediately orthogonalised correction
    # vectors
    Smat = work3(cache, 1:nnew, 1:nnew)

    BLAS.gemm!('C', 'N', one, bnew, bnew, zero, Smat)

    # Inverse square root of the overlap matrix
    # Note that invsqrt_matrix will overwrite the Smat matrix
    Sinvsq = work4(cache, 1:nnew, 1:nnew)
    invsqrt_matrix!(Sinvsq, Smat, cache)

    # Symmetric orthogonalisation of the intermediately orthogonalised
    # correction vectors amongst themselves
    ortho_bnew = work2(cache, 1:matdim, 1:nnew)
    BLAS.gemm!('N', 'N', one, bnew, Sinvsq, zero, ortho_bnew)
    copy!(bnew, ortho_bnew)

end

function invsqrt_matrix!(Ainvsq::AbstractMatrix{T},
                         A::AbstractMatrix{T}, cache::Cache
                         ) where T<:AllowedFloat

    #
    # N.B. this will overwrite the contents of the input matrix A
    #

    # Dimension of the input matrix
    dim = size(A)[1]

    # Eigenpairs of the input matrix
    jobz = "V"
    uplo = "L"
    n = dim
    lda = dim
    w = work1(cache, 1:dim)
    lwork = cache.lwork
    work = evwork(cache, 1:lwork)
    info = cache.info
    syev!(jobz, uplo, n, A, lda, w, work, lwork, info)

    # Inverse square root of the input matrix
    fill!(Ainvsq, 0.0)
    for k in 1:dim
        @inbounds λinvsq = 1.0 / sqrt(abs(w[k]))
        for j in 1:dim
            for i in 1:dim
                @inbounds Ainvsq[j,i] += A[i,k] * A[j,k] * λinvsq
            end
        end
    end
    
end

function invsqrt_matrix!(Ainvsq::AbstractMatrix{T},
                         A::AbstractMatrix{T}, cache::Cache
                         ) where T<:AllowedComplex

    #
    # N.B. this will overwrite the contents of the input matrix A
    #

    # Dimension of the input matrix
    dim = size(A)[1]

    # Eigenpairs of the input matrix
    jobz = "V"
    uplo = "L"
    n = dim
    lda = dim
    w = work1(cache, 1:dim)
    lwork = cache.lwork
    work = evwork(cache, 1:lwork)
    rwork = revwork(cache, 1:lwork)
    info = cache.info
    heev!(jobz, uplo, n, A, lda, w, work, lwork, rwork, info)

    # Inverse square root of the input matrix
    fill!(Ainvsq, 0.0)
    for k in 1:dim
        @inbounds λinvsq = 1.0 / sqrt(abs(w[k]))
        for j in 1:dim
            for i in 1:dim
                @inbounds Ainvsq[j,i] += conj(A[i,k]) * A[j,k] * λinvsq
            end
        end
    end

end

function subspace_collapse(cache::Cache)

    #
    # Compute the Ritz vectors. We will use the sigvec array as a
    # working array here to store the Ritz vectors
    #

    @unpack matdim, maxvec, blocksize = cache
    
    # Initialisation
    currdim = cache.currdim
    σ = sigvec(cache, 1:matdim, 1:maxvec)
    fill!(σ, 0.0)    
    α1 = alpha(cache, 1:currdim, 1:currdim)
    
    # Compute the Ritz vectors
    @unpack one, zero = cache

    b = bvec(cache, 1:matdim, 1:currdim)
    
    α = view(α1, 1:currdim, 1:blocksize)

    ritz = sigvec(cache, 1:matdim, 1:blocksize)
    
    BLAS.gemm!('N', 'N', one, b, α, zero, ritz)
    
    #
    # Collapse the subspace to be spanned by the lowest-lying Ritz vectors
    #

    # New subspace dimension
    cache.currdim = cache.blocksize
    cache.nnew = cache.blocksize

    # Save the Ritz vectors as the new subspace vectors
    b = bvec(cache, 1:matdim, 1:blocksize)
    ritz = sigvec(cache, 1:matdim, 1:blocksize)

    copy!(b, ritz)

end

function eigenvectors!(vectors::AbstractMatrix{T},
                       cache::Cache) where T<:AllowedTypes
    
    # Compute the Ritz vectors for the nroots lowest roots
    @unpack matdim, currdim, nroots, zero, one = cache

    α = alpha(cache, 1:currdim, 1:nroots)
    
    b = bvec(cache, 1:matdim, 1:currdim)

    v = view(vectors, 1:matdim, 1:nroots)
    
    BLAS.gemm!('N', 'N', one, b, α, zero, v)
    
end


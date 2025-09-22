function sil!(v::AbstractVector{T},
              Δt::Float64,
              ϵ::Float64,
              f::Function,
              maxvec::Int64;
              kwargs...) where {T<:AllowedTypes}

    # Real type
    T <: Allowed64 ? R = Float64 : R = Float32
    
    # Matrix dimension
    matdim = length(v)

    # Work arrays
    Twork, Rwork = sil_workarrays(T, matdim, maxvec)

    nproducts = sil!(v, Δt, ϵ, f, maxvec, Twork, Rwork; kwargs...)

    return nproducts
    
end
    
function sil!(v::AbstractVector{T},
              Δt1::Float64,
              ϵ::Float64,
              f::Function,
              maxvec::Int64,
              Twork::Vector{T},
              Rwork::Vector{R};
              kwargs...) where {T<:AllowedTypes, R<:AllowedFloat}

    # Absolute vale of the time step
    Δt = abs(Δt1)
    Δt1 < 0.0 ? backwards = true : backwards = false
    
    # Matrix dimension
    matdim = length(v)

    # SIL cache
    cache = SILCache{T, R}(f, matdim, maxvec, ϵ, Twork, Rwork)

    # Total number of matrix-vector products taken
    nproducts = 0
    
    # Perform SIL steps with an adaptive step size until
    # we reach a total time step of Δt
    t = 0.0
    while t != Δt

        # Target time step
        δ = Δt - t
    
        # Perform the Lanczos iterations
        converged = lanczos_iterations(cache, v, δ; kwargs...)

        # Update the total number of matrix-vector products taken
        nproducts += cache.Kdim
        
        # Compute F(A) * v
        δactual = Fv!(cache, v, converged, δ, backwards)

        # Update the total time step
        t += δactual
        
    end

    return nproducts
    
end

function lanczos_iterations(cache::SILCache,
                            v::AbstractVector{T},
                            δ::Float64;
                            kwargs...) where {T<:AllowedTypes}

    # Lanczos iterations:
    # r := H qⱼ - βⱼ₋₁ qⱼ₋₁
    # αⱼ := r† qⱼ
    # r := r - αⱼ qⱼ
    # βⱼ := ||r||
    # qⱼ₊₁ := r / βⱼ

    @unpack matdim, maxvec, f, ϵ = cache
    
    # Lanczos vectors
    q = qvec(cache, 1:matdim, 1:maxvec)

    # r-vector
    r = rvec(cache, 1:matdim)
    
    # On-diagonal subspace matrix elements
    α = alpha(cache, 1:maxvec)

    # Off-diagonal subspace matrix elements
    β = beta(cache, 1:maxvec)
    
    # Iteration 1
    @views copy!(q[:,1], v)

    if haskey(kwargs, :data)
        @views cache.f(q[:,1], r, kwargs[:data])
    else
        @views f(q[:,1], r)
    end

    @views α[1] = dot(q[:,1], r)
    for i in 1:matdim
        @inbounds r[i] -= α[1] * q[i,1]
    end
    β[1] = sqrt(dot(r, r))
    
    # Iterations 1,2,...
    Kdim = 0
    converged = false
    for j in 2:maxvec

        # Dimension of the Krylov subspace
        Kdim = copy(j)

        # Next Lanczos vector
        for i in 1:matdim
            @inbounds q[i,j] = r[i] / β[j-1]
        end

        # Lanczos recursion
        if haskey(kwargs, :data)
            @views cache.f(q[:,j], r, kwargs[:data])
        else
            @views f(q[:,j], r)
        end
        
        for i in 1:matdim
            @inbounds r[i] -= β[j-1] * q[i,j-1]
        end

        @views α[j] = dot(r, q[:,j])

        for i in 1:matdim
            @inbounds r[i] -= α[j] * q[i,j]
        end
        
        β[j] = sqrt(dot(r, r))

        # Error estimate
        @views error = abs(prod(β[1:j]) / factorial(j+2) * δ^(j+2))
        if error < ϵ
            converged = true
        end

        # Exit if we have converged
        if converged
            break
        end
        
    end

    # Save the subspace dimension
    cache.Kdim = copy(Kdim)

    return converged
    
end

function Fv!(cache::SILCache, v::AbstractVector{T},
             converged::Bool, δ::Float64,
             backwards::Bool) where {T<:AllowedTypes}

    # Lanzcos subspace matrix:
    # Gᵢⱼ = qᵢ† A qⱼ
    # Gᵢᵢ = αᵢ
    # Gᵢⱼ = βᵢ, |i-j| = 1
    # Gᵢⱼ = 0, |i-j| > 1

    @unpack matdim, maxvec, ϵ, Kdim = cache

    # On- and off-diagonal matrix elements of the
    # subspace matrix
    α = alpha(cache, 1:Kdim)
    β = beta(cache, 1:Kdim)
    
    # Determine the time step
    if converged
        δactual = copy(δ)
    else
        @views δactual = abs((ϵ * factorial(Kdim+2) /
            prod(β[1:Kdim])) ^ (1.0/(Kdim+2)))
    end

    # Subspace matrix
    G = Gmat(cache, 1:Kdim, 1:Kdim)
    fill!(G, 0.0)
    for i in 1:Kdim
        @inbounds G[i,i] = α[i]
    end
    for i in 1:Kdim-1
        @inbounds G[i,i+1] = conj(β[i])
        @inbounds G[i+1,i] = β[i]
    end

    # Diagonalise the subspace matrix
    subspace_diag(cache)

    # Compute the Lanczos representation of F(A)
    # Note that only the first column of this matrix is
    # needed as the Lanczos representation of F(A)*v is the column
    # vector (1,0,…,0)ᵀ: this is also the Lanczos representation
    # of F(A)*v
    C = Fvcoeff(cache, 1:Kdim)
    fill!(C, 0.0)

    U = eigvec(cache, 1:Kdim, 1:Kdim)
    λ = eigval(cache, 1:Kdim)

    if backwards
        a = complex(0.0, -1.0)
    else
        a = complex(0.0, 1.0)
    end
    
    for i ∈ 1:Kdim
        for j ∈ 1:Kdim
            C[i] += U[i,j] * conj(U[1,j]) *
                exp(-a * λ[j] * δactual)
        end
    end

    # Compute the Lanczos approximation to F(A)*v
    q = qvec(cache, 1:matdim, 1:Kdim)
    mul!(v, q, C)

    return δactual
    
end

function subspace_diag(cache::SILCache{T}) where T<:AllowedFloat

    @unpack Kdim, lwork, info = cache
    
    jobz = "V"
    uplo = "L"
    n = Kdim
    U = eigvec(cache, 1:Kdim, 1:Kdim)
    lda = Kdim
    λ = eigval(cache, 1:Kdim)
    work = evwork(cache, 1:lwork)

    G = Gmat(cache, 1:Kdim, 1:Kdim)

    copy!(U, G)

    # Call to ?syev
    syev!(jobz, uplo, n, U, lda, λ, work, lwork, info)

end

function subspace_diag(cache::SILCache{T}) where T<:AllowedComplex

    @unpack Kdim, lwork, info = cache

    jobz = "V"
    uplo = "L"
    n = Kdim
    U = eigvec(cache, 1:Kdim, 1:Kdim)
    lda = Kdim
    λ = eigval(cache, 1:Kdim)
    work = evwork(cache, 1:lwork)
    rwork = revwork(cache, 1:lwork)
   
    G = Gmat(cache, 1:Kdim, 1:Kdim)

    copy!(U, G)

    # Call to ?heev
    heev!(jobz, uplo, n, U, lda, λ, work, lwork, rwork, info)
    
end

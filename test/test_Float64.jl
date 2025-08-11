@testset "Float64" begin

    T = Float64
    R = Float64
    
    zero::T = 0.0
    one::T = 1.0
    
    # Matrix-vector multiplication function
    function sigma!(v::AbstractMatrix{T},
                    σ::AbstractMatrix{T})
        
        BLAS.gemm!('N', 'N', one, A, v, zero, σ)
        
    end
    
    # Dimensions
    matdim = 100
    nroots = 4
    
    # Random, sparse, symmetric matrix A
    A = Matrix{T}(undef, matdim, matdim)
    for i in 1:matdim
        A[i,i] = rand()
    end
    for i in 1:matdim-1
        for j in i+1:matdim
            A[j,i] = rand() * 0.01
            A[i,j] = A[j,i]
        end
    end
    
    # Diagonal of the matrix A
    diagA = diag(A)
    
    # Residual norm convergence threshold
    ϵ = 1e-4
    
    # Davidson eigensolver
    result = solver(sigma!, diagA, nroots, matdim; tol=ϵ)
    
    # LinaerAlgebra eigen function
    F = eigen(A)
    
    # Difference relative to the full diagonalisation results
    Δ = abs.(F.values[1:nroots] - result.values)

    @test any(i -> i > ϵ^2 * 100, Δ) == false

end
    

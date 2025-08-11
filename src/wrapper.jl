import LinearAlgebra.BLAS.@blasfunc

function syev!(jobz::String, uplo::String, n::Int64,
               a::AbstractMatrix{Float64}, lda::Int64,
               w::AbstractVector{Float64}, work::AbstractVector{Float64},
               lwork::Int64, info::Int64)

    ccall((@blasfunc(dsyev_), LinearAlgebra.libblastrampoline),
          Cvoid,
          (Cstring, Cstring, Ref{Int64}, Ptr{Float64}, Ref{Int64},
           Ptr{Float64}, Ptr{Float64}, Ref{Int64}, Ref{Int64}),
          jobz, uplo, n, a, lda, w, work, lwork, info)

end

function syev!(jobz::String, uplo::String, n::Int64,
               a::AbstractMatrix{Float32}, lda::Int64,
               w::AbstractVector{Float32}, work::AbstractVector{Float32},
               lwork::Int64, info::Int64)

    ccall((@blasfunc(ssyev_), LinearAlgebra.libblastrampoline),
          Cvoid,
          (Cstring, Cstring, Ref{Int64}, Ptr{Float64}, Ref{Int64},
           Ptr{Float64}, Ptr{Float64}, Ref{Int64}, Ref{Int64}),
          jobz, uplo, n, a, lda, w, work, lwork, info)
    
end

function heev!(jobz::String, uplo::String, n::Int64,
               a::AbstractMatrix{ComplexF64}, lda::Int64,
               w::AbstractVector{Float64}, work::AbstractVector{ComplexF64},
               lwork::Int64, rwork::AbstractVector{Float64}, info::Int64)

    ccall((@blasfunc(zheev_), LinearAlgebra.libblastrampoline),
          Cvoid,
          (Cstring, Cstring, Ref{Int64}, Ptr{ComplexF64}, Ref{Int64},
           Ptr{Float64}, Ptr{ComplexF64}, Ref{Int64}, Ptr{Float64},
           Ref{Int64}),
          jobz, uplo, n, a, lda, w, work, lwork, rwork, info)
    
end

function heev!(jobz::String, uplo::String, n::Int64,
               a::AbstractMatrix{ComplexF32}, lda::Int64,
               w::AbstractVector{Float32}, work::AbstractVector{ComplexF32},
               lwork::Int64, rwork::AbstractVector{Float32}, info::Int64)

    ccall((@blasfunc(cheev_), LinearAlgebra.libblastrampoline),
          Cvoid,
          (Cstring, Cstring, Ref{Int64}, Ptr{ComplexF32}, Ref{Int64},
           Ptr{Float32}, Ptr{ComplexF32}, Ref{Int64}, Ptr{Float32},
           Ref{Int64}),
          jobz, uplo, n, a, lda, w, work, lwork, rwork, info)
    
end

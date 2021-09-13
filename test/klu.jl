using SuiteSparse.KLU: klu, KLUITypes
using SuiteSparse
using SuiteSparse: increment!, decrement
using SparseArrays: SparseMatrixCSC, sparse, nnz
using LinearAlgebra
@testset "KLU Wrappers" begin
    Ap = increment!([0,4,1,1,2,2,0,1,2,3,4,4])
    Ai = increment!([0,4,0,2,1,2,1,4,3,2,1,2])
    Ax = [2.,1.,3.,4.,-1.,-3.,3.,6.,2.,1.,4.,2.]
    A0 = sparse(Ap, Ai, Ax)
    @testset "Core functionality for $Tv elements" for Tv in (Float64, ComplexF64)
        @testset "Core functionality for $Ti indices" for Ti ∈ Base.uniontypes(KLUITypes)
            A = convert(SparseMatrixCSC{Tv, Ti}, A0)
            klua = klu(size(A, 1), decrement(A.colptr), decrement(A.rowval), A.nzval)
            @test nnz(klua) == 18
            # This fails with one value wrong. No idea why
            R = Diagonal(Tv == ComplexF64 ? complex.(klua.Rs) : klua.Rs)
            @test R \ A[klua.P, klua.Q] ≈ (klua.L * klua.U + klua.F)

            b = [8., 45., -3., 3., 19.]
            x = klua \ b
            @test x ≈ float([1:5;])
            @test A*x ≈ b

            z = complex.(b)
            x = ldiv!(klua, z)
            @test x ≈ float([1:5;])
            @test z === x
            # Can't match UMFPACK's ldiv!(<OUTPUT>, <KLU>, <INPUT>)
            # since klu_solve(<KLU>, <INPUT>) modifies <INPUT>, and has no field for <OUTPUT>.
            @test A*x ≈ b

            b = [8., 20., 13., 6., 17.]
            x = klua'\b
            @test x ≈ float([1:5;])

            @test A'*x ≈ b
            z = complex.(b)
            x = ldiv!(adjoint(klua), z)
            @test x ≈ float([1:5;])
            @test x === z

            @test A'*x ≈ b
            x = transpose(klua) \ b
            @test x ≈ float([1:5;])
            @test transpose(A) * x ≈ b

            x = ldiv!(transpose(klua), complex.(b))
            @test x ≈ float([1:5;])
            @test transpose(A) * x ≈ b

            @inferred klua\fill(1, size(A, 2))
        end
    end
end

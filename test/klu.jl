using SuiteSparse.KLU: klu, klu!, KLUITypes, KLUFactorization, klu_analyze!, klu_factor!
using SuiteSparse
using SuiteSparse: increment!, decrement
using SparseArrays: SparseMatrixCSC, sparse, nnz
using LinearAlgebra
@testset "KLU Wrappers" begin
    Ap = increment!([0,4,1,1,2,2,0,1,2,3,4,4])
    Ai = increment!([0,4,0,2,1,2,1,4,3,2,1,2])
    Ax = [2.,1.,3.,4.,-1.,-3.,3.,6.,2.,1.,4.,2.]
    A0 = sparse(Ap, Ai, Ax)
    A1 = sparse(increment!([0,4,1,1,2,2,0,1,2,3,4,4]),
                increment!([0,4,0,2,1,2,1,4,3,2,1,2]),
                [2.,1.,3.,4.,-1.,-3.,3.,9.,2.,1.,4.,2.], 5, 5)
    @testset "Core functionality for $Tv elements" for Tv in (Float64, ComplexF64)
        @testset "Core functionality for $Ti indices" for Ti ∈ Base.uniontypes(KLUITypes)
            A = convert(SparseMatrixCSC{Tv, Ti}, A0)
            klua = klu(size(A, 1), decrement(A.colptr), decrement(A.rowval), A.nzval)
            @test klua.F == klu(A).F
            @test nnz(klua) == 18
            # This fails with one value wrong. No idea why
            R = Diagonal(Tv == ComplexF64 ? complex.(klua.Rs) : klua.Rs)
            @test R \ A[klua.p, klua.q] ≈ (klua.L * klua.U + klua.F)

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

            @testset "Utility functions" begin
                K = KLUFactorization(A);
                @test size(K) == (5, 5)
                @test_throws ArgumentError K.symbolic
                @test_throws ArgumentError K.numeric
                klu_analyze!(K);
                @test K.symbolic.nz == 12
                @test K.symbolic.nzoff == 4
                klu_factor!(K)
                @test K.numeric.lnz == 7 == K.numeric.unz
            end
            @testset "Refactorization" begin
                B = convert(SparseMatrixCSC{Tv, Ti}, A1)
                b = Tv[8., 45., -3., 3., 19.]
                F = klu(A)
                klu!(F, B)
                @test F\b ≈ B\b ≈ Matrix(B)\b
            end
            @testset "Singular matrix" begin
                S = sparse([1 2; 0 0])
                S = convert(SparseMatrixCSC{Tv, Ti}, S)
                @test_throws SingularException klu(S)

            end
        end
    end
end

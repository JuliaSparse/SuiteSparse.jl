module KLU

using SparseArrays
import Base: (\), size, getproperty, setproperty!
import ..increment, ..increment!, ..decrement, ..decrement!
import ..LibSuiteSparse:
    klu_l_common,
    klu_l_defaults,
    klu_l_symbolic,
    klu_l_free_symbolic,
    klu_l_numeric,
    klu_l_free_numeric,
    klu_zl_free_numeric,
    KLU_OK,
    KLU_SINGULAR,
    KLU_OUT_OF_MEMORY,
    KLU_INVALID,
    KLU_TOO_LARGE,
    klu_l_analyze,
    klu_l_factor,
    klu_zl_factor,
    klu_l_solve,
    klu_zl_solve,
    klu_l_tsolve,
    klu_zl_tsolve,
    klu_l_extract,
    klu_zl_extract,
    klu_l_sort,
    klu_zl_sort
using LinearAlgebra
import LinearAlgebra: Factorization
const KLUTypes = Union{Float64, ComplexF64}

function kluerror(status::Integer)
    if status == KLU_OK
        return
    elseif status == KLU_SINGULAR
        throw(LinearAlgebra.SingularException(0))
    elseif status == KLU_OUT_OF_MEMORY
        throw(OutOfMemoryError())
    elseif status == KLU_INVALID
        throw(ArgumentError("Invalid Status"))
    elseif status == KLU_TOO_LARGE
        throw(OverflowError("Integer overflow has occured"))
    else
        throw(ErrorException("Unknown KLU error code: $status"))
    end
end

kluerror(common::klu_l_common) = kluerror(common.status)

macro isok(A)
    :(kluerror($(esc(A))))
end

function _common()
    common = klu_l_common()
    ok = klu_l_defaults(Ref(common))
    if ok == 1
        return common
    else
        throw(ErrorException("Could not initialize klu_common struct."))
    end
    print(common)
end

mutable struct KLUFactorization{T<:KLUTypes} <: Factorization{T}
    common::klu_l_common
    symbolic::Ptr{klu_l_symbolic}
    numeric::Ptr{klu_l_numeric}
    n::Int
    colptr::Vector{Int64}
    rowval::Vector{Int64}
    nzval::Vector{T}
    function KLUFactorization(n, colptr, rowval, nzval, common = _common())
        T = eltype(nzval)
        obj = new{T}(common, C_NULL, C_NULL, n, colptr, rowval, nzval)
        # This finalizer may fail if common is C_NULL.
        function f(klu)
            klu_l_free_symbolic(Ref(klu.symbolic), Ref(klu.common))
            if T <: AbstractFloat
            klu_l_free_numeric(Ref(klu.numeric), Ref(klu.common))
            elseif T <: Complex
                klu_zl_free_numeric(Ref(klu.numeric), Ref(klu.common))
            end
        end
        return finalizer(f, obj)
    end
end

function KLUFactorization(A::SparseMatrixCSC{T, Int64}) where {T<:KLUTypes}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("KLU only accepts square matrices."))
    return KLUFactorization(n, decrement(A.colptr), decrement(A.rowval), A.nzval)
end

size(K::KLUFactorization) = (K.n, K.n)
function size(K::KLUFactorization, dim::Integer)
    if dim < 1
        throw(ArgumentError("size: dimension $dim out of range"))
    elseif dim == 1 || dim == 2
        return Int(K.n)
    else
        return 1
    end
end

Base.adjoint(F::KLUFactorization) = Adjoint(F)
Base.transpose(F::KLUFactorization) = Transpose(F)

function setproperty!(klu::KLUFactorization, ::Val{:symbolic}, x)
    current = getfield(klu, :symbolic)
    if current != C_NULL
        klu_l_free_symbolic(Ref(current), Ref(getfield(klu, :common)))
    end
    setfield!(klu, :symbolic, x)
end

function setproperty!(klu::KLUFactorization{T}, ::Val{:numeric}, x) where {T}
    current = getfield(klu, :numeric)
    if T == Float64
        klu_l_free_numeric(Ref(current), Ref(getfield(klu, :common)))
    else
        klu_zl_free_numeric(Ref(current), Ref(getfield(klu, :common)))
    end
    setfield!(klu, :numeric, x)
end

# Certain sets of inputs must be non-null *together*:
# [Lp, Li, Lx], [Up, Ui, Ux], [Fp, Fi, Fx]
function _extract!(
    klu::KLUFactorization{T};
    Lp = C_NULL, Li = C_NULL, Up = C_NULL, Ui = C_NULL, Fp = C_NULL, Fi = C_NULL,
    P = C_NULL, Q = C_NULL, R = C_NULL, Lx = C_NULL, Lz = C_NULL, Ux = C_NULL, Uz = C_NULL,
    Fx = C_NULL, Fz = C_NULL, Rs = C_NULL
    ) where {T<:KLUTypes}
    if T == Float64
        klu_l_sort(klu.symbolic, klu.numeric, Ref(klu.common))
        ok = klu_l_extract(
            klu.numeric, klu.symbolic,
            Lp, Li, Lx, Up, Ui, Ux, Fp, Fi, Fx, P, Q, Rs, R,
            Ref(klu.common)
            )
    elseif T == ComplexF64
        klu_zl_sort(klu.symbolic, klu.numeric, Ref(klu.common))
        ok = klu_zl_extract(
            klu.numeric, klu.symbolic,
            Lp, Li, Lx, Lz, Up, Ui, Ux, Uz, Fp, Fi, Fx, Fz, P, Q, Rs, R,
            Ref(klu.common)
        )
    end
    if ok == 1
        return nothing
    else
        kluerror(klu.common)
    end
    return nothing
end

function getproperty(klu::KLUFactorization{T}, s::Symbol) where {T<:KLUTypes}
    # Forwards to the numeric struct:
    if s ∈ [:lnz, :unz, :nzoff]
        klu.numeric != C_NULL || throw(ArgumentError("This KLUFactorization has no available numeric object."))
        numeric = unsafe_load(klu.numeric)
        return getproperty(numeric, s)
    end
    if s ∈ [:nblocks, :maxblock]
        klu.symbolic != C_NULL || throw(ArgumentError("This KLUFactorization has no available symbolic object."))
        symbolic = unsafe_load(klu.symbolic)
        return getproperty(symbolic, s)
    end
    # Non-overloaded parts:
    if s ∉ [:L, :U, :F, :P, :Q, :R, :Rs, :(_L), :(_U), :(_F)]
        return getfield(klu, s)
    end
    # Factor parts:
    if s === :(_L)
        lnz = klu.lnz
        Lp = Vector{Int64}(undef, klu.n + 1)
        Li = Vector{Int64}(undef, lnz)
        Lx = Vector{Float64}(undef, lnz)
        Lz = T == Float64 ? C_NULL : Vector{Float64}(undef, lnz)
        _extract!(klu; Lp, Li, Lx, Lz)
        return Lp, Li, Lx, Lz
    elseif s === :(_U)
        unz = klu.unz
        Up = Vector{Int64}(undef, klu.n + 1)
        Ui = Vector{Int64}(undef, unz)
        Ux = Vector{Float64}(undef, unz)
        Uz = T == Float64 ? C_NULL : Vector{Float64}(undef, unz)
        _extract!(klu; Up, Ui, Ux, Uz)
        return Up, Ui, Ux, Uz
    elseif s === :(_F)
        fnz = klu.nzoff
        Fp = Vector{Int64}(undef, klu.n + 1)
        Fi = Vector{Int64}(undef, fnz)
        Fx = Vector{Float64}(undef, fnz)
        Fz = T == Float64 ? C_NULL : Vector{Float64}(undef, fnz)
        _extract!(klu; Fp, Fi, Fx, Fz)
        return Fp, Fi, Fx, Fz
    end
    if s ∈ [:Q, :P, :R, :Rs]
        if s === :Rs
            out = Vector{Float64}(undef, klu.n)
        elseif s === :R
            out = Vector{Int64}(undef, klu.nblocks + 1)
        else
            out = Vector{Int64}(undef, klu.n)
        end
        # This tuple construction feels hacky, there's a better way I'm sure.
        _extract!(klu; NamedTuple{(s,)}((out,))...)
        if s ∈ [:Q, :P, :R]
            increment!(out)
        end
        return out
    end
    if s ∈ [:L, :U, :F]
        if s === :L
            p, i, x, z = klu._L
        elseif s === :U
            p, i, x, z = klu._U
        elseif s === :F
            p, i, x, z = klu._F
        end
        if T == Float64
            return SparseMatrixCSC(klu.n, klu.n, increment!(p), increment!(i), x)
        else
            return SparseMatrixCSC(klu.n, klu.n, increment!(p), increment!(i), Complex.(x, z))
        end
    end
end

function klu_analyze!(K::KLUFactorization)
    if K.symbolic != C_NULL return K end
    sym = klu_l_analyze(K.n, K.colptr, K.rowval, Ref(K.common))
    println(sym)
    if sym == C_NULL
        kluerror(K.common)
    else
        K.symbolic = sym
    end
    return K
end

function klu_factor!(K::KLUFactorization{T}) where {T}
    K.symbolic == C_NULL  && klu_analyze!(K)
    if T == Float64
        num = klu_l_factor(K.colptr, K.rowval, K.nzval, K.symbolic, Ref(K.common))
    elseif T == ComplexF64
        num = klu_zl_factor(K.colptr, K.rowval, K.nzval, K.symbolic, Ref(K.common))
    else
        throw(ArgumentError("$T is not a supported eltype."))
    end
    if num == C_NULL
        kluerror(K.common)
    else
        K.numeric = num
    end
    return K
end



function klu(n, colptr::Vector{Int64}, rowval::Vector{Int64}, nzval::Vector{T}) where {T<:AbstractFloat}
    if nzval != Float64
        nzval = convert(Vector{Float64}, nzval)
    end
    K = KLUFactorization(n, colptr, rowval, nzval)
    return klu_factor!(K)
end

function klu(n, colptr::Vector{Int64}, rowval::Vector{Int64}, nzval::Vector{T}) where {T<:Complex}
    if nzval != ComplexF64
        nzval = convert(Vector{ComplexF64}, nzval)
    end
    K = KLUFactorization(n, colptr, rowval, nzval)
    return klu_factor!(K)
end

function klu(A::SparseMatrixCSC{T, Int64}) where {T<:KLUTypes}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch())
    return klu(n, decrement(A.colptr), decrement(A.rowval), A.nzval)
end

#B is the modified argument here. To match with the math it should be (num, B). But convention is
# modified first. Thoughts?
function solve!(klu::KLUFactorization{T}, B::StridedVecOrMat{T}) where {T<:KLUTypes}
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    klu.numeric == C_NULL && klu_factor!(klu)
    size(B, 1) == size(klu, 1) || throw(DimensionMismatch())
    T == Float64 ? (f = klu_l_solve) : (f = klu_zl_solve)
    isok = f(klu.symbolic, klu.numeric, size(B, 1), size(B, 2), B, Ref(klu.common))
    isok == 0 && kluerror(klu.common)
    return B
end

function solve!(klu::LinearAlgebra.AdjOrTrans{T, KLUFactorization{T}}, B::StridedVecOrMat{T}) where {T}
    if klu isa Adjoint
        conj = 1
    elseif klu isa Transpose
        conj = 0
    end
    klu = parent(klu)
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    klu.numeric == C_NULL && klu_factor!(klu)
    size(B, 1) == size(klu, 1) || throw(DimensionMismatch())
    if T == Float64
        # Float solve doesn't take a conjugate argument
        isok = klu_l_tsolve(klu.symbolic, klu.numeric, size(B, 1), size(B, 2), B, Ref(klu.common))
    else
        # conj determines whether to use conjugate transpose (1) or transpose (0).
        isok = klu_zl_tsolve(klu.symbolic, klu.numeric, size(B, 1), size(B, 2), B, conj, Ref(klu.common))
    end
    isok == 0 && kluerror(klu.common)
    return B
end


function solve(klu, B)
    X = copy(B)
    return solve!(klu, X)
end

end

module KLU

using SparseArrays
import ..increment, ..increment!, ..decrement, ..decrement!
import ..LibSuiteSparse:
    klu_l_common_struct,
    klu_l_defaults,
    klu_l_symbolic,
    klu_l_free_symbolic,
    klu_l_numeric,
    klu_l_free_numeric,
    KLU_OK,
    KLU_SINGULAR,
    KLU_OUT_OF_MEMORY,
    KLU_INVALID,
    KLU_TOO_LARGE,
    klu_l_analyze,
    klu_l_factor,
    klu_zl_factor,
    klu_l_solve
using LinearAlgebra
import LinearAlgebra: Factorization
const klu_common = klu_l_common_struct()
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

macro isok(A)
    :(kluerror($(esc(A))))
end

mutable struct Symbolic
    ptr::Ptr{klu_l_symbolic}
    function Symbolic(ptr::Ptr{klu_l_symbolic})
        if ptr == C_NULL
            throw(ArgumentError("Symbolic construction failed for " *
            "unknown reasons. Please submit a bug report."))
        end
        obj = new(ptr)
        return finalizer((x) -> klu_l_free_symbolic(Ref(x.ptr), Ref(klu_common)), obj)
    end
end
Base.unsafe_convert(::Type{Ptr{klu_l_symbolic}}, s::Symbolic) = s.ptr
# I'd like to leave this unrestricted, so non-SparseMatrixCSC types can integrate easily.
# In particular C arrays. Should perhaps restrict to Ptrs, but this is already checked in the ccall.
function Symbolic(n::Integer, Ap, Ai, common::klu_l_common_struct = klu_common)
    return Symbolic(klu_l_analyze(n, Ap, Ai, Ref(common)))
end

function Symbolic(A::SparseMatrixCSC{Float64, Int64}, common::klu_l_common_struct = klu_common)
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("KLU only accepts square matrices."))
    return Symbolic(n, decrement(A.colptr), decrement(A.rowval), common)
end

mutable struct Numeric{T<:KLUTypes} <: Factorization{T}
    ptr::Ptr{klu_l_numeric}
    function Numeric{T}(ptr::Ptr{klu_l_numeric}) where {T}
        if ptr == C_NULL
            throw(ArgumentError("Numeric construction failed for " *
            "unknown reasons. Please submit a bug report."))
        end
        obj = new(ptr)
        return finalizer((x) -> klu_l_free_numeric(Ref(x.ptr), Ref(klu_common)), obj)
    end
end
Base.unsafe_convert(::Type{Ptr{klu_l_numeric}}, n::Numeric) = n.ptr

function Numeric(
    Ap, Ai, Ax, T,
    symbolic::Symbolic, common::klu_l_common_struct = klu_common
)
    println(Ap)
    println(Ai)
    println(Ax)
    println(T)
    if T == Float64
        n = klu_l_factor(Ap, Ai, Ax, symbolic, Ref(common))
    elseif T == ComplexF64
        n = klu_zl_factor(Ap, Ai, Ax, symbolic, Ref(common))
    end
    return Numeric{T}(n)
end

function Numeric(
    A::SparseMatrixCSC{T, Int64},
    symbolic::Symbolic, common::klu_l_common_struct = klu_common
) where {T<:KLUTypes}
    return Numeric(decrement(A.colptr), decrement(A.rowval), A.nzval, T, symbolic, common)
end

#B is the modified argument here. To match with the math it should be (num, B). But convention is
# modified first. Thoughts?
function solve!(
    sym::Symbolic, num::Numeric{T}, B::StridedVecOrMat{T},
    common::klu_l_common_struct = klu_common
) where {T<:KLUTypes}
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    if T == Float64
        klu_l_solve(sym, num, size(B, 1), size(B, 2), B, Ref(common))
    elseif T == ComplexF64
        throw(ArgumentError("ComplexF64 not yet implmented."))
    end
    return B
end

function solve(
    sym::Symbolic, num::Numeric{T}, B::StridedVecOrMat{T},
    common::klu_l_common_struct = klu_common
) where {T<:KLUTypes}
    return solve!(sym, num, copy(B), common)
end

function solve(
    A::SparseMatrixCSC{Float64, Int64}, B::StridedVecOrMat{Float64},
    common::klu_l_common_struct = klu_common
)
    s = Symbolic(A, common)
    n = Numeric(A, s, common)
    return solve(s, n, B)
end

function __init__()
    klu_l_defaults(Ref(klu_common))
end


end

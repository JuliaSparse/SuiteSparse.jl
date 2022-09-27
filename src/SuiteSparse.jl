# This file is a part of Julia. License is MIT: https://julialang.org/license

module SuiteSparse

using SparseArrays
const increment = SparseArrays.increment
const increment! = SparseArrays.increment!
const decrement = SparseArrays.decrement
const decrement! = SparseArrays.decrement!

const LibSuiteSparse = SparseArrays.LibSuiteSparse

if Base.USE_GPL_LIBS
    const CHOLMOD = SparseArrays.CHOLMOD
    const SPQR = SparseArrays.SPQR
    const UMFPACK = SparseArrays.UMFPACK
end

end # module SuiteSparse

## Low-rank Certification for Sums-of-Squares (on Projective Varieties)
module LowRankSOS

# import modules for math operations and computations
import LinearAlgebra, SparseArrays

# define the option of enabling dense methods
const DENSE_METHODS = true

# define common constants
const NUM_DIG = 8
const VAL_TOL = 1.0e-8
const VAL_PEN = 1.0
const NUM_MAX_ITER = 1000
const NUM_MEM_SIZE = 20

# include the source files of basic types and methods
include("types.jl")
include("methods.jl")


# include optional methods based on dense arrays and matrices
if DENSE_METHODS
    include("dense_methods/basics.jl")
    include("dense_methods/solver_based.jl")
    include("dense_methods/direct_search.jl")
    include("dense_methods/fiber_escape.jl")
    include("dense_methods/trap_bypass.jl")
end


end

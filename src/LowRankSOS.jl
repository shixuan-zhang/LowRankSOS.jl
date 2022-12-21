## Low-rank Certification for Sums-of-Squares (on Projective Varieties)
module LowRankSOS

# import modules for math operations and computations
import LinearAlgebra, SparseArrays

# define common constants
const NUM_DIG = 8
const VAL_TOL = 1.0e-8
const VAL_PEN = 1.0
const NUM_MAX_ITER = 1000

# include the source files of basic types and methods
include("types.jl")
include("methods.jl")
# include different algorithms for solving the SOS certification
include("solver_based.jl")
include("direct_search.jl")
include("fiber_escape.jl")
include("trap_bypass.jl")


end

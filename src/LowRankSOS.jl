## Low-rank Certification for Sums-of-Squares (on Projective Varieties)
module LowRankSOS

# import modules for math operations and computations
using LinearAlgebra, Polynomials, SparseArrays
# import modules for auxiliary functions
using Formatting

# export types and methods for application programming interface
export CoordinateRing2
export idx_sym, get_sos, build_Jac_mat, build_Hess_mat
export embed_tuple, get_BP_bound
export solve_gradient_descent, solve_BFGS_descent, solve_lBFGS_descent
export solve_CG_descent, solve_CG_push_descent
export move_direct_path
export call_NLopt, call_JuMP_solver

# define common constants
const NUM_DIG = 8
const VAL_TOL = 1.0e-8
const VAL_PEN = 1.0e4
const NUM_MEM_SIZE = 20   # default for limited-memory quasi-Newton methods
const NUM_MAX_MOVE = 10   # default for direct path search methods
const NUM_MAX_ITER = 2000 # default for local (line search) descent methods

# include the source files of basic types and methods
include("types.jl")
include("methods.jl")
include("line_search.jl")
include("solver_call.jl")
include("direct_path.jl")


end

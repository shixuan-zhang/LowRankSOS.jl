# test of fiber escape method using ternary quartics

include("../example/Veronese_surface/Veronese_surface.jl")

using LinearAlgebra
using Test


function test_fiber_escape()
    # define the ideal of embedded ternary quartics
    dim = 6
    num_square = 3
    # define the quadratic ideal
    ideal_Veronese_surface = LowRankSOS.QuadraticIdeal(dim, define_Veronese_surface_ideal())
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_Veronese_surface)
    # import the starting linear forms and target quadratic forms
    include("../example/Veronese_surface/data/grad_int_stat_point.jl")
    # test the algorithms
    mat_solution = LowRankSOS.solve_gradient_method_with_escapes(num_square, mat_target, map_quotient, ideal_Veronese_surface, 
                                                                 mat_linear_forms=mat_start, 
                                                                 str_line_search="interpolation", 
                                                                 lev_print=-1)
    return LowRankSOS.compute_norm_proj(mat_solution'*mat_solution-mat_target, map_quotient) < LowRankSOS.VAL_TOL 
end

@test test_fiber_escape()

# test of fiber escape method using ternary quartics

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra
using Test


function test_fiber_escape()
    # define the ideal of embedded ternary quartics
    dim = 6
    # the entries correspond to monomials
    # x², xy, xz, y², yz, z²
    # there are 6 generators in the ideal
    mat_gen = Vector{Matrix{Float64}}(undef, 6)
    # generator 1: x²⋅y² - (xy)²
    mat_gen[1] = zeros(dim,dim)
    mat_gen[1][1,4] = 1/2
    mat_gen[1][4,1] = 1/2
    mat_gen[1][2,2] = -1
    # generator 2: x²⋅z² - (xz)²
    mat_gen[2] = zeros(dim,dim)
    mat_gen[2][1,6] = 1/2
    mat_gen[2][6,1] = 1/2
    mat_gen[2][3,3] = -1
    # generator 3: y²⋅z² - (yz)²
    mat_gen[3] = zeros(dim,dim)
    mat_gen[3][4,6] = 1/2
    mat_gen[3][6,4] = 1/2
    mat_gen[3][5,5] = -1
    # generator 4: xy⋅xz - x²⋅yz
    mat_gen[4] = zeros(dim,dim)
    mat_gen[4][2,3] = 1/2
    mat_gen[4][3,2] = 1/2
    mat_gen[4][1,5] = -1/2
    mat_gen[4][5,1] = -1/2
    # generator 5: xy⋅yz - y²⋅xz
    mat_gen[5] = zeros(dim,dim)
    mat_gen[5][2,5] = 1/2
    mat_gen[5][5,2] = 1/2
    mat_gen[5][4,3] = -1/2
    mat_gen[5][3,4] = -1/2
    # generator 6: xz⋅yz - z²⋅xy
    mat_gen[6] = zeros(dim,dim)
    mat_gen[6][3,5] = 1/2
    mat_gen[6][5,3] = 1/2
    mat_gen[6][2,6] = -1/2
    mat_gen[6][6,2] = -1/2
    # define the quadratic ideal
    ideal_ternary_quartics = LowRankSOS.QuadraticIdeal(dim, mat_gen)
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_ternary_quartics)
    # choose a target quadratic form corresponding to x⁴+x²y²+y⁴+z⁴
    mat_target = zeros(dim,dim)
    mat_target[1,1] = 1
    mat_target[2,2] = 1
    mat_target[4,4] = 1
    mat_target[6,6] = 1
    # set the number of squares and starting point corresponding to the tuple (x²,xy,y²)
    num_square = 3
    mat_start = [1.0 0.0 0.0 0.0 0.0 0.0;
                 0.0 0.9 0.0 0.0 0.0 0.0;
                 0.0 0.0 0.0 1.0 0.1 0.0]
    # test the algorithms
    println("Start the fiber-escape algorithm on the ternary quartics...")
    mat_solution = LowRankSOS.solve_gradient_method_with_escapes(num_square, mat_target, map_quotient, ideal_ternary_quartics, 
                                                                 mat_linear_forms=mat_start, 
                                                                 str_line_search="interpolation", 
                                                                 lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_solution'*mat_solution-mat_target, map_quotient))

    return true # if the algorithm terminates successfully
end

@test test_fiber_escape()

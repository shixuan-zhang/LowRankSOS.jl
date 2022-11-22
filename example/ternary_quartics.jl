# example of ternary quartics (embedded as a projective variety)

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra

function define_ternary_quartics_ideal()
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
    return mat_gen
end

function test_ternary_quartics()
    println("\n\nStart the test of low-rank sum-of-squares certification on the ternary quartics...")
    # define a quadratic ideal corresponding to ternary quartics
    dim = 6
    ideal_ternary_quartics = LowRankSOS.QuadraticIdeal(dim, define_ternary_quartics_ideal())
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_ternary_quartics)
    # set the target rank
    rank = 3
    println("The dimension of the problem is ", dim, ", and the sought rank is ", rank)
    # choose a target quadratic form corresponding to x⁴+x²y²+y⁴+z⁴
    mat_aux = randn(dim, dim)
    mat_target = LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_aux' * mat_aux))
    mat_target = zeros(dim,dim)
    mat_target[1,1] = 1
    mat_target[2,2] = 1
    mat_target[4,4] = 1
    mat_target[6,6] = 1
    # choose a starting point
    mat_start = randn(rank, dim)
    mat_start = [1.0 0.0 0.0 0.0 0.0 0.0;
                 0.0 2.0 0.0 0.0 0.0 0.0;
                 0.0 -.1 -.1 3.0 0.0 0.0]
    println("The projected norm of the difference between the initial solution and the target is ", 
            LowRankSOS.compute_norm_proj(mat_start'*mat_start-mat_target, map_quotient))
    # add timer for profiling
    time_start = time()
    # solve the nonlinear optimization model
    mat_nonlinear = LowRankSOS.solve_nonlinear_model(rank, mat_target, ideal_ternary_quartics, mat_linear_forms=mat_start)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_nonlinear'*mat_nonlinear-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the semidefinite optimization model
    mat_semidefinite = LowRankSOS.solve_semidefinite_model(mat_target, ideal_ternary_quartics)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_semidefinite-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the gradient method with backtracking line search
    #=
    mat_grad_backtracking = LowRankSOS.solve_gradient_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, str_line_search="backtracking", lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_grad_backtracking'*mat_grad_backtracking-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)
    =#

    # solve the gradient method with interpolation line search
    mat_grad_interpolation = LowRankSOS.solve_gradient_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, str_line_search="interpolation", lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_grad_interpolation'*mat_grad_interpolation-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the pushforward direction method with interpolation line search
    mat_push_interpolation = LowRankSOS.solve_push_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_push_interpolation'*mat_push_interpolation-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # check if the gradient method returns the same solution as the nonlinear solver
    println()
    println("The distance between the solver-returned and our Gram matrices is ", LinearAlgebra.norm(mat_grad_interpolation'*mat_push_interpolation - mat_semidefinite))
    println()

end

test_ternary_quartics()

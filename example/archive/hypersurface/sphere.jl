# example on spheres

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra

function test_sphere(
        dim::Int
    )
    println("\n\nStart the test of low-rank sum-of-squares certification on a sphere...")
    # define a sphere quadratic ideal
    mat_sphere = diagm([-1.0; ones(dim-1)])
    ideal_sphere = LowRankSOS.QuadraticIdeal(dim, [mat_sphere])
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_sphere)
    # generate a target quadratic form randomly
    mat_aux= randn(dim-1, dim)
    mat_target = mat_aux' * mat_aux
    # set the target rank
    rank = dim-1
    println("The dimension of the problem is ", dim, ", and the sought rank is ", rank)
    # choose a starting point
    mat_start = randn(rank, dim)
    println("The projected norm of the difference between the initial solution and the target is ", 
            LowRankSOS.compute_norm_proj(mat_start'*mat_start-mat_target, map_quotient))
    # add timer for profiling
    time_start = time()

    # solve the nonlinear optimization model
    mat_nonlinear = LowRankSOS.solve_nonlinear_model(rank, mat_target, ideal_sphere, mat_linear_forms=mat_start)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_nonlinear'*mat_nonlinear-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the semidefinite optimization model
    mat_semidefinite = LowRankSOS.solve_semidefinite_model(mat_target, ideal_sphere)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_semidefinite-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the gradient method with backtracking line search
    mat_grad_backtracking = LowRankSOS.solve_gradient_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, str_line_search="backtracking", lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_grad_backtracking'*mat_grad_backtracking-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)
    #
    # solve the gradient method with interpolation line search
    mat_grad_interpolation = LowRankSOS.solve_gradient_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, str_line_search="interpolation", lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_grad_interpolation'*mat_grad_interpolation-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the pushforward direction method with interpolation line search
    mat_push_interpolation = LowRankSOS.solve_push_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_push_interpolation'*mat_push_interpolation-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

end

test_sphere(40)

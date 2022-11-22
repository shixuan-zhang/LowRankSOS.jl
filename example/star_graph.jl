# example of varieties associated with star graphs

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra

function define_star_graph_ideal(
        dim::Int
    )
    # calculate the number of non-edges
    num_non_edge = (dim-1)*(dim-2) ÷ 2
    # initialize the output
    vec_mat_gen = Vector{Matrix{Float64}}(undef, num_non_edge)
    for i =1:num_non_edge
        vec_mat_gen[i] = zeros(dim,dim)
    end
    # loop over all non-edges
    for i = 1:(dim-1)
        for j = 1:(i-1)
            idx_mat = j + (i-1)*(i-2) ÷ 2
            vec_mat_gen[idx_mat][i,j] = 1
            vec_mat_gen[idx_mat][j,i] = 1
        end
    end
    return vec_mat_gen
end

function test_star_graph(
        dim::Int
    )
    println("\n\nStart the test of low-rank sum-of-squares certification on the variety corresponding to a star graph...")
    # define a quadratic ideal corresponding to the star graph
    ideal_star = LowRankSOS.QuadraticIdeal(dim, define_star_graph_ideal(dim))
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_star)
    # set the target rank
    rank = 2 # 1+ceil(Int, sqrt(2*dim))
    println("The dimension of the problem is ", dim, ", and the sought rank is ", rank)
    # generate a target quadratic form randomly
    mat_aux = randn(dim, dim)
    mat_target = LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_aux' * mat_aux))
    # choose a starting point
    mat_start = randn(rank, dim)
    println("The projected norm of the difference between the initial solution and the target is ", 
            LowRankSOS.compute_norm_proj(mat_start'*mat_start-mat_target, map_quotient))
    # add timer for profiling
    time_start = time()
    # solve the nonlinear optimization model
    mat_nonlinear = LowRankSOS.solve_nonlinear_model(rank, mat_target, ideal_star, mat_linear_forms=mat_start)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_nonlinear'*mat_nonlinear-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the semidefinite optimization model
    mat_semidefinite = LowRankSOS.solve_semidefinite_model(mat_target, ideal_star)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_semidefinite-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the gradient method with backtracking line search
    mat_grad_backtracking = LowRankSOS.solve_gradient_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, str_line_search="backtracking", lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_grad_backtracking'*mat_grad_backtracking-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

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


test_star_graph(50)

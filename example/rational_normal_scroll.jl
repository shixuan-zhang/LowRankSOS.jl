# example of rational normal scrolls

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra

function define_scroll_ideal(
        vec_deg::Vector{Int}
    )
    # calculate the dimension of the ambient space
    num_col = sum(vec_deg)
    num_block = length(vec_deg)
    dim = num_col + num_block
    # construct the array of starting indices of each block
    idx_block = [0; cumsum(sort(vec_deg))]
    # initialize the output
    vec_mat_gen = Matrix{Float64}[]
    # loop over all possible first column
    for i = 1:(num_col-1)
        # get the block index for the first column
        ib = findfirst(k->(k>=i), idx_block) - 1
        ic = i + (ib-1)
        # loop over all possible second column
        for j = (i+1):num_col
            # get the block index for the second column
            jb = findfirst(k->(k>=j), idx_block) - 1
            jc = j + (jb-1)
            # create a temporary matrix to fill the entries
            mat_temp = zeros(dim,dim)
            mat_temp[ic,jc+1] += 1/2
            mat_temp[jc+1,ic] += 1/2
            mat_temp[ic+1,jc] -= 1/2
            mat_temp[jc,ic+1] -= 1/2
            # store the matrix
            push!(vec_mat_gen, mat_temp)
        end
    end
    return vec_mat_gen
end


function test_rational_normal_scroll(
        vec_deg::Vector{Int}
    )
    println("\n\nStart the test of low-rank sum-of-squares certification on a rational normal scroll of block sizes ", vec_deg, "...")
    # define a quadratic ideal corresponding to the rational normal scroll
    dim = sum(vec_deg) + length(vec_deg)
    ideal_scroll = LowRankSOS.QuadraticIdeal(dim, define_scroll_ideal(vec_deg))
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_scroll)
    # set the target rank
    rank = length(vec_deg) + 1
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
    mat_nonlinear = LowRankSOS.solve_nonlinear_model(rank, mat_target, ideal_scroll, mat_linear_forms=mat_start)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_nonlinear'*mat_nonlinear-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the semidefinite optimization model
    mat_semidefinite = LowRankSOS.solve_semidefinite_model(mat_target, ideal_scroll)
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

test_rational_normal_scroll([2,3,4,5,6])

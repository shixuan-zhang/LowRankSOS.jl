# example of rational normal scrolls

include("../../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra, ForwardDiff

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


function compare_methods_on_scroll(
        vec_deg::Vector{Int};
        mat_start::Matrix{Float64} = zeros(0,0),
        mat_target::Matrix{Float64} = zeros(0,0),
        num_max_iter::Int = LowRankSOS.NUM_MAX_ITER
    )
    println("\n\nStart the comparison of low-rank sum-of-squares certification methods",
            " on a rational normal scroll of block sizes ", vec_deg, "...")
    # define a quadratic ideal corresponding to the rational normal scroll
    dim = sum(vec_deg) + length(vec_deg)
    ideal_scroll = LowRankSOS.QuadraticIdeal(dim, define_scroll_ideal(vec_deg))
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_scroll)
    # set the target rank
    rank = length(vec_deg) + 1
    println("The dimension of the problem is ", dim, ", and the sought rank is ", rank)
    # check if the starting linear forms and the target quadratic form are supplied
    if size(mat_start) != (rank,dim)
        # choose a starting point
        println("The starting linear forms are chosen randomly...")
        mat_start = randn(rank, dim)
    end
    if size(mat_target) != (dim,dim)
        # generate a target quadratic form randomly
        println("The target quadratic form is chosen randomly...")
        mat_aux = randn(dim, dim)
        mat_target = mat_aux' * mat_aux
    end
    mat_target = LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_target))
    println("The projected norm of the difference between the initial solution and the target is ", 
            LowRankSOS.compute_norm_proj(mat_start'*mat_start-mat_target, map_quotient))
    # define the anonymous objective function for Hessian computation
    func_obj_val = (mat_temp)->LinearAlgebra.norm(LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_target - mat_temp'*mat_temp), dim=dim))^2
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

    # solve the pushforward direction method with interpolation line search
    mat_push_interpolation = LowRankSOS.solve_push_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, num_max_iter=num_max_iter, lev_print=1)
    val_norm_push = LowRankSOS.compute_norm_proj(mat_push_interpolation'*mat_push_interpolation-mat_target, map_quotient)
    println("The projected norm of the residue is ", val_norm_push)
    if val_norm_push > LowRankSOS.VAL_TOL
        println("Spurious stationary point encountered at the linear forms ", round.(mat_push_interpolation, digits=LowRankSOS.NUM_DIG))
        mat_Hessian_temp = ForwardDiff.hessian(func_obj_val, mat_push_interpolation)
        println("The smallest eigenvalue of the Hessian is ", LinearAlgebra.eigmin(mat_Hessian_temp + mat_Hessian_temp'))
    end
    println("The total elapsed time is ", time() - time_start)

    # solve the gradient method with interpolation line search
    mat_grad_interpolation = LowRankSOS.solve_gradient_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, str_line_search="interpolation", num_max_iter=num_max_iter,  lev_print=1)
    val_norm_grad = LowRankSOS.compute_norm_proj(mat_grad_interpolation'*mat_grad_interpolation-mat_target, map_quotient)
    println("The projected norm of the residue is ", val_norm_grad)
    if val_norm_grad > LowRankSOS.VAL_TOL
        println("Spurious stationary point encountered at the linear forms ", round.(mat_grad_interpolation, digits=LowRankSOS.NUM_DIG))
        mat_Hessian_temp = ForwardDiff.hessian(func_obj_val, mat_grad_interpolation)
        println("The smallest eigenvalue of the Hessian is ", LinearAlgebra.eigmin(mat_Hessian_temp + mat_Hessian_temp'))
    end
    println("The total elapsed time is ", time() - time_start)

    # solve the gradient method with fiber movement to escape stationary points
    mat_grad_fiber = LowRankSOS.solve_gradient_method_with_escapes(rank, mat_target, map_quotient, ideal_scroll, mat_linear_forms=mat_start, str_line_search="interpolation", num_max_iter=num_max_iter,  lev_print=1)
    val_norm_fiber = LowRankSOS.compute_norm_proj(mat_grad_fiber'*mat_grad_fiber-mat_target, map_quotient)
    println("The projected norm of the residue is ", val_norm_fiber)
    if val_norm_fiber > LowRankSOS.VAL_TOL
        println("Spurious stationary point encountered at the linear forms ", round.(mat_grad_fiber, digits=LowRankSOS.NUM_DIG))
        mat_Hessian_temp = ForwardDiff.hessian(func_obj_val, mat_grad_fiber)
        println("The smallest eigenvalue of the Hessian is ", LinearAlgebra.eigmin(mat_Hessian_temp + mat_Hessian_temp'))
    end
    println("The total elapsed time is ", time() - time_start)
    println()
end

function test_batch_on_scroll(
        vec_deg::Vector{Int};
        mat_target::Matrix{Float64} = zeros(0,0),
        str_method::String = "gradient",
        num_square::Int = 0,
        num_sample::Int = 100,
        num_max_iter::Int = LowRankSOS.NUM_MAX_ITER
    )
    println("\n\nStart the batch experiment of low-rank sum-of-squares certification using the ",
            str_method, " method on a rational normal scroll of block sizes ", vec_deg, "...")
    # define a quadratic ideal corresponding to the rational normal scroll
    dim = sum(vec_deg) + length(vec_deg)
    ideal_scroll = LowRankSOS.QuadraticIdeal(dim, define_scroll_ideal(vec_deg))
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_scroll)
    # set the target rank
    rank = min(num_square, dim)
    if rank <= 0
        rank = length(vec_deg) + 1
    end
    println("The dimension of the problem is ", dim, ", and the sought rank is ", rank)
    # check if the starting linear forms and the target quadratic form are supplied
    if size(mat_target) != (dim,dim)
        # generate a target quadratic form randomly
        println("The target quadratic form is chosen randomly...")
        mat_aux = randn(dim, dim)
        mat_target = mat_aux' * mat_aux
    end
    mat_target = LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_target))
    # define the anonymous objective function for Hessian computation
    func_obj_val = (mat_temp)->LinearAlgebra.norm(LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_target - mat_temp'*mat_temp), dim=dim))^2

    # prepare the arrays for output storage
    vec_runtime = fill(NaN, num_sample)
    vec_residue = fill(NaN, num_sample)
    ctr_residue = 0
    vec_mineig = fill(NaN, num_sample)
    # start the main loop of experiments
    for idx_sample = 1:num_sample
        # choose randomly a starting point
        mat_start = randn(rank, dim)
        # initialize the matrix of linear forms
        mat_linear_forms = zeros(rank, dim)
        # add timer for profiling
        time_start = time()
        # select the method based on the input string
        if str_method == "pushforward"
            mat_linear_forms = LowRankSOS.solve_push_method(rank, mat_target, map_quotient, 
                                                            mat_linear_forms=mat_start, 
                                                            num_max_iter=num_max_iter,
                                                            lev_print=-1)
        elseif str_method == "gradient"
            mat_linear_forms = LowRankSOS.solve_gradient_method(rank, mat_target, map_quotient, 
                                                                mat_linear_forms=mat_start, 
                                                                str_line_search="interpolation",
                                                                num_max_iter=num_max_iter,
                                                                lev_print=-1)
        elseif str_method == "gradient+fiber"
            mat_linear_forms = LowRankSOS.solve_gradient_method_with_escapes(rank, mat_target, map_quotient, ideal_scroll,
                                                                             mat_linear_forms=mat_start, 
                                                                             str_line_search="interpolation",
                                                                             num_max_iter=num_max_iter,
                                                                             lev_print=-1)
        else
            error("Unsupported optimization method for sum of squares certification!")
        end
        # save the computation time
        time_finish = time() - time_start
        vec_runtime[idx_sample] = time_finish
        # compute the residual norm
        val_residue = LowRankSOS.compute_norm_proj(mat_linear_forms'*mat_linear_forms-mat_target, map_quotient)
        vec_residue[idx_sample] = val_residue
        # check the second order stationary criterion for nonzero residues
        if val_residue > LowRankSOS.VAL_TOL
            ctr_residue += 1
            mat_Hessian_temp = ForwardDiff.hessian(func_obj_val, mat_linear_forms)
            vec_mineig[idx_sample] = LinearAlgebra.eigmin(mat_Hessian_temp + mat_Hessian_temp')
        end
    end
    # print some statistics about the result
    println("There is a nonzero residue in ", ctr_residue, " out of ", num_sample, " cases, ",
            "among which ", sum(vec_mineig.<0.0), " are not second order stationary points.")           
    println("The maximum computation time is ", maximum(vec_runtime),
            " and the average is ", sum(vec_runtime)/num_sample)
end

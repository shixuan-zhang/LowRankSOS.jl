# example of varieties associated with star graphs

include("../../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra, ForwardDiff

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
        dim::Int;
        mat_start::Matrix{Float64} = zeros(0,0),
        mat_target::Matrix{Float64} = zeros(0,0)
    )
    println("\n\nStart the test of low-rank sum-of-squares certification on the variety corresponding to a star graph...")
    # define a quadratic ideal corresponding to the star graph
    ideal_star = LowRankSOS.QuadraticIdeal(dim, define_star_graph_ideal(dim))
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_star)

    # set the target rank
    rank = 2 
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
    # define the anonymous objective function for Hessian computation
    func_obj_val = (mat_temp)->LinearAlgebra.norm(LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_target - mat_temp'*mat_temp), dim=dim))^2
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
    val_norm_grad = LowRankSOS.compute_norm_proj(mat_grad_interpolation'*mat_grad_interpolation-mat_target, map_quotient)
    println("The projected norm of the residue is ", val_norm_grad)
    if val_norm_grad > LowRankSOS.VAL_TOL
        println("Spurious stationary point encountered at the linear forms ", round.(mat_grad_interpolation, digits=LowRankSOS.NUM_DIG))
        println("The eigenvalues of the Hessian is ", LinearAlgebra.eigen(ForwardDiff.hessian(func_obj_val, mat_grad_interpolation)).values)
        println("The smallest eigenvalue of the Hessian is ", LinearAlgebra.eigmin(ForwardDiff.hessian(func_obj_val, mat_grad_interpolation)))
    end
    println("The total elapsed time is ", time() - time_start)

    # solve the pushforward direction method with interpolation line search
    mat_push_interpolation = LowRankSOS.solve_push_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_push_interpolation'*mat_push_interpolation-mat_target, map_quotient))
    val_norm_push = LowRankSOS.compute_norm_proj(mat_push_interpolation'*mat_push_interpolation-mat_target, map_quotient)
    println("The projected norm of the residue is ", val_norm_push)
    if val_norm_push > LowRankSOS.VAL_TOL
        println("Spurious stationary point encountered at the linear forms ", round.(mat_push_interpolation, digits=LowRankSOS.NUM_DIG))
        println("The eigenvalues of the Hessian is ", LinearAlgebra.eigen(ForwardDiff.hessian(func_obj_val, mat_push_interpolation)).values)
        println("The smallest eigenvalue of the Hessian is ", LinearAlgebra.eigmin(ForwardDiff.hessian(func_obj_val, mat_push_interpolation)))
    end
    println("The total elapsed time is ", time() - time_start)
    println()
end

function test_batch_on_star_graph(
        dim::Int;
        mat_target::Matrix{Float64} = zeros(0,0),
        str_method::String = "gradient",
        num_square::Int = 0,
        num_sample::Int = 100,
        num_max_iter::Int = LowRankSOS.NUM_MAX_ITER,
        val_tol_res::Float64 = sqrt(LowRankSOS.VAL_TOL)
    )
    println("\n\nStart the batch experiment of low-rank sum-of-squares certification using the ",
            str_method, " method on the star graph variety...")
    # define a quadratic ideal corresponding to the star graph variety
    ideal_star_graph = LowRankSOS.QuadraticIdeal(dim, define_star_graph_ideal(dim))
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_star_graph)
    # set the target rank
    rank = min(num_square, dim)
    if rank <= 0
        rank = 2
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
    vec_min_eigenval = fill(NaN, num_sample)
    vec_norm_grad = fill(NaN, num_sample)
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
            mat_linear_forms = LowRankSOS.solve_gradient_method_with_escapes(rank, mat_target, map_quotient, ideal_star_graph,
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
        if val_residue > val_tol_res
            ctr_residue += 1
            vec_gradient_temp = ForwardDiff.gradient(func_obj_val, mat_linear_forms)
            vec_norm_grad[idx_sample] = LinearAlgebra.norm(vec_gradient_temp)
            if vec_norm_grad[idx_sample] < LowRankSOS.VAL_TOL
                mat_Hessian_temp = ForwardDiff.hessian(func_obj_val, mat_linear_forms)
                vec_min_eigenval[idx_sample] = LinearAlgebra.eigmin(mat_Hessian_temp + mat_Hessian_temp')
            end
        end
    end
    # print some statistics about the result
    println("The results of ", num_sample, " independent experiments with random starting points are summarized below:")
    println("  Number of significant residues:      ", ctr_residue)
    println("  Number of nonvanishing gradients:    ", sum(vec_norm_grad.>LowRankSOS.VAL_TOL))
    println("  Number of indefinite Hessians:       ", sum(vec_min_eigenval.<-LowRankSOS.VAL_TOL))
    println("  Average computation time in seconds: ", sum(vec_runtime)/num_sample)
end

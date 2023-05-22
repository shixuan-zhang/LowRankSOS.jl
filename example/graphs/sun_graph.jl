# example of varieties associated with sun graphs

include("../../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra, ForwardDiff

function define_sun_graph_ideal(
        cycle::Int,
        clique::Int = 3
    )
    # set the dimension
    dim = cycle * (clique - 1)
    # calculate the number of non-edges
    num_non_edge = dim*(dim-1) ÷ 2 - (clique-1)*clique ÷ 2 * cycle
    # initialize the output
    vec_mat_gen = Vector{Matrix{Float64}}(undef, num_non_edge)
    for i =1:num_non_edge
        vec_mat_gen[i] = zeros(dim,dim)
    end
    # loop over all non-edges
    idx_mat = 1
    # loop over all other vertices paired with the first vertex
    let i = 1
        for j = (clique+1):(dim-clique+1)
            vec_mat_gen[idx_mat][i,j] = 1
            vec_mat_gen[idx_mat][j,i] = 1
            idx_mat += 1
        end
    end
    # loop over all other vertex pairs
    for i = 2:dim
        idx_clique = div(i-1,clique-1)+1
        for j = (idx_clique*(clique-1)+2):dim
            vec_mat_gen[idx_mat][i,j] = 1
            vec_mat_gen[idx_mat][j,i] = 1
            idx_mat += 1
        end
    end
    # check if the total number of matrices matches the calculation
    if idx_mat != num_non_edge+1
        println("The estimated number of non-edges is ", num_non_edge)
        println("The looped number of non-edges is ", idx_mat-1)
        error("ERROR: incorrect number of non-edges for the sun graph!")
    end
    return vec_mat_gen
end

function test_sun_graph(
        cycle::Int,
        clique::Int = 3;
        mat_start::Matrix{Float64} = zeros(0,0),
        mat_target::Matrix{Float64} = zeros(0,0)
    )
    println("\n\nStart the test of low-rank sum-of-squares certification on the variety corresponding to a sun graph")
    println("with cycle size ", cycle, " and clique sizes ", clique, " ...")
    # set the dimension
    dim = cycle * (clique - 1)
    # define a quadratic ideal corresponding to the sun graph
    ideal_sun_graph = LowRankSOS.QuadraticIdeal(dim, define_sun_graph_ideal(cycle,clique))
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_sun_graph)

    # set the target rank
    rank = max(dim+3-cycle, clique)
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
    mat_nonlinear = LowRankSOS.solve_nonlinear_model(rank, mat_target, ideal_sun_graph, mat_linear_forms=mat_start)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_nonlinear'*mat_nonlinear-mat_target, map_quotient))
    println("The total elapsed time is ", time() - time_start)

    # solve the semidefinite optimization model
    mat_semidefinite = LowRankSOS.solve_semidefinite_model(mat_target, ideal_sun_graph)
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
        mat_Hessian_temp = ForwardDiff.hessian(func_obj_val, mat_grad_interpolation)
        println("The smallest eigenvalue of the Hessian is ", LinearAlgebra.eigmin(mat_Hessian_temp + mat_Hessian_temp'))
    end
    println("The total elapsed time is ", time() - time_start)
    
    # solve the limited memory quasi-second-order method with interpolation line search
    mat_limited_memory = LowRankSOS.solve_limited_memory_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, str_line_search="backtracking", lev_print=1)
    val_norm_grad = LowRankSOS.compute_norm_proj(mat_limited_memory'*mat_limited_memory-mat_target, map_quotient)
    println("The projected norm of the residue is ", val_norm_grad)
    if val_norm_grad > LowRankSOS.VAL_TOL
        println("Spurious stationary point encountered at the linear forms ", round.(mat_limited_memory, digits=LowRankSOS.NUM_DIG))
        mat_Hessian_temp = ForwardDiff.hessian(func_obj_val, mat_limited_memory)
        println("The smallest eigenvalue of the Hessian is ", LinearAlgebra.eigmin(mat_Hessian_temp + mat_Hessian_temp'))
    end
    println("The total elapsed time is ", time() - time_start)
    
    # solve the pushforward direction method with interpolation line search
    mat_push_interpolation = LowRankSOS.solve_push_method(rank, mat_target, map_quotient, mat_linear_forms=mat_start, lev_print=1)
    println("The projected norm of the residue is ", LowRankSOS.compute_norm_proj(mat_push_interpolation'*mat_push_interpolation-mat_target, map_quotient))
    val_norm_push = LowRankSOS.compute_norm_proj(mat_push_interpolation'*mat_push_interpolation-mat_target, map_quotient)
    println("The projected norm of the residue is ", val_norm_push)
    if val_norm_push > LowRankSOS.VAL_TOL
        println("Spurious stationary point encountered at the linear forms ", round.(mat_push_interpolation, digits=LowRankSOS.NUM_DIG))
        mat_Hessian_temp = ForwardDiff.hessian(func_obj_val, mat_push_interpolation)
        println("The smallest eigenvalue of the Hessian is ", LinearAlgebra.eigmin(mat_Hessian_temp + mat_Hessian_temp'))
    end
    println("The total elapsed time is ", time() - time_start)
    println()
end

function test_batch_on_sun_graph(
        cycle::Int,
        clique::Int = 3;
        mat_target::Matrix{Float64} = zeros(0,0),
        str_method::String = "gradient",
        num_square::Int = 0,
        num_sample::Int = 100,
        num_max_iter::Int = LowRankSOS.NUM_MAX_ITER,
        val_tol_res::Float64 = sqrt(LowRankSOS.VAL_TOL)
    )
    println("\n\nStart the batch experiment of low-rank sum-of-squares certification using the ",
            str_method, " method on the sun graph variety...")
    # set the dimension
    dim = cycle * (clique - 1)
    # define a quadratic ideal corresponding to the sun graph variety
    ideal_sun_graph = LowRankSOS.QuadraticIdeal(dim, define_sun_graph_ideal(cycle,clique))
    # get the orthogonal projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_sun_graph)
    # set the target rank
    rank = min(num_square, dim)
    if rank <= 0
        rank = max(dim+3-cycle, clique)
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
        elseif str_method == "quasiNewton"
            mat_linear_forms = LowRankSOS.solve_limited_memory_method(rank, mat_target, map_quotient, 
                                                                      mat_linear_forms=mat_start, 
                                                                      str_line_search="backtracking",
                                                                      num_max_iter=num_max_iter,
                                                                      lev_print=-1)
        elseif str_method == "gradient+fiber"
            mat_linear_forms = LowRankSOS.solve_gradient_method_with_escapes(rank, mat_target, map_quotient, ideal_sun_graph,
                                                                             mat_linear_forms=mat_start, 
                                                                             str_line_search="interpolation",
                                                                             num_max_iter=num_max_iter,
                                                                             lev_print=-1)
        elseif str_method == "gradient+bypass"
            mat_linear_forms = LowRankSOS.solve_gradient_method_with_penalty(rank, mat_target, map_quotient, 
                                                                             mat_linear_forms=mat_start, 
                                                                             str_line_search="interpolation",
                                                                             num_max_iter=num_max_iter,
                                                                             lev_print=-1)
        elseif str_method == "pushforward+bypass"
            mat_linear_forms = LowRankSOS.solve_push_method_with_penalty(rank, mat_target, map_quotient, 
                                                                         mat_linear_forms=mat_start, 
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

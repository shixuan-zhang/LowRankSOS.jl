function exec_multiple(
        coord_ring::CoordinateRing2,
        set_num_sq::Vector{Int};
        solver_comp::Vector{String} = String[]
    )
    # record the main experiment results
    result_success = Dict{Int,Vector{Int}}()
    result_seconds = Dict{Int,Vector{Float64}}()
    result_residue = Dict{Int,Vector{Float64}}()
    for num in set_num_sq
        result_success[num] = zeros(Int,num_repeat)
        result_seconds[num] = zeros(num_repeat)
        result_residue[num] = zeros(num_repeat)
    end
    # record the comparison(s) against semidefinite programming
    num_compare = length(solver_comp)
    compare_time = Dict(name => Float64[] for name in solver_comp)
    compare_rank = Dict(name => Float64[] for name in solver_comp)
    for name in solver_comp
        compare_time[name] = zeros(num_repeat)
        compare_rank[name] = zeros(Int,num_repeat)
    end
    for idx in 1:num_repeat
        println("\n" * "="^80)
        # choose randomly a target
        tuple_random = rand(coord_ring.dim1*coord_ring.dim1)
        target_sos = get_sos(tuple_random, coord_ring)
        # rescale the target to avoid numerical issues
        target_sos ./= norm(target_sos)
        # find the minimum number of squares
        num_sq_min = minimum(set_num_sq)
        # choose randomly a starting point
        tuple_min_sq = rand(num_sq_min*coord_ring.dim1)
        # loop over different numbers of squares
        for num_square in set_num_sq
            println("Use ", num_square, " squares for the local optimization method...")
            # embed the starting tuple of linear forms
            tuple_start = embed_tuple(tuple_min_sq, num_sq_min, num_square, random=true)
            # rescale the starting point to avoid numerical issues
            tuple_start ./= norm(tuple_start)
            # solve the problem and record the time
            time_start = time()
            vec_sol, val_res, flag_conv = call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, num_max_eval=coord_ring.dim1*REL_MAX_ITER, val_threshold=VAL_TOL, print_level=1) 
            time_end = time()
            # check the optimal value
            if val_res < VAL_TOL
                result_success[num_square][idx] = 1
                result_seconds[num_square][idx] = time_end-time_start
            else
                result_seconds[num_square][idx] = NaN
                if flag_conv
                    result_success[num_square][idx] = -1
                    result_residue[num_square][idx] = val_res / max(1.0, norm(target_sos))
                    # check the optimality conditions
                    vec_sos = get_sos(vec_sol, coord_ring)
                    mat_Jac = build_Jac_mat(vec_sol, coord_ring)
                    vec_grad = 2*mat_Jac'*(vec_sos-target_sos)
                    mat_Hess = build_Hess_mat(num_square, vec_sol, target_sos, coord_ring)
                    printfmtln("Stationary point encountered with grad norm = {:<10.4e} and the min Hessian eigenval = {:<10.4e}",
                               norm(vec_grad), minimum(eigen(mat_Hess).values))
                    # start the adaptive moves along a direct path connecting the quadrics
                    println("Re-solve the problem using the direct path method...")
                    vec_sol, val_res = move_direct_path(num_square, target_sos, coord_ring, 
                                                        tuple_linear_forms=tuple_start, 
                                                        str_descent_method="lBFGS-NLopt", 
                                                        print_level=1, 
                                                        val_threshold=VAL_TOL)
                    vec_sos = get_sos(vec_sol, coord_ring)
                    if norm(vec_sos-target_sos) < VAL_TOL
                        result_success[num_square][idx] = 2
                        result_residue[num_square][idx] = 0.0
                    end
                end
            end
        end
        # compare against the standard semidefinite programming solver(s)
        for solver_name in solver_comp
            flush(stdout)
            println()
            println("Run for comparison " * solver_name * " solver with up to ", coord_ring.dim1, " squares...")
            time_start = time()
            vec_sol, val_res, flag_conv, sol_rank = call_JuMP_solver(target_sos, coord_ring, val_threshold=VAL_TOL, print_level=1, solver_name=solver_name)
            time_end = time()
            if flag_conv
                compare_time[solver_name][idx] = time_end - time_start
                compare_rank[solver_name][idx] = sol_rank
            else
                compare_time[solver_name][idx] = NaN
                compare_rank[solver_name][idx] = -1
            end
        end
    end
    for num in set_num_sq
        println("\nResult summary for ", num, " squares:")
        println("Global optima are found in ", count(x->x>0, result_success[num]), " out of ", num_repeat, " experiment runs.")
        println("Restricted-path method is used in ", count(x->x>1, result_success[num]), " experiment runs.")
        println("The average wall clock time for test runs is ", mean(filter(!isnan, result_seconds[num])), " seconds.")
    end
    # return the arrays of successful runs and the average time for batch experiments
    SUCC = [count(x->x>0, result_success[num]) for num in set_num_sq]
    FAIL = [count(x->x<0, result_success[num]) for num in set_num_sq]
    TIME = [mean(filter(!isnan, result_seconds[num])) for num in set_num_sq]
    DIST = [maximum(result_residue[num]) for num in set_num_sq]
    SDPTIME = Dict([name => [mean(compare_time[name]) for _ in set_num_sq] for name in solver_comp])
    SDPRANK_MIN = Dict([name => [minimum(filter(x->x>0,compare_rank[name])) for _ in set_num_sq] for name in solver_comp])
    SDPRANK_MED = Dict([name => [median(filter(x->x>0,compare_rank[name])) for _ in set_num_sq] for name in solver_comp])
    return SUCC, FAIL, TIME, DIST, SDPTIME, SDPRANK_MIN, SDPRANK_MED
end
